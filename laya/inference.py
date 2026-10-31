"""Shared validation and scan provenance for the Torch and ONNX runtimes."""
import copy
import json
from typing import Any, Dict

from .hooks import _as_sequence
from .inference_common import QTYPES, _resolve_noul_labels, state_room


def _option_count(qdef: Dict) -> int:
    """How many options a validated question definition renders to.

    Mirrors `render_options`: a choice has one option per criterion, a score one per level,
    and a noul is always the pair [false, true].
    """
    if qdef.get("type") == "noul":
        return 2
    crit = qdef.get("criteria")
    return len(crit) if isinstance(crit, (dict, list, tuple)) else 0


def _scan_answer_values(answer):
    # Reliability/action annotations do not change which prediction the model made. Keep the
    # prediction fields independent of in-place edits to probabilities or a score's legend.
    return {key: answer[key] for key in ("type", "choice", "noul", "score", "legend",
                                         "probabilities", "confidence", "answer_confidence")
            if key in answer}


def _scan_answer_index(evidence, qid, answer, fallback):
    recorded = evidence["answers"]
    if recorded is None:
        return fallback                         # a replacement batch never dispatched hooks
    record = recorded.get((qid, id(answer)))
    if record is not None and record[0] is answer and record[2] == _scan_answer_values(answer):
        return record[1]
    return None                                 # replaced, edited, or shared across windows


def _start_evidence(track_answers=False):
    """A recorder for `predict_long`: what the start-hook chain left for inference to run on.

    Why a recorder at all: `ctx.skip()` only assigns `ctx.results` (see `laya.hooks.PredictContext`),
    so outside `predict_batch` the only account of what the hooks did is the context they were
    handed. The result count cannot tell a hook that answered the document from one that replaced
    the window list -- both return fewer results than there were windows, and the two need opposite
    readings: the first scored no window, the second scored the ones it left behind and booked
    tokens for them.

    The probe belongs last in the chain because `compose_hooks` orders defaults, then installed,
    then per-call hooks, and `dispatch` calls them in order, so a probe appended after the caller's
    own start hooks observes exactly what `predict_batch` is about to act on.

    Returns the probe and the dict it fills: `answered` is whether a hook replaced the call before
    inference, `states` is a snapshot of the states that reached it (`None` if the probe never ran,
    which means `predict_batch` was replaced and no hook chain was dispatched).
    """
    evidence = {"answered": False, "states": None, "question_types": None, "answers": None,
                "max_len": None, "head_max_len": None, "questions": None}

    def remember(recorded, qid, answer, index, values):
        key = (qid, id(answer))
        # Keep the object alive as well as its id. One object emitted for more than
        # one window has no unique origin, even when all prediction fields agree.
        recorded[key] = None if key in recorded else (answer, index, copy.deepcopy(values))

    def record_results(ctx):
        recorded = {}
        for index, result in enumerate(ctx.results):
            for qid, answer in result["answers"].items():
                remember(recorded, qid, answer, index, _scan_answer_values(answer))
        evidence["answers"] = recorded

    def record_cache_results(results, computed, replayed, dupes):
        # A partial DecisionCache call scores only misses. Its end hook restores full-window
        # order, so translate miss-local origins and record retained answers before later hooks.
        recorded = evidence["answers"]
        if recorded is None:
            return
        for key, record in list(recorded.items()):
            if record is not None:
                recorded[key] = (record[0], computed[record[1]], record[2])
        for index in replayed:
            for qid, answer in results[index]["answers"].items():
                remember(recorded, qid, answer, index, _scan_answer_values(answer))
        for index, source in dupes.items():
            for qid, answer in results[index]["answers"].items():
                original = results[source]["answers"].get(qid)
                record = recorded.get((qid, id(original)))
                if (record is not None and record[0] is original
                        and record[2] == _scan_answer_values(original) == _scan_answer_values(answer)):
                    remember(recorded, qid, answer, index, record[2])

    def probe(ctx):
        evidence["answered"] = ctx.results is not None
        evidence["states"] = list(ctx.states)
        # Snapshot only the inference schema; leave malformed questions to predict_batch's
        # validator, and do not inspect questions when a hook has already answered the call.
        if ctx.results is None and isinstance(ctx.questions, dict):
            evidence["question_types"] = {
                qid: qdef.get("type") if isinstance(qdef, dict) else None
                for qid, qdef in ctx.questions.items()
            }
        # Also what the budget and the questions ACTUALLY are once the chain has run, for
        # `_check_scan_budget`. Recorded, never judged here: a hook that raises is governed by the
        # caller's `hooks_raise`/`hooks_timeout`, so a correctness check that raises from inside the
        # chain can be switched off by a policy meant for third-party telemetry. Measured: with
        # `hooks_raise=False` -- what `docs/hooks/tracing.md` recommends -- a refusal became a
        # RuntimeWarning blaming `_StartAdapter`, and the scan proceeded.
        evidence["max_len"] = ctx.max_len
        evidence["head_max_len"] = ctx.head_max_len
        evidence["questions"] = dict(ctx.questions) if ctx.questions else {}
        if track_answers:
            # predict_batch calls this before ANY end hook, including defaults and installed
            # hooks. An end observer appended to the hook list would already see reordered rows.
            ctx._record_scan_results = record_results
            ctx._record_scan_cache_results = record_cache_results

    return probe, evidence


def _replayed_from_memory(evidence, windows, results):
    """Whether a decision cache answered the scan per window, in whole or in part.

    Its signature is one result per window either way: every window answered from memory (the
    start chain answered with that many results), or the windows it did not hold sent on in their
    order while it put the rest back. A hook that answers the document returns one result, and one
    that drops windows returns fewer, so neither reads as a replay.
    """
    if len(results) != len(windows) or len(windows) < 2:
        return False
    if evidence["answered"]:
        return True
    sent = evidence["states"]
    if sent is None or len(sent) >= len(windows):
        return False
    rest = iter(windows)
    return all(any(state == window for window in rest) for state in sent)


def _check_scan_budget(agent, evidence, sized, cfg_max_len, cfg_head_max_len, asked=None):
    """Raise when the budget or questions in force leave less room than the scan was sized for.

    `predict_long` sizes its windows from the agent's config, before any hook has run. A start hook
    may then set `ctx.max_len`/`ctx.head_max_len`, or rewrite `ctx.questions` -- both documented
    powers -- and `build_sequence` uses whatever it finds. When the head widens faster than
    `max_len`, or the questions get more options, the real room SHRINKS: every window is
    re-truncated on the way in, and once the stride exceeds the real room consecutive windows stop
    touching. Measured on the English checkpoint with `widen_for_high_cardinality` from
    `docs/hooks/patterns.md` at 50 options, windows sized 303 against a room of 253; at 100 options
    a reviewer measured 43.4% of a document reaching no model, against the 37.6% this change exists
    to remove. Silently worse than not windowing at all.

    Called by `predict_long` AFTER the chain has run, not raised from inside it. A hook that raises
    is subject to the caller's `hooks_raise` and `hooks_timeout`, so the first version of this check
    could be switched off by `hooks_raise=False` -- which `docs/hooks/tracing.md` recommends -- and
    could blow a `hooks_timeout` the caller set for their own hooks. The cost is that the forward
    pass has already happened when this fires: a misconfigured scan is refused rather than answered
    wrongly, but it is not refused for free. Sizing the windows after the chain instead would change
    when hooks run and what they see, which is a larger change than this one.
    """
    if evidence.get("answered"):
        return                                 # a hook answered the document; no window was scored
    questions = evidence.get("questions")
    if not questions:
        return                                 # nothing to fit, as `window_budget` also concludes
    eff_max_len = evidence.get("max_len")
    eff_head = evidence.get("head_max_len")
    eff_max_len = cfg_max_len if eff_max_len is None else eff_max_len
    eff_head = cfg_head_max_len if eff_head is None else eff_head
    if eff_max_len == cfg_max_len and eff_head == cfg_head_max_len and (
            asked is None or questions == asked):
        # Nothing the scan was sized against moved, so there is nothing to recompute -- and
        # recomputing anyway costs a tokenization of every question head, which `window_budget`
        # already paid and `_encode_state` will pay again. That showed up as a third head
        # tokenization in `test_question_token_reuse`, which asserts two.
        return
    try:
        internal = [agent._to_internal(q) for q in questions.values()]
    except Exception:
        return                                 # malformed questions are the validator's to report
    if not internal:
        return
    room = min(state_room(agent.tok, q, eff_max_len, eff_head) for q in internal)
    if room >= sized:
        return
    raise ValueError(
        "predict_long: after the start hooks ran, the questions and token budget (max_len=%d "
        "head_max_len=%d) leave %d state tokens per window, but the scan was sized for %d from the "
        "agent's config (max_len=%d head_max_len=%d). Every window would be re-truncated and parts "
        "of the document would reach no model. Pass window=/stride= explicitly, or call "
        "predict_batch directly, if you need a start hook to change either."
        % (eff_max_len, eff_head, room, sized, cfg_max_len, cfg_head_max_len))


def _with_start_probe(hook_kwargs, probe):
    """`hook_kwargs` with `probe` appended after the caller's own start hooks.

    `probe` may be one hook or a list of them, appended in order.
    """
    kwargs = dict(hook_kwargs)
    extra = list(probe) if isinstance(probe, (list, tuple)) else [probe]
    kwargs["on_predict_start"] = list(_as_sequence(hook_kwargs.get("on_predict_start"))) + extra
    return kwargs


def _check_question(qid: str, qdef: Any) -> None:
    """Reject a question that cannot be answered, naming it and what to fix.

    `render_options` reads `criteria` in the shape the question's type expects and the decision
    head needs at least one option, so a malformed definition used to surface from three frames
    down as something that names neither the question nor the problem: `AttributeError:
    'NoneType' object has no attribute 'items'`, `KeyError: 'bool'`, or a `selected index k out
    of range` raised inside the model for a question that ended up with no options at all.
    """
    if qid is None:
        raise ValueError("question id must not be None")
    if not isinstance(qid, (str, int)) or (isinstance(qid, str) and not qid.strip()):
        raise ValueError("question id must be a non-empty string, got %r" % (qid,))
    if not isinstance(qdef, dict):
        raise ValueError("question %r: definition must be a dict, got %s"
                         % (qid, type(qdef).__name__))
    t = qdef.get("type")
    if not isinstance(t, str) or t not in QTYPES:
        raise ValueError("question %r: unknown type %r; use one of %s" % (qid, t, sorted(QTYPES)))
    if "instructions" not in qdef:
        raise ValueError("question %r: no 'instructions'; add the text the model should answer" % (qid,))
    ins = qdef["instructions"]
    if ins is None:
        raise ValueError("question %r: 'instructions' must not be None; add the text the model should answer" % (qid,))
    if isinstance(ins, str) and not ins.strip():
        raise ValueError("question %r: 'instructions' must not be empty; add the text the model should answer" % (qid,))
    if isinstance(ins, (list, dict)) and not ins:
        raise ValueError("question %r: 'instructions' must not be empty; add the text the model should answer" % (qid,))
    if not isinstance(ins, (str, dict, list, int, float)):
        raise ValueError("question %r: 'instructions' must be a string, dict, or list, got %s"
                         % (qid, type(ins).__name__))
    crit = qdef.get("criteria")
    if t == "choice":
        if not isinstance(crit, (dict, list)):
            raise ValueError("question %r: a choice question takes 'criteria' as a dict of "
                             "label -> description, or a list of labels" % (qid,))
        if not crit:
            raise ValueError("question %r: a choice question needs at least one criterion" % (qid,))
        # A label is used as a dict key when a list of labels is normalised, so an unhashable
        # label raised `TypeError` from three frames down -- which names neither the question nor
        # the label, and which `serve` cannot classify as a caller error, so over HTTP it became
        # a 500 "inference failed" instead of a 422. Requiring the documented scalars exactly
        # also catches the hashable-but-not-scalar shapes (`tuple`, `frozenset`, `bytes`,
        # `complex`) that the old deny-list let through to the same 500. Labels are rendered as
        # option text, so a nested structure has no meaning here.
        for i, label in enumerate(crit if isinstance(crit, list) else crit.keys()):
            if label is not None and not isinstance(label, (str, int, float, bool)):
                raise ValueError(
                    "question %r: choice label %d is a %s; a label is rendered as option text "
                    "and used as the answer key, so it must be a scalar (a string, number or "
                    "bool), got %r" % (qid, i, type(label).__name__, label))
            if label is None:
                # The score path rejects a null level with the same argument one question type
                # over: a label is option text AND the answer key, and a null one can be
                # neither. `_to_internal` normalises a list of labels to `{label: None}`, so a
                # null label became the option text "None" (`str(None)`) while the answer key
                # and the `probabilities` key for that same option were the JSON string "null",
                # because a dict key has to be a str. A client then cannot tell whether that
                # option was the *string* `"null"` or JSON `null`, and the answer key is
                # unreachable: `criteria[answer["choice"]]` yields `None`, and
                # `answer["choice"] == "null"` is False for it. `""` stays accepted, because
                # unlike a null it does round-trip.
                raise ValueError(
                    "question %r: choice label %d is null; a label is rendered as option text "
                    "and used as the answer key, so it must be a string, number or bool -- a "
                    "null label renders as the text \"None\" while its answer key is \"null\""
                    % (qid, i))
        # `_to_internal` normalises the list form to `{label: None}`, so those labels become
        # the answer keys. Two entries that land on one key made the model score fewer options
        # than the caller wrote, and the response carry fewer probabilities than their list,
        # without a word -- the same silent-shape class as the two checks above. Python
        # treats values as one key whenever they compare equal, so `[1, 1.0]` and `[True, 1]`
        # collapse as well as an exact repeat. An unhashable label raised
        # `TypeError: cannot use 'tuple' as a dict key` from `_to_internal`, three frames down,
        # which names neither the question nor the label -- and which `serve` cannot classify
        # as a caller error, so over HTTP it became a 500 "inference failed" instead of a 422.
        if isinstance(crit, list):
            keys: Dict[Any, int] = {}
            for i, label in enumerate(crit):
                try:
                    first = keys[label]
                except TypeError as exc:
                    raise ValueError(
                        "question %r: choice label %d (%r) cannot be an answer key because it is "
                        "unhashable; a label is rendered as option text and used as the answer "
                        "key" % (qid, i, label)) from exc
                except KeyError:
                    keys[label] = i
                else:
                    raise ValueError(
                        "question %r: choice label %d (%r) repeats label %d; the labels are the "
                        "answer keys, so every option needs its own (1, 1.0 and True are one "
                        "key)" % (qid, i, label, first))
    elif t == "score":
        if not isinstance(crit, list):
            raise ValueError("question %r: a score question takes 'criteria' as a list of level "
                             "descriptions, index 0 first" % (qid,))
        if not crit:
            raise ValueError("question %r: a score question needs at least one level" % (qid,))
        if None in crit:
            raise ValueError("question %r: score level %d is null; give every level a description, "
                             "index 0 first" % (qid, crit.index(None)))
    elif crit is not None and not isinstance(crit, dict):
        raise ValueError("question %r: a noul question takes 'criteria' as a dict with optional "
                         "'true'/'false' descriptions, or omits it" % (qid,))
    elif isinstance(crit, dict):
        # `render_options` reads these two descriptions out by name -- `crit.get("false")` and
        # `crit.get("true")` -- so a dict keyed any other way is not a noul description at all.
        # It used to be substituted with the default pair without a word, so a caller saw their
        # descriptions accepted and never reach the model (#156). `labels` just below has
        # rejected the same mistake since #163; this is the same rule on the other parameter,
        # and a noul is a boolean question either way, so those are the only two keys it can have.
        keys = {str(k).lower() for k in crit}
        if not keys <= {"true", "false"}:
            raise ValueError(
                "question %r: a noul question takes 'criteria' keyed only 'true'/'false' (either "
                "or both, and omitted is fine), got %s. Those keys are the option texts the model "
                "reads; any other key was silently dropped and replaced with the defaults. If you "
                "want the answer worded differently, keep 'criteria' keyed 'true'/'false' and set "
                "'labels' instead." % (qid, sorted(keys)))
    if "option_order" in qdef:
        # Slot s shows option `order[s]`. Anything other than a permutation of the option
        # indices would either drop an option or show one twice, so reject it here rather
        # than let it reach the encoder.
        order = qdef["option_order"]
        n = _option_count(qdef)
        if (not isinstance(order, (list, tuple))
                or len(order) != n
                or sorted(int(i) for i in order if isinstance(i, int) and not isinstance(i, bool)) != list(range(n))):
            raise ValueError(
                "question %r: 'option_order' must be a permutation of range(%d) -- one slot per "
                "option, each option once -- got %r" % (qid, n, order))
    if "labels" in qdef:
        if t != "noul":
            raise ValueError("question %r: 'labels' is only supported for noul questions" % (qid,))
        try:
            _resolve_noul_labels(qdef["labels"])
        except ValueError as e:
            raise ValueError("question %r: %s" % (qid, e)) from e


def _to_internal(qdef: Dict) -> Dict:
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    elif t == "noul" and isinstance(crit, dict):
        # Normalize boolean literal keys to string keys ("true"/"false")
        crit = {str(k).lower(): v for k, v in crit.items()}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        # `ensure_ascii=False`, matching `serialize_state` and `render_criterion` in
        # common.py and the instructions path in shortlist.py. The default escaped
        # non-ASCII to literal `\uXXXX`, which the tokenizer then read as escape text:
        # on the English checkpoint one German question answered noul=0.1652 as a dict
        # and noul=0.2650 as the identical plain string.
        ins = json.dumps(ins, ensure_ascii=False)
    q = {"t": t, "ins": ins, "crit": crit}
    if "labels" in qdef:
        q["labels"] = qdef["labels"]
    if "option_order" in qdef:
        q["option_order"] = [int(i) for i in qdef["option_order"]]
    return q
