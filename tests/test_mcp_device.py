"""LAYA_DEVICE normalisation *and* parseability: the laya.mcp.device env contract.

The variable is read once, in `env_device`, and the same string is handed to torch
(``Router(device=...)``) and reported by `resolve_device` / `laya_status`. Case and
whitespace normalisation was already covered (``CUDA:0 -> cuda:0``); what this suite
adds is the other half of "torch can use what we return": ``cuda:`` (or ``gpu``, or
``cuda: 0``) normalises cleanly and is still refused by torch -- so on a lazy Router it
survived startup, was reported as the configured device, and the refusal landed inside
the first request that built a checkpoint. `env_device` now parses the value it returns
and raises a named ValueError where the environment is read.

The other half of the contract is its scope: `force=` is passed straight through, unchecked, so the
parse guarantee covers the environment and the auto-detection chain and not that hook -- and
`agent_device`, the answer for a *loaded* checkpoint, reports a device type without its index while
the plain-string form it accepts for fakes keeps whatever it was given.

No model weights, no network, no GPU: torch is imported for its parser only.
Run: python tests/test_mcp_device.py
"""
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401  (the parser under test)
except ImportError:
    print("SKIP: torch not importable, so there is no device parser to check")
    sys.exit(0)

from laya.mcp.device import agent_device, env_device, resolve_device  # noqa: E402

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append("%s%s" % (name, (": " + detail) if detail and not cond else ""))
    print("   %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""), flush=True)


def env_device_with(raw):
    """`env_device()` with LAYA_DEVICE set to raw (None = unset); the old value is restored."""
    old = os.environ.get("LAYA_DEVICE")
    try:
        if raw is None:
            os.environ.pop("LAYA_DEVICE", None)
        else:
            os.environ["LAYA_DEVICE"] = raw
        return env_device()
    finally:
        if old is None:
            os.environ.pop("LAYA_DEVICE", None)
        else:
            os.environ["LAYA_DEVICE"] = old


def rejection(raw):
    """The ValueError message `env_device()` raises for raw, or "" if it does not raise."""
    try:
        env_device_with(raw)
    except ValueError as exc:
        return str(exc)
    except BaseException as exc:  # noqa: BLE001 - a wrong exception is a failed case, not an abort
        return "wrong exception %r" % (exc,)
    return ""


def test_unset_and_blank_stay_none():
    ok("unset -> None", env_device_with(None) is None)
    ok("empty -> None", env_device_with("") is None)
    ok("spaces -> None", env_device_with("   ") is None)


def test_normalisation_still_applies_first():
    ok("cpu verbatim", env_device_with("cpu") == "cpu")
    ok("case + spaces + index", env_device_with("  CUDA:0 ") == "cuda:0")
    ok("index preserved", env_device_with("cuda:1") == "cuda:1")


def test_the_parser_is_the_judge_not_an_allowlist():
    # torch.device("meta") parses; a hand-written {cpu, cuda, mps} allowlist would not,
    # and would make a device this module has never heard of impossible to configure.
    ok("meta (valid for torch, never loadable here)", env_device_with("meta") == "meta")


def test_unparseable_values_are_rejected_by_name():
    for raw in ("cuda:", "gpu", "cuda: 0"):
        message = rejection(raw)
        ok("%r -> ValueError" % raw, bool(message) and "wrong exception" not in message, message or "no error")
        ok("%r names LAYA_DEVICE" % raw, "LAYA_DEVICE" in message, message)
        ok("%r quotes the offending value" % raw, raw.strip() in message, message)
    ok("'cuda:' carries torch's own reason",
       "Invalid device string" in rejection("cuda:"), rejection("cuda:"))
    ok("'gpu' carries torch's own list of device types",
       "Expected one of" in rejection("gpu"), rejection("gpu"))


def test_the_env_path_raises_and_force_passes_through():
    old = os.environ.get("LAYA_DEVICE")
    try:
        os.environ["LAYA_DEVICE"] = "cuda:"
        try:
            resolve_device()
        except ValueError:
            ok("resolve_device() refuses it too", True)
        else:
            ok("resolve_device() refuses it too", False, "returned a label for 'cuda:'")
    finally:
        if old is None:
            os.environ.pop("LAYA_DEVICE", None)
        else:
            os.environ["LAYA_DEVICE"] = old
    # force= is the documented tests hook: it is returned verbatim, exactly like the
    # CLI --device flag, which never reads the environment. Guarded, because the check is
    # about the pass-through: a version that parses `force` has to fail by name, not abort.
    try:
        forced = resolve_device("cuda:")
    except Exception as exc:
        ok("force= hook stays verbatim", False, "raised %s: %s" % (type(exc).__name__, exc))
    else:
        ok("force= hook stays verbatim", forced == "cuda:", repr(forced))


def parses(value):
    """Whether torch's own parser accepts `value` -- the judge this module defers to."""
    try:
        torch.device(value)
    except Exception:
        return False
    return True


def test_force_is_the_only_path_that_can_label_a_refused_device():
    # The two paths the docstring claims for never hand back a label torch would refuse:
    # the environment either parses or raises, and the auto-detection chain answers with a
    # device type torch names. Dropping `env_device`'s parse check turns the refused rows of
    # the first loop into returned labels, which is what this catches.
    for raw in ("cpu", "CUDA:0", "cuda:1", "mps", "meta", "  xpu:2  ",
                "cuda:", "gpu", "cuda: 0", "cuda:0:1", "1cuda", "cuda-0"):
        try:
            value = env_device_with(raw)
        except ValueError:
            ok("env %r raises instead of labelling" % raw, True)
            continue
        except Exception as exc:  # a wrong exception type is a failed check, not an aborted run
            ok("env %r raises ValueError or returns a label" % raw, False,
               "raised %s: %s" % (type(exc).__name__, exc))
            continue
        ok("env %r labels a device torch accepts" % raw, parses(value), repr(value))
    old = os.environ.get("LAYA_DEVICE")
    try:
        os.environ.pop("LAYA_DEVICE", None)
        label = resolve_device()
        ok("auto-detected label parses", parses(label), repr(label))
    finally:
        if old is None:
            os.environ.pop("LAYA_DEVICE", None)
        else:
            os.environ["LAYA_DEVICE"] = old
    # force= is the one path that can, and it is returned verbatim on purpose. Checked
    # defensively: a future edit that parses `force` too must fail a named check, not the run.
    for raw in ("cuda:", "gpu", "cuda: 0"):
        try:
            given = resolve_device(raw)
        except Exception as exc:
            ok("force=%r comes back as given" % raw, False, "raised %s: %s" % (type(exc).__name__, exc))
            continue
        ok("force=%r comes back as given" % raw, given == raw, repr(given))
        ok("force=%r is a label torch refuses" % raw, not parses(given), repr(given))

    # The claim itself, read off the docstring: whatever states the parse guarantee has to
    # name `force` as the path it does not cover. A docstring that reverts to "there is no
    # path on which this function labels a device torch would refuse" fails both halves --
    # and the second half is what the line above pins as true.
    tail = inspect.getdoc(resolve_device).split("torch cannot parse", 1)
    ok("the docstring still states the parse guarantee", len(tail) == 2, inspect.getdoc(resolve_device))
    claim = tail[1] if len(tail) == 2 else ""
    ok("the parse claim names force as the exception", "`force`" in claim, claim)
    ok("the parse claim is scoped, not universal", "no path" not in claim, claim)


def test_agent_device_answers_by_input_shape():
    # A loaded agent's `device` is a torch.device, whose `type` carries no index; the plain
    # string the suite accepts for fakes has no such split, so the two shapes do not answer
    # the same question. Both halves are what `agent_device`'s docstring now says.
    def held(device):
        return type("Agent", (), {"device": device})()

    for raw in ("cuda:1", "cuda:0", "cpu:0"):
        ok("torch.device(%r) loses its index" % raw, agent_device(held(torch.device(raw))) == raw.split(":")[0],
           repr(agent_device(held(torch.device(raw)))))
        ok("the same string keeps its index", agent_device(held(raw)) == raw, repr(agent_device(held(raw))))
    ok("a type-only device is unchanged", agent_device(held(torch.device("mps"))) == "mps"
       and agent_device(held("mps")) == "mps")
    # The asymmetry above is invisible to a caller who reads only the docstring, so the prose
    # is pinned to it: whatever describes this function has to say that an index is dropped.
    doc = inspect.getdoc(agent_device) or ""
    ok("the docstring states that the index is not reported", "index" in doc, doc)


def test_reported_equals_handed_to_torch():
    old = os.environ.get("LAYA_DEVICE")
    try:
        os.environ["LAYA_DEVICE"] = "CUDA:1"
        ok("env_device == resolve_device", env_device() == resolve_device() == "cuda:1")
    finally:
        if old is None:
            os.environ.pop("LAYA_DEVICE", None)
        else:
            os.environ["LAYA_DEVICE"] = old


def main():
    for fn in (test_unset_and_blank_stay_none,
               test_normalisation_still_applies_first,
               test_the_parser_is_the_judge_not_an_allowlist,
               test_unparseable_values_are_rejected_by_name,
               test_the_env_path_raises_and_force_passes_through,
               test_force_is_the_only_path_that_can_label_a_refused_device,
               test_agent_device_answers_by_input_shape,
               test_reported_equals_handed_to_torch):
        fn()
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for name in FAIL:
        print("  FAIL " + name)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
