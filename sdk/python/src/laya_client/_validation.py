"""Request and response checks for the self-hosted Pro wire contract."""
import math
import re
import threading
from typing import Any

from .errors import LayaResponseError, LayaValidationError


MIN_CONFIDENCE_KEYS = frozenset(["default"] + [kind + ":" + size
    for kind in ("choice", "score", "noul") for size in ("2", "3-5", "6-10", "11+")])


def finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def probability(value: Any) -> bool:
    return finite(value) and 0 <= value <= 1


def timeout_value(value: Any) -> float:
    if not finite(value) or not 0 < value <= threading.TIMEOUT_MAX:
        raise LayaValidationError("timeout must be positive, finite and within the platform timeout limit")
    return float(value)


def json_value(value: Any, ancestors=None) -> None:
    if value is None or type(value) in (bool, int):
        return
    if isinstance(value, str):
        value.encode("utf-8")  # Reject unpaired surrogates before sending.
        return
    if type(value) is float and math.isfinite(value):
        return
    if not isinstance(value, (list, dict)):
        raise LayaValidationError("request must contain only finite JSON values")
    ancestors = set() if ancestors is None else ancestors
    if id(value) in ancestors:
        raise LayaValidationError("request contains a circular reference")
    ancestors.add(id(value))
    try:
        if isinstance(value, dict):
            for key in value:
                if not isinstance(key, str):
                    raise LayaValidationError("JSON object keys must be strings")
                key.encode("utf-8")
            values = value.values()
        else:
            values = value
        for item in values:
            json_value(item, ancestors)
    finally:
        ancestors.remove(id(value))


def questions_value(value: Any) -> None:
    if not isinstance(value, dict) or not value:
        raise LayaValidationError("questions must be a nonempty object")
    for qid, question in value.items():
        if not isinstance(qid, str) or not qid.strip():
            raise LayaValidationError("question IDs must be nonempty strings")
        if not isinstance(question, dict) or question.get("type") not in ("choice", "score", "noul"):
            raise LayaValidationError("each question must have type choice, score or noul")
        if set(question) - {"type", "instructions", "criteria", "labels", "option_order"}:
            raise LayaValidationError("questions.%s: unknown question field" % qid)
        instructions = question.get("instructions")
        if (instructions is None or (isinstance(instructions, str) and not instructions.strip())
                or (isinstance(instructions, (dict, list)) and not instructions)
                or not isinstance(instructions, (str, dict, list, int, float))):
            raise LayaValidationError("questions.%s: nonempty instructions are required" % qid)
        kind, criteria = question["type"], question.get("criteria")
        if "labels" in question and kind != "noul":
            raise LayaValidationError("questions.%s: labels require a noul question" % qid)
        if kind == "choice":
            if not isinstance(criteria, (dict, list)) or not criteria:
                raise LayaValidationError("questions.%s: choice requires nonempty criteria" % qid)
            if not all(isinstance(label, str) for label in criteria):
                raise LayaValidationError("questions.%s: choice labels must be strings" % qid)
            if isinstance(criteria, list) and len(set(criteria)) != len(criteria):
                raise LayaValidationError("questions.%s: choice labels must be unique" % qid)
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria or any(level is None for level in criteria):
                raise LayaValidationError("questions.%s: score requires nonempty, non-null levels" % qid)
        else:
            if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
                raise LayaValidationError("questions.%s: noul criteria may only describe true and false" % qid)
            labels = question.get("labels")
            if labels is not None and (not isinstance(labels, dict) or set(labels) != {"true", "false"}
                    or not all(isinstance(label, str) and label.strip() for label in labels.values())
                    or labels["true"].strip() == labels["false"].strip()):
                raise LayaValidationError("questions.%s: labels must contain distinct false and true strings" % qid)
        order = question.get("option_order")
        if "option_order" in question:
            count = 2 if kind == "noul" else len(criteria)
            if (not isinstance(order, list) or any(type(index) is not int for index in order)
                    or sorted(order) != list(range(count))):
                raise LayaValidationError("questions.%s: option_order must permute the option indices" % qid)


def expect(condition: bool, field: str) -> None:
    if not condition:
        raise LayaResponseError("Invalid Laya response: " + field)


def is_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def model_name(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-z0-9_.-]+", value) is not None and value != "auto"


def health(value: Any) -> None:
    expect(isinstance(value, dict) and value.get("status") == "ok", "health")
    if "device" in value:
        expect(isinstance(value["device"], str), "health.device")
    if "loaded" in value:
        expect(isinstance(value["loaded"], list) and all(model_name(name) for name in value["loaded"]),
               "health.loaded")


def usage_value(value: Any) -> None:
    expect(isinstance(value, dict), "usage")
    for key in ("input_tokens", "output_tokens", "state_tokens", "state_tokens_dropped"):
        if key in value or key in ("input_tokens", "output_tokens"):
            expect(is_count(value.get(key)), "usage." + key)
    if "truncated" in value:
        expect(type(value["truncated"]) is bool, "usage.truncated")
    if "truncated_questions" in value:
        expect(isinstance(value["truncated_questions"], list)
               and all(isinstance(qid, str) for qid in value["truncated_questions"]), "usage.truncated_questions")
    if "options" in value:
        expect(isinstance(value["options"], dict), "usage.options")
        for collapse in value["options"].values():
            expect(isinstance(collapse, dict), "usage.options")
            expect(is_count(collapse.get("total")) and is_count(collapse.get("distinct"))
                   and collapse["distinct"] <= collapse["total"], "usage.options")
            expect("tokens_per_option" in collapse and (collapse["tokens_per_option"] is None
                   or is_count(collapse["tokens_per_option"])), "usage.options.tokens_per_option")


def routing_value(value: Any) -> None:
    expect(isinstance(value, dict), "routing")
    expect(model_name(value.get("model")) and isinstance(value.get("repo"), str)
           and isinstance(value.get("reason"), str), "routing model/repo/reason")
    expect("workflow" in value and (value["workflow"] is None or isinstance(value["workflow"], str)),
           "routing workflow")
    expect("detection" in value, "routing detection")
    detection = value["detection"]
    if detection is None:
        return
    expect(isinstance(detection, dict), "routing detection")
    expect(isinstance(detection.get("script"), str) and isinstance(detection.get("script_profile"), dict),
           "script detection")
    expect(all(finite(item) for item in detection["script_profile"].values()), "script profile")
    expect("language" in detection and (detection["language"] is None or isinstance(detection["language"], str)),
           "detected language")
    for key in ("is_english", "language_undecided"):
        expect(type(detection.get(key)) is bool, "routing." + key)
    for key in ("diacritic_rate", "non_latin_fraction"):
        expect(probability(detection.get(key)), "routing." + key)
    if "mixed_segment" in detection:
        expect(detection["mixed_segment"] is None or isinstance(detection["mixed_segment"], str),
               "routing mixed_segment")


def batch_prediction(value: Any, questions: dict, count: int) -> None:
    expect(isinstance(value, dict), "batch prediction object")
    results = value.get("results")
    expect(isinstance(results, list), "results")
    expect(len(results) == count, "results count does not match states")
    usage = value.get("total_usage")
    expect(isinstance(usage, dict), "total_usage")
    for key in ("input_tokens", "output_tokens"):
        expect(is_count(usage.get(key)), "total_usage." + key)
    for i, result in enumerate(results):
        try:
            prediction(result, questions)
        except LayaResponseError as exc:
            raise LayaResponseError("Invalid Laya response: results[%d]: %s" % (i, exc)) from exc


def prediction(value: Any, questions: dict) -> None:
    expect(isinstance(value, dict), "prediction object")
    expect(isinstance(value.get("model"), str), "model")
    answers, usage = value.get("answers"), value.get("usage")
    expect(isinstance(answers, dict), "answers")
    usage_value(usage)
    if "routing" in value:
        routing_value(value["routing"])
    for qid, question in questions.items():
        answer = answers.get(qid)
        field = "answers." + qid
        expect(isinstance(answer, dict) and answer.get("type") == question["type"], field + ".type")
        kind = answer["type"]
        for key in ("confidence", "answer_confidence", "x_jev_confidence", "abstention_threshold"):
            if key in answer or (key == "confidence" and kind != "noul"):
                expect(probability(answer.get(key)), field + "." + key)
        if "low_confidence" in answer:
            expect(answer["low_confidence"] is True, field + ".low_confidence")
        if "x_jev_confidence" in answer:
            expect(kind != "noul", field + ".x_jev_confidence")
        if "abstention" in answer or "abstention_threshold" in answer:
            # The gate reports a state and the threshold that produced it together.
            expect(answer.get("abstention") in ("passed", "abstained", "unevaluated"), field + ".abstention")
            expect(probability(answer.get("abstention_threshold")), field + ".abstention_threshold")
        if "action" in answer:
            expect(isinstance(answer["action"], dict) and probability(answer["action"].get("act_probability")),
                   field + ".action")
        if "x_tournament" in answer:
            meta = answer["x_tournament"]
            expect(kind == "choice" and isinstance(meta, dict), field + ".x_tournament")
            labels = list(question["criteria"])
            expect(type(meta.get("rounds")) is int and meta["rounds"] > 0, field + ".x_tournament.rounds")
            finalists = meta.get("finalists")
            expect(isinstance(finalists, list) and bool(finalists)
                   and all(isinstance(label, str) and label in labels for label in finalists)
                   and len(set(finalists)) == len(finalists) and answer.get("choice") in finalists,
                   field + ".x_tournament.finalists")
        if kind == "noul":
            expect(probability(answer.get("noul")), field + ".noul")
            continue
        probs = answer.get("probabilities")
        expect(isinstance(probs, dict) and bool(probs) and all(probability(p) for p in probs.values()),
               field + ".probabilities")
        criteria = question.get("criteria")
        if kind == "choice":
            expect(isinstance(answer.get("choice"), str) and answer["choice"] in probs, field + ".choice")
            labels = list(criteria) if isinstance(criteria, dict) else criteria
            if isinstance(labels, list) and all(isinstance(label, str) for label in labels):
                expect(answer["choice"] in labels and all(label in probs for label in labels),
                       field + ".choice labels")
        else:
            expect(finite(answer.get("score")) and answer["score"] >= 0, field + ".score")
            expect(isinstance(answer.get("legend"), dict), field + ".legend")
            if isinstance(criteria, list):
                expect(answer["score"] <= len(criteria) - 1, field + ".score")
                expect(all(str(i) in probs and str(i) in answer["legend"] for i in range(len(criteria))),
                       field + ".rubric levels")
