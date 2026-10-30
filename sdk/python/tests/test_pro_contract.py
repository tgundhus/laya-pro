"""Pro wire checks that the permissive upstream transport did not enforce."""
import copy
import json
import unittest
from unittest.mock import patch

from laya_client import LayaClient, LayaResponseError, LayaValidationError
from laya_client._validation import MIN_CONFIDENCE_KEYS, prediction
from _support import PREDICTION, QUESTIONS, Server


class ProContractTests(unittest.TestCase):
    def test_choice_lists_are_normalized_without_mutating_inputs(self):
        questions = copy.deepcopy(QUESTIONS)
        questions["team"]["criteria"] = ["billing", "technical"]
        questions["team"]["option_order"] = [1, 0]
        original = copy.deepcopy(questions)
        with Server() as server:
            LayaClient(base_url=server.url).predict("hello", questions)
        wire = json.loads(server.requests[0][3])["questions"]
        self.assertEqual(wire["team"]["criteria"], {"billing": None, "technical": None})
        self.assertEqual(wire["team"]["option_order"], [1, 0])
        self.assertEqual(questions, original)

    def test_malformed_question_shapes_are_refused_before_sending(self):
        cases = []
        for key, value in (("instructions", None), ("instructions", " "), ("instructions", []),
                           ("criteria", []), ("criteria", ["a", "a"]), ("criteria", [1, "a"]),
                           ("future_field", True), ("labels", {"true": "yes", "false": "no"}),
                           ("option_order", [False, 1]), ("option_order", [0, 0]),
                           ("option_order", [1.0, 0]), ("option_order", ["a", "b"]),
                           ("option_order", None), ("option_order", {})):
            question = copy.deepcopy(QUESTIONS["team"])
            question[key] = value
            cases.append(question)
        cases.extend([
            {"type": "score", "instructions": "?", "criteria": [None]},
            {"type": "score", "instructions": "?", "criteria": {}},
            {"type": "noul", "instructions": "?", "criteria": {"yes": "approve"}},
            {"type": "noul", "instructions": "?", "labels": {"true": "same", "false": "same "}},
            {"type": "noul", "instructions": "?", "labels": {"true": "yes"}},
        ])
        client = LayaClient()
        with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("request was sent")):
            for question in cases:
                for method, state in ((client.predict, "hello"), (client.predict_batch, ["hello"])):
                    with self.subTest(question=question, method=method.__name__), self.assertRaises(LayaValidationError):
                        method(state, {"q": question})

    def test_unknown_confidence_buckets_never_send(self):
        client = LayaClient()
        with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("request was sent")):
            for key in ("future:bucket", "choice:2-5", "choice:1", "noul:3", "", 1):
                with self.subTest(key=key), self.assertRaises(LayaValidationError):
                    client.predict("hello", QUESTIONS, min_confidence={key: 0.9})
        with Server() as server:
            LayaClient(base_url=server.url).predict("hello", QUESTIONS,
                                                 min_confidence=dict.fromkeys(MIN_CONFIDENCE_KEYS, 0.0))
        self.assertEqual(set(json.loads(server.requests[0][3])["min_confidence"]), MIN_CONFIDENCE_KEYS)

    def test_tournament_metadata_matches_the_reported_choice(self):
        valid = {"finalists": ["billing"], "rounds": 2}
        invalid = [None, {}, {"finalists": [], "rounds": 1}]
        for rounds in ([], True, 0, -1, 1.5):
            invalid.append(dict(valid, rounds=rounds))
        for finalists in (["technical"], ["billing", "billing"], [1], ["unknown"], "billing", {}):
            invalid.append(dict(valid, finalists=finalists))
        for meta in invalid:
            payload = copy.deepcopy(PREDICTION)
            payload["answers"]["team"]["x_tournament"] = meta
            with self.subTest(meta=meta), self.assertRaises(LayaResponseError):
                prediction(payload, QUESTIONS)
        for qid in ("urgency", "refund"):
            payload = copy.deepcopy(PREDICTION)
            payload["answers"][qid]["x_tournament"] = valid
            with self.subTest(qid=qid), self.assertRaises(LayaResponseError):
                prediction(payload, QUESTIONS)

    def test_optional_answer_metadata_is_checked(self):
        for key, value in (("low_confidence", False), ("action", {}),
                           ("action", {"act_probability": True}), ("answer_confidence", True),
                           ("x_jev_confidence", float("inf")), ("abstention_threshold", False)):
            payload = copy.deepcopy(PREDICTION)
            payload["answers"]["team"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(LayaResponseError):
                prediction(payload, QUESTIONS)
        payload = copy.deepcopy(PREDICTION)
        payload["answers"]["refund"]["x_jev_confidence"] = 0.9
        with self.assertRaises(LayaResponseError):
            prediction(payload, QUESTIONS)

    def test_route_and_detection_fields_are_typed(self):
        valid = {"model": "custom", "repo": "local", "reason": "explicit",
                 "workflow": None, "detection": None}
        payload = copy.deepcopy(PREDICTION)
        payload["routing"] = valid
        prediction(payload, QUESTIONS)
        for route in ({"model": "custom"}, dict(valid, model="auto"), dict(valid, workflow=False),
                      dict(valid, reason=None), dict(valid, detection=[])):
            payload["routing"] = route
            with self.subTest(route=route), self.assertRaises(LayaResponseError):
                prediction(payload, QUESTIONS)
        detection = {"script": "Latin", "script_profile": {"Latin": 1.0}, "language": "en",
                     "is_english": True, "language_undecided": False, "diacritic_rate": 0.0,
                     "non_latin_fraction": 0.0, "mixed_segment": None}
        payload["routing"] = dict(valid, detection=detection)
        prediction(payload, QUESTIONS)
        for key, value in (("script_profile", {"Latin": True}), ("is_english", 1),
                           ("language_undecided", None), ("diacritic_rate", True),
                           ("non_latin_fraction", 1.1), ("mixed_segment", False)):
            payload["routing"] = dict(valid, detection=dict(detection, **{key: value}))
            with self.subTest(key=key), self.assertRaises(LayaResponseError):
                prediction(payload, QUESTIONS)

    def test_usage_extensions_are_typed_but_legacy_minimal_usage_is_valid(self):
        prediction(PREDICTION, QUESTIONS)
        cases = [("state_tokens", True), ("state_tokens_dropped", -1), ("truncated", 0),
                 ("truncated_questions", [1]), ("options", []),
                 ("options", {"q": {"total": 2, "distinct": 3, "tokens_per_option": None}}),
                 ("options", {"q": {"total": True, "distinct": 1, "tokens_per_option": None}}),
                 ("options", {"q": {"total": 2, "distinct": 1}}),
                 ("options", {"q": {"total": 2, "distinct": 1, "tokens_per_option": False}})]
        for key, value in cases:
            payload = copy.deepcopy(PREDICTION)
            payload["usage"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(LayaResponseError):
                prediction(payload, QUESTIONS)

    def test_health_metadata_is_checked_when_present(self):
        for payload in ({"status": "ok", "device": False}, {"status": "ok", "loaded": ["auto"]},
                        {"status": "ok", "loaded": [False]}):
            with self.subTest(payload=payload), Server(payload) as server:
                with self.assertRaises(LayaResponseError):
                    LayaClient(base_url=server.url).health()

    def test_overflowing_response_number_is_not_valid_json(self):
        raw = json.dumps(PREDICTION).replace('"model": "custom-checkpoint"',
                                              '"future": 1e999, "model": "custom-checkpoint"').encode()
        with Server(raw=raw) as server, self.assertRaises(LayaResponseError):
            LayaClient(base_url=server.url).predict("hello", QUESTIONS)


if __name__ == "__main__":
    unittest.main()
