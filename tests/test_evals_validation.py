"""Regression checks for evaluation artifact and runner contract validation. No weights."""
from collections import OrderedDict
from copy import deepcopy
import argparse
import json
from types import SimpleNamespace

import pytest

import laya
from laya import evals, evals_cli, evidence


QUESTIONS = {"q": {"type": "choice", "instructions": "Choose.", "criteria": ["a", "b"]}}


@pytest.mark.parametrize("baseline", [None, [], "bad", {}, {"overall": {}},
    {"overall": None}, {"overall": []}, {"overall": {"choice_accuracy": "0.9"}},
    {"overall": {"choice_accuracy": True}}, {"overall": {"choice_accuracy": None}}])
def test_compare_refuses_baselines_that_cannot_be_measurements(baseline):
    with pytest.raises(evals.EvalError, match="baseline"):
        evals.EvalReport(overall={"choice_accuracy": 0.9}).compare(baseline)


@pytest.mark.parametrize("overall", [None, [], {"choice_accuracy": "0.9"},
                                      {"choice_accuracy": True}])
def test_compare_validates_the_candidate_metric_map(overall):
    with pytest.raises(evals.EvalError, match="report"):
        evals.EvalReport(overall=overall).compare({"choice_accuracy": 0.9})


def test_metric_names_must_support_metric_lookup():
    with pytest.raises(evals.EvalError, match="metric name"):
        evals.EvalReport(overall={"choice_accuracy": 0.9}).compare({3: 0.9})


@pytest.mark.parametrize("tolerances", [False, 0, 0.0, "", (), [], "bad",
    {"choice_accuracy": None}, {"choice_accuracy": True}, {"choice_accuracy": "0.1"},
    {"choice_accuracy": []}, {"choice_accuracy": float("inf")}])
def test_compare_refuses_tolerances_that_cannot_gate(tolerances):
    with pytest.raises(evals.EvalError, match="tolerance"):
        evals.EvalReport(overall={"choice_accuracy": 0.8}).compare(
            {"choice_accuracy": 0.9}, tolerances)


def test_compare_keeps_numeric_failures_and_bare_maps():
    report = evals.EvalReport(overall={"choice_accuracy": 0.9})
    assert report.compare({"choice_accuracy": 0.9})[0]
    for value in (float("nan"), -1.0, -float("inf")):
        assert not report.compare({"choice_accuracy": 0.9}, {"choice_accuracy": value})[0]
    assert not evals.EvalReport().compare({"choice_accuracy": 0.9})[0]


@pytest.mark.parametrize("on_error", ["fail", "skip"])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("form", ["states", "requests"])
def test_batch_results_must_match_every_input_row(count, form, on_error):
    class StatesRunner:
        def predict(self, state, questions, model=None):
            return {"answers": {"q": {"type": "choice", "choice": "a"}}}

        def predict_batch(self, states, questions, model=None, batch_size=None):
            return (self.predict(None, questions) for _ in range(count))

    class RequestsRunner(StatesRunner):
        def predict_batch(self, requests, batch_size=None):
            return [self.predict(None, QUESTIONS) for _ in range(count)]

    dataset = evals.Dataset([evals.Example(state, QUESTIONS, {"q": "a"}) for state in ("a", "b")])
    runner = StatesRunner() if form == "states" else RequestsRunner()
    with pytest.raises(evals.EvalError, match=r"%d results.*2 rows" % count):
        evals.evaluate(runner, dataset, batch_size=4, on_error=on_error)


def test_option_order_changes_question_identity_without_changing_legacy_hash():
    def fingerprint(questions):
        return evals.questions_fingerprint(evals.Dataset([evals.Example("s", questions, {"q": "a"})]))

    legacy = fingerprint(QUESTIONS)
    # Pin the pre-option_order digest so adding an unused field cannot invalidate old baselines.
    assert legacy == "a1d78866f7d8dfd29ad8c395f8a9ebca6548efda038e3e1604d6b094aeed92ea"
    hashes = {legacy}
    for order in ([0, 1], [1, 0]):
        questions = deepcopy(QUESTIONS)
        questions["q"]["option_order"] = order
        hashes.add(fingerprint(questions))
    assert len(hashes) == 3


@pytest.mark.parametrize("example", [None, {}, "row", SimpleNamespace(expected={}),
    evals.Example("s", [], {}), evals.Example("s", QUESTIONS, []),
    evals.Example("s", QUESTIONS, {3: "a", "other": "a"})])
def test_python_dataset_rows_fail_at_the_shared_boundary(example):
    with pytest.raises(evals.EvalError, match="example 0"):
        evals.Dataset([example])


def test_dataset_keeps_duck_typed_rows_and_typed_keys():
    questions = {3: {"type": "choice", "instructions": "Choose.", "criteria": {4: "Accept"}}}
    row = SimpleNamespace(state="s", questions=questions, expected={3: 4}, tags=(), language=None,
                          model=None)
    assert evals.Dataset([row]).examples == [row]


@pytest.mark.parametrize("shape", ["directory", "missing", "encoding", "depth"])
@pytest.mark.parametrize("command", ["validate", "run"])
def test_unreadable_dataset_is_a_usage_error(tmp_path, capsys, shape, command):
    path = tmp_path / "dataset.jsonl"
    if shape == "directory":
        path.mkdir()
    elif shape == "encoding":
        path.write_bytes(b"\xff\xfe")
    elif shape == "depth":
        path.write_text("[" * 5000 + "0" + "]" * 5000, encoding="utf-8")
    assert evals_cli.main([command, str(path)]) == 2
    assert path.name in capsys.readouterr().err


def test_dataset_row_errors_keep_the_line_number_and_bom_is_accepted(tmp_path):
    path = tmp_path / "dataset.jsonl"
    row = {"state": "s", "questions": QUESTIONS, "expected": {"q": "a"}}
    path.write_text(json.dumps(row), encoding="utf-8-sig")
    assert len(evals.Dataset.from_jsonl(str(path))) == 1
    path.write_text("# comment\nnull\n", encoding="utf-8")
    with pytest.raises(evals.EvalError, match=r"dataset.jsonl:2 must be an object"):
        evals.Dataset.from_jsonl(str(path))


@pytest.mark.parametrize("shape", ["directory", "missing", "encoding", "depth", "malformed", "array"])
def test_unreadable_report_is_a_usage_error(tmp_path, capsys, shape):
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"overall":{"choice_accuracy":0.9}}', encoding="utf-8")
    path = tmp_path / "baseline.json"
    if shape == "directory":
        path.mkdir()
    elif shape == "encoding":
        path.write_bytes(b"\xff\xfe")
    elif shape != "missing":
        text = {"depth": "[" * 5000 + "0" + "]" * 5000, "malformed": "{broken", "array": "[]"}[shape]
        path.write_text(text, encoding="utf-8")
    assert evals_cli.main(["compare", str(candidate), "--baseline", str(path)]) == 2
    assert path.name in capsys.readouterr().err


def test_saved_report_accepts_a_utf8_bom(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"overall":{"choice_accuracy":0.9}}', encoding="utf-8-sig")
    assert evals_cli.main(["compare", str(path), "--baseline", str(path)]) == 0


def test_agreement_saved_runs_accept_a_utf8_bom(tmp_path, capsys):
    paths = []
    for name in ("reference", "repeat", "student"):
        run = {"schema": "laya-agreement-run/1", "run_id": name,
               "provenance": {"model": "local", "revision": None, "settings": {}},
               "cases": [{"id": "one", "state": "s", "questions": QUESTIONS, "answers": {"q": "a"}}]}
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(run), encoding="utf-8-sig")
        paths.append(str(path))
    assert evals_cli.main(["agreement", *paths, "--bootstrap-samples", "100"]) == 0
    assert json.loads(capsys.readouterr().out)["overall"]["student_teacher_agreement"] == 1.0


@pytest.mark.parametrize("raw", ["inf", "-inf", "1e999", "nan"])
def test_cli_gate_parameters_must_be_finite(raw):
    with pytest.raises(evals.EvalError):
        evals_cli._parse_pairs(["choice_accuracy=" + raw])
    with pytest.raises(argparse.ArgumentTypeError):
        evals_cli._limit(raw)


def test_threshold_names_its_own_metric_and_accuracy_shorthand_resolves(tmp_path, monkeypatch, capsys):
    class Router:
        def __init__(self, **kwargs):
            pass

        def predict(self, state, questions, model=None):
            return {"answers": {"q": {"type": "noul", "noul": 0.9}}}

        def predict_batch(self, requests, **kwargs):
            return [self.predict(None, None) for _ in requests]

    monkeypatch.setattr(laya, "Router", Router)
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps({"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}},
                                "expected": {"q": True}}), encoding="utf-8")
    assert evals_cli.main(["run", str(path), "--min-accuracy", "0.9"]) == 0
    assert evals_cli.main(["run", str(path), "--min", "choice_accuracy=0.9"]) == 1
    assert "choice_accuracy' is not in the report" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["--json", "--markdown"])
def test_unwritable_report_is_a_usage_error(tmp_path, monkeypatch, name, capsys):
    monkeypatch.setattr(laya, "Router", lambda **kwargs: object())
    monkeypatch.setattr(evals, "evaluate", lambda *args, **kwargs: evals.EvalReport())
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps({"state": "s", "questions": QUESTIONS, "expected": {"q": "a"}}),
                    encoding="utf-8")
    assert evals_cli.main(["run", str(path), name, str(tmp_path)]) == 2
    assert "cannot write" in capsys.readouterr().err


def test_confidence_flags_are_not_probabilities():
    assert evals._answer_confidence({"type": "choice", "answer_confidence": True,
                                     "confidence": 0.4}) == 0.4
    assert evals._answer_confidence({"type": "choice", "confidence": False}) is None
    assert evals._answer_confidence({"type": "noul", "confidence": False, "noul": 0.8}) == 0.8


def test_markdown_slice_labels_do_not_change_the_table_shape():
    report = evals.EvalReport(slices={"tag": {"risk\\|high\r\nnext": {"choice_accuracy": 0.5}}})
    assert "| risk\\\\\\|high  next | 0.5000 |" in report.to_markdown()


@pytest.mark.parametrize("incomparable", [False, True])
def test_regression_assertion_names_non_numeric_failure(incomparable):
    report = evals.EvalReport(config={"coverage_metric_definition": evals.COVERAGE_METRIC_DEFINITION},
                             overall={"aurc": 0.1 if incomparable else float("nan")})
    with pytest.raises(AssertionError, match="aurc") as exc:
        evals.assert_regression(report, {"overall": {"aurc": 0.1}})
    if incomparable:
        assert "incomparable" in str(exc.value)


@pytest.mark.parametrize("pins", [["english=a", "english=b"], ["en=a", "English=b"]])
def test_revision_aliases_cannot_hide_conflicting_checkpoint_pins(pins):
    with pytest.raises(evals.EvalError, match="two commits"):
        evals_cli._parse_revisions(pins)
    assert evals_cli._parse_revisions(["en=a", "english=a"]) == (None, {"en": "a", "english": "a"})


@pytest.mark.parametrize("items", [True, False, "1", 1.0, None])
def test_calibration_item_counts_must_be_integers(items):
    result = evidence.inspect_calibration({"training": {"laya_train_calibration": {
        "choice": {"items": items, "issues": []}}}})
    assert result["types"]["choice"]["state"] == evidence.UNKNOWN


def test_evidence_reads_identity_from_either_report_location():
    report = {"schema": "laya-evals-report/1", "dataset_sha256": "a", "questions_sha256": "b"}
    assert evidence.inspect_report(report)["state"] == evidence.PRESENT
    assert evidence.relate({"dataset_sha256": "other"}, report)["state"] == evidence.INCOMPARABLE
    report["config"] = {"dataset_sha256": "other"}
    assert evidence.relate({"dataset_sha256": "other"}, report)["state"] == evidence.UNKNOWN


@pytest.mark.parametrize("artifact", ["config", "report"])
@pytest.mark.parametrize("shape", ["bom", "malformed", "encoding", "depth", "array"])
def test_evidence_artifacts_name_the_file_or_accept_bom(tmp_path, artifact, shape):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "rl_agent_config.json"
    config.write_text("{}", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    target = config if artifact == "config" else report
    if shape == "encoding":
        target.write_bytes(b"\xff\xfe")
    else:
        value = {"bom": "{}", "malformed": "{broken", "depth": "[" * 5000 + "0" + "]" * 5000,
                 "array": "[]"}[shape]
        target.write_text(value, encoding="utf-8-sig" if shape == "bom" else "utf-8")
    if shape == "bom":
        assert evidence.inspect_checkpoint(str(checkpoint), str(report))
    else:
        with pytest.raises(ValueError, match=target.name.replace(".", r"\.")) as exc:
            evidence.inspect_checkpoint(str(checkpoint), str(report))
        if shape == "depth":
            assert "nested deeper" in str(exc.value)


def test_not_fitted_calibration_is_not_described_as_fitted():
    result = evidence.inspect_calibration({"training": {"laya_train_calibration": {
        "choice": {"items": 5, "issues": ["not fitted: too few items"]}}}})
    assert result["types"]["choice"]["reason"].startswith("limitations recorded:")


def test_strict_json_accepts_ordered_containers_without_widening_scalars():
    from laya import _agreement

    class Rows(list):
        pass

    class Number(int):
        pass

    assert _agreement._json(OrderedDict([("x", Rows([1, True]))]), "input") == '{"x":[1,true]}'
    with pytest.raises(evals.EvalError, match="strict JSON"):
        _agreement._json(OrderedDict([(1, "numeric key")]), "input")
    with pytest.raises(evals.EvalError, match="strict JSON"):
        _agreement._json({"x": Number(1)}, "input")
