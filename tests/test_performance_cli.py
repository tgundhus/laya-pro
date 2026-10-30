"""Startup presets stay separate from prediction controls and JSON output."""
import json
from types import SimpleNamespace

import pytest

from laya import cli


@pytest.fixture(autouse=True)
def clean_mode(monkeypatch):
    monkeypatch.delenv("LAYA_PERFORMANCE_MODE", raising=False)
    monkeypatch.delenv("LAYA_THREADS", raising=False)


@pytest.mark.parametrize("level", range(1, 6))
def test_presets_reach_router_without_becoming_prediction_controls(level, monkeypatch, capsys):
    constructors = []
    calls = []

    class Router:
        def __init__(self, **kwargs):
            constructors.append(kwargs)
            self.performance_config = {"mode": level, "batch_size": level}

        def predict(self, state, questions, **kwargs):
            calls.append(kwargs)
            return {"answers": {"team": {"choice": "billing"}}}

    monkeypatch.setattr(cli.laya, "Router", Router)
    assert cli.main(["--performance-mode", str(level), "--predict", "--json", "Billing help"]) == 0
    out = capsys.readouterr()
    assert constructors == [{"device": None, "performance_mode": level}]
    assert "performance_mode" not in calls[0]
    assert json.loads(out.out)["answers"]["team"]["choice"] == "billing"
    assert "laya performance:" in out.err


def test_routing_never_preloads_checkpoints_even_with_dedicated_preset(monkeypatch):
    constructors = []
    monkeypatch.setattr(cli.laya, "Router", lambda **kwargs: constructors.append(kwargs) or object())
    cli.make_router(cli.build_parser().parse_args(["--performance-mode", "5", "Hello"]))
    assert constructors == [{"device": None, "performance_mode": 5, "preload": False}]


def test_omitted_mode_preserves_lazy_cli_constructor(monkeypatch):
    constructors = []
    monkeypatch.setattr(cli.laya, "Router", lambda **kwargs: constructors.append(kwargs) or object())
    cli.make_router(cli.build_parser().parse_args(["--predict", "Hello"]))
    assert constructors == [{"device": None, "preload": False}]


def test_environment_preset_and_explicit_default(monkeypatch):
    constructors = []
    monkeypatch.setenv("LAYA_PERFORMANCE_MODE", "4")
    monkeypatch.setattr(cli.laya, "Router", lambda **kwargs: constructors.append(kwargs) or object())
    cli.make_router(cli.build_parser().parse_args(["--predict", "Hello"]))
    cli.make_router(cli.build_parser().parse_args(["--performance-mode", "default", "--predict", "Hello"]))
    assert constructors == [{"device": None, "performance_mode": 4},
                            {"device": None, "performance_mode": "default", "preload": False}]


@pytest.mark.parametrize("value", ["0", "6", "-1", "true", "2.5"])
def test_invalid_cli_mode_fails_before_router_construction(value, monkeypatch):
    monkeypatch.setattr(cli, "make_router", lambda args: pytest.fail("invalid mode reached startup"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--performance-mode", value, "Hello"])
    assert exc.value.code == 2


def test_invalid_environment_is_a_friendly_error(monkeypatch, capsys):
    monkeypatch.setenv("LAYA_PERFORMANCE_MODE", "99")
    monkeypatch.setattr(cli, "make_router", lambda args: pytest.fail("invalid mode reached startup"))
    assert cli.main(["Hello"]) == 2
    assert "performance" in capsys.readouterr().err.lower()


def test_profile_load_failures_use_cli_error_handling(monkeypatch, capsys):
    def fail(args):
        raise RuntimeError("not enough memory")

    monkeypatch.setattr(cli, "make_router", fail)
    assert cli.main(["--performance-mode", "5", "--predict", "Hello"]) == 2
    assert "not enough memory" in capsys.readouterr().err


def test_explicit_batch_override_reaches_prediction(monkeypatch, capsys):
    calls = []
    router = SimpleNamespace(performance_config={"mode": 1, "batch_size": 1})
    router.predict_batch = lambda requests, **kwargs: calls.append(kwargs) or [
        {"answers": {}} for _ in requests]
    args = cli.build_parser().parse_args(["--performance-mode", "1", "--batch-size", "7", "--predict", "--json"])
    assert cli.run_batch(["first", "second"], args, router) == 0
    assert calls[0]["batch_size"] == 7
    assert len(capsys.readouterr().out.splitlines()) == 2
