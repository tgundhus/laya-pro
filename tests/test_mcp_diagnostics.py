"""Opt-in MCP routing diagnostics, including actual MCP argument parsing; no weights."""
import asyncio
import copy
import json

import pytest

pytest.importorskip("mcp")

from laya import BaseHook, Router  # noqa: E402
from laya.mcp import server as server_mod  # noqa: E402
from laya.mcp.tools import ToolError, laya_route, laya_route_batch  # noqa: E402

STATE = {"message": "Turn lights off"}
QUESTIONS = {"pick": {"type": "choice", "instructions": "Pick a team", "criteria": {"A": None, "B": None}}}
DIAGNOSTIC = {"stage": "whole_state", "reason": "insufficient_words", "source": None}
ROUTING = {"model": "english", "repo": "example/english", "reason": "test route"}
INVALID_FLAGS = [None, 0, 1, 0.0, "false", "true", "", [], {}]


def call(tool, router, **options):
    if tool == "laya_route":
        return laya_route(STATE, QUESTIONS, router=router, **options)
    return laya_route_batch([{"state": STATE, "questions": QUESTIONS}], router=router, **options)


@pytest.mark.parametrize("tool", ["laya_route", "laya_route_batch"])
def test_default_keeps_legacy_call_shape_and_requested_explain_is_not_dropped(tool):
    class Legacy:
        calls = 0

        def route(self, state, questions):
            self.calls += 1
            return dict(ROUTING)

        def route_batch(self, requests):
            self.calls += 1
            return [dict(ROUTING)]

    router = Legacy()
    first = call(tool, router)
    assert call(tool, router, explain=False) == first
    decision = first if tool == "laya_route" else first["decisions"][0]
    assert decision == ROUTING
    with pytest.raises(ToolError) as error:
        call(tool, router, explain=True)
    assert error.value.code == "unsupported_explain"
    assert router.calls == 2


@pytest.mark.parametrize("tool", ["laya_route", "laya_route_batch"])
@pytest.mark.parametrize("bad", INVALID_FLAGS)
def test_boolean_validation_precedes_dispatch(tool, bad):
    class Unused:
        def route(self, *args, **kwargs):
            raise AssertionError("invalid flag reached routing")

        route_batch = route

    with pytest.raises(ToolError) as error:
        call(tool, Unused(), explain=bad)
    assert error.value.code == "invalid_explain"


def test_core_diagnostic_preserves_route_and_reports_nested_source_without_text():
    state = {"note": "This is an English note about the order. " * 20,
             "payload": {"customer": "Ich kann mich nicht anmelden und brauche Hilfe"}}
    original = copy.deepcopy(state)
    router = Router()
    plain = laya_route(state, QUESTIONS, router=router)
    explained = laya_route(state, QUESTIONS, router=router, explain=True)
    assert explained.pop("diagnostic") == {
        "stage": "segment", "reason": "named_segment",
        "source": {"leaf_index": 1, "path": ["payload", "customer"], "line": 1},
    }
    assert explained == plain
    assert set(explained) == {"model", "repo", "reason"}
    assert state == original


@pytest.mark.parametrize("override", [{"model": "english"}, {"task": "typed_decisions"},
                                     {"lang": "de"}, {"lang_guess": "de"}])
def test_routing_overrides_return_null_diagnostic(override):
    router = Router()
    out = laya_route(STATE, QUESTIONS, router=router, explain=True, **override)
    assert out.pop("diagnostic") is None
    assert out == laya_route(STATE, QUESTIONS, router=router, **override)


def test_batch_uses_final_hook_decisions_and_preserves_timeout_and_counts():
    class Replace(BaseHook):
        def on_route(self, ctx):
            if ctx.states[0].get("replace"):
                ctx.decision = dict(ROUTING, detection={"private": "not exposed"})

    router = Router(hooks=[Replace()])
    requests = [{"state": STATE, "questions": QUESTIONS},
                {"state": {"replace": True}, "questions": QUESTIONS}]
    out = laya_route_batch(requests, hooks_timeout=2.5, explain=True, router=router)
    assert out["decisions"] == [dict(laya_route(STATE, QUESTIONS, router=router), diagnostic=DIAGNOSTIC),
                                dict(ROUTING, diagnostic=None)]
    assert out["model_counts"] == {"english": 2}
    assert "not exposed" not in json.dumps(out)


@pytest.mark.parametrize("tool", ["laya_route", "laya_route_batch"])
def test_internal_type_errors_are_not_retried_or_relabelled(tool):
    error = TypeError("inner.route() got an unexpected keyword argument 'explain'")

    class Broken:
        calls = 0

        def route(self, *args, **kwargs):
            self.calls += 1
            raise error

        route_batch = route

    router = Broken()
    with pytest.raises(TypeError) as info:
        call(tool, router, explain=True)
    assert info.value is error
    assert router.calls == 1


@pytest.mark.parametrize("tool", ["laya_route", "laya_route_batch"])
@pytest.mark.parametrize("flag", [False, True, *INVALID_FLAGS])
def test_real_mcp_schema_and_strict_boolean_parsing(tool, flag, monkeypatch):
    from mcp.server.mcpserver.exceptions import ToolError as McpToolError

    router = Router()
    monkeypatch.setattr(server_mod, "_ROUTER", router)
    schema = next(t for t in asyncio.run(server_mod.server.list_tools()) if t.name == tool).input_schema
    assert schema["properties"]["explain"]["type"] == "boolean"
    assert schema["properties"]["explain"]["default"] is False
    arguments = {"state": STATE, "questions": QUESTIONS} if tool == "laya_route" else {
        "requests": [{"state": STATE, "questions": QUESTIONS}],
    }
    arguments["explain"] = flag
    if type(flag) is not bool:
        with pytest.raises(McpToolError, match="explain"):
            asyncio.run(server_mod.server.call_tool(tool, arguments))
        return
    result = asyncio.run(server_mod.server.call_tool(tool, arguments))
    assert not result.is_error
    out = json.loads(result.content[0].text)
    decision = out if tool == "laya_route" else out["decisions"][0]
    assert ("diagnostic" in decision) is flag
    if flag:
        assert decision["diagnostic"] == DIAGNOSTIC


def test_explanation_does_not_leak_mutable_diagnostics_into_later_routes():
    state = {"note": "This is an English note about the order. " * 20,
             "payload": {"customer": "Ich kann mich nicht anmelden und brauche Hilfe"}}
    router = Router()
    first = router.route(state, QUESTIONS, explain=True)
    first["detection"]["diagnostic"]["source"]["path"].append("caller mutation")
    again = router.route(state, QUESTIONS, explain=True)
    assert again["detection"]["diagnostic"]["source"]["path"] == ["payload", "customer"]
    assert "diagnostic" not in router.route(state, QUESTIONS)["detection"]
