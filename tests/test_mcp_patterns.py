"""MCP tournament and long-state adapters, with real routing and weight-free agents."""
import asyncio
import copy
import json

import pytest

pytest.importorskip("mcp")

from laya import BaseHook, Router  # noqa: E402
from laya.mcp import server as server_mod  # noqa: E402
from laya.mcp.tools import (  # noqa: E402
    ToolError, laya_predict_batch, laya_predict_long, laya_route_batch, laya_tournament,
)

STATE = {"body": "I need help with my payment"}
QUESTIONS = {"pick": {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}}}


class RecordingAgent:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def predict(self, state, questions, **options):
        self.calls.append((copy.deepcopy(questions), options))
        answers = {}
        for name, spec in questions.items():
            if spec["type"] == "choice":
                labels = list(spec["criteria"])
                answers[name] = {"type": "choice", "choice": labels[0], "confidence": 1.,
                                 "answer_confidence": 1., "probabilities": {
                                     label: float(i == 0) for i, label in enumerate(labels)}}
            else:
                answers[name] = {"type": "noul", "noul": 0.8, "confidence": 0.8, "answer_confidence": 0.8}
        return {"answers": answers, "usage": {"input_tokens": 10, "output_tokens": 0}}

    system_one = predict

    def predict_batch(self, states, questions, **options):
        return [self.predict(state, questions, **options) for state in states]

    def predict_long(self, state, questions, **options):
        result = self.predict(state, questions, **options)
        result["usage"]["windows"] = 3
        for answer in result["answers"].values():
            answer["window"] = {"index": 2, "token_start": 20, "token_end": 40}
        return result


@pytest.mark.parametrize("tool", [laya_tournament, laya_predict_long])
@pytest.mark.parametrize("control", ["model", "task"])
def test_registered_models_are_resolved_for_new_tools(tool, control):
    agent = RecordingAgent()
    router = Router()
    router.attach("Special", agent)
    out = tool(STATE, QUESTIONS, router=router, **{control: "SPECIAL"})
    assert out["routing"]["model"] == "special"
    assert len(agent.calls) == 1


def test_batch_task_accepts_registered_names_and_refuses_unknown_names_before_running():
    router = Router()
    router.attach("special", RecordingAgent())
    request = {"state": STATE, "questions": QUESTIONS, "task": "SPECIAL"}
    assert laya_route_batch([request], router=router, explain=True)["decisions"][0] == {
        "model": "special", "repo": None, "reason": "explicit task='special'", "diagnostic": None,
    }
    assert laya_predict_batch([request], router=router)["model_counts"] == {"special": 1}
    request["task"] = "unknown"
    with pytest.raises(ToolError) as error:
        laya_route_batch([request], router=router)
    assert error.value.code == "invalid_task"


def test_tournament_routes_original_workflow_and_pins_model_and_language_for_every_round():
    agent = RecordingAgent()
    routes = []

    class Observe(BaseHook):
        def on_route(self, ctx):
            routes.append((set(ctx.questions), ctx.decision["model"]))

    router = Router(auto_task_detection=True, hooks=[Observe()])
    router.attach("typed-decisions", agent)
    questions = {name: {"type": "noul", "instructions": "Relevant?"}
                 for name in ["action", "category", "churn_risk", "needs_human", "urgency"]}
    questions["category"] = {"type": "choice", "instructions": "Pick a category",
                             "criteria": {"opt%d" % i: None for i in range(10)},
                             "option_order": list(reversed(range(10)))}
    original = copy.deepcopy(questions)
    with pytest.warns(RuntimeWarning, match="option_order"):
        out = laya_tournament(STATE, questions, group_size=3, lang="de", max_len=128,
                              head_max_len=64, router=router)
    assert out["routing"]["model"] == "typed-decisions"
    assert routes[0][0] == set(questions)
    assert all(model == "typed-decisions" for _, model in routes)
    assert len(agent.calls) == 3
    assert all(options == {"lang": "de", "max_len": 128, "head_max_len": 64}
               for _, options in agent.calls)
    assert out["tournament"]["category"] == {"labels": ["opt0", "opt5"], "n": 10, "rounds": 2}
    assert out["option_order_dropped"] == {"category": {"n": 10, "kept": 2}}
    assert out["device"] == "cpu"
    assert set(out["answers"]) == set(questions)
    assert questions == original


@pytest.mark.parametrize("bad", [None, "outside", 7])
def test_tournament_refuses_missing_or_out_of_group_winners_before_more_passes(bad):
    class Invalid(RecordingAgent):
        def predict(self, state, questions, **options):
            out = super().predict(state, questions, **options)
            if bad is None:
                out["answers"].clear()
            else:
                for answer in out["answers"].values():
                    answer["choice"] = bad
            return out

    agent = Invalid()
    questions = {"pick": dict(QUESTIONS["pick"], criteria={str(i): None for i in range(8)})}
    with pytest.raises(ToolError) as error:
        laya_tournament(STATE, questions, group_size=2, model="english", agent=agent)
    assert error.value.code == "internal_error"
    assert len(agent.calls) == 1


@pytest.mark.parametrize(("tool", "option", "value", "code"), [
    (laya_tournament, "group_size", True, "invalid_group_size"),
    (laya_tournament, "group_size", 1, "invalid_group_size"),
    (laya_tournament, "group_size", 2.0, "invalid_group_size"),
    (laya_predict_long, "window", False, "invalid_window"),
    (laya_predict_long, "stride", 0, "invalid_stride"),
    (laya_predict_long, "window", "64", "invalid_window"),
])
def test_new_controls_are_type_preserving_and_checked_before_inference(tool, option, value, code):
    agent = RecordingAgent()
    with pytest.raises(ToolError) as error:
        tool(STATE, QUESTIONS, agent=agent, **{option: value})
    assert error.value.code == code
    assert agent.calls == []


@pytest.mark.parametrize("bad_type", [[], {}])
def test_container_question_type_is_a_caller_error(bad_type):
    agent = RecordingAgent()
    with pytest.raises(ToolError) as error:
        laya_predict_long(STATE, {"q": {"type": bad_type, "instructions": "Pick"}}, agent=agent)
    assert error.value.code == "invalid_questions"
    assert agent.calls == []


def test_long_scan_preserves_window_attribution_and_only_forwards_set_controls():
    agent = RecordingAgent()
    out = laya_predict_long(STATE, QUESTIONS, agent=agent, window=24, stride=12, lang="de")
    assert out["windows"] == 3
    assert out["answers"]["pick"]["window"] == {"index": 2, "token_start": 20, "token_end": 40}
    assert out["routing"]["model"] is None
    assert "device" not in out
    assert agent.calls[0][1] == {"window": 24, "stride": 12, "lang": "de"}
    laya_predict_long(STATE, QUESTIONS, model="english", agent=agent)
    assert agent.calls[1][1] == {}
    with pytest.raises(ToolError) as error:
        laya_predict_long(STATE, QUESTIONS, agent=object())
    assert error.value.code == "internal_error"


@pytest.mark.parametrize("tool", ["laya_tournament", "laya_predict_long"])
@pytest.mark.parametrize("model", [{}, {"model": None}, {"model": "auto"}])
def test_new_tools_are_callable_through_real_mcp_parser(tool, model, monkeypatch):
    router = Router()
    router.attach("english", RecordingAgent())
    monkeypatch.setattr(server_mod, "_ROUTER", router)
    result = asyncio.run(server_mod.server.call_tool(tool, {"state": STATE, "questions": QUESTIONS, **model}))
    assert not result.is_error
    out = json.loads(result.content[0].text)
    assert out["routing"]["model"] == "english"
    assert out["answers"]["pick"]["choice"] == "a"
