"""Resource presets, startup ownership and inference chunking, without model weights.

Run: python tests/test_performance_modes.py
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from laya import performance  # noqa: E402
from laya.agent import Agent, load  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402
from laya.router import Router  # noqa: E402


QUESTIONS = {"q": {"type": "choice", "instructions": "?", "criteria": {"yes": "y", "no": "n"}}}
STATE_STATS = {"state_tokens": 3, "state_tokens_used": 3, "state_tokens_dropped": 0, "truncated": False}


class _Tokenizer:
    pad_token_id = 0
    mask_token = "[M]"
    cls_token_id, sep_token_id, mask_token_id = 101, 102, 103

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = list(range(len(text)))
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}

    def decode(self, ids):
        return "x" * len(ids)


def _runtime(cls, mode=None):
    runtime = cls.__new__(cls)
    runtime.cfg = {"max_len": 100, "head_max_len": 20}
    runtime.tok = _Tokenizer()
    runtime.performance_config = performance.resolve_performance_mode(mode or "default")
    runtime.performance_mode = mode
    runtime.calls = []
    runtime._encode_state = lambda state, ids, internal, *args, **kwargs: [
        {"ids": [1, 2, 3], "markers": [0, 1], "qtype": 0, "state_stats": STATE_STATS} for _ in ids]
    runtime._decode_answers = lambda logits, act, items, ids, internal, row, **kwargs: {
        qid: {"type": "choice", "choice": "yes", "confidence": 0.8, "answer_confidence": 0.9} for qid in ids}

    def forward(batch):
        n = batch["input_ids"].shape[0]
        runtime.calls.append(n)
        return np.zeros((n, 2), dtype=np.float32), np.zeros((n, 2), dtype=np.float32)

    if cls is Agent:
        runtime._forward = forward
    else:
        runtime.session = SimpleNamespace(run=lambda outputs, inputs: forward(inputs))
    return runtime


class PerformanceModesTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"LAYA_PERFORMANCE_MODE": "", "LAYA_THREADS": "", "LAYA_CPU_AMP": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_parser_and_fresh_presets(self):
        for value in (None, "", "  ", "default", "DEFAULT"):
            self.assertIsNone(performance.parse_performance_mode(value))
        for level in range(1, 6):
            self.assertEqual(performance.parse_performance_mode(" %d " % level), level)
            config = performance.resolve_performance_mode(level)
            self.assertEqual(config["mode"], level)
            self.assertEqual(set(config), {"mode", "name", "threads", "batch_size", "max_loaded",
                                           "preload", "max_batch_tokens", "max_concurrent"})
            config["batch_size"] = 999
            self.assertNotEqual(performance.resolve_performance_mode(level)["batch_size"], 999)
        for value in (True, False, 1.0, 0, 6, -1, "1.0", "fast", [], {}, "²"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                performance.parse_performance_mode(value)

    def test_environment_and_thread_override(self):
        with patch.dict(os.environ, {"LAYA_PERFORMANCE_MODE": "4"}), \
                patch.object(performance, "_available_cpus", return_value=3):
            self.assertEqual(performance.resolve_performance_mode()["threads"], 3)
            self.assertEqual(performance.resolve_performance_mode(1)["mode"], 1)
            self.assertIsNone(performance.resolve_performance_mode("default"))
            self.assertIsNone(performance.resolve_performance_mode(""))
            for threads in ("7", " 2 "):
                with patch.dict(os.environ, {"LAYA_THREADS": threads}):
                    self.assertEqual(performance.resolve_performance_mode()["threads"], int(threads))
            for threads in ("0", "-1", "bad"):
                with patch.dict(os.environ, {"LAYA_THREADS": threads}):
                    self.assertEqual(performance.resolve_performance_mode()["threads"], 3)
        with patch.dict(os.environ, {"LAYA_PERFORMANCE_MODE": "6"}), self.assertRaises(ValueError):
            Router()

    def test_affinity_caps_logical_cpus(self):
        with patch.object(performance.os, "cpu_count", return_value=32), \
                patch.object(performance.os, "process_cpu_count", return_value=32, create=True), \
                patch.object(performance.os, "sched_getaffinity", return_value={0, 1}, create=True):
            self.assertEqual(performance.resolve_performance_mode(5)["threads"], 2)

    def test_default_router_stays_lazy(self):
        script = """
import os, sys
import laya
r = laya.Router()
assert r.performance_mode is None and r.performance_config is None
assert r.max_loaded == 2 and r.loaded == []
os.environ['LAYA_PERFORMANCE_MODE'] = '5'
r = laya.Router(performance_mode='default')
assert r.performance_config is None and r.loaded == []
r.route('Please refund this invoice')
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    def test_router_overrides_and_bounded_preload(self):
        with patch("torch.set_num_threads") as threads, \
                patch.object(Router, "_build", return_value=SimpleNamespace()):
            router = Router(performance_mode=4, max_loaded=1, default="multilingual")
            self.assertEqual(router.loaded, ["multilingual"])
            self.assertEqual(router.max_loaded, 1)
            self.assertEqual(router.performance_config["max_loaded"], 1)
            self.assertTrue(router.performance_config["preload"])
            self.assertEqual(threads.call_count, 1)
            router = Router(performance_mode=5, preload=False)
            self.assertEqual(router.loaded, [])
            self.assertFalse(router.performance_config["preload"])
            router = Router(performance_mode=1, preload=True)
            self.assertEqual(len(router.loaded), 3)
            self.assertEqual(router.performance_config["max_loaded"], 3)
            self.assertTrue(router.performance_config["preload"])
        with self.assertRaisesRegex(ValueError, "Router sets.*performance_mode"):
            Router(performance_mode=2, agent_kwargs={"performance_mode": 1})

    def test_router_lazy_load_does_not_reapply_global_threads(self):
        def build(repo, **kwargs):
            config = performance.resolve_performance_mode(kwargs["performance_mode"])
            performance._apply_torch_threads(config)
            return SimpleNamespace(performance_mode=config["mode"], performance_config=config)

        with patch("torch.set_num_threads") as threads, \
                patch.object(performance, "_available_cpus", return_value=16), \
                patch("laya.agent.Agent", side_effect=build):
            router = Router(performance_mode=2)
            threads.assert_called_once_with(2)
            with patch.dict(os.environ, {"LAYA_PERFORMANCE_MODE": "5", "LAYA_THREADS": "7"}):
                agent = router.load("english")
            threads.assert_called_once_with(2)
            self.assertEqual(agent.performance_mode, 2)
            self.assertEqual(agent.performance_config["threads"], 2)

    def test_batch_presets_and_explicit_overrides(self):
        for cls in (Agent, ONNXAgent):
            for mode, size, expected in ((None, None, [5]), (2, None, [2, 2, 1]),
                                          (1, 3, [3, 2]), (1, 0, [5])):
                with self.subTest(cls=cls.__name__, mode=mode, size=size):
                    runtime = _runtime(cls, mode)
                    result = runtime.predict_batch(["x"] * 5, QUESTIONS, batch_size=size)
                    self.assertEqual(len(result), 5)
                    self.assertEqual(runtime.calls, expected)

    def test_long_scans_apply_preset_and_explicit_override(self):
        for cls in (Agent, ONNXAgent):
            for size in (None, 3, 0):
                with self.subTest(cls=cls.__name__, size=size):
                    runtime = _runtime(cls, 1)
                    result = runtime.predict_long("x" * 250, QUESTIONS, batch_size=size)
                    windows = result["usage"]["windows"]
                    self.assertGreater(windows, 3)
                    limit = 1 if size is None else size
                    self.assertLessEqual(max(runtime.calls), limit if limit else windows)
                    if size == 0:
                        self.assertEqual(runtime.calls, [windows])

    def test_router_batch_and_long_forward_effective_limit(self):
        agent = _runtime(Agent)
        with patch("torch.set_num_threads"):
            router = Router(performance_mode=2)
        router.attach("english", agent)
        requests = [{"state": "x", "questions": QUESTIONS, "model": "english"}] * 5
        router.predict_batch(requests)
        self.assertEqual(agent.calls, [2, 2, 1])
        agent.calls.clear()
        router.predict_batch(requests, batch_size=0)
        self.assertEqual(agent.calls, [5])
        agent.calls.clear()
        router.predict_long("x" * 250, QUESTIONS, model="english")
        self.assertLessEqual(max(agent.calls), 2)

    def test_load_forwards_mode_to_both_backends(self):
        with patch("laya.agent.Agent", return_value="torch") as build:
            self.assertEqual(load("./checkpoint", performance_mode="default"), "torch")
            self.assertEqual(build.call_args.kwargs["performance_mode"], "default")
        with patch("laya.onnx_agent.ONNXAgent", return_value="onnx") as build:
            self.assertEqual(load("./checkpoint", backend="onnx", performance_mode=3), "onnx")
            self.assertEqual(build.call_args.kwargs["performance_mode"], 3)

    def test_agent_startup_and_onnx_session_settings(self):
        class Model:
            encoder = SimpleNamespace(config=SimpleNamespace())

            def load_state_dict(self, weights, strict):
                pass

            def to(self, device):
                return self

            def eval(self):
                return self

        sessions = []

        def session(path, **kwargs):
            sessions.append(kwargs)
            return SimpleNamespace()

        ort = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=session,
                              GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=99),
                              get_available_providers=lambda: ["CPUExecutionProvider"])
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: _Tokenizer()))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "rl_agent_config.json").write_text(json.dumps({"temperature": [1.0] * 3}))
            (checkpoint / "model.safetensors").touch()
            graph = checkpoint / "laya.onnx"
            graph.touch()
            with patch("torch.set_num_threads") as threads, \
                    patch("laya.agent.build_model", return_value=Model()), \
                    patch("laya.agent._load_tokenizer", return_value=_Tokenizer()), \
                    patch("laya.agent._verify_compatibility"), \
                    patch("safetensors.torch.load_file", return_value={}), \
                    patch.dict(sys.modules, {"onnxruntime": ort, "transformers": transformers}), \
                    patch.dict(os.environ, {"LAYA_PERFORMANCE_MODE": "2", "LAYA_THREADS": "3"}):
                agent = Agent(directory, device="cpu")
                self.assertEqual(agent.performance_mode, 2)
                self.assertEqual(agent.performance_config["threads"], 3)
                threads.assert_called_once_with(3)
                self.assertEqual(agent.backend, "eager")
                self.assertEqual(str(agent.dtype), "torch.float32")
                legacy = Agent(directory, device="cpu", performance_mode="default")
                self.assertEqual(agent._checkpoint_signature, legacy._checkpoint_signature)
                self.assertEqual(agent.dtype, legacy.dtype)
                threads.assert_called_once_with(3)
                onnx = ONNXAgent(directory, str(graph))
                self.assertEqual(onnx.performance_mode, 2)
                self.assertEqual(sessions[-1]["sess_options"].intra_op_num_threads, 3)
                self.assertEqual(sessions[-1]["providers"], ["CPUExecutionProvider"])
                ONNXAgent(directory, str(graph), performance_mode="default")
                self.assertFalse(hasattr(sessions[-1]["sess_options"], "intra_op_num_threads"))
                threads.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
