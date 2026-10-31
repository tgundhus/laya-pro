"""Run installed-package inference with heavy imports actively forbidden."""
import importlib.abc
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import unittest


class NoHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "transformers", "laya"):
            raise AssertionError("light runtime imported " + fullname)


sys.meta_path.insert(0, NoHeavyImports())
import laya_onnx  # noqa: E402


FIXTURE = Path(sys.argv[1])
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {2: "billing", 7: "help"}, "option_order": [1, 0]},
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "severity": {"type": "score", "instructions": "How severe?", "criteria": ["low", "high"]},
}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.agent = laya_onnx.ONNXAgent(str(FIXTURE), str(FIXTURE / "model.onnx"), performance_mode=1)

    def test_distribution_is_independent_and_has_only_light_dependencies(self):
        for name in ("torch", "transformers", "laya-pro", "onnx", "onnxscript", "safetensors"):
            with self.assertRaises(importlib.metadata.PackageNotFoundError):
                importlib.metadata.distribution(name)
        self.assertEqual(importlib.metadata.version("laya-pro-onnx"), laya_onnx.__version__)
        self.assertEqual(self.agent.session.get_providers(), ["CPUExecutionProvider"])
        self.assertEqual(self.agent.performance_mode, 1)
        self.assertIs(laya_onnx.Agent, laya_onnx.ONNXAgent)
        loaded = laya_onnx.load(str(FIXTURE), onnx_path=str(FIXTURE / "model.onnx"), performance_mode="default")
        self.assertIsInstance(loaded, laya_onnx.ONNXAgent)
        self.assertIsNone(loaded.performance_mode)

    def test_predict_batch_preserves_typed_labels_and_order(self):
        states = ["hello billing", "billing help", "hello " * 12]
        answers = self.agent.predict_batch(states, QUESTIONS)
        self.assertEqual(len(answers), len(states))
        for result in answers:
            self.assertEqual(result["answers"]["team"]["choice"], 7)
            self.assertEqual(set(result["answers"]["team"]["probabilities"]), {2, 7})
            self.assertIsInstance(result["answers"]["urgent"]["noul"], float)
            self.assertIsInstance(result["answers"]["severity"]["score"], float)

    def test_hooks_and_retention_replay_without_running_graph(self):
        original = self.agent.session
        calls = []

        class Session:
            def run(self, *args, **kwargs):
                calls.append(1)
                return original.run(*args, **kwargs)

        self.agent.session = Session()
        with tempfile.TemporaryDirectory() as tmp:
            cache = laya_onnx.DecisionCache(str(Path(tmp) / "decisions.sqlite"), ttl=30)
            try:
                with self.agent.hooks_installed(cache):
                    first = self.agent.predict("billing help", QUESTIONS)
                    count = len(calls)
                    self.assertEqual(self.agent.predict("billing help", QUESTIONS)["answers"], first["answers"])
                    self.assertEqual(len(calls), count)
            finally:
                cache.close()

    def test_long_scan_calibration_and_structured_decisions(self):
        scanned = self.agent.predict_long("hello billing help " * 30, QUESTIONS, window=16, stride=8)
        self.assertGreater(scanned["usage"]["windows"], 1)
        self.assertTrue(all("window" in answer for answer in scanned["answers"].values()))
        schema = {"type": "object", "properties": {"urgent": {"type": "boolean"}}}
        self.assertIsInstance(self.agent.decide("billing help", schema)["urgent"], bool)
        with tempfile.TemporaryDirectory() as tmp:
            calibration = Path(tmp) / "calibration.json"
            calibration.write_text(json.dumps({"temperature": [1.5, 2, 1]}), encoding="utf-8")
            self.agent.load_calibration(str(calibration))
            self.assertEqual(self.agent.temperature, [1.5, 2, 1])
            self.agent.predict("hello", QUESTIONS)
        self.assertFalse(any(name.split(".")[0] in ("torch", "transformers", "laya") for name in sys.modules))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
