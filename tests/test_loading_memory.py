"""Storage reuse must preserve strict copying semantics. Run: python tests/test_loading_memory.py"""
import copy
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import weakref

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from laya._loading import _checkpoint_identity, _load_weights  # noqa: E402


HAS_ASSIGN = "assign" in inspect.signature(torch.nn.Module.load_state_dict).parameters


class DenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, 3))
        self.bias = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
        self.register_buffer("temperature", torch.ones(3))
        self.register_buffer("count", torch.tensor(0))
        self.register_buffer("rotary", torch.arange(4.0), persistent=False)

    def forward(self, inputs):
        return (inputs @ self.weight + self.bias) / self.temperature


def weights_for(model):
    return {name: torch.full_like(value, index + 1)
            for index, (name, value) in enumerate(model.state_dict().items())}


def storage_bytes(tensors):
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
    return sum(storages.values())


class LoadingMemoryTests(unittest.TestCase):
    @unittest.skipUnless(HAS_ASSIGN, "torch 2.0 keeps the copy path")
    def test_matching_storage_is_reused_without_changing_values_or_buffers(self):
        model = DenseModel()
        expected = copy.deepcopy(model)
        weights = weights_for(model)
        expected.load_state_dict(weights, strict=True)
        old_parameters = [weakref.ref(p) for p in model.parameters()]
        rotary = model.rotary
        checkpoint_bytes = storage_bytes(weights.values())
        self.assertEqual(storage_bytes(list(model.state_dict().values()) + list(weights.values())),
                         2 * checkpoint_bytes)

        self.assertEqual(_load_weights(model, weights), "assign")

        self.assertTrue(all(ref() is None for ref in old_parameters))
        # These are the real live model/checkpoint allocations, not a calculated estimate:
        # strict copying leaves two storages per entry; assignment leaves one.
        self.assertEqual(storage_bytes(list(model.state_dict().values()) + list(weights.values())),
                         checkpoint_bytes)
        self.assertIs(model.temperature, weights["temperature"])
        self.assertIs(model.rotary, rotary)
        self.assertTrue(torch.equal(model.rotary, torch.arange(4.0)))
        self.assertTrue(model.weight.requires_grad)
        self.assertFalse(model.bias.requires_grad)
        for name, value in model.state_dict().items():
            self.assertEqual(value.dtype, expected.state_dict()[name].dtype)
            self.assertTrue(torch.equal(value, expected.state_dict()[name]))
        inputs = torch.arange(8.0).reshape(2, 4)
        self.assertTrue(torch.equal(model(inputs), expected(inputs)))

    def test_mixed_dtypes_keep_model_dtype_and_copy_cast_values(self):
        model = DenseModel()
        expected = copy.deepcopy(model)
        weights = weights_for(model)
        weights["weight"] = torch.linspace(-1, 1, 12).reshape(4, 3).to(torch.bfloat16)
        weights["temperature"] = torch.tensor([1.1, 2.2, 3.3], dtype=torch.float64)
        weights["count"] = torch.tensor(7, dtype=torch.int32)
        parameter, temperature = model.weight, model.temperature
        expected.load_state_dict(weights, strict=True)

        self.assertEqual(_load_weights(model, weights), "copy")

        self.assertIs(model.weight, parameter)
        self.assertIs(model.temperature, temperature)
        for name, value in model.state_dict().items():
            self.assertEqual(value.dtype, expected.state_dict()[name].dtype)
            self.assertTrue(torch.equal(value, expected.state_dict()[name]))

    def test_tied_parameters_buffers_and_checkpoint_aliases_keep_copy_semantics(self):
        for alias in ("parameter", "buffer", "nonpersistent", "checkpoint"):
            with self.subTest(alias=alias):
                model = DenseModel()
                if alias == "parameter":
                    model.other = model.weight
                elif alias in ("buffer", "nonpersistent"):
                    model.register_buffer("other", model.weight.detach(), persistent=alias == "buffer")
                weights = weights_for(model)
                if alias == "checkpoint":
                    weights["temperature"] = weights["bias"]
                expected = copy.deepcopy(model)
                if alias in ("buffer", "nonpersistent"):
                    expected.other = expected.weight.detach()
                expected.load_state_dict(weights, strict=True)
                parameter = model.weight

                self.assertEqual(_load_weights(model, weights), "copy")

                self.assertIs(model.weight, parameter)
                if alias != "checkpoint":
                    self.assertEqual(model.weight.data_ptr(), model.other.data_ptr())
                if alias == "parameter":
                    self.assertIs(model.weight, model.other)
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, expected.state_dict()[name]))

    def test_strides_views_and_parameter_metadata_keep_copy_path(self):
        for special in ("target_stride", "source_stride", "source_view", "metadata", "grad_hook"):
            with self.subTest(special=special):
                model = DenseModel()
                if special == "target_stride":
                    model.weight = torch.nn.Parameter(torch.zeros(3, 4).t())
                elif special == "metadata":
                    model.weight.marker = "retained"
                elif special == "grad_hook":
                    model.weight.register_hook(lambda grad: grad)
                weights = weights_for(model)
                if special == "source_stride":
                    weights["weight"] = torch.ones(3, 4).t()
                elif special == "source_view":
                    weights["weight"] = torch.ones(5, 3)[1:]
                parameter, stride = model.weight, model.weight.stride()

                self.assertEqual(_load_weights(model, weights), "copy")

                self.assertIs(model.weight, parameter)
                self.assertEqual(model.weight.stride(), stride)
                self.assertTrue(torch.equal(model.weight, weights["weight"]))
                if special == "metadata":
                    self.assertEqual(model.weight.marker, "retained")

    def test_missing_extra_and_wrong_shapes_have_the_original_strict_errors(self):
        for defect in ("missing_parameter", "missing_buffer", "extra", "shape"):
            with self.subTest(defect=defect):
                model = DenseModel()
                weights = weights_for(model)
                if defect == "missing_parameter":
                    del weights["weight"]
                elif defect == "missing_buffer":
                    del weights["temperature"]
                elif defect == "extra":
                    weights["extra"] = torch.zeros(1)
                else:
                    weights["weight"] = torch.zeros(1)
                with self.assertRaises(RuntimeError) as original:
                    copy.deepcopy(model).load_state_dict(weights, strict=True)
                with self.assertRaises(RuntimeError) as optimized:
                    _load_weights(model, weights)
                self.assertEqual(str(optimized.exception), str(original.exception))

    def test_legacy_torch_without_assign_keeps_the_copy_path(self):
        original = torch.nn.Module.load_state_dict
        calls = []

        def legacy_load(model, weights, strict=True):
            calls.append(strict)
            return original(model, weights, strict=strict)

        model = DenseModel()
        weights = weights_for(model)
        parameter = model.weight
        with patch.object(torch.nn.Module, "load_state_dict", legacy_load):
            self.assertEqual(_load_weights(model, weights, "not-opened.safetensors", ()), "copy")
        self.assertEqual(calls, [True])
        self.assertIs(model.weight, parameter)
        self.assertTrue(torch.equal(model.weight, weights["weight"]))

    def test_loading_hooks_keep_copy_semantics_and_errors_are_not_retried(self):
        model = DenseModel()
        weights = weights_for(model)
        parameter = model.weight
        calls = []

        def pre_hook(module, state, *args):
            calls.append(module)
            self.assertIs(module.weight, parameter)
            raise TypeError("loader error")

        model.register_load_state_dict_pre_hook(pre_hook)
        with self.assertRaisesRegex(TypeError, "loader error"):
            _load_weights(model, weights)
        self.assertEqual(calls, [model])
        self.assertIs(model.weight, parameter)

    def test_assignment_does_not_trigger_parameter_registration_hooks(self):
        register = getattr(torch.nn.modules.module, "register_module_parameter_registration_hook", None)
        if register is None:
            self.skipTest("this torch has no global parameter registration hook")
        model = DenseModel()
        weights, calls = weights_for(model), []
        parameter = model.weight
        handle = register(lambda *args: calls.append(True))
        try:
            self.assertEqual(_load_weights(model, weights), "copy")
        finally:
            handle.remove()
        self.assertEqual(calls, [])
        self.assertIs(model.weight, parameter)

    @unittest.skipUnless(HAS_ASSIGN, "torch 2.0 keeps the copy path")
    def test_real_mixed_checkpoint_copies_exactly_and_releases_each_mapping(self):
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file

        model = DenseModel()
        expected = copy.deepcopy(model)
        weights = weights_for(model)
        weights["weight"] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 1 / 3,
                                          2, -2, 65504, -65504, 0.0001, -0.0001], dtype=torch.float16).reshape(4, 3)
        weights["temperature"] = torch.tensor([1.1, 2.2, 3.3], dtype=torch.bfloat16)
        parameter, temperature, rotary = model.weight, model.temperature, model.rotary
        reads, streamed_refs = [], []
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.safetensors")
            save_file(weights, path)
            weights = load_file(path)
            expected.load_state_dict(weights, strict=True)
            initial_refs = [weakref.ref(tensor) for tensor in weights.values()]

            @contextmanager
            def tracked_open(*args, **kwargs):
                self.assertTrue(all(ref() is None for ref in initial_refs))
                self.assertTrue(all(ref() is None for ref in streamed_refs))
                with safe_open(*args, **kwargs) as checkpoint:
                    class Reader:
                        def get_tensor(self, name):
                            tensor = checkpoint.get_tensor(name)
                            reads.append(name)
                            streamed_refs.append(weakref.ref(tensor))
                            return tensor

                    yield Reader()
                    # A source must already be gone before its mapping's context closes.
                    self.assertTrue(all(ref() is None for ref in streamed_refs))

            with patch("safetensors.safe_open", side_effect=tracked_open):
                mode = _load_weights(model, weights, path, _checkpoint_identity(path))
        self.assertEqual(mode, "stream")
        self.assertEqual(weights, {})
        self.assertEqual(reads, list(model.state_dict()))
        self.assertIs(model.weight, parameter)
        self.assertIs(model.temperature, temperature)
        self.assertIs(model.rotary, rotary)
        self.assertFalse(model.bias.requires_grad)
        for name, value in model.state_dict().items():
            old = expected.state_dict()[name]
            self.assertEqual(value.dtype, old.dtype)
            self.assertEqual(value.stride(), old.stride())
            self.assertTrue(torch.equal(value.reshape(-1).view(torch.uint8), old.reshape(-1).view(torch.uint8)))

    @unittest.skipUnless(HAS_ASSIGN, "torch 2.0 keeps the copy path")
    def test_streaming_rejects_checkpoint_changes_before_and_between_reads(self):
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file

        for change in ("before", "replace", "malformed"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                model = DenseModel()
                path = str(Path(directory) / "model.safetensors")
                weights = weights_for(model)
                weights["weight"] = weights["weight"].half()
                save_file(weights, path)
                weights = load_file(path)
                identity = _checkpoint_identity(path)
                reads = []

                @contextmanager
                def changing_open(*args, **kwargs):
                    with safe_open(*args, **kwargs) as checkpoint:
                        reads.append(True)
                        yield checkpoint
                    if change == "replace":
                        replacement = path + ".replacement"
                        save_file(weights_for(model), replacement)
                        os.replace(replacement, path)
                    elif change == "malformed":
                        Path(path).write_bytes(b"malformed checkpoint")

                if change == "before":
                    stat = os.stat(path)
                    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
                with patch("safetensors.safe_open", side_effect=changing_open):
                    with self.assertRaisesRegex(RuntimeError, "Checkpoint changed while loading"):
                        _load_weights(model, weights, path, identity)
                self.assertEqual(len(reads), 0 if change == "before" else 1)

    def test_custom_loader_and_hooks_keep_full_strict_copy_with_a_path(self):
        from safetensors.torch import load_file, save_file

        class CustomModel(DenseModel):
            def _load_from_state_dict(self, *args, **kwargs):
                self.loaded = True
                return super()._load_from_state_dict(*args, **kwargs)

        class CustomAssignment(DenseModel):
            def __setattr__(self, name, value):
                super().__setattr__(name, value)

        for custom in ("loader", "assignment", "pre_hook", "post_hook", "alias"):
            with self.subTest(custom=custom), tempfile.TemporaryDirectory() as directory:
                model = (CustomModel() if custom == "loader" else CustomAssignment()
                         if custom == "assignment" else DenseModel())
                calls = []
                if custom == "pre_hook":
                    model.register_load_state_dict_pre_hook(lambda *args: calls.append("pre"))
                elif custom == "post_hook":
                    model.register_load_state_dict_post_hook(lambda *args: calls.append("post"))
                elif custom == "alias":
                    model.other = model.weight
                weights = weights_for(model)
                weights["weight"] = weights["weight"].half()
                path = str(Path(directory) / "model.safetensors")
                save_file(weights, path)
                weights = load_file(path)
                parameter = model.weight
                with patch("safetensors.safe_open", side_effect=AssertionError("must keep stock loader")):
                    self.assertEqual(_load_weights(model, weights, path, _checkpoint_identity(path)), "copy")
                self.assertIs(model.weight, parameter)
                self.assertTrue(weights)
                if custom == "loader":
                    self.assertTrue(model.loaded)
                elif custom == "assignment":
                    self.assertFalse(calls)
                elif custom == "alias":
                    self.assertIs(model.weight, model.other)
                else:
                    self.assertEqual(calls, ["pre" if custom == "pre_hook" else "post"])

    def test_agent_releases_mapping_before_compile_placement_fallback_and_warmup(self):
        from laya.agent import Agent
        from safetensors.torch import load_file as original_load_file, save_file

        class Weights(dict):
            pass

        for mixed_dtype in (False, True):
            with self.subTest(mixed_dtype=mixed_dtype), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "rl_agent_config.json").write_text(json.dumps({"encoder": "offline", "head_layers": 0}))
                mapping_refs, events = [], []

                def released(event):
                    self.assertIsNone(mapping_refs[0]())
                    events.append(event)

                class ProbeModel(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        for name in ("encoder", "type_emb", "scorer", "act_head"):
                            setattr(self, name, torch.nn.Linear(2, 2))
                        self.encoder.config = SimpleNamespace()

                    def to(self, device):
                        released(device.type)
                        if device.type == "cuda":
                            raise RuntimeError("simulated device allocation failure")
                        return super().to(device)

                model = ProbeModel()
                checkpoint_weights = weights_for(model)
                if mixed_dtype:
                    checkpoint_weights["encoder.weight"] = checkpoint_weights["encoder.weight"].to(torch.float16)
                save_file(checkpoint_weights, root / "model.safetensors")

                def load_file(path):
                    weights = Weights(original_load_file(path))
                    mapping_refs.append(weakref.ref(weights))
                    return weights

                def compile_model(model, **kwargs):
                    released("compile")
                    return model

                with (patch("laya.agent.build_model", return_value=model),
                      patch("laya.agent._load_tokenizer", return_value=object()),
                      patch("safetensors.torch.load_file", side_effect=load_file),
                      patch("laya.agent.compile_model", side_effect=compile_model),
                      patch.object(Agent, "warmup", side_effect=lambda: released("warmup")),
                      patch("torch.cuda.is_available", return_value=True),
                      patch("torch.cuda.get_device_capability", return_value=(8, 0)),
                      warnings.catch_warnings(record=True) as caught):
                    agent = Agent(directory, device="cuda", compile=True)
                self.assertEqual(events, ["compile", "cuda", "cpu", "warmup"])
                self.assertEqual(agent.device.type, "cpu")
                self.assertEqual(agent.dtype, torch.float32)
                self.assertFalse(agent.amp_enabled)
                self.assertFalse(model.training)
                self.assertTrue(any("simulated device allocation failure" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
