"""Offline checks for the shared ONNX runtime and its native tokenizer boundary."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laya.checkpoint import _NativeTokenizer  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402


class NativeONNXAgent(ONNXAgent):
    _tokenizer_backend = "tokenizers"


TEXTS = [
    "Hello, world! I'm testing punctuation . Don't stop !",
    "Hva skjer? blåbær og ærlig. русский запрос",
    "中文请求。日本語の質問 اختبار باللغة العربية",
    "choice question: Choose the category score question: Rate urgency noul question: Is it urgent",
    "false: no, the statement does not hold true: yes, the statement holds billing support low high",
]
QUESTIONS = {
    "category": {"type": "choice", "instructions": "Choose the category",
                 "criteria": {"billing": "billing", "support": "support"}, "option_order": [1, 0]},
    "urgency": {"type": "score", "instructions": "Rate urgency", "criteria": ["low", "high"]},
    "urgent": {"type": "noul", "instructions": "Is it urgent"},
}


@pytest.fixture(params=["WordPiece", "BPE"])
def checkpoint(tmp_path, request):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tmp_path = tmp_path / "checkpoint"
    tmp_path.mkdir()
    special = ["[UNK]", "[CLS]", "[SEP]", "[MASK]", "[PAD]"]
    if request.param == "WordPiece":
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.WordPiece()
        trainer = trainers.WordPieceTrainer(vocab_size=512, special_tokens=special)
    else:
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=512, special_tokens=special,
                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(TEXTS, trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    # A saved tokenizer may carry training-time padding/truncation; inference must override it.
    tokenizer.enable_truncation(max_length=7)
    tokenizer.enable_padding(length=9, pad_id=4, pad_token="[PAD]")
    tok_dir = tmp_path / "tokenizer"
    tok_dir.mkdir()
    tokenizer.save(str(tok_dir / "tokenizer.json"))
    metadata = {name: token for name, token in zip(
        ["unk_token", "cls_token", "sep_token", "mask_token", "pad_token"], special)}
    metadata.update(tokenizer_class="PreTrainedTokenizerFast", model_max_length=512,
                    clean_up_tokenization_spaces=False, model_input_names=["input_ids", "attention_mask"])
    (tok_dir / "tokenizer_config.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tok_dir / "special_tokens_map.json").write_text(json.dumps(
        {name: {"content": metadata[name]} for name in ("cls_token", "sep_token", "mask_token", "pad_token")}),
        encoding="utf-8")
    (tmp_path / "rl_agent_config.json").write_text(json.dumps(
        {"encoder": "must-not-download", "max_len": 96, "head_max_len": 40}), encoding="utf-8")
    (tmp_path / "laya.onnx").write_bytes(b"test graph")
    return tmp_path


def fake_ort():
    class Session:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def run(self, names, inputs):
            self.calls += 1
            mask = inputs["marker_mask"]
            return [np.where(mask, np.arange(mask.shape[1], dtype=np.float32), -1e9),
                    np.zeros((len(mask), 2), dtype=np.float32)]

    return types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"],
                                 SessionOptions=types.SimpleNamespace,
                                 GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=99),
                                 InferenceSession=Session)


def exercise_native(path):
    from laya.consistency import DecisionCache
    from laya.calibration import apply_calibration_payload

    sys.modules["onnxruntime"] = fake_ort()
    agent = NativeONNXAgent(str(path), str(path / "laya.onnx"))
    results = agent.predict_batch(TEXTS[:3], QUESTIONS, batch_size=2, sort_by_length=True)
    assert [list(r["answers"]) for r in results] == [list(QUESTIONS)] * 3
    assert results[0]["answers"]["category"]["choice"] == "billing"  # reversed slot order
    assert results[0]["answers"]["urgent"]["noul"] > 0.5
    assert isinstance(results[0]["answers"]["urgency"]["score"], float)
    assert agent.decide(TEXTS[0], {"type": "object", "properties": {
        "urgent": {"type": "boolean"}}}) == {"urgent": True}
    with agent.hooks_installed(DecisionCache()):
        first = agent.predict(TEXTS[0], QUESTIONS)
        calls = agent.session.calls
        assert agent.predict(TEXTS[0], QUESTIONS)["answers"] == first["answers"]
        assert agent.session.calls == calls
    scanned = agent.predict_long(" ".join(TEXTS * 3), QUESTIONS, window=32, stride=16, batch_size=2)
    assert scanned["usage"]["windows"] > 1
    assert all("window" in a for a in scanned["answers"].values())
    apply_calibration_payload(agent, {"temperature": [1.5, 2, 1],
        "binning_map": {"choice:2": {"bins": 1, "values": [0.6]}}})
    assert agent.predict(TEXTS[0], QUESTIONS)["answers"]["category"]["answer_confidence"] == 0.6


def test_native_runtime_never_imports_torch_or_transformers(checkpoint):
    code = '''
import importlib.abc, runpy, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers'):
            raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0, Block())
ns = runpy.run_path(sys.argv[1])
ns['exercise_native'](Path(sys.argv[2]))
assert not any(n.split('.')[0] in ('torch','transformers') for n in sys.modules)
'''
    run = subprocess.run([sys.executable, "-c", code, __file__, str(checkpoint)],
                         cwd=ROOT, text=True, capture_output=True)
    assert run.returncode == 0, run.stdout + run.stderr


def test_native_tokenizer_matches_auto_tokenizer(checkpoint):
    from transformers import AutoTokenizer

    native = _NativeTokenizer(str(checkpoint / "tokenizer"))
    auto = AutoTokenizer.from_pretrained(str(checkpoint / "tokenizer"), local_files_only=True)
    before = native._tokenizer.to_str()
    for text in TEXTS:
        for kwargs in ({}, {"add_special_tokens": False},
                       {"add_special_tokens": False, "truncation": True, "max_length": 8}):
            assert native(text, **kwargs) == dict(auto(text, **kwargs))
            ids = native(text, **kwargs)["input_ids"]
            assert native.decode(ids, skip_special_tokens=True) == auto.decode(ids, skip_special_tokens=True)
    assert native._tokenizer.to_str() == before
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(lambda text: native(text, add_special_tokens=False), TEXTS * 5))
    assert got == [native(text, add_special_tokens=False) for text in TEXTS * 5]
    assert native._tokenizer.to_str() == before


def test_native_checkpoint_resolution_and_integrity(checkpoint, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort())
    calls = []
    snapshot = tmp_path / "hub" / "snapshots" / ("a" * 40)
    shutil.copytree(checkpoint, snapshot)

    def download(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return str(snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=download))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_TOKEN", "environment-token")
    monkeypatch.delenv("LAYA_REVISION", raising=False)
    agent = NativeONNXAgent("owner/private", str(checkpoint / "laya.onnx"))
    assert agent.revision == "a" * 40
    assert calls[-1][1] == {"token": "environment-token", "allow_patterns": [
        "rl_agent_config.json", "tokenizer/*", "encoder/*"]}
    NativeONNXAgent("owner/private", str(checkpoint / "laya.onnx"), token="explicit-token", revision="stable")
    assert calls[-1][1]["token"] == "explicit-token" and calls[-1][1]["revision"] == "stable"
    with pytest.raises(FileNotFoundError, match="Local model path not found"):
        NativeONNXAgent(str(checkpoint / "missing"), str(checkpoint / "laya.onnx"))
    cfg = checkpoint / "rl_agent_config.json"
    digest = hashlib.sha256(cfg.read_bytes()).hexdigest()
    NativeONNXAgent(str(checkpoint), str(checkpoint / "laya.onnx"),
                    expected_sha256={"rl_agent_config.json": digest})
    cfg.write_text("invalid json", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        NativeONNXAgent(str(checkpoint), str(checkpoint / "laya.onnx"),
                        expected_sha256={"rl_agent_config.json": digest})


def test_native_missing_tokenizer_refuses_encoder_fallback(checkpoint, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort())
    (checkpoint / "tokenizer" / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="native ONNX tokenizer requires"):
        NativeONNXAgent(str(checkpoint), str(checkpoint / "laya.onnx"))


def test_native_metadata_precedence_and_left_truncation(checkpoint):
    from transformers import AutoTokenizer

    cfg_path = checkpoint / "tokenizer" / "tokenizer_config.json"
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    config.update(mask_token="[UNK]", truncation_side="left")
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    native = _NativeTokenizer(str(checkpoint / "tokenizer"))
    auto = AutoTokenizer.from_pretrained(str(checkpoint / "tokenizer"), local_files_only=True)
    assert native.mask_token_id == auto.mask_token_id == 3  # legacy special_tokens_map wins
    before = native._tokenizer.to_str()
    assert native(TEXTS[0], truncation=True, max_length=8) == dict(auto(TEXTS[0], truncation=True, max_length=8))
    with pytest.raises(TypeError):
        native(object(), truncation=True, max_length=8)
    assert native._tokenizer.to_str() == before
    # Modern metadata carries its own AddedToken definitions and supersedes the legacy map.
    raw = json.loads((checkpoint / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
    config["added_tokens_decoder"] = {str(t["id"]): {k: v for k, v in t.items() if k != "id"}
                                      for t in raw["added_tokens"]}
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    assert _NativeTokenizer(str(checkpoint / "tokenizer")).mask_token_id == 0


def test_native_requires_existing_special_ids(checkpoint):
    path = checkpoint / "tokenizer" / "special_tokens_map.json"
    special = json.loads(path.read_text(encoding="utf-8"))
    special["mask_token"] = {"content": "[MISSING]"}
    path.write_text(json.dumps(special), encoding="utf-8")
    with pytest.raises(ValueError, match="requires mask_token metadata"):
        _NativeTokenizer(str(checkpoint / "tokenizer"))


def test_full_onnx_keeps_auto_tokenizer_default(checkpoint, monkeypatch):
    from transformers import AutoTokenizer

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort())
    sentinel = object()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda path: sentinel)
    assert ONNXAgent(str(checkpoint), str(checkpoint / "laya.onnx")).tok is sentinel
