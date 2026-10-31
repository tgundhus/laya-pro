"""Local ONNX decisions with native tokenization and no training dependencies."""
__version__ = "0.3.29"

from ._runtime.onnx_agent import ONNXAgent as _ONNXAgent
from ._runtime.consistency import DecisionCache, DecisionStore, decision_margins
from ._runtime.hooks import AsyncHook, BaseHook, Hook, PredictContext, PredictHook
from ._runtime.performance import parse_performance_mode, resolve_performance_mode
from ._runtime.structured import DecisionResult, decide, decide_batch


class ONNXAgent(_ONNXAgent):
    """Run an exported Laya checkpoint using ONNX Runtime and its tokenizer.json."""
    _tokenizer_backend = "tokenizers"


Agent = ONNXAgent


def load(model_id_or_path, **kwargs):
    """Load an ONNX graph with the tokenizer and configuration from its source checkpoint."""
    return ONNXAgent(model_id_or_path, **kwargs)


__all__ = [
    "ONNXAgent", "Agent", "load", "DecisionCache", "DecisionStore", "decision_margins",
    "AsyncHook", "BaseHook", "Hook", "PredictContext", "PredictHook", "DecisionResult",
    "decide", "decide_batch", "parse_performance_mode", "resolve_performance_mode",
]
