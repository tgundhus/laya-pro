"""Exceptions shared by the framework integrations."""
from typing import Any, Dict


class LayaLowConfidenceError(ValueError):
    """Raised by CrewAI or LlamaIndex selection when the caller enables low-confidence errors.

    Requires a positive threshold, `raise_on_low_confidence=True`, and no fallback index.
    """

    def __init__(self, message: str, confidence: float, threshold: float, raw_decision: Dict[str, Any]):
        super().__init__(message)
        self.confidence = confidence
        self.threshold = threshold
        self.raw_decision = raw_decision
