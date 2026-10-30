"""Opt-in local resource budgets; importing this module never imports a backend.

Modes change scheduling and residency, never the model, precision or predictions' schema.
Torch's intra-op thread budget is process-global: select a mode at process startup.
"""
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Optional


_PRESETS = {
    1: ("minimal", 1, 1, 1, False, 8192, 1),
    2: ("economy", 2, 2, 1, False, 16384, 2),
    3: ("balanced", 4, 4, 2, False, 32768, 4),
    4: ("throughput", 8, 8, 2, True, 65536, 8),
    5: ("dedicated", 16, 16, 3, True, 131072, 16),
}
_THREADS_OWNED = ContextVar("laya_performance_threads_owned", default=False)


def parse_performance_mode(value: Any) -> Optional[int]:
    """Parse a level 1..5, or None/blank/``default`` for legacy behavior."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() == "default":
            return None
        if value.isascii() and value.isdigit():
            value = int(value)
    if type(value) is not int or value not in _PRESETS:
        raise ValueError("performance_mode must be 'default' or an integer from 1 to 5")
    return value


def _available_cpus() -> int:
    count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    try:
        count = min(count, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError, NotImplementedError):
        pass
    return max(1, count)


def resolve_performance_mode(mode: Any = None) -> Optional[Dict[str, Any]]:
    """Return a fresh effective preset, reading LAYA_PERFORMANCE_MODE only for None.

    An explicit ``default`` (or blank string) suppresses the environment. Preset threads
    are capped by available logical CPUs/affinity; a valid positive LAYA_THREADS overrides
    that cap, preserving the existing explicit thread setting's meaning.
    """
    level = parse_performance_mode(os.environ.get("LAYA_PERFORMANCE_MODE", "") if mode is None else mode)
    if level is None:
        return None
    keys = ("name", "threads", "batch_size", "max_loaded", "preload", "max_batch_tokens", "max_concurrent")
    config = dict(zip(keys, _PRESETS[level]))
    config["mode"] = level
    config["threads"] = min(config["threads"], _available_cpus())
    try:
        threads = int(os.environ.get("LAYA_THREADS", ""))
    except ValueError:
        threads = 0
    if threads > 0:
        config["threads"] = threads
    return config


def _apply_torch_threads(config, threads=None):
    """Apply the process-global startup budget; lazy Router loads inherit it."""
    if threads is None:
        threads = config["threads"] if config else None
    if threads is None or _THREADS_OWNED.get():
        return threads
    import torch

    torch.set_num_threads(threads)
    return threads


@contextmanager
def _without_torch_threads():
    token = _THREADS_OWNED.set(True)
    try:
        yield
    finally:
        _THREADS_OWNED.reset(token)
