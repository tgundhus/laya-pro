"""Apply persisted calibration maps without importing the fitting runtime."""
import warnings
from typing import Any, Dict

import numpy as np

from .inference_common import QTYPES, TEMP_MAX, TEMP_MIN, clamp_temperature, temperature_rejected

# Payloads without a version key predate recorded checkpoint identity (version 1).
CALIBRATION_VERSION = 2
N_QTYPES = len(QTYPES)
_CONFIG_IDENTITY_SKIP = ("temperature", "temperature_by_options")


def apply_binning_map(confidence: float, bucket: str,
                      binning_map: Dict[str, Dict[str, Any]]) -> float:
    """Recalibrate one `answer_confidence` for its option-count `bucket` (`common.temp_bucket`).

    Returns the confidence unchanged when the map has no entry for the bucket, so a bucket the map
    was not fit for passes through rather than being forced to a wrong value.
    """
    entry = binning_map.get(bucket)
    if not entry:
        return float(confidence)
    bins = int(entry["bins"])
    b = min(bins - 1, max(0, int(float(confidence) * bins)))
    return float(entry["values"][b])


def _config_identity(cfg):
    """Checkpoint config without the temperature fields this file itself stores."""
    if cfg is None:
        return None
    return {str(k): cfg[k] for k in cfg if k not in _CONFIG_IDENTITY_SKIP}


def calibration_payload(
    temperature,
    temperature_by_options,
    model_id_or_path=None,
    subfolder=None,
    config=None,
    binning_map=None,
) -> Dict[str, Any]:
    """JSON body written by `Agent.save_calibration` (no weights).

    `version` is `CALIBRATION_VERSION`. Payloads written before identity was recorded have
    no version key; `apply_calibration_payload` treats that as version 1 and still loads.

    `model_id_or_path`, `subfolder`, and `config` say which checkpoint the map was fitted
    against. `config` is `rl_agent_config.json` without `temperature` /
    `temperature_by_options` (those live at the top of this payload).

    `binning_map` is the optional histogram-binning recalibration map fitted by
    `fit_binning_map`; the key is omitted when no map is installed, which is `None`, not an empty
    map. A map that fitted nothing -- `{}`, the return value when no bucket reached the sample
    floor -- is still written, so the round trip through :func:`apply_calibration_payload` gives
    `{}` back rather than `None`.
    """
    payload = {
        "version": CALIBRATION_VERSION,
        "temperature": [float(x) for x in temperature],
        "temperature_by_options": {
            str(k): float(v) for k, v in dict(temperature_by_options).items()
        },
        "model_id_or_path": model_id_or_path,
        "subfolder": subfolder,
        "config": _config_identity(config),
    }
    if binning_map is not None:
        payload["binning_map"] = binning_map
    return payload


def _warn_if_identity_mismatch(obj, payload) -> None:
    recorded_model = payload.get("model_id_or_path")
    recorded_sub = payload.get("subfolder")
    recorded_cfg = payload.get("config")
    current_model = getattr(obj, "model_id_or_path", None)
    current_sub = getattr(obj, "subfolder", None)
    current_cfg = _config_identity(getattr(obj, "cfg", None))
    if (recorded_model, recorded_sub, recorded_cfg) == (current_model, current_sub, current_cfg):
        return
    cfg_note = ""
    if recorded_cfg != current_cfg:
        cfg_note = " Checkpoint config identity differs."
    warnings.warn(
        "calibration was fitted for model_id_or_path=%r subfolder=%r but this agent is "
        "model_id_or_path=%r subfolder=%r.%s Loading the temperatures anyway."
        % (recorded_model, recorded_sub, current_model, current_sub, cfg_note),
        UserWarning,
        stacklevel=3,
    )


def _rejected_temperatures(entries):
    """Names whose raw value is not the number `clamp_temperature` kept, matching checkpoint load."""
    rejected = []
    for name, raw, applied in entries:
        if not temperature_rejected(raw, applied):
            continue
        rejected.append("%s=%r -> %g" % (name, raw, applied))
    return rejected


def _install_temperatures(obj, temperature, temperature_by_options, warn: bool = True) -> None:
    """Store a map the way checkpoint load does: raw values, and the clamped ones inference reads.

    `warn=False` is the fitter path: those values were already passed through `clamp_temperature`,
    so a second pass must not report them as a bad file.
    """
    raw_t = list(temperature)
    applied_t = [clamp_temperature(t) for t in raw_t]
    raw_b = {str(k): v for k, v in dict(temperature_by_options).items()}
    applied_b = {k: clamp_temperature(v) for k, v in raw_b.items()}
    obj.temperature_raw = raw_t
    obj.temperature = applied_t
    obj.temperature_by_options_raw = raw_b
    obj.temperature_by_options = applied_b
    if not warn:
        return
    entries = [("temperature[%d]" % i, raw, applied_t[i]) for i, raw in enumerate(raw_t)]
    entries += [(key, raw_b[key], applied_b[key]) for key in raw_b]
    rejected = _rejected_temperatures(entries)
    if rejected:
        warnings.warn(
            "laya: calibration temperatures are invalid or outside [%g, %g]; "
            "using %s. Treat confidence from the affected entries as uncalibrated."
            % (TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
            RuntimeWarning,
            stacklevel=3,
        )


def apply_calibration_payload(obj, payload: Dict[str, Any]) -> None:
    """Copy `temperature`, `temperature_by_options` and `binning_map` onto `obj`.

    Those are the three fields this reads and the three it writes. `binning_map` is installed
    whatever the payload holds, including nothing: a payload with no ``binning_map`` key -- every
    file written before histogram binning existed, and every agent that saved before
    :meth:`Agent.fit_binning` ran -- installs `None`, which clears a map this object already
    fitted. The file is the whole calibration state, not a patch onto the current one.

    A missing `version` is version 1 (temperatures only, no checkpoint identity). Version
    `CALIBRATION_VERSION` records the checkpoint the map was fitted for; a mismatch warns
    and still loads, so an older file never becomes a hard failure. Each temperature is passed
    through `clamp_temperature`, so a non-numeric or out-of-range entry cannot crash a later
    forward the way an unclamped zero used to.

    `temperature` and `binning_map` take opposite value policies, and that is the sentence to read
    before writing a file by hand: a bad *temperature* is forgiven and clamped into
    `[TEMP_MIN, TEMP_MAX]`, while a bad *binning value* is refused. Nothing recalibrates a
    confidence the way an out-of-range binning value would, so there is no defensible fallback
    for it.

    The payload's *shape* is checked before any of that, and refused with a `ValueError` naming
    the field: the values may be junk the clamp forgives, but the payload itself not being an
    object, a `version` that is not an integer, a `temperature` that is not a list of three, a
    `temperature_by_options` that is not an object, a `binning_map` that is not an object, or a
    `binning_map` entry that is not an object with an integer `bins` >= 1 and `values` of exactly
    that length holding finite numbers in [0, 1] is a file this code cannot read -- JSON gives
    `int`, `str`, `list` and `dict` for each of those mistakes just as happily as it gives the
    right shapes, and each used to fail with a raw `TypeError`/`AttributeError` from
    `len()`/`dict()`, or worse, to load: a `{"a": 1, "b": 2, "c": 3}` or `"abc"` has `len` 3 and
    used to pass the length check and install its *keys* as temperatures.
    """
    if not isinstance(payload, dict):
        raise ValueError("calibration JSON must be an object, got %s" % type(payload).__name__)
    version = payload.get("version", 1)
    if version is None:
        version = 1
    if isinstance(version, bool) or not isinstance(version, (int, float, str)):
        raise ValueError("calibration JSON version must be an integer, got %r" % (version,))
    try:
        version = int(version)
    except (TypeError, ValueError, OverflowError) as exc:   # Overflow: JSON's 1e999 is inf
        raise ValueError(
            "calibration JSON version must be an integer, got %r" % (version,)) from exc
    temps = payload.get("temperature")
    if not isinstance(temps, (list, tuple)) or len(temps) != N_QTYPES:
        raise ValueError("calibration JSON must contain temperature: [3 floats]")
    by_options = payload.get("temperature_by_options")
    if by_options is None:   # absent, or an explicit null: a file from before bucket temperatures
        by_options = {}
    if not isinstance(by_options, dict):
        raise ValueError(
            "calibration JSON temperature_by_options must be an object of bucket -> float, "
            "got %s" % type(by_options).__name__)
    binning = payload.get("binning_map")
    if binning is not None:
        if not isinstance(binning, dict):
            raise ValueError(
                "calibration JSON binning_map must be an object of bucket -> {bins, values}, "
                "got %s" % type(binning).__name__)
        for name, entry in binning.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    "calibration JSON binning_map[%r] must be an object with \"bins\" and \"values\", "
                    "got %s" % (name, type(entry).__name__))
            n_bins = entry.get("bins")
            values = entry.get("values")
            if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
                raise ValueError(
                    "calibration JSON binning_map[%r] must have an integer \"bins\" >= 1, got %r" % (name, n_bins))
            if not isinstance(values, (list, tuple)) or len(values) != n_bins:
                raise ValueError(
                    "calibration JSON binning_map[%r] must have \"values\" of length \"bins\" (%d)"
                    % (name, n_bins))
            try:
                parsed_values = [float(v) for v in values]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "calibration JSON binning_map[%r] values must be numbers, got %r" % (name, values)) from exc
            for v in values:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(
                        "calibration JSON binning_map[%r] values must be numbers, got %r" % (name, values))
            for v in parsed_values:
                if not np.isfinite(v):
                    raise ValueError(
                        "calibration JSON binning_map[%r] values must be finite, got %r" % (name, values))
                if not 0.0 <= v <= 1.0:
                    raise ValueError(
                        "calibration JSON binning_map[%r] values must be in [0, 1], got %r" % (name, values))
    obj.binning_map = binning
    if version >= CALIBRATION_VERSION:
        _warn_if_identity_mismatch(obj, payload)
    _install_temperatures(obj, temps, by_options)
