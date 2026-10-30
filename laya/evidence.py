"""Read-only evidence inspection over existing Laya artifacts.

This module reports what evidence exists for a checkpoint -- and what is missing,
insufficient, or incomparable -- without loading model weights, without needing
torch, and without writing a new manifest. It only reads artifacts the upstream
train/eval path already persists:

- a checkpoint directory's ``rl_agent_config.json`` (``training.laya_train`` and
  ``training.laya_train_calibration`` as written by merged #931/#933);
- an existing ``laya-evals`` report JSON (identity semantics from `laya.evals`).

Every state it assigns is either derived directly from an upstream artifact or
marked UNKNOWN. It never manufactures a healthy-looking state from absent input.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

PRESENT = "PRESENT"
MISSING = "MISSING"
INSUFFICIENT = "INSUFFICIENT"
UNKNOWN = "UNKNOWN"
INCOMPARABLE = "INCOMPARABLE"


def _entry(state: str, reason: str, detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {"state": state, "reason": reason}
    if detail:
        out["detail"] = detail
    return out


def inspect_calibration(config: Dict[str, Any]) -> Dict[str, Any]:
    """Evidence state of the persisted calibration fit in a checkpoint config."""
    training = config.get("training")
    if not isinstance(training, dict):
        return _entry(UNKNOWN, "legacy checkpoint: no training metadata persisted; cannot tell whether calibration ran")
    calibration = training.get("laya_train_calibration")
    if not isinstance(calibration, dict):
        return _entry(UNKNOWN, "training metadata present but no calibration block; checkpoint predates #933")
    types: Dict[str, Any] = {}
    for name, entry in calibration.items():
        if (not isinstance(entry, dict) or isinstance(entry.get("items"), bool)
                or not isinstance(entry.get("items"), int)):
            types[name] = _entry(UNKNOWN, "calibration entry is malformed")
            continue
        items = entry["items"]
        if items == 0:
            types[name] = _entry(MISSING, "question type present but zero calibration items", {"items": 0})
            continue
        # Persisted issues are the authoritative limitation text (#933 writes
        # not-fitted/below-floor/clamped/unchanged-fit there); we do not keep a
        # second numeric source of truth.
        issues = entry.get("issues")
        if not isinstance(issues, list):
            types[name] = _entry(UNKNOWN, "calibration entry has no issues list; cannot tell limitations")
        elif issues:
            types[name] = _entry(INSUFFICIENT, "limitations recorded: " + "; ".join(str(i) for i in issues),
                                 {"items": items, "issues": issues})
        else:
            types[name] = _entry(PRESENT, "fitted with no recorded issues on %d items" % items,
                                 {"items": items, "temperature": entry.get("temperature")})
    present = sum(1 for t in types.values() if t["state"] == PRESENT)
    overall = PRESENT if types and present == len(types) else (
        MISSING if types and all(t["state"] == MISSING for t in types.values()) else INSUFFICIENT)
    return {
        "state": overall,
        "types": types,
    }


# Copy the eval identity vocabulary to keep this read-only path free of numpy and torch.
# tests/test_evidence.py checks it against laya.evals._IDENTITY_KEYS.
_IDENTITY_KEYS = ("schema", "dataset_sha256", "questions_sha256")
_PROVENANCE_KEYS = ("laya_version",)
_IDENTITY_CONFLICT_KEYS = ("dataset_sha256", "questions_sha256")


def _report_identity(report: Dict[str, Any]) -> Dict[str, Any]:
    """Read config first, then top-level identity, matching laya.evals._identity_of."""
    config = report.get("config")
    found = dict(config) if isinstance(config, dict) else {}
    for key in _IDENTITY_KEYS + _PROVENANCE_KEYS:
        if found.get(key) is None and report.get(key) is not None:
            found[key] = report[key]
    return found


def inspect_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Evidence state of the report identity and optional provenance fields."""
    identity = _report_identity(report)
    fields = {}
    for key in _IDENTITY_KEYS + _PROVENANCE_KEYS:
        value = identity.get(key)
        fields[key] = _entry(PRESENT, "recorded", {"value": value}) if value is not None else _entry(UNKNOWN, "not recorded in this report")
    overall = PRESENT if all(fields[key]["state"] == PRESENT for key in _IDENTITY_KEYS) else UNKNOWN
    return {"state": overall, "fields": fields}


def _field(config_like: Any, key: str) -> Any:
    if not isinstance(config_like, dict):
        return None
    if config_like.get(key) is not None:
        return config_like.get(key)
    nested = config_like.get("training")
    if isinstance(nested, dict):
        for inner in (nested.get("laya_train"),):
            if isinstance(inner, dict) and inner.get(key) is not None:
                return inner.get(key)
    return None


def relate(config: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic checkpoint <-> report relationship.

    v1 is deliberately conservative: it emits INCOMPARABLE only when both artifacts
    expose the same deterministic identity key with conflicting values, and it never
    infers a PRESENT match from `laya_version` (which names the software, not the
    checkpoint). Everything else is UNKNOWN -- current artifacts expose no immutable
    shared checkpoint identity.
    """
    report_config = _report_identity(report)
    conflicts = []
    for key in _IDENTITY_CONFLICT_KEYS:
        a = _field(config, key)
        b = _field(report_config, key)
        if a is not None and b is not None and str(a) != str(b):
            conflicts.append(key)
    if conflicts:
        return _entry(INCOMPARABLE, "checkpoint and report disagree on: " + ", ".join(conflicts))
    return _entry(UNKNOWN, "no immutable shared checkpoint identity between this checkpoint artifact and the report")


def _read_json_object(path: str) -> Dict[str, Any]:
    """Read an artifact or refuse it with a message naming the file."""
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError("%s is not valid JSON: %s" % (path, exc)) from exc
    except UnicodeDecodeError as exc:
        raise ValueError("%s is not UTF-8 text: %s" % (path, exc)) from exc
    except RecursionError as exc:
        raise ValueError("%s is JSON nested deeper than it can be read: %s" % (path, exc)) from exc
    if not isinstance(document, dict):
        raise ValueError("%s is not a JSON object" % path)
    return document


def inspect_checkpoint(checkpoint_dir: str, report_path: Optional[str] = None) -> Dict[str, Any]:
    """Full read-only evidence inspection for one checkpoint (optionally one report)."""
    config_path = os.path.join(checkpoint_dir, "rl_agent_config.json")
    if not os.path.isdir(checkpoint_dir) or not os.path.isfile(config_path):
        raise FileNotFoundError("no rl_agent_config.json under %r" % checkpoint_dir)
    config = _read_json_object(config_path)
    out: Dict[str, Any] = {"checkpoint": checkpoint_dir, "calibration": inspect_calibration(config)}
    if report_path is not None:
        report = _read_json_object(report_path)
        out["report"] = inspect_report(report)
        out["relationship"] = relate(config, report)
    return out


def format_summary(result: Dict[str, Any]) -> str:
    lines: List[str] = ["evidence summary for %s" % result["checkpoint"],
                        "(reports evidence and limitations found in existing artifacts; this is not a deployment verdict)"]
    cal = result["calibration"]
    lines.append("calibration: %s" % cal["state"])
    for name, t in (cal.get("types") or {}).items():
        lines.append("  %-24s %s -- %s" % (name, t["state"], t["reason"]))
    if cal.get("state") == UNKNOWN and "reason" in cal:
        lines.append("  " + cal["reason"])
    if "report" in result:
        rep = result["report"]
        lines.append("eval report identity: %s" % rep["state"])
        for key, f in rep.get("fields", {}).items():
            lines.append("  %-18s %s" % (key, f["state"]))
        lines.append("checkpoint<->report: %s -- %s" % (result["relationship"]["state"], result["relationship"]["reason"]))
    return "\n".join(lines)
