"""Compare temperature fitting with a Git baseline on seeded CPU-only cases.

Measures the fitter and each fitted temperature's true, unpadded NLL. No checkpoint
is loaded. Small synthetic cases show numerical regressions, not model accuracy.
"""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from laya import calibrate  # noqa: E402


def baseline_fitter(ref):
    source = subprocess.check_output(["git", "show", ref + ":laya/calibrate.py"], cwd=ROOT)
    namespace = {"__name__": "laya._calibration_benchmark_baseline", "__package__": "laya"}
    exec(compile(source, "<baseline-calibrate>", "exec"), namespace)
    return namespace["fit_one_temperature"], hashlib.sha256(source).hexdigest()


def argmax_rows(width, n, seed):
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        logits = rng.randn(width).astype(np.float32)
        target = np.zeros(width, np.float32)
        target[logits.argmax()] = 1.0
        rows.append((logits, target))
    return rows


def mixed_rows(width, n, seed):
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        label = int(rng.randint(width))
        predicted = label if rng.rand() < 0.7 else int(rng.randint(width))
        logits = np.full(width, -1.0, np.float32)
        logits[predicted] = 1.0
        target = np.zeros(width, np.float32)
        target[label] = 1.0
        rows.append((logits, target))
    return rows


def cases():
    low = [(np.array([-20000 - i, -20001 - i], np.float32), np.array([1, 0], np.float32))
           for i in range(12)]
    return {
        "argmax_width2": argmax_rows(2, 24, 0),
        "argmax_width9": argmax_rows(9, 24, 0),
        "argmax_ragged_2_9": argmax_rows(2, 24, 0) + argmax_rows(9, 24, 0),
        "negative_ragged_2_4": low + argmax_rows(4, 12, 0),
        "interior_width20": mixed_rows(20, 400, 293),
        "interior_ragged_3_20": mixed_rows(3, 400, 51) + mixed_rows(20, 400, 293),
    }


def true_nll(rows, temperature):
    total = 0.0
    for logits, target in rows:
        scaled = logits.astype(np.float64) / temperature
        shifted = scaled - scaled.max()
        logp = shifted - np.log(np.exp(shifted).sum())
        total -= float(np.dot(target, logp))
    return total / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-ref", default="3357dee78bb7933268a6db2d1f0215d1d0ea2dc7")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1 or args.threads < 1:
        parser.error("warmups must be nonnegative; repeats and threads must be positive")
    torch.set_num_threads(args.threads)
    before, before_hash = baseline_fitter(args.baseline_ref)
    implementations = {"before": before, "after": calibrate.fit_one_temperature}
    outputs = {}
    for name, rows in cases().items():
        measured = {key: {"temperatures": [], "samples_ms": []} for key in implementations}
        for fitter in implementations.values():
            for _ in range(args.warmups):
                fitter(rows, min_n=1)
        for repetition in range(args.repeats):
            order = list(implementations) if repetition % 2 == 0 else list(reversed(implementations))
            for key in order:
                started = time.perf_counter()
                fitted = implementations[key](rows, min_n=1)
                measured[key]["samples_ms"].append((time.perf_counter() - started) * 1000)
                measured[key]["temperatures"].append(fitted)
        for measurement in measured.values():
            fitted = measurement["temperatures"][0]
            if any(value != fitted for value in measurement["temperatures"]):
                raise RuntimeError("repeated fits returned different temperatures")
            measurement["temperature"] = fitted
            measurement["true_nll"] = true_nll(rows, fitted)
            measurement["median_ms"] = statistics.median(measurement["samples_ms"])
        outputs[name] = {"rows": len(rows), "widths": sorted({len(logits) for logits, _ in rows}), **measured}
    current_hash = hashlib.sha256((ROOT / "laya/calibrate.py").read_bytes()).hexdigest()
    result = {
        "baseline_ref": subprocess.check_output(["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip(),
        "baseline_calibrate_sha256": before_hash,
        "after_calibrate_sha256": current_hash,
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "processor": platform.processor(), "python": platform.python_version(),
                        "torch": torch.__version__, "numpy": np.__version__, "torch_threads": args.threads},
        "protocol": {"warmups_per_case_per_implementation": args.warmups, "measured_repeats": args.repeats,
                     "order": "before/after on even repetitions; after/before on odd repetitions",
                     "min_n": 1, "target_nll_dtype": "float64", "input_dtype": "float32",
                     "argmax_seed": 0, "mixed_width3_seed": 51, "mixed_width20_seed": 293,
                     "mixed_target_probability": 0.7, "mixed_logit_magnitude": 1.0},
        "caution": "Small synthetic fitter measurements on an active workstation; not inference latency or accuracy.",
        "cases": outputs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, values in outputs.items():
        before, after = values["before"], values["after"]
        print("%s: T %.9f -> %.9f; NLL %.9f -> %.9f; median %.3f -> %.3f ms" % (
            name, before["temperature"], after["temperature"], before["true_nll"], after["true_nll"],
            before["median_ms"], after["median_ms"]))


if __name__ == "__main__":
    main()
