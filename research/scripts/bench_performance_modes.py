"""Measure default and five performance modes on one real, cached CPU checkpoint.

Run from the repository: python research/scripts/bench_performance_modes.py \
    --model /path/to/pinned/checkpoint --out research/results/performance_modes_cpu.json
Each configuration runs in its own process. No downloads or decision cache are used.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this request?",
             "criteria": {"billing": "payments, invoices and refunds",
                          "technical": "errors, outages and account access",
                          "sales": "new purchases and pricing", "other": "other requests"}},
    "urgent": {"type": "noul", "instructions": "Does this request need immediate action?"},
}
CASES = [
    {"state": "My account was charged twice. Please refund the duplicate charge.",
     "team": "billing", "urgent": False},
    {"state": "Our production service is down and customers cannot sign in. We need help immediately.",
     "team": "technical", "urgent": True},
    {"state": "We are comparing plans for next quarter. Please send a quote for twenty new licenses. "
              "There is no deadline this week; a reply whenever you have the pricing information is fine.",
     "team": "sales", "urgent": False},
    {"state": "Thank you for the update and for your help with the earlier request. "
              "Everything is working again and there are no outstanding problems. "
              "This message is only to confirm that the case can be closed. "
              "No further work or reply is needed from your team.",
     "team": "other", "urgent": False},
]


def peak_memory():
    if os.name == "nt":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"peak_working_set_bytes": counters.PeakWorkingSetSize,
                "peak_commit_bytes": counters.PeakPagefileUsage, "source": "Windows GetProcessMemoryInfo"}
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_working_set_bytes": rss if sys.platform == "darwin" else rss * 1024,
            "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss"}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def child(args):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    import transformers
    from laya.agent import Agent
    module = importlib.import_module("laya.agent")
    torch.manual_seed(29018)
    started = time.perf_counter()
    agent = Agent(str(args.model), device="cpu", backend="eager", performance_mode=args.mode)
    cold_load = time.perf_counter() - started
    report = {"mode": args.mode, "performance_config": agent.performance_config,
        "cold_load_s": cold_load, "environment": {
        "python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(), "torch": torch.__version__, "numpy": np.__version__,
        "transformers": transformers.__version__, "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(), "dtype": str(agent.dtype),
        "amp_enabled": agent.amp_enabled}, "workloads": []}
    for name, cases in (("single_2_questions", CASES[:1]), ("batch4_2_questions", CASES)):
        states = [case["state"] for case in cases]
        def run():
            if len(states) == 1:
                return [agent.predict(states[0], QUESTIONS, max_len=192, head_max_len=96)]
            return agent.predict_batch(states, QUESTIONS, max_len=192, head_max_len=96)
        for _ in range(args.warmups):
            run()
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = run()
            samples.append((time.perf_counter() - start) * 1000)
        probabilities, shapes = [], []
        original_decode, original_forward = agent._decode_answers, agent._forward
        def decode(logits, act, items, ids, internal, offset, lang=None):
            rows = module._option_logits(logits, items, offset)
            for j, qid in enumerate(ids):
                qt = module.QTYPES[internal[qid]["t"]]
                k = len(items[j]["markers"])
                scale = agent.temperature_by_options.get(module.temp_bucket(qt, k), agent.temperature[qt])
                values = rows[j] / scale
                probs = np.exp(values - values.max())
                probs /= probs.sum()
                probabilities.append(probs.tolist())
            return original_decode(logits, act, items, ids, internal, offset, lang)
        def forward(batch):
            shapes.append(list(batch["input_ids"].shape))
            return original_forward(batch)
        with patch.object(agent, "_decode_answers", decode), patch.object(agent, "_forward", forward):
            parity_result = run()
        assert parity_result == result, "repeated predictions changed within one configuration"
        labels = [value for row in result for value in (
            row["answers"]["team"]["choice"], row["answers"]["urgent"]["noul"] >= 0.5)]
        expected = [value for case in cases for value in (case["team"], case["urgent"])]
        report["workloads"].append({"name": name, "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
            "requests_per_s": len(states) * len(samples) * 1000 / sum(samples),
            "decisions_per_s": len(states) * len(QUESTIONS) * len(samples) * 1000 / sum(samples),
            "states": len(states), "questions_per_state": len(QUESTIONS), "forward_shapes": shapes,
            "raw_probabilities": probabilities, "labels": labels,
            "matches_synthetic_labels": sum(a == b for a, b in zip(labels, expected)),
            "label_count": len(labels), "outputs": result})
    report["memory"] = peak_memory()
    args.child_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def markdown(report):
    lines = ["# Performance modes: cached English checkpoint on CPU", "",
             "One fresh subprocess per configuration; CPU fp32, eager, decision cache disabled. "
             "%d warm-up and %d timed calls per workload. p95 is nearest-rank and, with this small "
             "sample, is the slowest call. Process peak memory includes imports, loading and inference." % (report["warmups"], report["repeats"]),
             "", "| mode | threads | single median / p95 ms | batch4 median / p95 ms | batch requests/s | load s | peak working set MiB | max probability difference | labels vs default |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["configurations"]:
        single, batch = row["workloads"]
        lines.append("| %s | %d | %.1f / %.1f | %.1f / %.1f | %.2f | %.2f | %.0f | %.3g | %d/%d |" % (
            row["mode"], row["environment"]["torch_threads"], single["median_ms"], single["p95_ms"],
            batch["median_ms"], batch["p95_ms"], batch["requests_per_s"], row["cold_load_s"],
            row["memory"]["peak_working_set_bytes"] / 2**20,
            max(w["max_probability_difference_vs_default"] for w in row["workloads"]),
            sum(w["labels_agree_with_default"] for w in row["workloads"]),
            sum(w["label_count"] for w in row["workloads"])))
    env = report["configurations"][0]["environment"]
    lines += ["", "Host: %s; %d logical CPUs; Python %s; Torch %s; transformers %s." % (
        env["cpu"], env["logical_cpus"], env["python"].split()[0], env["torch"], env["transformers"]),
        "", "Checkpoint snapshot: `%s`; weight SHA-256: `%s`." % (
            report["checkpoint"]["snapshot_revision"], report["checkpoint"]["weights_sha256"]),
        "", "The four hand-written support cases have mixed lengths and two questions each. "
        "They test numerical parity and resource choices, not accuracy on a representative dataset. "
        "The benchmark loads the same single checkpoint directly in every mode: it does not measure "
        "multi-model residency, startup preloading, HTTP admission, GPU performance, or older CPUs. "
        "Cold load is a new Python Agent load, not a flushed filesystem cache. Workstation load was uncontrolled.",
        "", "Raw samples, probabilities, outputs, tensor shapes, source hashes and reproduction command "
        "are in the adjacent JSON file. Larger mode numbers are resource budgets, not speed rankings.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--mode", choices=["default", "1", "2", "3", "4", "5"])
    parser.add_argument("--child-out", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 1:
        parser.error("repeats and warmups must be positive")
    if args.child_out:
        child(args)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    weights = args.model / "model.safetensors"
    weights_hash = file_sha256(weights)
    report = {"command": subprocess.list2cmdline(sys.argv), "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode_order": ["default", "1", "2", "3", "4", "5"], "seed": 29018,
        "repeats": args.repeats, "warmups": args.warmups,
        "checkpoint": {"path": str(args.model.resolve()), "snapshot_revision": args.model.name,
                       "weights_sha256": weights_hash,
                       "config_sha256": hashlib.sha256((args.model / "rl_agent_config.json").read_bytes()).hexdigest()},
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
            "laya/agent.py", "laya/performance.py", "research/scripts/bench_performance_modes.py")},
        "questions": QUESTIONS, "cases": CASES, "configurations": []}
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", USE_TF="0")
    for key in ("LAYA_PERFORMANCE_MODE", "LAYA_THREADS", "LAYA_CPU_AMP", "LAYA_CACHE", "LAYA_CACHE_PATH",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.pop(key, None)
    for mode in ("default", "1", "2", "3", "4", "5"):
        child_out = args.out.with_name(args.out.stem + "_" + mode + ".tmp.json")
        command = [sys.executable, str(Path(__file__).resolve()), "--model", str(args.model), "--out", str(args.out),
                   "--repeats", str(args.repeats), "--warmups", str(args.warmups),
                   "--mode", mode, "--child-out", str(child_out)]
        subprocess.run(command, env=env, check=True, cwd=ROOT)
        row = json.loads(child_out.read_text(encoding="utf-8"))
        child_out.unlink()
        reference = report["configurations"][0] if report["configurations"] else row
        for workload, baseline in zip(row["workloads"], reference["workloads"]):
            workload["max_probability_difference_vs_default"] = max(
                abs(a - b) for aa, bb in zip(workload["raw_probabilities"], baseline["raw_probabilities"])
                for a, b in zip(aa, bb))
            workload["labels_agree_with_default"] = sum(a == b for a, b in zip(workload["labels"], baseline["labels"]))
        report["configurations"].append(row)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("mode %s: single %.1f ms, batch4 %.1f ms" % (
            mode, row["workloads"][0]["median_ms"], row["workloads"][1]["median_ms"]), flush=True)
    args.out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
