"""Compare a frozen checkout with current Torch or standalone ONNX loading, offline.

Each sample uses a fresh process, the same checkpoint, fp32 CPU execution and no
decision cache. This measures process startup with a warm filesystem cache, not a
cold boot. Run --self-check for the small report-comparison check without loading a model.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = "6867bc3df62e40f656fc7062f34fca0fa7fdd252"
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this request?",
             "criteria": {"billing": "payments, invoices and refunds",
                          "technical": "errors, outages and account access",
                          "sales": "new purchases and pricing", "other": "other requests"}},
    "urgent": {"type": "noul", "instructions": "Does this request need immediate action?"},
    "severity": {"type": "score", "instructions": "How severe is this problem?",
                 "criteria": ["no active problem", "minor inconvenience", "service unavailable"]},
}
STATES = [
    "My account was charged twice. Please refund the duplicate charge.",
    "Our production service is down and customers cannot sign in. We need help immediately.",
    "We are comparing plans for next quarter. Please send a quote for twenty new licenses. "
    "There is no deadline this week; a reply whenever you have the pricing information is fine.",
    "Thank you for the update and for your help with the earlier request. "
    "Everything is working again and there are no outstanding problems. "
    "This message is only to confirm that the case can be closed. No further work or reply is needed.",
]


def memory():
    if os.name != "nt":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"peak_working_set_bytes": peak if sys.platform == "darwin" else peak * 1024,
                "source": "resource.getrusage; current RSS and private commit unavailable"}
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                "PrivateUsage")]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"working_set_bytes": counters.WorkingSetSize, "private_commit_bytes": counters.PrivateUsage,
            "peak_working_set_bytes": counters.PeakWorkingSetSize,
            "peak_commit_bytes": counters.PeakPagefileUsage, "page_faults": counters.PageFaultCount,
            "source": "Windows GetProcessMemoryInfo PROCESS_MEMORY_COUNTERS_EX"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_tree(revision, destination=None):
    destination = (destination or ROOT / ".venv" / "loading-baseline").resolve()
    if not destination.is_relative_to((ROOT / ".venv").resolve()):
        raise ValueError("Baseline must stay under the checkout's ignored .venv directory")
    marker = destination / ".benchmark-revision"
    if marker.exists():
        if marker.read_text(encoding="utf-8").strip() != revision:
            raise ValueError("Existing baseline has a different revision: %s" % destination)
        return destination
    archive = destination.with_suffix(".zip")
    subprocess.run(["git", "archive", "--format=zip", "--output", str(archive), revision, "laya"],
                   check=True, cwd=ROOT)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive member escapes baseline directory: %s" % item.filename)
        bundle.extractall(destination)
    marker.write_text(revision + "\n", encoding="utf-8")
    return destination


def child(args):
    sys.path.insert(0, str(args.source))
    import numpy as np

    if args.runtime == "torch":
        import torch
        from laya.agent import Agent
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(29018)
    elif args.variant == "baseline":
        from laya.onnx_agent import ONNXAgent as Agent
    else:
        from laya_onnx import ONNXAgent as Agent
    imported_at = time.perf_counter()
    stages = {"after_imports": memory()}
    load_at = time.perf_counter()
    if args.runtime == "torch":
        agent = Agent(str(args.model), device="cpu", backend="eager", performance_mode="default")
        assert agent.dtype == torch.float32 and not agent.amp_enabled
    else:
        agent = Agent(str(args.model), str(args.onnx), performance_mode=3)
        assert agent.session.get_providers() == ["CPUExecutionProvider"]
        assert agent.session.get_session_options().intra_op_num_threads == args.threads
    loaded_at = time.perf_counter()
    stages["after_load"] = memory()

    def predict(state):
        return agent.predict(state, QUESTIONS, max_len=192, head_max_len=96)

    first_at = time.perf_counter()
    first = predict(STATES[0])
    answered_at = time.perf_counter()
    stages["after_first_answer"] = memory()
    for _ in range(args.warmups):
        predict(STATES[0])
    prediction_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = predict(STATES[0])
        prediction_ms.append((time.perf_counter() - start) * 1000)
    assert first == result
    stages["after_warm_predictions"] = memory()

    # Capture inputs and valid logits only on untimed calls. Timed loading and first
    # prediction run the unmodified runtime; no synthetic checkpoint or patched loader.
    records, feeds = [], []
    decode = agent._decode_answers

    def capture_decode(logits, act, items, ids, internal, offset, lang=None):
        for j, qid in enumerate(ids):
            kind = {"choice": 0, "score": 1, "noul": 2}[internal[qid]["t"]]
            count = len(items[j]["markers"])
            row = np.array(logits[offset + j, :count], copy=True)
            temperature = agent.temperature_by_options.get("%s:%s" % (
                internal[qid]["t"], "2" if count == 2 else "3-5" if count <= 5 else
                "6-10" if count <= 10 else "11+"), agent.temperature[kind])
            values = row / temperature
            probs = np.exp(values - values.max())
            probs /= probs.sum()
            records.append({"question": qid, "logits": row.tolist(), "probabilities": probs.tolist(),
                            "action_probabilities": act[offset + j].tolist()})
        return decode(logits, act, items, ids, internal, offset, lang)

    agent._decode_answers = capture_decode
    if args.runtime == "torch":
        forward = agent._forward

        def capture_forward(batch):
            feeds.append(batch)
            return forward(batch)

        agent._forward = capture_forward
    else:
        session = agent.session

        class CaptureSession:
            def run(self, names, feed):
                feeds.append(feed)
                return session.run(names, feed)

        agent.session = CaptureSession()
    outputs = [predict(state) for state in STATES]
    agent._decode_answers = decode
    if args.runtime == "torch":
        agent._forward = forward
        @torch.no_grad()
        def direct_forward():
            return forward(feeds[0])
        parameter_dtypes = sorted({str(value.dtype) for value in agent.model.parameters()})
        versions = {"torch": torch.__version__}
    else:
        agent.session = session
        direct_forward = lambda: session.run(["logits", "act_logits"], feeds[0])
        parameter_dtypes = ["fp32 graph; see graph artifact identity"]
        import onnxruntime
        versions = {"onnxruntime": onnxruntime.__version__}
    forward_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        direct_forward()
        forward_ms.append((time.perf_counter() - start) * 1000)
    stages["after_parity_and_forward"] = memory()
    report = {"variant": args.variant, "runtime": args.runtime, "source": str(args.source),
              "import_s": imported_at - args.parent_start,
              "agent_load_s": loaded_at - load_at, "first_predict_s": answered_at - first_at,
              "startup_to_first_answer_s": answered_at - args.parent_start,
              "prediction_ms": prediction_ms, "forward_ms": forward_ms,
              "memory_stages": stages, "parameter_dtypes": parameter_dtypes,
              "loading_mode": getattr(agent, "_loading_mode", "legacy_copy") if args.runtime == "torch" else "onnx",
              "records": records, "outputs": outputs,
              "environment": {"python": sys.version, "platform": platform.platform(),
                  "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
                  "threads": args.threads, "torch_interop_threads": 1 if args.runtime == "torch" else None,
                  "numpy": np.__version__, **versions},
              "imported_torch": "torch" in sys.modules,
              "imported_transformers": "transformers" in sys.modules}
    args.child_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def difference(left, right):
    if len(left) != len(right):
        raise AssertionError("Different number of parity records")
    deltas = {}
    for field in ("logits", "probabilities", "action_probabilities"):
        values = []
        for a, b in zip(left, right):
            assert a["question"] == b["question"] and len(a[field]) == len(b[field])
            assert all(math.isfinite(value) for value in a[field] + b[field]), "Non-finite parity value"
            values.extend(abs(x - y) for x, y in zip(a[field], b[field]))
        deltas["max_abs_" + field] = max(values)
    return deltas


def markdown(report):
    rows = report["runs"]
    lines = ["# Loading and memory: %s" % report["runtime"], "",
             "%d paired fresh-process runs, identical local artifacts, fp32 CPU execution, "
             "%d threads, no decision cache. Filesystem caches were left intact. "
             "Startup includes Python, imports, model loading and the first three-question answer." % (
                 report["process_runs"], report["threads"]),
             "", "| variant | import s | load s | first predict s | startup to first answer s | warm predict ms | forward ms | peak working set MiB | peak private commit MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in ("baseline", "current"):
        selected = [r for r in rows if r["variant"] == variant]
        values = [statistics.median(r[key] for r in selected) for key in
                  ("import_s", "agent_load_s", "first_predict_s", "startup_to_first_answer_s")]
        values += [statistics.median(x for r in selected for x in r[key])
                   for key in ("prediction_ms", "forward_ms")]
        values += [statistics.median(r["memory_stages"]["after_parity_and_forward"].get(key, 0)
                                     for r in selected) / 2**20
                   for key in ("peak_working_set_bytes", "peak_commit_bytes")]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.2f | %.2f | %.1f | %.1f |" %
                     (variant, *values))
    lines += ["", "Numbers are medians across process runs; warm timings pool %d calls per process. "
              "Peak counters cover the entire process, including imports and untimed parity calls. "
              "Working set includes resident mapped pages; private commit excludes shared file mappings. "
              "Neither is the model's file size." % report["repeats"],
              "", "| memory stage | baseline working / private MiB | current working / private MiB |",
              "|---|---:|---:|"]
    for stage in rows[0]["memory_stages"]:
        cells = []
        for variant in ("baseline", "current"):
            selected = [r["memory_stages"][stage] for r in rows if r["variant"] == variant]
            cells.append("%.1f / %.1f" % tuple(statistics.median(r.get(key, 0) for r in selected) / 2**20
                         for key in ("working_set_bytes", "private_commit_bytes")))
        lines.append("| %s | %s | %s |" % (stage, *cells))
    parity = report["parity_vs_first_baseline"]
    lines += ["", "Maximum absolute differences against the first baseline: logits %.9g, probabilities %.9g, "
              "action probabilities %.9g. Typed answer payloads agree in every run: %s." % (
                  max(r["max_abs_logits"] for r in parity), max(r["max_abs_probabilities"] for r in parity),
                  max(r["max_abs_action_probabilities"] for r in parity),
                  all(r["answers_equal"] for r in parity)),
              "", "The four hand-written states cover choice, score and yes/no decoding; this is numerical "
              "parity, not an accuracy evaluation. Loading is not timed separately inside the constructor: "
              "the report samples memory after imports and after the complete Agent load. Any work deferred "
              "to the first inference remains in startup-to-first-answer. The workstation was otherwise "
              "uncontrolled; small timing changes are not evidence of an inference speedup.",
              "", "Baseline commit: `%s`. Artifact hashes, raw samples, process order, stage counters, "
              "unrounded logits/probabilities and the command are in the adjacent JSON." % report["baseline_revision"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runtime", choices=("torch", "onnx"), default="torch")
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--light-source", type=Path)
    parser.add_argument("--baseline-revision", default=BASE)
    parser.add_argument("--baseline-directory", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--process-runs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--variant", choices=("baseline", "current"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--parent-start", type=float)
    parser.add_argument("--child-out", type=Path)
    args = parser.parse_args()
    if args.self_check:
        row = {"question": "q", "logits": [1., 2.], "probabilities": [.25, .75],
               "action_probabilities": [.1, .9]}
        assert all(value == 0 for value in difference([row], [row]).values())
        changed = dict(row, logits=[1., 3.])
        assert difference([row], [changed])["max_abs_logits"] == 1
        assert memory()["peak_working_set_bytes"] > 0
        print("Loading benchmark self-check passed.")
        return
    if not args.model or not args.out:
        parser.error("--model and --out are required")
    if min(args.process_runs, args.repeats, args.threads) < 1 or args.warmups < 0:
        parser.error("runs, repeats and threads must be positive; warmups must be nonnegative")
    if args.runtime == "onnx" and (not args.onnx or not args.light_source):
        parser.error("ONNX comparison requires --onnx and --light-source")
    if args.child_out:
        child(args)
        return
    baseline = baseline_tree(args.baseline_revision, args.baseline_directory)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    weight_path = args.model / "model.safetensors"
    with weight_path.open("rb") as stream:
        header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
    dtypes = sorted({entry["dtype"] for key, entry in header.items() if key != "__metadata__"})
    artifacts = [weight_path, args.model / "rl_agent_config.json"]
    artifacts += sorted((args.model / "tokenizer").glob("*.json"))
    if args.onnx:
        artifacts += sorted(args.onnx.parent.glob(args.onnx.name + "*"))
    report = {"command": subprocess.list2cmdline(sys.argv), "started_utc": datetime.now(timezone.utc).isoformat(),
              "baseline_revision": args.baseline_revision, "runtime": args.runtime, "threads": args.threads,
              "process_runs": args.process_runs, "repeats": args.repeats, "warmups": args.warmups,
              "checkpoint_dtypes": dtypes,
              "artifacts": {str(p.resolve()): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in artifacts},
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "laya").glob("*.py"))},
              "harness_sha256": sha256(Path(__file__)), "questions": QUESTIONS, "states": STATES,
              "runs": [], "parity_vs_first_baseline": []}
    if args.light_source:
        report["standalone_source_sha256"] = {
            str(p.relative_to(args.light_source)): sha256(p)
            for p in sorted((args.light_source / "laya_onnx").rglob("*.py"))}
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               USE_TF="0", LAYA_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads))
    for key in ("LAYA_PERFORMANCE_MODE", "LAYA_CPU_AMP", "LAYA_CACHE", "LAYA_CACHE_PATH"):
        env.pop(key, None)
    for repeat in range(args.process_runs):
        order = ("baseline", "current") if repeat % 2 == 0 else ("current", "baseline")
        for variant in order:
            source = baseline if variant == "baseline" else args.light_source if args.runtime == "onnx" else ROOT
            child_out = args.out.with_name(args.out.stem + ".child.json")
            command = [sys.executable, str(Path(__file__).resolve()), "--runtime", args.runtime,
                       "--variant", variant, "--model", str(args.model.resolve()), "--out", str(args.out),
                       "--source", str(source.resolve()), "--threads", str(args.threads),
                       "--repeats", str(args.repeats), "--warmups", str(args.warmups),
                       "--child-out", str(child_out.resolve())]
            if args.onnx:
                command += ["--onnx", str(args.onnx.resolve()), "--light-source", str(args.light_source.resolve())]
            launched = time.perf_counter()
            command += ["--parent-start", str(launched)]
            subprocess.run(command, env=env, check=True, cwd=ROOT)
            row = json.loads(child_out.read_text(encoding="utf-8"))
            child_out.unlink()
            row["process_repeat"] = repeat + 1
            reference = report["runs"][0] if report["runs"] else row
            parity = difference(row["records"], reference["records"])
            parity["answers_equal"] = [r["answers"] for r in row["outputs"]] == [r["answers"] for r in reference["outputs"]]
            report["parity_vs_first_baseline"].append(parity)
            report["runs"].append(row)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("%s %s run %d: startup %.3f s; peak working set %.1f MiB" % (
                args.runtime, variant, repeat + 1, row["startup_to_first_answer_s"],
                row["memory_stages"]["after_parity_and_forward"]["peak_working_set_bytes"] / 2**20), flush=True)
    args.out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
