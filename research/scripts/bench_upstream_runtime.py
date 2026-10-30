"""Compare selected runtime candidates with the actual Pro baseline, offline on two CPU threads.

Run: .venv/Scripts/python.exe research/scripts/bench_upstream_runtime.py --out research/results/upstream_runtime.json
Add --model <cached English checkpoint> for real-checkpoint comparisons.
The trusted baseline functions are executed in isolated module namespaces; no refs or files change.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import types
import warnings
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import laya.common as common  # noqa: E402
from laya.agent import Agent  # noqa: E402
from laya.onnx_agent import ONNXAgent, _collate_items_numpy  # noqa: E402

agent_module = importlib.import_module("laya.agent")
onnx_module = importlib.import_module("laya.onnx_agent")
BASE = "55efeaa87ea10c334eefd0f7b5ac5fc84a1772ad"


def source_module(path, name):
    source = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT, text=True)
    if path != "laya/common.py":
        source = source.replace("from .common import (", "from ._adoption_common import (")
        source = source.replace("from laya.common import (", "from laya._adoption_common import (")
    module = types.ModuleType("laya." + name)
    module.__package__ = "laya"
    module.__file__ = str(ROOT / path) + "@" + BASE
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def summary(pairs):
    before = [p["before"] for p in pairs]
    after = [p["after"] for p in pairs]
    med = lambda values: statistics.median(values)
    mad = lambda values: med([abs(v - med(values)) for v in values])
    return {"before_median_ms": med(before), "after_median_ms": med(after),
            "before_mad_ms": mad(before), "after_mad_ms": mad(after),
            "median_paired_saving_ms": med([a - b for a, b in zip(before, after)]),
            "after_faster_pairs": sum(b < a for a, b in zip(before, after)),
            "pairs_ms": pairs}


def measure(before, after, *, loops=30, repeats=9, warmups=3):
    for _ in range(warmups):
        before()
        after()
    pairs = []
    for i in range(repeats):
        pair = {}
        for key in (("before", "after") if i % 2 == 0 else ("after", "before")):
            fn = before if key == "before" else after
            start = time.perf_counter()
            for _ in range(loops):
                fn()
            pair[key] = (time.perf_counter() - start) * 1000 / loops
        pairs.append(pair)
    return {"loops_per_sample": loops, "warmups": warmups, **summary(pairs)}


def python_peak(fn):
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, result


def old_onnx_collate(base_common, batch, pad):
    result = base_common.collate_items(batch, pad)
    feeds = {key: result[key].numpy().astype(bool if key == "marker_mask" else np.int64)
             for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}
    if feeds["marker_pos"].shape[1] < 2:
        for key in ("marker_pos", "marker_mask"):
            feeds[key] = np.pad(feeds[key], ((0, 0), (0, 2 - feeds[key].shape[1])))
    return feeds


@contextlib.contextmanager
def runtime_mode(before, base_agent, base_onnx):
    cls, baseline = (ONNXAgent, base_onnx.ONNXAgent) if isinstance(before, ONNXAgent) else (Agent, base_agent.Agent)
    with contextlib.ExitStack() as stack:
        for name in ("predict_batch", "_encode_state", "_decode_answers"):
            stack.enter_context(patch.object(cls, name, getattr(baseline, name)))
        if cls is ONNXAgent:
            stack.enter_context(patch.object(cls, "_infer_batch", baseline._infer_batch))
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--regressions-only", action="store_true")
    parser.add_argument("--entropy-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    bc = source_module("laya/common.py", "_adoption_common")
    ba = source_module("laya/agent.py", "_adoption_agent")
    bo = source_module("laya/onnx_agent.py", "_adoption_onnx")
    report = {"baseline": BASE, "seed": 29018, "candidate_source_sha256": {
                  p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                  for p in ("laya/agent.py", "laya/common.py", "laya/onnx_agent.py")},
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "cpu": platform.processor(), "torch": torch.__version__, "numpy": np.__version__,
                              "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                              "omp_threads": os.environ["OMP_NUM_THREADS"], "mkl_threads": os.environ["MKL_NUM_THREADS"]},
              "microbenchmarks": [], "checkpoint": None,
              "memory_note": "tracemalloc is Python-visible peak, not native Torch allocator or process RSS"}

    def record(name, before, after, **kwargs):
        row = {"name": name, **measure(before, after, **kwargs)}
        report["microbenchmarks"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "pairs_ms"}), flush=True)
        return row

    if args.entropy_only:
        commit = "0a899ea3be19b7c1d4fcdb809f27712d4dfa1d03"
        source = subprocess.check_output(["git", "show", commit + ":laya/common.py"], cwd=ROOT, text=True)
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "confidence_from_probs")
        namespace = dict(bc.__dict__)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "entropy_loop@" + commit, "exec"), namespace)
        loop = namespace["confidence_from_probs"]
        rng = np.random.default_rng(29018)
        for width in (2, 11, 64, 256):
            distributions = rng.dirichlet(np.ones(width), size=1000)
            deltas = [abs(loop(p, width) - bc.confidence_from_probs(p, width)) for p in distributions]
            p = distributions[0]
            row = record("entropy_loop_vs_vectorized_k%d" % width,
                         lambda: loop(p, width), lambda: common.confidence_from_probs(p, width), loops=1000)
            row.update(before_source=commit, after_source="retained vectorized entropy and native scalar clamp",
                       parity_distributions=len(distributions), loop_exact_mismatches=sum(d != 0 for d in deltas),
                       loop_max_abs_difference=max(deltas),
                       rounded_4dp_mismatches=sum(round(loop(p, width), 4) != round(bc.confidence_from_probs(p, width), 4)
                                                 for p in distributions))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        return

    if args.regressions_only:
        hooks = importlib.import_module("laya.hooks")
        bh = source_module("laya/hooks.py", "_adoption_hooks")
        source = (ROOT / "tests/test_predict_long.py").read_text(encoding="utf-8")
        # Reuse only the existing weight-free fixture definitions, not its top-level suite.
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("_Tok", "make_real_agent", "_question_agent")]
        namespace = {"Agent": Agent, "np": np, "threading": ba.threading,
                     "NO_STATE_STATS": {"state_tokens": 0, "state_tokens_used": 0,
                                        "state_tokens_dropped": 0, "truncated": False}}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "long_scan_fixtures", "exec"), namespace)
        questions = {"dept": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}},
                     "flag": {"type": "noul", "instructions": "?"}}
        outcomes = {}
        for label, cm, hm, cls in (("before", bc, bh, ba.Agent), ("after", common, hooks, Agent)):
            result = {}
            try:
                cm.ece_score(np.array([0.9]), np.array([0.0]), bins=0)
                result["ece_rejects_zero_bins"] = False
            except ValueError:
                result["ece_rejects_zero_bins"] = True
            try:
                hm.validate_timeout(True)
                result["timeout_rejects_bool"] = False
            except ValueError:
                result["timeout_rejects_bool"] = True
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                cm.resolve_lang_temperatures({"de": {"temperature": [True, 1.0, 1.0]}}, [1.0] * 3)
            result["warns_on_bool_language_temperature"] = len(seen) == 1
            namespace["Agent"] = cls
            make = namespace["_question_agent"]
            expected = make().predict_long("x" * 300, questions)
            reordered = make().predict_long("x" * 300, questions, on_predict_end=lambda ctx: ctx.results.reverse())
            result["reorder_preserves_deciding_span"] = expected["answers"]["dept"] == reordered["answers"]["dept"]
            def add(ctx):
                ctx.questions["temporary"] = dict(questions["flag"])
            def remove(ctx):
                for row in ctx.results:
                    del row["answers"]["temporary"]
            try:
                actual = make().predict_long("x" * 300, dict(questions), on_predict_start=add, on_predict_end=remove)
                result["allows_uniform_removal_of_start_added_question"] = actual["answers"] == expected["answers"]
            except KeyError:
                result["allows_uniform_removal_of_start_added_question"] = False
            outcomes[label] = result
        assert not any(outcomes["before"].values())
        assert all(outcomes["after"].values()), outcomes
        report["correctness_regressions"] = outcomes
        # Preserve Pro's pre-existing full/partial replay contract on both backends as well.
        from laya.consistency import DecisionCache
        source = (ROOT / "tests/test_consistency.py").read_text(encoding="utf-8")
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in ("CharTok", "make_scanner", "make_onnx_scanner")]
        namespace = {"np": np, "threading": ba.threading}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "cached_scan_fixtures", "exec"), namespace)
        team = {"team": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"billing": "charges", "tech": "errors"}}}
        doc = " ".join("w%d" % i for i in range(120))
        cache_contract = {}
        for label, cls, onnx_cls in (("before", ba.Agent, bo.ONNXAgent), ("after", Agent, ONNXAgent)):
            namespace.update(Agent=cls, ONNXAgent=onnx_cls)
            for backend, factory in (("torch", namespace["make_scanner"]), ("onnx", namespace["make_onnx_scanner"])):
                scanner = factory([DecisionCache()])
                first = scanner.predict_long(doc, team, window=80, stride=50)
                full = scanner.predict_long(doc, team, window=80, stride=50)
                partial = scanner.predict_long(doc + " and a tail the first document did not have", team, window=80, stride=50)
                cache_contract[label + "_" + backend] = {
                    "full_hit_span_matches": full["answers"] == first["answers"],
                    "full_hit_inference_windows": full["usage"]["windows"],
                    "partial_hit_span": partial["answers"]["team"].get("window"),
                    "partial_hit_inference_windows": partial["usage"]["windows"]}
                assert full["answers"] == first["answers"] and full["usage"]["windows"] == 0
                assert 0 < partial["usage"]["windows"] < partial["answers"]["team"]["window"]["count"]
        report["preserved_pro_cache_contract"] = cache_contract
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(outcomes), flush=True)
        return

    if args.schema_only:
        router_module = importlib.import_module("laya.router")
        source = subprocess.check_output(["git", "show", BASE + ":laya/router.py"], cwd=ROOT, text=True)
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Router")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "predict_batch")
        namespace = dict(router_module.__dict__)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "router.py@" + BASE, "exec"), namespace)
        old_batch = namespace["predict_batch"]
        class StubAgent:
            def predict_batch(self, states, questions, **kwargs):
                return [{"answers": {"state": state}, "usage": {}} for state in states]
        router = router_module.Router(default="english")
        router.load = lambda model: StubAgent()
        router.route_batch = lambda requests, **kwargs: [{"model": "english"} for _ in requests]
        for count, width, shared in ((4, 2, True), (64, 20, True), (64, 20, False)):
            questions = {"intent": {"type": "choice", "instructions": "Pick a department",
                                    "criteria": {str(i): {"description": "department %d" % i} for i in range(width)}}}
            requests = [{"state": str(i), "questions": questions if shared else json.loads(json.dumps(questions))}
                        for i in range(count)]
            before = lambda: old_batch(router, requests)
            after = lambda: router.predict_batch(requests)
            assert before() == after()
            row = record("router_schema_%dstates_%doptions_shared%s" % (count, width, shared), before, after, loops=30)
            row["outputs_exact"] = True
            calls = []
            real_schema = router_module._question_schema
            def counted_schema(questions):
                calls.append(questions)
                return real_schema(questions)
            counts = {}
            for label, fn in (("before", before), ("after", after)):
                calls.clear()
                with patch.dict(old_batch.__globals__, _question_schema=counted_schema), patch.object(
                        router_module, "_question_schema", counted_schema):
                    fn()
                counts[label] = len(calls)
            row["schema_serializations"] = counts
            row["question_schema_identity"] = "unchanged typed, order-preserving _question_schema; call-local reference only"
        report["candidate_source_sha256"]["laya/router.py"] = hashlib.sha256((ROOT / "laya/router.py").read_bytes()).hexdigest()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        return

    for rows, length, width, parallel in ((1, 64, 2, False), (48, 256, 11, False), (24, 768, 64, True)):
        batch = [[{"ids": list(range(length - i % 17)), "markers": list(range(width - i % 2)),
                   "qtype": i % 3, "target": [1.0] + [0.0] * (width - i % 2 - 1),
                   **({"layout": {"position_ids": list(range(length - i % 17)),
                                   "option_ids": [0] * (length - i % 17)}} if parallel else {})}
                  for i in range(rows)]]
        before = lambda: bc.collate_items(batch, 0)
        after = lambda: common.collate_items(batch, 0)
        left, right = before(), after()
        assert left.keys() == right.keys()
        assert all(torch.equal(left[k], right[k]) if isinstance(left[k], torch.Tensor) else left[k] == right[k]
                   for k in left)
        row = record("collate_%dx%d_k%d_parallel%s" % (rows, length, width, parallel), before, after)
        row["tensor_and_metadata_parity"] = True
        row["python_peak_bytes"] = {"before": python_peak(before)[0], "after": python_peak(after)[0]}
        counts = {}
        real_tensor = torch.tensor
        for label, fn in (("before", before), ("after", after)):
            with patch.object(torch, "tensor", wraps=real_tensor) as calls:
                fn()
                counts[label] = calls.call_count
        row["torch_tensor_factory_calls"] = counts
        if not parallel:
            ob = lambda: old_onnx_collate(bc, batch, 0)
            oa = lambda: _collate_items_numpy(batch, 0)
            assert all(np.array_equal(ob()[key], oa()[key]) for key in oa())
            row = record("onnx_feeds_%dx%d_k%d" % (rows, length, width), ob, oa)
            row["all_feeds_exactly_equal"] = True
            row["python_peak_bytes"] = {"before": python_peak(ob)[0], "after": python_peak(oa)[0]}
            row["feed_storage_bytes"] = sum(value.nbytes for value in oa().values())
            row["baseline_additional_torch_feed_storage_bytes"] = row["feed_storage_bytes"]

    rng = np.random.default_rng(29018)
    parity = 0
    for dtype in (np.float32, np.float64):
        for width in (1, 2, 11, 64, 256):
            distributions = rng.dirichlet(np.ones(width), size=200).astype(dtype)
            for p in distributions:
                for name in ("answer_confidence", "confidence_from_probs"):
                    assert getattr(bc, name)(p, width) == getattr(common, name)(p, width)
                    parity += 1
    for width in (2, 11, 64, 256):
        p = rng.dirichlet(np.ones(width))
        row = record("confidence_k%d" % width,
                     lambda: (bc.answer_confidence(p, width), bc.confidence_from_probs(p, width)),
                     lambda: (common.answer_confidence(p, width), common.confidence_from_probs(p, width)), loops=1000)
        row["exact_scalar_parity_cases_total"] = parity

    if args.model:
        agent = Agent(args.model, device="cpu", fast=False, compile=False)
        config_bytes = (Path(args.model) / "rl_agent_config.json").read_bytes()
        report["checkpoint"] = {"path": args.model, "snapshot_revision": Path(args.model).name,
                                "weights_size": (Path(args.model) / "model.safetensors").stat().st_size,
                                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                                "dtype": str(agent.dtype), "workloads": [],
                                "caveat": "choice:11+ temperature 0.1005828 is clamped to 0.5; parity is not accuracy/calibration evidence"}
        questions = {"team": {"type": "choice", "instructions": "Which team should handle this request?",
                              "criteria": {"billing": {"reason": "payments and refunds"},
                                           "technical": {"reason": "errors and outages"},
                                           "sales": {"reason": "purchase and pricing"},
                                           "other": {"reason": "other requests"}}},
                     "urgent": {"type": "noul", "instructions": "Does the request require urgent action?"}}
        states = ["My account was charged twice and I need a refund.",
                  "Our production service is unavailable. Please help urgently.",
                  "Please send a quote for twenty new licenses.", "Thank you for your help."]
        ids = list(questions)
        internal = {key: agent._to_internal(value) for key, value in questions.items()}
        seq_before = lambda: ba.Agent._encode_state(agent, states[0], ids, internal, 160, 96)
        seq_after = lambda: agent._encode_state(states[0], ids, internal, 160, 96)
        assert seq_before() == seq_after()
        record("encode_state_real_tokenizer_structured_criteria", seq_before, seq_after, loops=100)
        # Assess the omitted ONNX decorator against Pro's existing question-head cache.
        # Both sides use exactly the same candidate encoder; only the new scope differs.
        def prepare():
            return [ONNXAgent._encode_state(agent, state, ids, internal, 160, 96) for state in states * 16]
        reused = common._reuse_question_tokens(prepare)
        class CountingTokenizer:
            def __init__(self, tok):
                self.tok, self.calls = tok, 0
            def __getattr__(self, name):
                return getattr(self.tok, name)
            def __call__(self, *args, **kwargs):
                self.calls += 1
                return self.tok(*args, **kwargs)
        original_tok = agent.tok
        agent.tok = counter = CountingTokenizer(original_tok)
        try:
            for cold in (False, True):
                def prepare_before():
                    if cold:
                        common._QUESTION_HEADS.clear()
                    return prepare()
                def prepare_after():
                    if cold:
                        common._QUESTION_HEADS.clear()
                    return reused()
                assert prepare_before() == prepare_after()
                row = record("proposed_onnx_token_reuse_64states_%s_heads" % ("cold" if cold else "warm"),
                             prepare_before, prepare_after, loops=10)
                counts = {}
                for label, fn in (("before", prepare_before), ("after", prepare_after)):
                    counter.calls = 0
                    fn()
                    counts[label] = counter.calls
                row["tokenizer_calls"] = counts
                row["encoded_rows_exact"] = True
                row["disposition"] = "deferred: Pro's existing head cache already avoids repeated question tokenization"
        finally:
            agent.tok = original_tok
        for name, fn, workload in (
                ("single_2_questions", lambda: agent.system_one(states[0], questions, max_len=160, head_max_len=96),
                 {"states": states[:1], "max_len": 160, "head_max_len": 96}),
                ("batch4_2_questions", lambda: agent.predict_batch(states, questions, batch_size=4, max_len=160, head_max_len=96),
                 {"states": states, "batch_size": 4, "max_len": 160, "head_max_len": 96})):
            def old():
                with runtime_mode(agent, ba, bo):
                    return fn()
            expected = old()
            actual = fn()
            assert expected == actual
            raw = {}
            original = agent._forward
            def capture(batch):
                output = original(batch)
                raw.setdefault("inputs", []).append({k: v.clone() for k, v in batch.items() if isinstance(v, torch.Tensor)})
                raw.setdefault("outputs", []).append(tuple(v.copy() for v in output))
                return output
            with patch.object(agent, "_forward", capture):
                old()
                before_raw, raw = raw, {}
                fn()
            assert all(torch.equal(a[k], b[k]) for a, b in zip(before_raw["inputs"], raw["inputs"]) for k in a)
            assert all(np.array_equal(a, b) for aa, bb in zip(before_raw["outputs"], raw["outputs"]) for a, b in zip(aa, bb))
            row = {"name": name, "workload": workload, "questions": questions,
                   "input_tensors_exact": True, "logits_and_action_probabilities_exact": True, "outputs_exact": True,
                   "logits_max_abs_difference": 0.0, "action_probabilities_max_abs_difference": 0.0,
                   "reference_output": expected,
                   **measure(old, fn, loops=1, repeats=args.repeats, warmups=2)}
            report["checkpoint"]["workloads"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("pairs_ms", "questions", "workload", "reference_output")}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
