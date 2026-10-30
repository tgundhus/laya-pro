# Selected upstream runtime adoption

The comparison baseline is **Laya Pro `55efeaa87ea10c334eefd0f7b5ac5fc84a1772ad`**,
not an earlier upstream release. The upstream review boundary is
`118d361a774c9a28cd02a78d6cf4d2af0614f422`. All work stays local; no dependency,
checkpoint weight, deployed calibration, public signature, or default changes.

## Correctness fixes

- `502bd88c`: a question added by start hooks may be removed uniformly by end
  hooks during a long scan. Missing caller answers and inconsistent window schemas
  raise `ValueError` instead of falling through to a `KeyError` or disappearing.
- `b2047e18`: unchanged answer objects retain their original deciding spans after
  end hooks reorder results. Cloned, replaced, prediction-edited, shared, or
  start-rewritten answers have no asserted span. Reliability metadata is retained;
  aggregation no longer adds a window to a hook/cache-owned answer object.
  The pre-end recorder is integrated with DecisionCache's restore point: miss-local
  indexes map back to full scan indexes, retained answers get recorded before later
  end hooks, and valid duplicate-window copies retain their restored origins.
- `37f2620b`, `d552d9a8`: the loaders and calibration payload path share the clamp's
  rejection predicate, including `True`. Language overrides report replaced
  entries once per language, without repeating inherited base temperatures.
- `6b37ae71`: ECE rejects boolean, noninteger, and nonpositive bucket counts.
- `ae132f47`: boolean hook deadlines are refused as boolean values; numeric
  integers and numeric strings preserve their previous behavior.

The weight-free reproduction reports five failing invariants on the actual Pro
baseline and five passing invariants after adoption in
[`upstream_runtime_regressions.json`](upstream_runtime_regressions.json).
The expanded Torch and ONNX long-scan suites additionally cover caller-answer
removal, partial removal, reordering at every hook tier, ties, cloned result
wrappers, nested prediction edits, and ambiguous shared answer objects.
The shared gate caught a partial-cache attribution regression during adoption;
the original failing consistency assertion is retained. The reproduction now also
confirms the actual Pro baseline and corrected runtime both preserve full/partial
cache-hit spans. Thirty-two added consistency checks exercise cached and computed
winners with nonzero original indexes, reordered full/partial replays, later clones
and nested prediction edits, and coalesced duplicate windows on both backends.

## Performance selection

Measurements use Windows 11, Python 3.12.14, Torch 2.14.1 CPU, NumPy 2.5.3,
two intra-op/OMP/MKL threads and one inter-op thread. Microbenchmarks use three
warmups and nine alternating before/after pairs, with repeated operations per
sample. Raw individual samples, median absolute deviations, workloads, equality
checks and source fingerprints are saved beside this report.

| Candidate | Disposition and measured evidence |
| --- | --- |
| `e54649f5` NumPy-backed Torch collation | Retain. Mixed training rows, padding, targets, dtypes, metadata and Pro parallel layouts match exactly. At 48 rows x 256 tokens / 11 options, median 1.916 -> 0.267 ms; MAD 0.031 / 0.0015 ms; faster in 9/9 pairs. At 24 x 768 / 64 options with parallel layout, 4.528 -> 0.873 ms; faster in 9/9 pairs. |
| `c100e182` direct ONNX feeds | Retain. All five feed arrays match exactly, including the masked second marker needed for one-option graphs. At 48 x 256 / 11, 1.894 -> 0.230 ms; MAD 0.0059 / 0.0017 ms; faster in 9/9 pairs. |
| `0e07dc46` reuse sequence option statistics | Retain. Reuses the already-computed option count without adding state or caches. Real English tokenizer with structured criteria: 0.0636 -> 0.0567 ms; MAD 0.0013 / 0.0009 ms; faster in 8/9 pairs. Encoded rows match exactly. |
| `0a899ea3` decoder + `6df985c7` follow-up | Retain in-place softmax and scalar clamps, retaining vectorized entropy. Combined confidence helpers at k=11: 7.39 -> 3.33 microseconds; faster in 9/9 pairs, with exact parity in 4,000 seeded scalar comparisons across float32/float64 and k=1,2,11,64,256. |
| Original `0a899ea3` Python entropy loop | Reject. Although faster at k=2/11, it is slower at k=64/256. At k=256, loop 21.72 vs retained vectorized 3.57 microseconds. Of 4,000 distributions, 1,832 differ from baseline entropy at the last bits (maximum 1.44e-15); rounded public values agree. The retained implementation preserves the reduction order. |
| `ee136b57` ONNX question-token scope | Defer. Pro already caches question heads weakly per tokenizer. Across 64 states, warm calls are 64 -> 64 and cold calls 72 -> 72. Warm and cold timing differences do not establish a win. No additional decorator or retention is introduced. |
| `c907368e` Router schema serialization | Retain only consecutive identity reuse after every start hook has completed. The typed, order-preserving `_question_schema` stays unchanged; there is no cross-call cache. A shared 20-option schema across 64 requests reduces serializations 64 -> 1 and orchestration median 1.377 -> 0.282 ms, faster in 9/9 pairs. Distinct dictionaries retain 64 -> 64 serializations; no speedup is claimed for that control. |

Main samples and checkpoint parity are in
[`upstream_runtime.json`](upstream_runtime.json); schema orchestration is in
[`upstream_runtime_schema.json`](upstream_runtime_schema.json); the entropy-loop
comparison is in [`upstream_runtime_entropy.json`](upstream_runtime_entropy.json).

Allocation evidence is deliberately narrow: Torch collation's `torch.tensor`
factory calls fall from 146 to zero for the 48-row workload and from 122 to zero
for the parallel workload. Direct ONNX feeds avoid at least 201,744 bytes of
simultaneously held intermediate Torch feed storage in the 48-row case. Python
`tracemalloc` includes NumPy's buffers but excludes Torch's native allocator, so
its higher reported NumPy peak is not comparable total-memory evidence. No RSS,
GPU-memory, throughput or universal latency reduction is claimed.

## Real English checkpoint

The offline snapshot is `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, English
ModernBERT-large, with 842,609,210-byte weights (SHA-256
`891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`).
The harness executes baseline code in isolated namespaces with the baseline's
own common helpers and head/token caches, while sharing the same loaded model.
Model inference is never stubbed for these comparisons.

Single-state and four-state batch workloads each ask a four-option choice and a
noul question at max_len=160/head_max_len=96. Two warmups and seven alternating
pairs produce exactly identical input tensors, raw logits, action probabilities,
and final decisions/probabilities/usage. Raw output differences are zero.

There is **no demonstrated end-to-end speedup** on this CPU workload: single
state medians are 416.8 -> 417.9 ms (MAD 1.2 / 5.0 ms); four-state batch medians
are 1822.1 -> 1839.9 ms (MAD 60.6 / 90.0 ms), each faster in 4/7 paired samples.
The preprocessing savings are small relative to model inference and host noise.
This supports retaining the measured preparation wins without promoting them as
an inference-speed result.

The checkpoint's `choice:11+` temperature 0.1005828 is still clamped to 0.5.
Neither parity nor these illustrative inputs establish accuracy or calibration.
Real inference parity was measured on Torch CPU; ONNX feed parity is exact,
but this change does not claim a new real-checkpoint ONNX throughput result.

## Reproduction and checks

```powershell
.venv/Scripts/python.exe research/scripts/bench_upstream_runtime.py --model C:/path/to/cached/English/snapshot --out research/results/upstream_runtime.json
.venv/Scripts/python.exe research/scripts/bench_upstream_runtime.py --schema-only --out research/results/upstream_runtime_schema.json
.venv/Scripts/python.exe research/scripts/bench_upstream_runtime.py --entropy-only --out research/results/upstream_runtime_entropy.json
.venv/Scripts/python.exe research/scripts/bench_upstream_runtime.py --regressions-only --out research/results/upstream_runtime_regressions.json
```

Targeted checks passed: consistency (279), Torch long scan (349), ONNX long scan (112), hooks (267),
calibration (248), language temperatures (75), training (66), ONNX batch (86),
runtime fixes, confidence (224), parallel options (21), question-token scope
(13 unittest cases), temperature loading (10 unittest cases), calibration
persistence (7 unittest cases), new collation contracts (3 unittest cases),
and Router batching (52 pytest cases). The changed Python files pass the
required selected Ruff rules and compilation. The parent workflow runs the
final shared CI and strict documentation build.
