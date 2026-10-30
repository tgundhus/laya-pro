---
title: Consistency and speed work, September 2026
description: What was measured and changed to make Laya's decisions repeatable and its requests faster, how it compares with Jev, and what is still open.
type: report
specificity: "Laya 0.3.20, branch claude/blissful-keller-kuw2r7 (commits b5376a0 to e544f92, on top of 4066d5d). Consistency figures come from the archived Feishu run (64 cases x 3 repeats, Jev 1.13.0 against laya-multilingual on an Apple M4). Timings come from a 4-core Intel Xeon at 2.8 GHz, on CPU with 4 threads."
credibility: "Every number links to a script and a raw result in research/, or to a test that recomputes it. The CPU timings used checkpoints with the published architectures and random weights, because the machine could not reach Hugging Face: they are right about where time goes, and say nothing about accuracy. No GPU or Apple-silicon timings were taken."
---

# Consistency and speed work, September 2026

This work set out to answer three questions about Laya's decisions:

1. Are they more repeatable than Jev's?
2. Can a repeated request be guaranteed the answer it got the first time?
3. Where does a request's time go, and how much of it can be removed?

The [October 2026 production review](production-review.md) records the later upstream merge,
SDK and cache fixes, and checks on Windows.

The [10 October upstream review](upstream-review-2026-10-10.md) records selected changes
reviewed through Laya 0.4.2, their measured integration costs and the remaining validation limits.

The [12 October upstream review](upstream-review-2026-10-12.md) covers selected changes through
Laya 0.4.6, mixed-width calibration fixes, TypeScript embedding reuse and tournament integration.

The [30 October upstream review](upstream-review-2026-10-30.md) covers selected changes through
0.4.18, runtime preparation measurements, HTTP resource bounds, MCP tools and SDK contracts.

The detailed September reports:

- [Consistency: Jev against Laya](consistency-jev-vs-laya.md): what varies between runs, and
  how voting compares with replaying decisions.
- [Where a request's time goes](request-performance.md): the stage profile, the speed-ups made,
  the CPU fast path and the encode-once spike.
- [Decision cache engineering](decision-cache-engineering.md): how the cache works, what it
  costs, retention, shared stores, concurrent requests and sharing across backends.
- [Real checkpoints on an Apple M4 Max](real-checkpoints-m4-max.md): these claims re-measured on
  the published checkpoints against `main`, plus retention, restarts, concurrency, serving and
  requests that differ only in formatting.

The user guides these reports feed are [Decision consistency](../consistency.md) and
[Where a request's time goes](../performance.md).

## Findings

| question | answer |
|---|---|
| Is Laya more repeatable than Jev? | Yes. Over three runs of the same 64 requests, 128 of 128 Laya responses were byte-identical, against 29 of 128 (23%) for Jev 1.13.0. |
| Did Laya need the cache for that? | No. 128 of 128 is Laya's default behaviour: its forward pass samples nothing. The cache covers what can still move an answer, such as other hardware, precision, batch shape or a model upgrade. |
| Can a repeat be guaranteed its first answer? | Yes, with `DecisionCache`. The first decision stored for a request is replayed to every caller while it lives, across threads, processes, machines with a shared store, and fp32, int8 and ONNX backends. |
| Does voting fix run-to-run variation? | Only partly. With a per-run flip rate q, a three-run majority still flips with probability 3q² − 2q³, which is 26% at q = 1/3. Replaying removes it. |
| Where does a request's time go? | To the encoder: 89-95% of it on CPU. Everything else adds up to about 2.5 ms for three questions on a ticket. |
| How much faster is a repeated request? | About 7,000 times faster: a hit takes 73 µs against 503 ms computed, for three questions on a ticket. |
| How much faster is a new request? | ONNX int8 is 2.07x faster (geometric mean over 13 requests) and int8 PyTorch 1.89x, at a cost in arithmetic. Behind the cache, each answer is fixed the first time it is computed. |
| What would make new requests much faster? | Encoding the state once instead of once per question: up to 4.29x at 10 questions, but it needs a retrained model. |

## Is this version better than the original?

It is better on consistency, and no worse on anything else measured:

- **Consistency.** It can give every repeat the answer it got first, which the original could
  not do across hardware, precision or model upgrades. It also reports how close each answer is
  to flipping (`decision_margins`), so borderline answers can go to review.
- **Speed on repeats.** Hits skip the model entirely.
- **Speed on new requests.** Fp32 output is unchanged, byte for byte. The profile-driven fixes
  took some milliseconds off the stages around the encoder, and an opt-in int8 path roughly
  halves CPU time. The int8 path is the one change that moves answers, and a cache in front of
  it fixes each one on first compute.
- **Dependencies.** None hosted. Every feature runs in the user's own process. A Redis store is
  an example, not a dependency.

## What changed

| commit | change |
|---|---|
| b5376a0 | `decision_margins`, and `DecisionCache` replaying decisions per request |
| dd3c694 | faster language detection, with identical output over 267,449 comparisons |
| 2e0289b | tokenize each question's head once per tokenizer: 0.71 → 0.13 ms for three questions |
| 59931c5 | cache docs: what a hit and a miss store |
| a043d62, b35b693 | stage profiler, cache benchmark and question-head timings, with results |
| 09276b1 | docs: decision consistency, and where a request's time goes |
| 9d471f8 | the Router remembers recent language detections: a hit went from 121 to 73 µs |
| e00ee8d | per-decision retention, renewal on use, and a pluggable `DecisionStore` |
| a543856 | ONNX export strips redundant attention NaN guards, and gains `--int8` |
| 74420d2, dcca178 | CPU fast-path harness and encode-once spike, with results and docs |
| 470f87e | concurrent identical requests share one forward pass |
| e544f92 | docs: coalescing, one cache across backends, pre-warming, trimming |

## Open items

- **Real checkpoints and other hardware.** Done on Apple silicon, CPU and MPS: see [Real
  checkpoints on an Apple M4 Max](real-checkpoints-m4-max.md), which changes the int8 findings.
  Every other timing here used random weights on a Xeon.
  Run `research/scripts/bench_stages.py --device mps` and `bench_cpu_fast_path.py` on an Apple
  M4 and on a GPU. Check int8's decision agreement on trained checkpoints before serving it.
- **Fused ONNX attention.** ONNX fp32 is 0.90x of PyTorch on a 1,125-character email, because
  the export unfuses attention. Fusing it into ONNX Runtime's attention operators is the next
  step.
- **Encode-once model.** Worth training, by distilling from today's model, if requests
  typically ask three or more questions.
- **A flaky pre-existing test.** A concurrency test in `tests/test_hooks.py` (id reuse) fails
  about 10 in 300 runs, on the base commit too. It is flagged and not fixed here.
