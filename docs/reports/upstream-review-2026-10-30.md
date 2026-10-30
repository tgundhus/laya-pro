---
title: Selected upstream review, 30 October 2026
description: Laya 0.4.18 adoption, measured runtime preparation, HTTP bounds, MCP and SDK validation.
---

# Selected upstream review, 30 October 2026

This selection follows the review of **724 commits, including 516 non-merge commits**, from
[`c73908d` (0.4.6)](https://github.com/NandhaKishorM/laya/commit/c73908d96ff73c0062c340a11bf7c54ec7d8bc4d)
through [`118d361` (0.4.18)](https://github.com/NandhaKishorM/laya/commit/118d361a774c9a28cd02a78d6cf4d2af0614f422).
The measurement baseline is Pro commit `55efeaa87ea10c334eefd0f7b5ac5fc84a1772ad`.
Laya-Pro keeps its English fallback, typed schema identity, strict agreement validation and
decision retention policies. Model weights and deployed calibration are unchanged.

## Changes selected

- **Runtime:** preserve long-scan attribution when hooks reorder results, handle removed auxiliary
  questions, report rejected calibration values consistently, and refuse boolean hook deadlines
  and invalid ECE bucket counts.
- **Evaluation:** validate baseline arithmetic, tolerances, exact thresholds and batch result
  cardinality. Question identity includes `option_order`. Artifact readers handle BOMs and report
  unreadable inputs; malformed Python rows and boolean confidence values are refused.
- **HTTP:** bound body delivery with `LAYA_BODY_READ_TIMEOUT` (10 seconds), cap caller batch sizes
  at the server token plan, and disable interactive docs/schema routes when authentication is
  configured. Batch tournament inference retains Pro's original-schema routing, language,
  64-question generated-call cap and ordered-choice limits. Compose and Nix forward the timeout.
- **MCP:** expose opt-in routing diagnostics, in-process long scans and tournaments. Registered
  checkpoints remain valid on the new paths; remote-only deployments refuse unsupported tools
  before loading a model. Remote requests identify themselves as `laya-pro/<version>` by default.
- **Integrations:** forward routing hints through the wrappers and reject malformed decisions.
  The optional Agno 1.x adapter supports local runners and self-hosted HTTP; its remote routing
  hints now reach the wire. Unknown agents and incomplete guardrail answers are refused.
- **SDKs:** native TypeScript adds structured batches, answer confidence, strict option ordering,
  language calibration forwarding and awaited batch hooks. The standalone Python HTTP client
  provides single/batch calls and health checks without installing the inference stack.

The confirmed TypeScript permutation defect is fixed: the original order must have the exact
length, with each integer index present once. Regression tests include the malformed order that
previously produced a confident wrong label. The new Go cache-clear race remains outside Pro
because the Go SDK is deferred; existing Pro embedding caches retain their generation guards.

## Performance measurements

Windows CPU measurements use Python 3.12.14, Torch 2.14.1, NumPy 2.5.3 and two threads.
Microbenchmarks use three warmups and nine alternating before/after pairs; checkpoint runs use
two warmups and seven pairs. The harness records raw samples, variability and source identities.

| Workload | Before | After | Interpretation |
|---|---:|---:|---|
| Torch collation, 48 rows x 256 tokens / 11 options | 1.916 ms | 0.267 ms | exact tensors and metadata; faster in 9/9 pairs |
| ONNX feed preparation, same shape | 1.894 ms | 0.230 ms | exact feed arrays; faster in 9/9 pairs |
| Router orchestration, 64 requests sharing a 20-option schema | 1.377 ms | 0.282 ms | serializations fall from 64 to 1; reuse is call-local |
| Confidence helpers, 11 options | 7.39 µs | 3.33 µs | exact parity in 4,000 seeded comparisons |
| Real English checkpoint, one state | 416.8 ms | 417.9 ms | no demonstrated end-to-end speedup |
| Real English checkpoint, four-state batch | 1822.1 ms | 1839.9 ms | no demonstrated end-to-end speedup |

Checkpoint inputs, logits, action probabilities and returned outputs matched exactly. Single-state
median absolute deviations were 1.2/5.0 ms; batch deviations were 60.6/90.0 ms. Each after path
was faster in 4/7 pairs. Preparation savings are small beside model inference and host noise.
These measurements establish no accuracy, calibration or universal latency improvement.

Collation avoids 146 intermediate Torch tensor factory calls on the 48-row workload. ONNX feeds
avoid at least 201,744 bytes of intermediate Torch storage there. This is allocation evidence,
not a total-memory/RSS measurement.

[Runnable harness](https://github.com/tgundhus/laya-pro/blob/main/research/scripts/bench_upstream_runtime.py),
[main samples](https://github.com/tgundhus/laya-pro/blob/main/research/results/upstream_runtime.json),
[schema samples](https://github.com/tgundhus/laya-pro/blob/main/research/results/upstream_runtime_schema.json),
[full runtime notes](https://github.com/tgundhus/laya-pro/blob/main/research/results/upstream_runtime_review.md).

## Validation and limits

Native TypeScript passed 1,214 tests, public type checks and an external packed consumer. MCP
passed 1,006 script checks, 94 device checks and 162 focused pytest cases. The HTTP suite passed
326 tests; the Python API contract passed 769 checks. Agno passed 219 checks against the actual
1.8.4 framework. The new evaluation regression suite passed 93 cases; 82 fail on the original Pro
baseline, with 11 controls passing.

The shared gate caught a cache/long-scan interaction during adoption: cache restores changed
row indexes after attribution was recorded. The fix preserves cached and computed provenance
before later hooks. All original assertions remain, with 32 additional consistency checks.
The shared gate passed all 99 suites; its pytest group passed 859 tests with one
checkpoint-dependent skip. Ruff, compilation, strict documentation and 50-URL crawl checks passed.

Independent review found one further Agno issue: incomplete score distributions could be read
as safe. The correction requires the configured rubric's keys, bounded values and a normalized
total, allowing four-decimal rounding. The absent-probabilities score fallback remains supported.
All 28 focused adapter regressions and 219 real-framework checks pass after this correction.

The Python client wheel and rebuilt source distribution each passed 65 tests with one Windows
symlink-privilege skip, then extended/tournament and strict single/batch HTTP integration. The
consumer environments contain no inference or third-party HTTP packages. CI covers Python
3.10 and 3.13 in addition to the local Python 3.12 run.

The new Go, Rust and C++ ports remain deferred. Option-stability features, tournament
evaluation attribution and staged exporter changes are also deferred. ONNX token reuse adds no
saved calls over Pro's existing head cache; the Python entropy loop was rejected for slower
large-option performance and changed floating-point reduction. Neither is imported.

Real-checkpoint parity here uses Torch CPU. Native accelerator behavior, real ONNX throughput
and Nix validation remain unverified locally. Confidence thresholds still need representative
labelled data; tournament confidence describes the surviving finalists.
