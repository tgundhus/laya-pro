---
title: Selected upstream review, 12 October 2026
description: Laya 0.4.6 adoption decisions, calibration fixes, embedding cache measurements and follow-up validation.
---

# Selected upstream review, 12 October 2026

This review covers **210 commits, including 128 non-merge commits**, from upstream
[`68804629` (0.4.2)](https://github.com/NandhaKishorM/laya/commit/68804629e8ccd9d616d48a40e87de9fabbeae069)
through [`c73908d` (0.4.6)](https://github.com/NandhaKishorM/laya/commit/c73908d96ff73c0062c340a11bf7c54ec7d8bc4d).
The [comparison](https://github.com/NandhaKishorM/laya/compare/68804629e8ccd9d616d48a40e87de9fabbeae069...c73908d96ff73c0062c340a11bf7c54ec7d8bc4d)
records the boundary. Laya-Pro remains version 0.3.29 with selected upstream changes;
it keeps its English fallback, typed schema identity and decision retention policies.
Checkpoint weights and shipped calibration files are unchanged.

## Changes adopted

| Area | Result and source |
|---|---|
| Calibration | Fit mixed option counts with a row-weighted objective over real columns; reject falsy non-object temperature maps. [Fitter](https://github.com/NandhaKishorM/laya/commit/97bb3a6), [loading](https://github.com/NandhaKishorM/laya/commit/63eef56) |
| Confidence | A zero threshold removes stale flags. TypeScript gates choice/score answers on calibrated confidence rather than a selected score value. Export the existing strict `jev_confidence` helper at the Python root. [Python](https://github.com/NandhaKishorM/laya/commit/2863500), [TypeScript](https://github.com/NandhaKishorM/laya/commit/3c645e9), [export](https://github.com/NandhaKishorM/laya/commit/b937ce0) |
| Runtime | Restore eager forward correctly, reject null batch states before model loading, and report length sorting per actual checkpoint group. CLI question files can use `questions` as a bare question id. [Eager](https://github.com/NandhaKishorM/laya/commit/b738272), [batch](https://github.com/NandhaKishorM/laya/commit/8d57836), [CLI](https://github.com/NandhaKishorM/laya/commit/a69fa97), [question files](https://github.com/NandhaKishorM/laya/commit/6e5ad07) |
| Evaluation | Accept direct Agents without a `model` keyword; refuse explicit pins they cannot honor. Count actual gate abstentions, expose ONNX source revisions and report invalid CLI model names as usage errors. Pro typed schema grouping and strict agreement validation remain in force. [Runner compatibility](https://github.com/NandhaKishorM/laya/commit/747e62e), [abstention](https://github.com/NandhaKishorM/laya/commit/9b47d34), [provenance](https://github.com/NandhaKishorM/laya/commit/a3f62b7) |
| MCP | Nullable model schemas match automatic routing; invalid batch controls reach validation. Shortlist responses report dropped full-set orders. [Schemas](https://github.com/NandhaKishorM/laya/commit/b259749), [controls](https://github.com/NandhaKishorM/laya/commit/1454580), [orders](https://github.com/NandhaKishorM/laya/commit/bffb34b) |
| HTTP | Opt-in `LAYA_TOURNAMENT_GROUP` handles up to 255 choice labels by elimination. Strict projection removes tournament metadata; the TypeScript HTTP client validates it when present. Batch diagnostics are explicitly refused because an in-place Python report cannot cross JSON. [Tournament](https://github.com/NandhaKishorM/laya/commit/c3e631b), [group bound](https://github.com/NandhaKishorM/laya/commit/472f098), [diagnostics](https://github.com/NandhaKishorM/laya/commit/9f7a6a6) |
| Native TypeScript | Add a bounded embedding cache and tournament helper, and trim URL suffixes without regex backtracking. [Cache](https://github.com/NandhaKishorM/laya/commit/4739b93), [tournament](https://github.com/NandhaKishorM/laya/commit/987ac82), [URL handling](https://github.com/NandhaKishorM/laya/commit/885b273) |
| Configuration | Forward the HTTP tournament setting through Compose and Nix. Windows CI runs the same shared Python suite list as Linux and release checks. [Shared suites](https://github.com/NandhaKishorM/laya/commit/3fa307c) |

## Additional fixes

Python and TypeScript embedding caches now track a clear generation. An embedding started
before a clear can return its own snapshot, but cannot repopulate a cache belonging to a
replacement model. Locking the dimension check alone did not fix that race.

HTTP tournament eligibility also controls the raised option cap. An ordered choice that
requires the one-pass path retains its 100-option cap and is refused before inference when
oversized. Generated elimination calls contain at most 64 questions. MCP route deadlines and
batch sizes reject booleans instead of interpreting `False` as an unset control.

Independent review found that generated tournament question ids could change automatic workflow
routing between rounds. Python and TypeScript now resolve the original schema first and keep its
checkpoint and detected language throughout elimination. The review also caught two test errors
in the API annotation check and cached batch result shape; both are corrected.

## Measurements

The calibration comparison uses seeded synthetic rows, CPU Torch with two threads, one warmup
and three alternating before/after repeats. Mixed 2/9-option argmax rows fitted `T=1.0` before
and `T=0.5` after; true unpadded NLL fell from **0.765155 to 0.479609**. Mixed negative 2/4-option
rows fitted `T=5.0` before and `T=0.5` after, with NLL **0.881463 to 0.241116**. A fixed-width
interior optimum stayed at `T=0.522357`. These demonstrate fitter correctness; they do not
establish pretrained accuracy or justify replacing deployed calibration files.

Fitter medians for the mixed 2/9 and negative 2/4 panels were **16.179→2.285 ms** and
**61.440→1.307 ms**. The interior mixed 3/20 panel cost **42.816→46.518 ms** and changed NLL
only from 1.063982619 to 1.063981773; the fixed-width panel cost **50.778→51.049 ms**.
The fix adds width-group work when both fits already converge. These are small fitter-only
measurements, separate from inference latency.
[Harness](https://github.com/tgundhus/laya-pro/blob/main/research/scripts/bench_calibration_ragged.py),
[raw comparison](https://github.com/tgundhus/laya-pro/blob/main/research/results/calibration_ragged_windows_20261012.json).

The native TypeScript embedding benchmark uses the retained English checkpoint, a 20-label
catalog and two distinct requests per repeat, two warmups and three alternating timed repeats.
Reusing label vectors reduced measured encoder rows from **126 to 6**, with identical rankings,
and median request cost from **294.525 to 33.726 ms**. This measures embedding and ranking,
excluding decision inference. Two ONNX Runtime threads, a 64-token embedding window and graph
hashes are recorded with the
[raw samples](https://github.com/tgundhus/laya-pro/blob/main/laya-ts/examples/results/cached-embedder-windows-20261012.json).

The HTTP English smoke used checkpoint `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, two CPU Torch
threads, two warmups and three alternating repeats. One-pass cost was **344.143 ms**, versus
**584.469 ms** for HTTP tournament inference on the same 20-label request with groups of 16.
HTTP timing includes the transport; this demonstrates additional cost, not a speed claim.
Finalist probabilities matched direct Python tournament output exactly, and a 128-label request
completed with eight finalists. The checkpoint's 11+ temperature is clamped to 0.5 by the existing
loader and remains uncalibrated. Measurements ran on an active Windows workstation.
[Harness](https://github.com/tgundhus/laya-pro/blob/main/research/scripts/http_tournament_smoke.py),
[raw samples](https://github.com/tgundhus/laya-pro/blob/main/research/results/http_tournament_windows_20261012.json).

Native TypeScript passed **887 tests**, type checks and an external packed-package consumer.
The HTTP client passed **33 tests**, including packaging and typed tournament metadata.
Four real English tournament scenarios covered zero, one and two elimination rounds. Winners,
finalists, call counts and usage matched Python exactly; the largest returned numeric difference
was `1e-4`, one four-decimal rounding unit.
[Parity record](https://github.com/tgundhus/laya-pro/blob/main/laya-ts/examples/results/tournament-parity-windows-20261012.json).

## Validation

The shared 92-suite Python gate was exercised. Its pytest group passed 546 tests with one
checkpoint-dependent skip; after correcting a new mock's label set, the shortlist script passed
249 checks on rerun. The API contract passed 688 checks. Separate evaluation suites passed 237
tests, with 50 API and 40 ONNX runner checks. Ruff and compilation passed. The strict clean docs
build reported no issues, and identity/crawl checks covered 49 sitemap URLs.
Follow-up independent review verified all three corrections and found no further actionable
defects in the complete diff, including new files.

## Deferred changes and limits

The new Go port currently supplies language detection and internal tokenizer foundations;
its agent, router and ONNX inference layers are planned for later changes. It is not adopted
as a production decision SDK. The upstream multilingual default would change ambiguous-state
routing, so Laya-Pro retains English and supports explicit multilingual configuration.

Native TypeScript hook deadlines remain deferred. The newer formatting and timer cleanup do
not prevent an abandoned callback from modifying its shared context. Awaited hooks keep their
existing contract. Docker and Nix are unavailable on this Windows host. Docker build and
runtime checks run in CI; Nix validation remains unverified. ONNX evaluation revisions identify configuration/tokenizer source,
not a hash of separately supplied graphs.

Tournament probabilities and gates describe the surviving finalists. Elimination can change
the selected label, costs additional inference and needs workload-specific quality evaluation.
No new accuracy, calibration or cross-language quality claim is made.
