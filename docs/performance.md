---
title: Where a request's time goes
description: Stage-by-stage timings of a Laya request, the cost of the decision cache, and which runtime options make a request faster, measured with reproducible scripts.
type: reference
specificity: "Laya 0.3.20 on a 4-core Intel Xeon at 2.8 GHz (AVX-512 VNNI, no native BF16), PyTorch 2.14 on CPU, fp32, 4 threads. Checkpoints with the published architectures (421.29M and 321.91M parameters) and random weights, with a byte-level BPE tokenizer trained locally standing in for the published ones."
credibility: "Measured with research/scripts/bench_stages.py and research/scripts/bench_decision_cache.py; raw results in research/results/. Time depends on tensor shapes, not weight values, so the split between stages holds for the real checkpoints; accuracy and the size of probability changes do not carry over from random weights. GPU and Apple-silicon splits differ and are not measured here."
---

# Where a request's time goes

For opt-in resource budgets on current code, see [Performance modes](performance-modes.md).
Modes adjust threads, batching and model lifecycle without changing precision or input windows;
the historical stage measurements below use their own stated hardware and settings.

`research/scripts/bench_stages.py` wraps a timer around every stage a `Router.predict` call goes
through and times each scenario twice: once with the timers, to see where the time goes, and
once without, which is the latency to quote. The figures below come from a 4-core Xeon CPU. The
machine could not reach Hugging Face, so the checkpoints had the published architectures and
parameter counts but random weights; time depends on shapes, not weight values. Run the script
on your own hardware for your own split ([below](#measure-it-on-your-machine)).

## The stages

Milliseconds per request, median of 10 runs after 2 warm-ups. The English columns use the
`english` architecture (ModernBERT-large), the Spanish ones `multilingual` (mmBERT-base); the
Router chose the checkpoint in every case.

| stage | ticket, 1 question | ticket, 3 | ticket, 10 | email, 3 | 32 tickets, 3 | Spanish ticket, 3 | Spanish email, 3 |
|---|---|---|---|---|---|---|---|
| routing (language detection) | 0.21 | 0.21 | 0.20 | 0.97 | 1.84 | 0.22 | 0.43 |
| tokenize the state | 0.34 | 0.35 | 0.34 | 1.18 | 3.16 | 0.35 | 0.91 |
| build rows: tokenize questions and options | 0.41 | 1.02 | 2.59 | 0.97 | 22.7 | 0.94 | 0.97 |
| collate tensors | 0.17 | 0.26 | 0.51 | 0.38 | 3.70 | 0.29 | 0.35 |
| **encoder forward** | **260** | **505** | **1,390** | **2,239** | **12,034** | **155** | **1,300** |
| decision head forward | 18.5 | 27.1 | 80.2 | 167 | 771 | 17.0 | 101 |
| host copy and decoding | 0.25 | 0.43 | 0.64 | 0.49 | 4.12 | 0.42 | 0.50 |
| validation, hooks, results | 0.21 | 0.24 | 0.25 | 0.26 | 1.55 | 0.24 | 0.28 |
| encoder share | 92.8% | 94.5% | 94.2% | 92.9% | 93.7% | 88.8% | 92.6% |
| **latency without timers** | **311** | **503** | **1,542** | **2,237** | **12,289** | **163** | **749** |

The staged runs add their own overhead, so their stages sum to a little more than the plain
latency. These stages ran before the question-head cache below; routing already had the faster
language detection.

On a CPU the encoder is the request: 89-95% of it, and the decision head most of the rest.
Everything around them -- routing, tokenization, collation, decoding, hooks -- adds up to about
2.5 ms for three questions on a ticket. An accelerator shortens the encoder but not those
milliseconds: on a T4, where one question takes 33-40 ms
([BENCHMARKS.md](https://github.com/tgundhus/laya-pro/blob/main/BENCHMARKS.md)), they are a
visible share, and they are a place to look for the per-call overhead BENCHMARKS.md reports on a
GB10 and could not place.

Time grows with the number of rows, one per question and state, and with each row's length.
One question on the short ticket took 311 ms, three took 503 ms and ten took 1,542 ms; the same
three questions on a 1,125-character email took 2,237 ms. Every question row carries the whole
state, so trimming a state to what the questions need is the largest lever there is.

## What makes a request faster

Each option was timed against the baseline in alternating runs on the same two requests (10
each), so drift in the machine cancels out.

| option | ticket, 3 questions | email, 3 questions | answers against the fp32 baseline |
|---|---|---|---|
| int8 dynamic quantization of the encoder | **1.80x faster** | **1.64x faster** | shifted; 1 of 3 decisions changed |
| ONNX Runtime (`ONNXAgent`), export before the fix below | 1.17x faster | 1.30x slower | identical |
| `torch.inference_mode()` | 1.02x | 1.01x | identical |
| 2 threads instead of 4 | 1.48x slower | 1.65x slower | identical |
| 1 thread instead of 4 | 2.58x slower | 2.63x slower | identical |
| bf16 autocast on a CPU without native BF16 | 3.10x slower | 3.47x slower | shifted; 1 of 3 decisions changed |

Random weights put every option within about 1e-4 of the others, so any change in arithmetic
can flip which one wins. A trained checkpoint has far fewer answers that close, but int8 and
bf16 do change the arithmetic: check them against your own labels before you rely on them.

- **Replaying a decision.** A repeated request through a
  [`DecisionCache`](consistency.md#replaying-decisions) skips everything from tokenization to
  decoding: 0.106 ms for the ticket and 0.729 ms for the email in this run, against 503 ms and
  2,237 ms. Most of what was left was language detection, which the Router now remembers for
  recently seen states (below).
- **Trimming the state.** Every question row carries the whole state, so time follows its
  length: three questions took 503 ms on a 97-character ticket and 2,237 ms on a
  1,125-character email. For email, `laya.email.clean_email_body` drops quoted history,
  signatures and disclaimers and caps the rest at 3,000 characters. A reply quoting the
  benchmark email went from 1,263 characters to 27, and a two-deep thread from 2,498 to 27. The
  quoted history is gone too, so keep it when your questions depend on it.
- **Concurrent repeats.** With a `DecisionCache` installed, identical requests arriving
  together share one forward pass
  ([concurrent identical requests](consistency.md#concurrent-identical-requests)).
- **Batching.** 32 tickets in one `predict_batch` took 12.3 s, 384 ms per ticket against 503 ms
  one at a time. On a GPU batching is worth far more: BENCHMARKS.md has 7.2 ms per question
  batched on a T4 against 33 ms for one.
- **int8 dynamic quantization** of the encoder's linear layers is the largest CPU gain, and the
  one that changes answers. It is a trade: speed for arithmetic that differs from the checkpoint
  as trained.
- **ONNX Runtime** gives identical answers, but in fp32 it is faster only on short inputs (see
  [the CPU fast path](#the-cpu-fast-path)); its int8 model is the fastest CPU option measured.
- **Threads.** Use every physical core. BENCHMARKS.md adds that inter-op threads should be pinned
  to 1.
- **bf16** only pays on hardware with native BF16 (`LAYA_CPU_AMP=bf16`), as BENCHMARKS.md warns.

Three stages got cheaper in this change, with identical output:

- **Language detection** skips work that cannot change its answer: it counts ASCII letters in C,
  tests the non-Latin share before walking the text for non-Latin words, and scores each distinct
  word once. A 2,000-character English text went from 723 to 221 µs, a multi-line email from 957
  to 643 µs and a JSON ticket from 1,610 to 946 µs; Hindi and Chinese are unchanged, since they
  need the walk. 267,449 comparisons against the previous code, over 38,247 inputs, differ in
  none.
- **Question heads.** Each question's instructions and options were tokenized again on every
  request, though a fixed question set never changes. `build_sequence` now keeps that part per
  tokenizer and question, which took encoding for three questions on a ticket from 0.71 to
  0.13 ms, for ten from 2.07 to 0.21 ms, and for a batch of 32 tickets from 22.6 to 4.0 ms, with
  identical rows (`research/scripts/bench_question_heads.py`).
- **Routing a repeated state.** Language detection reads the state and nothing else, so the
  Router keeps recent results under a 16-byte hash of the state. Routing a state it has seen
  takes 7, 10 and 39 µs for 140, 2,000 and 20,000 characters, against 56, 246 and 478 µs, and a
  `DecisionCache` hit over an instant model 73, 76 and 167 µs against 121, 314 and 614 µs.

## The CPU fast path

`research/scripts/bench_cpu_fast_path.py` times each backend against PyTorch fp32 on 13 English
requests (1 to 10 questions, tickets to 3,000-token emails), interleaved over 5 repeats, and
compares every answer. `python scripts/export_onnx.py --model ID --output english.onnx --int8`
writes both ONNX models.

| backend | ticket, 3 questions | email, 3 questions | all 13, geometric mean | same decision |
|---|---|---|---|---|
| int8 PyTorch (encoder linear layers) | 1.85x | 1.68x | 1.89x | 44 of 56 |
| ONNX fp32, export before the fix | 1.04x | 0.79x | 1.01x | 56 of 56 |
| ONNX fp32, NaN guards stripped | 1.11x | 0.90x | 1.06x | 56 of 56 |
| **ONNX int8** | **1.96x** | **1.76x** | **2.07x** | 28 of 56 |

- **Why ONNX fp32 loses on long inputs.** The exporter breaks PyTorch's fused attention into
  separate operations that build a full [heads, length, length] tensor in every layer, so its
  cost grows with length squared, and it adds a NaN check after each softmax that cannot fire:
  masked positions hold finfo.min, not -inf. `export_onnx.py` now removes those checks (13% of
  an email forward pass), which took the email from 0.79x to 0.90x of PyTorch; the rest is the
  unfused attention. Fusing it into ONNX Runtime's attention operators is the next step.
- **int8 changes answers.** Every answer that changed had an fp32 margin of at most 0.0006:
  with random weights nearly every answer sits on its boundary, so these are coin flips, and a
  trained checkpoint will change far fewer. Run the harness on your own checkpoint and data
  (`--requests FILE.jsonl`, or `--feishu` for the 64 Chinese cases on `multilingual`) before
  serving int8, or put a `DecisionCache` in front so each answer is fixed the first time.

## Encoding the state once (research)

Every question row carries the whole state, so k questions put the state through the encoder k
times. `research/scripts/spike_encode_once.py` sizes the alternative, built from the model's own
modules: encode the state once and each question separately, then let the two head layers do the
cross-attention. The answers mean nothing until a model is trained that way; this is compute
only. Forward pass, median of 3, speedup over today:

| state tokens | English, 1 question | 3 | 10 | multilingual, 1 | 3 | 10 |
|---|---|---|---|---|---|---|
| 20 | 0.73x | 0.85x | 1.43x | 0.66x | 0.79x | 1.08x |
| 100 | 0.60x | 1.45x | 2.32x | 0.56x | 1.19x | 2.06x |
| 300 | 0.83x | 2.07x | **4.18x** | 0.84x | 1.61x | 3.46x |
| 476 / 800 | 0.86x | 2.19x | **4.29x** | 0.93x | 2.09x | 3.70x |

It pays from about three questions on a hundred-token state, and loses on one question or a
very short state, where two encoder calls cost more than they save. The head layers still read
every question with the full state, which caps the gain at roughly 15x (English) and 9x
(multilingual). It needs retraining, most likely by distilling from today's model; an
asymmetric attention mask inside the encoder would keep question-to-state attention in every
layer at about the same cost, and is described in the script.

## What the cache costs

`research/scripts/bench_decision_cache.py`, on the same machine, with 100,000 decisions held:

| step | cost |
|---|---|
| request key: canonical JSON and a 16-byte BLAKE2b hash, 140-character state | 15.6 µs |
| request key, 2,000-character state | 22.0 µs |
| request key, 20,000-character state | 83.5 µs |
| BLAKE2b against SHA-256, 2,000-character state | 4.9 against 8.2 µs |
| store a decision's JSON, zlib-compressed when smaller | 15-19 µs; 263-453 bytes become 179-182 |
| read it back | 7-9 µs |
| memory store: hit, and bytes per decision on the Python heap | 0.48 µs, 336 bytes |
| SQLite store: hit, and bytes per decision on disk (with expiry and recency indexes) | 7.5 µs, 319 bytes |
| SQLite store: store one decision in its own transaction | 115 µs, or 276 µs with `synchronous=FULL` |
| `Router.predict` over an instant fake model: no cache, memory hit, SQLite hit | 87, 118, 125 µs |

A decision is stored only on a miss, which has just paid for a forward pass. Writing it to
SQLite on the request path took 45 µs; handing it to a background writer thread took 1.4 µs. The
difference is 0.15% of a single-question T4 request and less on a CPU, and a background writer
would give up the guarantee that the first decision stored wins across processes, so the shipped
cache writes synchronously. A hit writes nothing.

## Measure it on your machine

The same scripts run on the published checkpoints, which download on first use:

```bash
python -m pip install -e .
python research/scripts/bench_stages.py --device mps --out stages_m4.json    # Apple silicon
python research/scripts/bench_stages.py --device cuda --out stages_cuda.json  # NVIDIA
python research/scripts/bench_decision_cache.py --out cache.json
```

On a GPU or Apple silicon the timers synchronise the device around the encoder and the head, so
their split is the device's own. `--variants` picks what to compare with the baseline: `threads`,
`bf16`, `int8`, `inference_mode`, `head_cache_off`, `mps_fp16` (fp16 autocast from one row
instead of five on Apple silicon), `onnx` with `--onnx english=laya.onnx`, and `compile`.
