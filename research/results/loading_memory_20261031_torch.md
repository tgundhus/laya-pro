# Loading and memory: torch

3 paired fresh-process runs, identical local artifacts, fp32 CPU execution, 4 threads, no decision cache. Filesystem caches were left intact. Startup includes Python, imports, model loading and the first three-question answer.

| variant | import s | load s | first predict s | startup to first answer s | warm predict ms | forward ms | peak working set MiB | peak private commit MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.162 | 3.194 | 0.251 | 4.606 | 245.41 | 244.90 | 2735.1 | 4219.4 |
| current | 1.179 | 3.193 | 0.253 | 4.621 | 245.95 | 244.71 | 1970.8 | 4219.5 |

Numbers are medians across process runs; warm timings pool 5 calls per process. Peak counters cover the entire process, including imports and untimed parity calls. Working set includes resident mapped pages; private commit excludes shared file mappings. Neither is the model's file size.

| memory stage | baseline working / private MiB | current working / private MiB |
|---|---:|---:|
| after_imports | 195.6 / 274.8 | 195.8 / 275.1 |
| after_load | 1931.6 / 2609.2 | 1931.7 / 2609.2 |
| after_first_answer | 1952.4 / 2648.9 | 1952.6 / 2649.1 |
| after_warm_predictions | 1957.2 / 2648.9 | 1957.2 / 2649.1 |
| after_parity_and_forward | 1970.8 / 2710.8 | 1970.8 / 2710.7 |

Maximum absolute differences against the first baseline: logits 0, probabilities 0, action probabilities 0. Typed answer payloads agree in every run: True.

The four hand-written states cover choice, score and yes/no decoding; this is numerical parity, not an accuracy evaluation. Loading is not timed separately inside the constructor: the report samples memory after imports and after the complete Agent load. Any work deferred to the first inference remains in startup-to-first-answer. The workstation was otherwise uncontrolled; small timing changes are not evidence of an inference speedup.

Baseline commit: `6867bc3df62e40f656fc7062f34fca0fa7fdd252`. Artifact hashes, raw samples, process order, stage counters, unrounded logits/probabilities and the command are in the adjacent JSON.
