# Loading and memory: onnx

3 paired fresh-process runs, identical local artifacts, fp32 CPU execution, 4 threads, no decision cache. Filesystem caches were left intact. Startup includes Python, imports, model loading and the first three-question answer.

| variant | import s | load s | first predict s | startup to first answer s | warm predict ms | forward ms | peak working set MiB | peak private commit MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.147 | 3.991 | 0.240 | 5.374 | 232.97 | 231.86 | 1827.5 | 1960.9 |
| current | 0.172 | 1.360 | 0.242 | 1.782 | 233.43 | 232.86 | 1566.5 | 1705.7 |

Numbers are medians across process runs; warm timings pool 5 calls per process. Peak counters cover the entire process, including imports and untimed parity calls. Working set includes resident mapped pages; private commit excludes shared file mappings. Neither is the model's file size.

| memory stage | baseline working / private MiB | current working / private MiB |
|---|---:|---:|
| after_imports | 195.4 / 273.9 | 35.6 / 121.7 |
| after_load | 1759.0 / 1829.3 | 1497.8 / 1574.0 |
| after_first_answer | 1785.6 / 1861.6 | 1524.6 / 1606.4 |
| after_warm_predictions | 1808.4 / 1894.7 | 1547.4 / 1639.5 |
| after_parity_and_forward | 1827.5 / 1960.9 | 1566.5 / 1705.7 |

Maximum absolute differences against the first baseline: logits 0, probabilities 0, action probabilities 0. Typed answer payloads agree in every run: True.

The four hand-written states cover choice, score and yes/no decoding; this is numerical parity, not an accuracy evaluation. Loading is not timed separately inside the constructor: the report samples memory after imports and after the complete Agent load. Any work deferred to the first inference remains in startup-to-first-answer. The workstation was otherwise uncontrolled; small timing changes are not evidence of an inference speedup.

Baseline commit: `6867bc3df62e40f656fc7062f34fca0fa7fdd252`. Artifact hashes, raw samples, process order, stage counters, unrounded logits/probabilities and the command are in the adjacent JSON.
