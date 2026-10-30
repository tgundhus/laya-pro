# Performance modes: cached English checkpoint on CPU

One fresh subprocess per configuration; CPU fp32, eager, decision cache disabled. 1 warm-up and 3 timed calls per workload. p95 is nearest-rank and, with this small sample, is the slowest call. Process peak memory includes imports, loading and inference.

| mode | threads | single median / p95 ms | batch4 median / p95 ms | batch requests/s | load s | peak working set MiB | max probability difference | labels vs default |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 24 | 124.8 / 126.8 | 772.6 / 775.6 | 5.19 | 1.86 | 2739 | 0 | 10/10 |
| 1 | 1 | 645.4 / 665.4 | 3244.6 / 3327.9 | 1.22 | 2.15 | 2737 | 6.56e-07 | 10/10 |
| 2 | 2 | 326.3 / 326.8 | 1700.5 / 1807.6 | 2.30 | 1.91 | 2737 | 7.15e-07 | 10/10 |
| 3 | 4 | 204.3 / 212.9 | 1655.1 / 1830.6 | 2.38 | 1.87 | 2737 | 7.15e-07 | 10/10 |
| 4 | 8 | 282.5 / 290.4 | 1508.2 / 1596.9 | 2.65 | 1.98 | 2739 | 8.34e-07 | 10/10 |
| 5 | 16 | 155.2 / 156.0 | 944.5 / 947.4 | 4.25 | 1.81 | 2739 | 1.19e-06 | 10/10 |

Host: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel; 32 logical CPUs; Python 3.12.14; Torch 2.14.1+cpu; transformers 5.19.0.

Checkpoint snapshot: `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`; weight SHA-256: `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.

The four hand-written support cases have mixed lengths and two questions each. They test numerical parity and resource choices, not accuracy on a representative dataset. The benchmark loads the same single checkpoint directly in every mode: it does not measure multi-model residency, startup preloading, HTTP admission, GPU performance, or older CPUs. Cold load is a new Python Agent load, not a flushed filesystem cache. Workstation load was uncontrolled.

Raw samples, probabilities, outputs, tensor shapes, source hashes and reproduction command are in the adjacent JSON file. Larger mode numbers are resource budgets, not speed rankings.
