---
title: Performance modes
description: Five opt-in resource budgets for local inference, with explicit overrides and reproducible CPU measurements.
---

# Performance modes

Performance modes choose resource budgets for the same models. They do not make a checkpoint
smaller or change its weights, precision, calibration, input window or decision-cache policy.
Leave the mode unset to retain the existing defaults. Higher modes allow more resources;
they are not a speed ranking.

| Mode | Name | CPU threads | States per batch | Resident models | Startup | HTTP padded tokens per forward | HTTP admission |
|---|---|---:|---:|---:|---|---:|---:|
| 1 | minimal | 1 | 1 | 1 | lazy | 8,192 | 1 |
| 2 | economy | 2 | 2 | 1 | lazy | 16,384 | 2 |
| 3 | balanced | 4 | 4 | 2 | lazy | 32,768 | 4 |
| 4 | throughput | 8 | 8 | 2 | preload | 65,536 | 8 |
| 5 | dedicated | 16 | 16 | 3 | preload | 131,072 | 16 |

Profile threads are capped at the logical CPUs available to the process, including affinity
where the platform exposes it. An explicit `LAYA_THREADS` overrides that cap. Model residency
is an LRU bound, not a RAM quota. A single checkpoint still needs its normal memory.
Padded-token and admission budgets apply to HTTP serving. Padded tokens count question rows
times their padded width; a larger batch is split across forwards rather than shortening its
inputs. The budget cannot split one state's question set, so that state's rows may exceed it.

## Python

```python
import laya
from laya import Agent, Router
from laya.onnx_agent import ONNXAgent

router = Router(performance_mode=3)
agent = Agent("/path/to/checkpoint", device="cpu", performance_mode=1)
agent = laya.load("/path/to/checkpoint", performance_mode=2)
onnx = ONNXAgent("/path/to/checkpoint", onnx_path="/path/to/laya.onnx", performance_mode=2)
```

`performance_mode=None` reads `LAYA_PERFORMANCE_MODE`. Integers `1` through `5` and their
string forms select a mode. Use `performance_mode="default"` to ignore that environment
variable for this instance. The environment accepts `default` too.

`laya.parse_performance_mode(value)` validates a value and returns its numeric level, or
`None` for the default. `laya.resolve_performance_mode(value=None)` also reads the environment
when omitted and returns a fresh effective configuration dictionary, or `None` for the default.
Neither helper loads a backend or changes thread settings.

Existing explicit settings take precedence. For example, `Router(performance_mode=1,
max_loaded=2)` allows up to two resident models, and `predict_batch(..., batch_size=8)` uses
eight states per batch. Explicit `LAYA_THREADS`, `LAYA_MAX_LOADED`, `LAYA_PRELOAD`,
`LAYA_MAX_CONCURRENT` and `LAYA_MAX_BATCH_TOKENS` override the corresponding profile settings
in entry points that support them. An explicit `preload()` or `LAYA_MODELS` preload list can
raise the resident-model cap to retain the requested models. Model and device selection
remain independent.

Torch thread counts apply to the whole process. Choose the mode before inference starts;
constructing another Torch agent with a different mode changes that process's thread count.
ONNX Runtime applies the thread budget to each session instead. A direct Agent loads one
checkpoint; residency, preload and HTTP admission settings apply to their respective Router
or server layers.

Batch size and thread count can change floating-point accumulation order. Compare probabilities
and labels on representative inputs when changing modes, particularly near decision boundaries.

## CLI, HTTP and MCP

```bash
laya "Payment failed twice" --preset triage --performance-mode 2
laya "Payment failed twice" --preset triage --performance-mode default
```

Set `LAYA_PERFORMANCE_MODE=2` in the environment of `laya-serve` or `laya-mcp-server` to use
the same profile, or pass `--performance-mode 2` to either entry point. On PowerShell:

```powershell
$env:LAYA_PERFORMANCE_MODE = "2"
laya-serve
```

HTTP `/health` and MCP `laya_status` report `performance_mode` and `performance_config`;
the latter includes effective settings and explicit overrides. With no profile, both are null.
The detailed HTTP health response requires authentication when `LAYA_API_KEY` is configured.

`create_app(performance_mode=2)` constructs and configures its own Router. When injecting an
existing Router, configure it first with `Router(performance_mode=2)`, then call
`create_app(router=router)`. An explicit app mode must match the injected Router's mode;
a mismatch raises `ValueError`. Omitting the app mode keeps the Router's existing profile,
including its thread, residency and preload settings, and ignores later changes to
`LAYA_PERFORMANCE_MODE`. HTTP-only `LAYA_MAX_BATCH_TOKENS` and `LAYA_MAX_CONCURRENT` are
captured when the app is created and take precedence over that profile's HTTP budgets.

When MCP uses `LAYA_BASE_URL`, resources belong to the HTTP server. Set the profile in the
server's environment. MCP status reports that server's mode; setting a local MCP profile does
not change server resources or load a local model. [HTTP configuration](http-api.md) and
[MCP configuration](cli-mcp.md) describe the other controls.

## Measure on your hardware

`research/scripts/bench_performance_modes.py` runs the default and modes 1–5 in separate
subprocesses with the same cached checkpoint, CPU fp32 and no decision cache. It records
cold load, median and p95 latency, throughput, process peak memory, raw probabilities and
label agreement against the default. Use a snapshot directory to pin the checkpoint:

```bash
python research/scripts/bench_performance_modes.py \
  --model /path/to/hub/snapshots/COMMIT \
  --out research/results/performance_modes_cpu.json
```

The short mixed-length support workload measures a single checkpoint. It does not measure
multi-model preloading, server admission, GPU performance or accuracy on your application's
data. The [request-stage measurements](performance.md) cover different workloads and hardware.

### Windows CPU check, 30 October 2026

The cached English checkpoint at revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`
ran on a 32-logical-CPU Intel Family 6 Model 183 host, Python 3.12.14, Torch 2.14.1+cpu
and transformers 5.19.0. Each row used a fresh process, one warm-up and three timed calls
for a single state and a four-state mixed-length batch, with two questions per state.
The order was default, then 1 through 5; background workstation load was uncontrolled.

| Mode | Actual threads | Single median ms | Batch4 median ms | Batch requests/s |
|---|---:|---:|---:|---:|
| default | 24 | 124.8 | 772.6 | 5.19 |
| 1 | 1 | 645.4 | 3,244.6 | 1.22 |
| 2 | 2 | 326.3 | 1,700.5 | 2.30 |
| 3 | 4 | 204.3 | 1,655.1 | 2.38 |
| 4 | 8 | 282.5 | 1,508.2 | 2.65 |
| 5 | 16 | 155.2 | 944.5 | 4.25 |

The existing 24-thread default was fastest in this run. All five profiles agreed with its
10 observed labels, including the repeated single state; the largest unrounded calibrated
probability difference was `1.19e-6`. All profiles matched six of the eight hand-written
batch labels, so agreement does not establish accuracy. Peak process working set was
2,737–2,739 MiB, including checkpoint loading: these profiles did not make the model smaller.
Cold Agent loads took 1.81–2.15 seconds with the OS file cache left intact.

The batch needed at most 760 padded tokens, so it did not exercise the HTTP token limits.
Torch inter-op threads stayed at the existing 24 in every process. These measurements do
not establish behavior on an older CPU or another backend.
[Raw samples, probabilities and peak memory](https://github.com/tgundhus/laya-pro/blob/main/research/results/performance_modes_cpu_20261030.json)
and the [full table with p95](https://github.com/tgundhus/laya-pro/blob/main/research/results/performance_modes_cpu_20261030.md)
include the checkpoint weight hash and measured source hashes. With three samples, p95 is
the slowest observed call.
