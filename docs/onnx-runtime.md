---
title: Lightweight ONNX runtime
description: Run exported Laya graphs locally without installing PyTorch or Transformers.
---

# Lightweight ONNX runtime

`laya-pro-onnx` is a separate inference distribution with the `laya_onnx` import.
It loads an exported ONNX graph, its checkpoint configuration and a local tokenizer.
PyTorch and Transformers are not runtime dependencies. The full `laya-pro` package
continues to provide PyTorch inference, export tools, routing, training and integrations.

The lightweight distribution is available from this checkout; it has not been published
to a package registry:

```bash
python -m pip install ./runtime/onnx
```

The two distributions use different import names and can be installed together. Installing
the lightweight package alone does not install the full engine's command-line tools,
HTTP server or MCP server. Optional typed-model helpers use the `structured` extra:

```bash
python -m pip install "./runtime/onnx[structured]"
```

## Prepare artifacts

Export with the full package in a build environment, then copy the graph and its external
data files, if any, to the inference machine. The lightweight package does not export or
quantize models.

```bash
python -m pip install ".[onnx]"
python scripts/export_onnx.py --model /models/english --output /models/english.onnx
```

The checkpoint directory must contain `rl_agent_config.json`, `tokenizer/tokenizer.json`,
and the tokenizer's special-token metadata in `tokenizer_config.json` or
`special_tokens_map.json` inside that tokenizer directory. Keep the exported graph paired
with the checkpoint that produced it. The runtime does not need `model.safetensors` to
execute the graph. Use a normal fp32 export
when comparing against the checkpoint; quantization changes model arithmetic and can
change decisions.

## Predict locally

```python
from laya_onnx import ONNXAgent

agent = ONNXAgent("/models/english", onnx_path="/models/english.onnx")
questions = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {"billing": "invoices and refunds", "technical": "errors and outages"},
    },
}
result = agent.predict("Please refund a duplicate charge.", questions)
print(result["answers"]["team"]["choice"])
```

The runtime supports the ONNX agent's choice, score and yes/no questions, batching,
long-input windows, calibration and prediction hooks. `DecisionCache` can retain exact
repeats through the same hook interface:

```python
from laya_onnx import DecisionCache, ONNXAgent

with DecisionCache("decisions.sqlite", ttl=86400, maxsize=10_000) as cache:
    agent = ONNXAgent("/models/english", "/models/english.onnx", hooks=[cache])
    first = agent.predict("Please refund a duplicate charge.", questions)
    repeated = agent.predict("Please refund a duplicate charge.", questions)
```

For thread budgets, use `performance_mode=1` through `5`; `LAYA_THREADS` overrides the
profile's ONNX session thread count. These budgets keep the same graph and precision.
See [Performance modes](performance-modes.md).

## Tokenizers and compatibility

The lightweight runtime uses the native `tokenizers` backend to read local tokenizer
artifacts. The cls, sep, mask and pad tokens must be named by the saved metadata and
present in the tokenizer vocabulary. Saved truncation direction, maximum length and
decode-cleanup settings are honored. A missing tokenizer fails locally; the loader does
not download an encoder tokenizer as a fallback. Tokenization is part of model behavior:
keep its configuration and special tokens with the graph, and verify outputs when moving
between runtimes.

The existing `from laya.onnx_agent import ONNXAgent` interface remains available in
the full distribution. Its default tokenizer continues to use Transformers.
The standalone package does not provide `Router`, PyTorch `Agent`, training, model
export or every full-package integration.

## Loading and memory measurements

A pinned real English checkpoint was measured on Windows CPU with four threads,
fp32, no decision cache and three paired fresh-process runs. The standalone runtime
used the same graph and numerical dependency versions as the full ONNX baseline.

| runtime | startup to first answer s | peak working set MiB | warm predict ms |
|---|---:|---:|---:|
| full ONNX baseline | 5.374 | 1827.5 | 232.97 |
| standalone `laya_onnx` | 1.782 | 1566.5 | 233.43 |

All 12 typed answers and unrounded logits/probabilities matched exactly in every run.
Warm inference was unchanged; the benefit was startup and memory used by imported
libraries. Separately, the PyTorch loading path reduced peak working set from 2735.1
to 1970.8 MiB while keeping private commit, steady memory and startup time unchanged.
The independently resolved lightweight installation had 134.4 MiB of dependency files,
against 879.7 MiB in the existing full `laya-pro[onnx]` environment. These are installed
file bytes, excluding project code, rather than download sizes or model RAM.
[Full measurements and limits](https://github.com/tgundhus/laya-pro/blob/main/BENCHMARKS.md#loading-memory-and-standalone-onnx-31-october-2026).

`research/scripts/bench_loading_memory.py` compares an unchanged baseline checkout with
the current runtime in fresh processes, using the same pinned local checkpoint and graph.
It records process memory after imports, loading, first prediction and warm calls, along
with operating-system peaks, startup-to-first-answer, direct forward latency and
unrounded option logits and probabilities.

```bash
python research/scripts/bench_loading_memory.py --runtime torch \
  --model /models/english --threads 4 \
  --out research/results/loading_memory_20261031_torch.json
python research/scripts/bench_loading_memory.py --runtime onnx \
  --model /models/english --onnx /models/english.onnx \
  --light-source /path/to/installed/site-packages --threads 4 \
  --out research/results/loading_memory_20261031_onnx.json
```

The script creates a frozen baseline under ignored `.venv/loading-baseline` from the
recorded Git revision. Each configuration has three fresh-process runs. Filesystem
caches are left intact: these are process-start measurements, not cold-boot timings.
Windows reports current and peak working set plus private commit. Resident mapped
checkpoint pages count toward working set while they are in RAM; private commit is
a different measure. Model file size is neither of these process-memory numbers.

The full PyTorch loader can reuse checkpoint tensor storage when tensor properties,
storage layout and loading behavior are compatible and the installed PyTorch supports
assignment. Compatible mixed floating-point dtypes use bounded per-tensor copies with
the same destination dtype as the existing loader. Custom loading behavior retains the
strict copy path. The temporary weight dictionary is released before device placement.
This does not change precision. Any page-in work deferred to inference remains
visible in startup-to-first-answer, so constructor time alone is not a startup result.
