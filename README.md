# Laya-Pro

Laya-Pro is a local AI decision engine for classification, scoring and yes/no questions.
Use it for ticket routing, moderation, guardrails and structured decisions in your own applications.
It runs on your hardware through Python or ONNX Runtime, with HTTP and local MCP interfaces.

Built on [Laya](https://github.com/NandhaKishorM/laya), this fork adds configurable decision
retention, replay across devices and precision modes, and optimizations around inference.
The pretrained models remain upstream Laya checkpoints; the fork maintains its own code,
SDKs, documentation and measurements.

Selected upstream changes are reviewed through Laya 0.4.6. The
[12 October review](https://tgundhus.github.io/laya-pro/reports/upstream-review-2026-10-12/)
records the adopted fixes, measured costs and deferred features.

[Documentation](https://tgundhus.github.io/laya-pro/) · [Integration](https://github.com/tgundhus/laya-pro/blob/main/docs/integration.md) ·
[Benchmarks](https://github.com/tgundhus/laya-pro/blob/main/BENCHMARKS.md) · [Deployment](https://github.com/tgundhus/laya-pro/blob/main/docs/production.md)

## Install

```bash
python -m pip install "laya-pro @ git+https://github.com/tgundhus/laya-pro.git"
# HTTP and MCP support
python -m pip install "laya-pro[serve,mcp] @ git+https://github.com/tgundhus/laya-pro.git"
```

Python 3.10–3.13. The distribution is `laya-pro`; imports and commands remain `laya`.
Install it separately from upstream `laya`, since both provide that import.
Source installation is supported until a Laya-Pro registry release is published.
For reproducible deployment, append a reviewed commit SHA to the Git URL.

The first model load downloads a checkpoint. Inference runs locally. For offline use,
download checkpoints beforehand and pass their directories to the Router.
[Installation and platform notes](https://github.com/tgundhus/laya-pro/blob/main/docs/guide.md#installation).

## First decision

```python
from laya import DecisionCache, Router, decision_margins

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments and refunds",
                     "technical": "bugs, outages and access problems",
                     "other": "other requests"},
    },
    "urgent": {"type": "noul", "instructions": "Does this need attention today?"},
}

with DecisionCache("decisions.sqlite", ttl=7 * 86400, maxsize=100_000) as cache:
    with Router(hooks=[cache]) as router:
        state = "We were billed twice. Please refund the duplicate today."
        result = router.predict(state, questions)
        print(result["answers"]["department"]["choice"])
        print(decision_margins(result))
        router.predict(state, questions)  # an exact repeat replays the stored decision
```

| Question | Returns |
|---|---|
| `choice` | Label, probabilities per option and confidence |
| `score` | Expected level on an ordered rubric and its distribution |
| `noul` | Probability of yes |

Questions in a request share a forward pass. `predict_batch` handles multiple states;
`predict_long` handles documents beyond the context window. Routing selects an English,
multilingual or task-specific checkpoint. Explicit `model` and `lang` overrides are available.

## What the fork adds

- **Decision retention:** fixed or per-decision TTL, optional renewal on use, bounded storage,
  memory or SQLite persistence, and a protocol for your own store.
- **Consistent replay:** the first stored decision wins for exact repeats sharing a store.
  Identical requests in one process can share an in-flight computation.
- **Decision margins:** inspect how close an answer is to its decision boundary.
- **Less inference overhead:** reuse question tokens and recent language detections.
- **Deployment controls:** enable the decision cache in HTTP and local MCP, with cache
  statistics and configured retention.

The cache stores request hashes and result payloads, including option labels and probabilities.
It does not store the original request text. Expiry prevents replay; physical deletion requires
pruning and a policy for database files, journals and backups. A cached answer can still be
wrong. [Cache behavior and retention](https://github.com/tgundhus/laya-pro/blob/main/docs/consistency.md).

## Integration

| Application | Interface |
|---|---|
| Python, in process | `Router`, `Agent`, `ONNXAgent`, typed schemas and prediction hooks |
| JavaScript / TypeScript, HTTP | [HTTP client](https://github.com/tgundhus/laya-pro/blob/main/sdk/typescript/README.md) for your own `laya-serve` |
| JavaScript / TypeScript, in process | [ONNX SDK](https://github.com/tgundhus/laya-pro/blob/main/laya-ts/README.md) |
| Java, in process | [Java SDK](https://github.com/tgundhus/laya-pro/blob/main/laya-java/README.md) |
| .NET, in process | [.NET SDK](https://github.com/tgundhus/laya-pro/blob/main/laya-dotnet/README.md) |
| Editor or agent tools | [MCP over stdio](https://github.com/tgundhus/laya-pro/blob/main/docs/cli-mcp.md), local inference or your own HTTP server |
| Agent frameworks | [LangChain / LangGraph](https://github.com/tgundhus/laya-pro/blob/main/docs/langchain.md), [LlamaIndex](https://github.com/tgundhus/laya-pro/blob/main/docs/llamaindex.md), [CrewAI](https://github.com/tgundhus/laya-pro/blob/main/docs/crewai.md) |

The SDKs have different capabilities. Pro's persistent decision cache runs in the Python engine;
HTTP clients and MCP can use that engine's cache. Native SDKs do not implement the same
persistence policy. [Capability table and examples](https://github.com/tgundhus/laya-pro/blob/main/docs/integration.md).

Start a local HTTP server with `laya-serve`, then call `POST /v1/systemone` or
`POST /v1/systemone/batch`. Set these variables in the server's environment for a seven-day cache:

```text
LAYA_CACHE_PATH=decisions.sqlite
LAYA_CACHE_TTL=604800
LAYA_CACHE_MAXSIZE=100000
```

The same cache variables apply to `laya-mcp-server`. Authentication, request bounds,
concurrency and offline setup are covered in the [deployment guide](https://github.com/tgundhus/laya-pro/blob/main/docs/production.md).

## Measurements

The October CPU check used a pinned English checkpoint, fp32 and four threads. Median single
inference was 571.5 ms; a SQLite repeat was 0.148 ms. All 12 answers agreed between serial and
grouped batch execution. Workstation load was uncontrolled and this small check does not
measure accuracy. [Current results and methods](https://github.com/tgundhus/laya-pro/blob/main/docs/reports/production-review.md).

These archived results used published checkpoints on an Apple M4 Max in September 2026,
against Laya 0.3.20/0.3.21. They describe those runs, not a guarantee for newer versions or
another machine. [Method and raw results](https://github.com/tgundhus/laya-pro/blob/main/docs/reports/real-checkpoints-m4-max.md).

| Workload | Upstream run | Laya-Pro run |
|---|---|---|
| Repeated three-question ticket, CPU | 143.9 ms | 0.037 ms, memory cache |
| Repeated three-question ticket, MPS | 34.1 ms | 0.058 ms, memory cache |
| Stream with 70% repeats, MPS | 16.8 requests/s | 61.0 requests/s |
| 16 identical new requests, CPU | 16 forward passes, 1.26 s | 1 forward pass, 0.22 s |
| Shared-model new request, CPU | 152.7 ms | 151.8 ms |
| Cached decisions replayed after SQLite restart | — | 20,000 of 20,000 |

New requests are dominated by inference; cache speed depends on repeat frequency.
Earlier parity checks covered 841 requests in 11 languages. Precision changes can change
decisions, so validate fp16, bf16 or int8 on your own data before adopting them.
The [October review](https://github.com/tgundhus/laya-pro/blob/main/docs/reports/production-review.md) records current checks, fixes,
measurements and verification limits separately from these archived results.

## Limits and version policy

This version incorporates upstream **0.3.29** and selected correctness and integration changes
reviewed through **0.4.18**. The distribution remains 0.3.29 and retains Laya-Pro's English
fallback for ambiguous Latin text. This is selected upstream coverage, not complete behavioral
parity; the [30 October review](https://github.com/tgundhus/laya-pro/blob/main/docs/reports/upstream-review-2026-10-30.md)
records adopted changes, measurements and limits.

Validate accuracy and confidence on representative labelled data. Large choice sets,
underspecified criteria and ordinal scores remain difficult. Replay improves repeatability,
not accuracy; an expired, evicted or unavailable entry is computed again.
See [measured limits](https://github.com/tgundhus/laya-pro/blob/main/docs/limits.md) and the [evaluation harness](https://github.com/tgundhus/laya-pro/blob/main/docs/evals.md).

## Development and license

[Contributing](https://github.com/tgundhus/laya-pro/blob/main/CONTRIBUTING.md) covers test suites, required lint and compile gates, and strict
documentation builds. Changes should include a runnable regression check and measured evidence
when they alter performance or numerical behavior.

Laya was created by Nandakishor M and developed by Convai Innovations. Laya-Pro's additions
are maintained by Tobias Gundhus. The engine, checkpoints and fork are licensed under
[Apache 2.0](https://github.com/tgundhus/laya-pro/blob/main/LICENSE). Laya-Pro is an independent fork, with no endorsement implied.
