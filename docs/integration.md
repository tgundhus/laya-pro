---
title: SDKs and local MCP integration
description: Connect Python, TypeScript, Java or .NET applications to Laya-Pro, or expose local decisions through MCP tools.
---

# SDKs and integration

Choose in-process inference when the application should own the model. Choose HTTP when
several applications should share one Python engine and its decision cache. MCP provides
tools to an editor or agent over stdio and can use either a local Router or your own server.

## Capabilities

| Interface | Inference | Batch | Long documents | Pro persistent cache |
|---|---|---|---|---|
| Python | PyTorch or ONNX, in process | yes | yes | memory, SQLite or your store |
| Python HTTP client | your Python server | yes | use Python directly | server policy |
| TypeScript HTTP client | your Python server | yes | use Python directly | server policy |
| TypeScript ONNX SDK | Node.js or browser | yes | no | no |
| Java SDK | ONNX, in process | yes | yes | no |
| .NET SDK | ONNX, in process | sequential calls | no | no |
| MCP | local Python or your HTTP server | yes | in-process scans | engine policy |

Native SDK feature parity varies. Each SDK's README describes its model-export requirements,
supported methods and calibration limits. A successful build with tiny random models verifies
transport and execution, not pretrained accuracy. The [review report](reports/production-review.md)
records which SDK tests ran in this environment.

## Python

```python
from laya import DecisionCache, Router, triage_questions

with DecisionCache("decisions.sqlite", ttl=86400) as cache:
    with Router(hooks=[cache]) as router:
        answers = router.predict_batch(
            ["Payment failed", "Cannot sign in"], triage_questions(), batch_size=8
        )
```

Reuse the Router and cache between requests. Creating a new Router for each call reloads
models and loses in-process reuse. `max_loaded` bounds resident checkpoints. For local
weights, pass `models={"english": "/path/to/checkpoint"}` and select `model="english"`.
If the workload uses other languages, configure their checkpoint paths too.

Use `Router(performance_mode=2)` for a smaller resource budget, or set
`LAYA_PERFORMANCE_MODE=2` before starting a Python, HTTP or local MCP engine.
With the variable unset, defaults are unchanged; `performance_mode="default"` ignores it for
an instance. Explicit existing settings take precedence. The [five modes](performance-modes.md)
control resources without changing models, precision or cache policy. Remote HTTP clients and
MCP sessions use the server's resource settings.

Schema helpers and framework adapters are covered in [structured decisions](structured.md),
[LangChain / LangGraph](langchain.md), [LlamaIndex](llamaindex.md) and [CrewAI](crewai.md).
The optional `agno` extra adds Agno 3.x routing and guardrails (tested with 3.1.12). `LayaAgnoRouter(agent=router)`
selects a member; the application executes it. Local and self-hosted HTTP paths support `task`
and `lang_guess`, and refuse malformed decisions before selecting an agent.
Set `AGNO_TELEMETRY=false` to disable the framework's telemetry.

### Python over HTTP

Install the standalone client with `python -m pip install ./sdk/python`. Its application environment
needs only the standard library; run the inference server separately.

```python
from laya_client import LayaClient

client = LayaClient(base_url="http://127.0.0.1:8000", api_key="your-server-token")
result = client.predict("Please refund the duplicate charge", questions, model="english")
batch = client.predict_batch(["Payment failed", "Cannot sign in"], questions, batch_size=2)
```

The [client README](https://github.com/tgundhus/laya-pro/tree/main/sdk/python) covers errors,
strict response validation, controls and isolated package verification.

## TypeScript over HTTP

Build the client from this repository; an upstream npm package does not include fork changes.

```bash
cd sdk/typescript
npm ci
npm test
npm pack
# In your application:
npm install /path/to/laya-pro/sdk/typescript/laya-pro-client-0.1.0.tgz
```

```typescript
import { Laya, triageQuestions } from 'laya-pro-client';

const client = new Laya({ baseURL: 'http://127.0.0.1:8000', timeoutMs: 120_000 });
const result = await client.predict('Payment failed', triageQuestions());
const batch = await client.predictBatch(
  ['Payment failed', 'Cannot sign in'], triageQuestions(), { batchSize: 8 }
);
```

The client validates requests and responses, preserves routing and confidence fields,
supports cancellation and deadlines, and reports server errors. It does not retry inference
automatically. [HTTP SDK source and full API](https://github.com/tgundhus/laya-pro/tree/main/sdk/typescript).

## Native SDKs

The TypeScript SDK now includes `predictTournament` and `cachedEmbedFn`. Tournaments compare
finalists after elimination; their probabilities describe that final set. The bounded embedding
cache reuses label vectors between requests and stays in memory. Clear it when the model changes;
it is separate from Python's decision retention policy.

- [TypeScript ONNX SDK](https://github.com/tgundhus/laya-pro/tree/main/laya-ts): export ONNX
  artifacts, build with `npm ci` and `npm run build`, then install the packed local package.
- [Java SDK](https://github.com/tgundhus/laya-pro/tree/main/laya-java): build and test with
  the checked-in Gradle wrapper and a compatible JDK. The source API namespace is retained
  for compatibility.
- [.NET SDK](https://github.com/tgundhus/laya-pro/tree/main/laya-dotnet): build the solution,
  then reference the local project or packed artifact. Follow its exporter and tokenizer guide.

Use artifacts built from this checkout for Laya-Pro. Maven Central, NuGet and npm names in
upstream examples identify upstream releases until this fork publishes its own artifacts.

## Local MCP

Install `laya-pro[mcp]` from the repository. Configure your editor with the absolute executable
path from the environment where it is installed:

```json
{
  "mcpServers": {
    "laya-pro": {
      "command": "/absolute/path/to/venv/bin/laya-mcp-server",
      "env": {
        "LAYA_DEVICE": "cpu",
        "LAYA_PRELOAD": "0",
        "LAYA_CACHE_PATH": "/absolute/path/to/decisions.sqlite",
        "LAYA_CACHE_TTL": "604800",
        "LAYA_CACHE_MAXSIZE": "10000"
      }
    }
  }
}
```

On Windows, use the environment's `Scripts/laya-mcp-server.exe`. Keep protocol output on
stdout and logs on stderr. `laya_status` reports runtime readiness and cache statistics when
enabled. `laya_predict` and `laya_predict_batch` evaluate questions; schema, shortlist and email
tools are also available.
For a separate server, set `LAYA_BASE_URL` and configure caching on that server.
[Tool contracts and remote mode](cli-mcp.md).
