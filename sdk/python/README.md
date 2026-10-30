# Laya-Pro Python HTTP client

A synchronous, standard-library client for a self-hosted Laya-Pro server. The
application environment needs no core `laya`, Torch, Transformers, or NumPy.
Python 3.10–3.13 is supported. Distribution: `laya-pro-client`; import: `laya_client`.

Install from this checkout (the package has not been published):

```bash
python -m pip install ./sdk/python
```

Run the server in a separate environment using the
[Laya-Pro HTTP guide](https://github.com/tgundhus/laya-pro/blob/main/docs/http-api.md).

```python
from laya_client import LayaClient, LayaAPIError
from laya_client.types import Questions

client = LayaClient(base_url="http://127.0.0.1:8000", api_key="your-server-token")
questions: Questions = {
    "team": {"type": "choice", "instructions": "Which team should handle this message?",
             "criteria": {"billing": "Payments and refunds", "technical": "Product support"}},
}
try:
    result = client.predict({"message": "I was charged twice"}, questions)
    print(result["answers"]["team"]["choice"])
    batch = client.predict_batch(["Please refund my order", "The app will not start"], questions,
                                 model="english", batch_size=2, sort_by_length=True)
    print(batch["results"], batch["total_usage"])
except LayaAPIError as error:
    print(error.status_code, error.detail)
```

`predict` calls `POST /v1/systemone`; `predict_batch` calls `POST /v1/systemone/batch`.
Both accept keyword controls `model`, `task`, `lang`, `lang_guess`, `max_len`,
`head_max_len`, `min_confidence`, and `timeout`. Batch calls also accept
`batch_size` and `sort_by_length`. Controls set to `None` are omitted so the server's
settings remain in charge. `min_confidence` accepts a number in `[0, 1]` or a
nonempty map with `default` and core bucket keys such as `choice:3-5` or `score:6-10`.
The option-count buckets are `2`, `3-5`, `6-10`, and `11+` for choice, score, and noul.

States are text, JSON objects, or arrays. Questions support choice, score, and noul
with explicit `option_order` permutations. Choice labels and question IDs are
strings so they round-trip over JSON; choice label lists are sent as dictionaries.
Malformed questions, non-finite numbers, booleans in numeric controls, cycles, and
non-JSON inputs are refused locally. Server model/workflow registries and configured
request limits remain authoritative.

Results are ordinary dictionaries with typed definitions in `laya_client.types`.
Unknown response fields are retained. Required fields, probabilities, usage counts,
optional routing/abstention metadata, and tournament finalists are validated.
Batch results must match the input count and remain in server order; an invalid item
fails the call. Tournament probabilities may be zero for eliminated options, and
the client preserves the final-round confidence without recalculating it.
Strict Jev responses may omit extensions: missing abstention metadata does not mean
a requested gate passed.

`client.health()` reads `GET /health`. A minimal `{"status": "ok"}` response proves
liveness, including on a server whose prediction endpoint requires authentication.

| Exception | Meaning |
| --- | --- |
| `LayaValidationError` | Invalid client settings or request, rejected before sending |
| `LayaAPIError` | Non-2xx response; `status_code` and `detail` preserve diagnostics |
| `LayaConnectionError` | Connection, TLS, DNS, or interrupted-response failure |
| `LayaTimeoutError` | Blocking I/O timed out; also a `LayaConnectionError` |
| `LayaResponseError` | A successful response did not satisfy the wire contract |

All errors inherit `LayaError`. Error details may be strings, objects, arrays, or
null; error bodies are read up to 64 KiB. URLs may include a reverse-proxy prefix,
but must be ASCII HTTP(S) URLs without credentials, whitespace, query, or fragment.
API keys are explicit printable ASCII tokens without whitespace. Redirects and
automatic retries are disabled. TLS verification and standard proxy settings apply.
The client logs neither requests nor credentials.

Timeouts are positive finite seconds, defaulting to 120. A per-call `None` inherits
that value. This is a blocking I/O timeout, not a total deadline or server
cancellation. Calls close their responses; there is no session to close. There is
no async API, local model loading, automatic batch splitting, or single-call fallback.

Build and verify from the repository root using installed `build` and setuptools:

```bash
python -m build --no-isolation sdk/python
python sdk/python/tests/verify_package.py sdk/python/dist
```

The verifier installs the wheel offline in a clean virtual environment, runs the
stdlib tests, then rebuilds the sdist and repeats the installed-package checks.
Archive verification requires a patched Python with `tarfile.data_filter`.
To also test real HTTP inference with a tiny offline checkpoint, supply the Python
from the separate server environment:

```bash
python sdk/python/tests/verify_package.py sdk/python/dist --server-python /path/to/server/python
```

This verifies transport and inference parity, not pretrained model quality.
Package publication is a separate release step.
