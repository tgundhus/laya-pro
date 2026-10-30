"""Decision consistency: `decision_margins` and the `DecisionCache` hook, without a model.

The Router path attaches counting fakes. The Agent path runs the real `predict_batch` with its
tokenizer and forward pass stubbed, as `test_hooks.py` does, so the cache is exercised against
the engine's own hook sequencing. The last section replays the archived Feishu runs to check
that a 0.1 margin flags every answer that changed between Jev's repeated runs.

Run: python tests/test_consistency.py
"""
import copy
import json
import os
import sys
import tempfile
import threading
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import laya  # noqa: E402
import laya.consistency as consistency  # noqa: E402
from laya import DecisionCache, Router, decision_margins  # noqa: E402
from laya.agent import Agent, _checkpoint_signature  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def check_raises(name, exc, fn):
    try:
        fn()
    except exc:
        PASS.append(name)
        return
    except BaseException as other:  # noqa: BLE001
        FAIL.append("%s: raised %r, want %r" % (name, other, exc))
        return
    FAIL.append("%s: did not raise %r" % (name, exc))


class Clock:
    """Stands in for `laya.consistency._now` so expiry is tested without sleeping."""

    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


clock = Clock()
consistency._now = clock

# --------------------------------------------------------------- decision_margins
choice = {"type": "choice", "choice": "todo", "probabilities": {"todo": 0.53, "noise": 0.44, "urgent": 0.03}}
check("margins/choice is the top-two gap", decision_margins({"answers": {"c": choice}}), {"c": 0.09})
check("margins/an answers mapping works too", decision_margins({"c": choice}), {"c": 0.09})
check("margins/one option cannot flip",
      decision_margins({"c": {"type": "choice", "probabilities": {"only": 1.0}}}), {"c": 1.0})
check("margins/score uses the level probabilities",
      decision_margins({"s": {"type": "score", "score": 1.2, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}}),
      {"s": 0.3})
check("margins/noul at the default threshold",
      decision_margins({"a": {"noul": 0.54}, "b": {"noul": 0.5}, "c": {"noul": 1.0}, "d": {"noul": 0.0}}),
      {"a": 0.08, "b": 0.0, "c": 1.0, "d": 1.0})
check("margins/noul equals the P(yes) - P(no) gap at 0.5",
      decision_margins({"n": {"noul": 0.8}})["n"],
      decision_margins({"c": {"probabilities": {"yes": 0.8, "no": 0.2}}})["c"])
check("margins/per-question thresholds, missing names use 0.5",
      decision_margins({"urgent": {"noul": 0.8}, "related": {"noul": 0.6}}, noul_threshold={"urgent": 0.75}),
      {"urgent": 0.1, "related": 0.2})
check("margins/one threshold for every noul", decision_margins({"n": {"noul": 0.8}}, noul_threshold=0.75),
      {"n": 0.1})
check("margins/entries without numbers are left out",
      decision_margins({"a": {"noul": None}, "b": "text", "c": {"probabilities": {}}, "d": {"choice": "x"}}), {})
check_raises("margins/threshold outside [0, 1]", ValueError, lambda: decision_margins({"n": {"noul": 0.5}}, 1.5))
check_raises("margins/per-question threshold outside [0, 1]", ValueError,
             lambda: decision_margins({"n": {"noul": 0.5}}, {"n": -0.1}))
check_raises("margins/not a mapping", TypeError, lambda: decision_margins(["a"]))
jev = {"model": "jev-1.13.0", "answers": {"related": {"type": "noul", "noul": 0.86},
                                          "category": {"type": "choice", "choice": "todo", "confidence": 0.37,
                                                       "probabilities": {"todo": 0.53, "noise": 0.44}}}}
check("margins/reads a Jev response", decision_margins(jev), {"related": 0.72, "category": 0.09})


# --------------------------------------------------------------- Router fakes
class CountingAgent:
    """Answers every noul question with `answer` and records each forward pass."""

    model_id = "fake/checkpoint"
    revision = "r1"

    def __init__(self, answer=0.7):
        self.answer = answer
        self.calls = []
        self.temperature = [1.0, 1.0, 1.0]
        self.cfg = {"max_len": 512, "head_max_len": 192}

    def predict_batch(self, states, questions, batch_size=None, **overrides):
        self.calls.append((list(states), overrides))
        return [{"model": "laya-rl-agent",
                 "answers": {q: {"type": "noul", "noul": self.answer} for q in questions},
                 "usage": {"input_tokens": len(str(s)), "output_tokens": 0}} for s in states]

    def system_one(self, state, questions, **overrides):
        return self.predict_batch([state], questions, **overrides)[0]


def router_with(cache, english=None, multilingual=None):
    r = Router(hooks=[cache])
    en = english or CountingAgent()
    ml = multilingual or CountingAgent(0.2)
    r.attach("english", en)
    r.attach("multilingual", ml)
    return r, en, ml


Q = {"billing": {"type": "noul", "instructions": "Is this about billing?"}}
STATE = "Hi, we were billed twice for March. Please refund the duplicate today."

# --------------------------------------------------------------- hits, misses, replays
cache = DecisionCache()
r, en, ml = router_with(cache)
first = r.predict(STATE, Q)
second = r.predict(STATE, Q)
check("cache/a repeat skips the forward pass", len(en.calls), 1)
check("cache/the replay is the stored answer", second["answers"], first["answers"])
check("cache/the replay keeps usage", second["usage"], first["usage"])
check("cache/the Router adds routing to a replay", second["routing"]["model"], "english")
check("cache/counters", cache.cache_info(),
      {"size": 1, "maxsize": 100_000, "hits": 1, "misses": 1, "conflicts": 0, "coalesced": 0, "errors": 0})

first["answers"]["billing"]["noul"] = -1.0
second["answers"]["billing"]["noul"] = -2.0
check("cache/a caller mutating a result does not change the store",
      r.predict(STATE, Q)["answers"]["billing"]["noul"], 0.7)

r.predict(STATE + " ", Q)
check("cache/a different state is a different request", len(en.calls), 2)
swapped = {"c": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}}}
reordered = {"c": {"type": "choice", "instructions": "?", "criteria": {"b": "y", "a": "x"}}}
r.predict(STATE, swapped)
r.predict(STATE, reordered)
check("cache/option order is part of the key (#166)", len(en.calls), 4)
r.predict(STATE, Q, max_len=128)
check("cache/a per-call token budget is part of the key", len(en.calls), 5)
r.predict(STATE, Q, model="multilingual")
check("cache/the routed checkpoint is part of the key", len(ml.calls), 1)
r.predict(STATE, Q, lang="de")
check("cache/without per-language temperatures lang cannot change the answer", len(ml.calls), 1)
before = len(en.calls)
r.predict(STATE, {})
check("cache/an empty question set is left to the engine", cache.cache_info()["size"], 6)
check("cache/nothing inferred for empty questions here", len(en.calls), before + 1)


class LangAgent(CountingAgent):
    def __init__(self):
        super().__init__(0.2)
        self.lang_temperatures = {"de": {"temperature": [1.0, 1.0, 2.0], "temperature_by_options": {}}}


cache = DecisionCache()
r, _, ml = router_with(cache, multilingual=LangAgent())
r.predict(STATE, Q, lang="de")
r.predict(STATE, Q, lang="fr")
check("cache/with per-language temperatures an explicit lang is part of the key", len(ml.calls), 2)
r.predict(STATE, Q, lang="de")
check("cache/the same explicit lang replays", len(ml.calls), 2)
r.predict("मुझसे मार्च में दो बार शुल्क लिया गया", Q)
r.predict("मुझसे मार्च में दो बार शुल्क लिया गया", Q)
check("cache/a detected language replays too", len(ml.calls), 3)

# --------------------------------------------------------------- fingerprint
cache = DecisionCache()
r, en, _ = router_with(cache)
r.predict(STATE, Q)
en.temperature = [1.0, 1.0, 2.0]
r.predict(STATE, Q)
check("fingerprint/new calibration starts fresh decisions", len(en.calls), 2)
en.revision = "r2"
r.predict(STATE, Q)
check("fingerprint/new revision starts fresh decisions", len(en.calls), 3)
en.subfolder = "multilingual"
r.predict(STATE, Q)
check("fingerprint/another checkpoint subfolder starts fresh decisions", len(en.calls), 4)
with tempfile.TemporaryDirectory() as directory:
    config_path = os.path.join(directory, "rl_agent_config.json")
    weights_path = os.path.join(directory, "model.safetensors")
    with open(config_path, "w") as file:
        file.write("{}")
    with open(weights_path, "wb") as file:
        file.write(b"first weights")
    local = CountingAgent()
    local.model_id, local.revision = directory, None
    local._checkpoint_signature = _checkpoint_signature(directory, weights_path, {})
    local_cache = DecisionCache()
    local_router, _, _ = router_with(local_cache, english=local)
    local_router.predict(STATE, Q)
    with open(weights_path, "wb") as file:
        file.write(b"replacement weights")
    local._checkpoint_signature = _checkpoint_signature(directory, weights_path, {})
    local_router.predict(STATE, Q)
    check("fingerprint/replacing a local checkpoint starts fresh decisions", len(local.calls), 2)
    encoder_dir = os.path.join(directory, "encoder")
    os.mkdir(encoder_dir)
    encoder_path = os.path.join(encoder_dir, "config.json")
    with open(encoder_path, "w") as file:
        file.write('{"hidden_act":"gelu"}')
    local._checkpoint_signature = _checkpoint_signature(directory, weights_path, {})
    local_router.predict(STATE, Q)
    with open(encoder_path, "w") as file:
        file.write('{"hidden_act":"relu"}')
    stat = os.stat(encoder_path)
    os.utime(encoder_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    local._checkpoint_signature = _checkpoint_signature(directory, weights_path, {})
    local_router.predict(STATE, Q)
    check("fingerprint/a local encoder config change starts fresh decisions", len(local.calls), 4)
    pinned_signature = _checkpoint_signature(directory, weights_path, {"model.safetensors": "a" * 64})
    with open(weights_path, "wb") as file:
        file.write(b"metadata changed")
    check("fingerprint/a verified artifact digest outranks volatile file metadata",
          _checkpoint_signature(directory, weights_path, {"model.safetensors": "a" * 64}), pinned_signature)
pinned = DecisionCache(fingerprint="policy-v1")
r, en, _ = router_with(pinned)
r.predict(STATE, Q)
en.temperature = [9.0, 9.0, 9.0]
en.revision = "r9"
r.predict(STATE, Q)
check("fingerprint/a fixed fingerprint replays across upgrades", len(en.calls), 1)

# --------------------------------------------------------------- expiry and size
cache = DecisionCache(ttl=60)
r, en, _ = router_with(cache)
r.predict(STATE, Q)
clock.t += 59
r.predict(STATE, Q)
check("ttl/replayed while younger than ttl", len(en.calls), 1)
clock.t += 1
r.predict(STATE, Q)
check("ttl/recomputed once ttl old", len(en.calls), 2)
check("ttl/the recomputed decision replaces the old one", cache.cache_info()["size"], 1)

cache = DecisionCache(maxsize=2)
r, en, _ = router_with(cache)
for s in ("one", "two", "three"):
    r.predict(s, Q)
check("maxsize/bounded", cache.cache_info()["size"], 2)
r.predict("one", Q)
check("maxsize/the oldest decision left first", len(en.calls), 4)
r.predict("three", Q)
check("maxsize/newer decisions stay", len(en.calls), 4)

cache = DecisionCache(maxsize=2, ttl=lambda questions, result: 1 if result["answers"]["billing"]["noul"] < 0.5 else 100)
r, en, _ = router_with(cache)
r.predict("live", Q)
en.answer = 0.2
r.predict("short-lived", Q)
clock.t += 2
en.answer = 0.7
r.predict("fresh", Q)
r.predict("live", Q)
check("maxsize/expired entries are reclaimed before a live decision is evicted", len(en.calls), 3)
check("maxsize/reclaiming expired entries keeps the bound", cache.cache_info()["size"], 2)

memory = consistency._MemoryStore(2)
memory.add([(b"expired", b"value", clock.t + 1)], clock.t)
memory.renew(b"expired", clock.t + 100, clock.t + 2)
check("renew/memory does not resurrect a decision that already expired", memory.get(b"expired", clock.t + 2), None)

cache = DecisionCache(ttl=10)
r, en, _ = router_with(cache)
r.predict("a", Q)
clock.t += 5
r.predict("b", Q)
clock.t += 6
check("prune/drops what is expired", cache.prune(), 1)
check("prune/keeps the rest", cache.cache_info()["size"], 1)
cache.cache_clear()
check("clear/empties and resets", cache.cache_info(),
      {"size": 0, "maxsize": 100_000, "hits": 0, "misses": 0, "conflicts": 0, "coalesced": 0, "errors": 0})

check_raises("args/ttl must be positive", ValueError, lambda: DecisionCache(ttl=0))
check_raises("args/ttl is not a bool", ValueError, lambda: DecisionCache(ttl=True))
check_raises("args/maxsize must be positive", ValueError, lambda: DecisionCache(maxsize=0))
check_raises("args/maxsize is an int", ValueError, lambda: DecisionCache(maxsize=2.5))
check_raises("args/fingerprint is a string", TypeError, lambda: DecisionCache(fingerprint=1))
check("args/no bound and no expiry are allowed",
      DecisionCache(ttl=None, maxsize=None).cache_info()["maxsize"], None)


# --------------------------------------------------------------- results that are not JSON
class SetAgent(CountingAgent):
    def predict_batch(self, states, questions, batch_size=None, **overrides):
        results = super().predict_batch(states, questions, batch_size, **overrides)
        for result in results:
            result["extra"] = {1, 2}
        return results


cache = DecisionCache()
r, en, _ = router_with(cache, english=SetAgent())
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out = r.predict(STATE, Q)
    r.predict(STATE, Q)
check("json/the caller still gets the result", out["extra"], {1, 2})
check("json/a result that cannot be replayed exactly is not cached", len(en.calls), 2)
check("json/warned once", sum("not plain JSON" in str(w.message) for w in caught), 1)


class LabelAgent(CountingAgent):
    def predict_batch(self, states, questions, batch_size=None, **overrides):
        results = super().predict_batch(states, questions, batch_size, **overrides)
        for result in results:
            result["answers"] = {
                qid: {"type": "choice", "choice": next(iter(question["criteria"])),
                      "probabilities": {label: 1.0 / len(question["criteria"]) for label in question["criteria"]}}
                for qid, question in questions.items()}
            result["extra"] = ("tuple", {False: "false", None: "none", 1.5: "decimal"})
        return results


class RenderableDescription:
    def __str__(self):
        return "first"


numeric_schema = {1: {"type": "choice", "instructions": "Choose a label", "criteria": {7: "first", "7": "second"}}}
string_schema = {"1": numeric_schema[1]}
for label, make in (("memory", lambda path: DecisionCache()), ("sqlite", lambda path: DecisionCache(path))):
    with tempfile.TemporaryDirectory() as directory:
        cache = make(os.path.join(directory, "typed.sqlite"))
        r, en, _ = router_with(cache, english=LabelAgent())
        first = r.predict(STATE, numeric_schema, model="english")
        replay = r.predict(STATE, numeric_schema, model="english")
        check("json/%s: numeric keys and tuples replay without changing their types" % label, replay, first)
        check("json/%s: numeric keys hit the cache" % label, len(en.calls), 1)
        r.predict(STATE, string_schema, model="english")
        check("json/%s: a string question id is a different schema" % label, len(en.calls), 2)
        r.predict(STATE, {1: {**numeric_schema[1], "criteria": {"7": "first", 7: "second"}}}, model="english")
        check("json/%s: numeric and string choice labels keep distinct keys" % label, len(en.calls), 3)
        descriptor = RenderableDescription()
        r.predict(STATE, {1: {**numeric_schema[1], "criteria": {7: descriptor, "7": "second"}}}, model="english")
        check("json/%s: typed request keys retain renderable descriptor tolerance" % label, len(en.calls), 3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = cache._encode({"answers": {1: {"type": "choice", "description": descriptor}}})
        check("json/%s: typed result payloads still refuse a non-JSON object" % label, value, None)
        check("json/%s: refusing a non-JSON typed result warns once" % label,
              sum("not plain JSON" in str(w.message) for w in caught), 1)
        cache.close()

# --------------------------------------------------------------- Router.predict_batch
cache = DecisionCache()
r, en, _ = router_with(cache)
r.predict("a", Q)
out = r.predict_batch([{"state": s, "questions": Q, "model": "english"} for s in ("a", "b", "b", "c")])
check("batch/only misses reach the model", en.calls[-1][0], ["b", "b", "c"])
check("batch/results stay in request order", [o["usage"]["input_tokens"] for o in out], [1, 1, 1, 1])
check("batch/every result is routed", [o["routing"]["model"] for o in out], ["english"] * 4)
check("batch/duplicates in one batch store one decision", cache.cache_info()["size"], 3)
r.predict_batch([{"state": s, "questions": Q, "model": "english"} for s in ("a", "b", "c")])
check("batch/a repeated batch is served from the cache", len(en.calls), 2)
check("batch/predict and predict_batch share entries", r.predict("c", Q)["answers"], out[3]["answers"])
check("batch/still no extra forward pass", len(en.calls), 2)


# --------------------------------------------------------------- first decision wins
class Racing(CountingAgent):
    """While computing its own answer, lets a sibling process store a different one first."""

    def __init__(self, answer, sibling):
        super().__init__(answer)
        self.sibling = sibling

    def predict_batch(self, states, questions, batch_size=None, **overrides):
        self.sibling.predict(states[0], questions)
        return super().predict_batch(states, questions, batch_size, **overrides)


with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "decisions.sqlite")
    cache_a, cache_b = DecisionCache(path), DecisionCache(path)
    r_a, _, _ = router_with(cache_a, english=CountingAgent(0.51))
    r_b, en_b, _ = router_with(cache_b, english=Racing(0.49, r_a))
    out = r_b.predict(STATE, Q)
    check("race/the request that stored first decides for both", out["answers"]["billing"]["noul"], 0.51)
    check("race/the replaced decision is counted", cache_b.cache_info()["conflicts"], 1)
    check("race/the replacement keeps its routing", out["routing"]["model"], "english")
    check("race/one decision stored", cache_b.cache_info()["size"], 1)
    check("race/later requests replay it", r_b.predict(STATE, Q)["answers"]["billing"]["noul"], 0.51)
    check("race/without another pass", len(en_b.calls), 1)
    cache_a.close()
    cache_b.close()

# --------------------------------------------------------------- concurrent identical requests
class Slow(CountingAgent):
    """Takes `delay` seconds per forward pass, so concurrent callers overlap it."""

    def __init__(self, answer=0.7, delay=0.3):
        super().__init__(answer)
        self.delay = delay
        self.mutex = threading.Lock()

    def predict_batch(self, states, questions, batch_size=None, **overrides):
        time.sleep(self.delay)
        with self.mutex:
            return super().predict_batch(states, questions, batch_size, **overrides)


def concurrently(n, call):
    """Run `call()` on n threads released together; return their results in thread order."""
    barrier, out = threading.Barrier(n), [None] * n

    def run(i):
        barrier.wait()
        out[i] = call()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return out


cache = DecisionCache()
r, slow, _ = router_with(cache, english=Slow())
outs = concurrently(8, lambda: r.predict(STATE, Q))
info = cache.cache_info()
check("coalesce/eight concurrent identical requests run one forward pass", len(slow.calls), 1)
check("coalesce/every caller gets the one decision", len({json.dumps(o["answers"]) for o in outs}), 1)
check("coalesce/every caller gets its routing", [o["routing"]["model"] for o in outs], ["english"] * 8)
check("coalesce/the others are hits", (info["hits"], info["misses"]), (7, 1))
check_true("coalesce/the waits are counted", info["coalesced"] >= 1, str(info))
check("coalesce/no claim outlives its request", cache._inflight, {})
check("coalesce/nothing conflicts", info["conflicts"], 0)

cache = DecisionCache(coalesce=False)
r, slow, _ = router_with(cache, english=Slow())
outs = concurrently(4, lambda: r.predict(STATE, Q))
check_true("coalesce/off: each request computes", len(slow.calls) > 1, str(len(slow.calls)))
check("coalesce/off: the first decision still wins", len({json.dumps(o["answers"]) for o in outs}), 1)

# different requests never wait for each other
cache = DecisionCache()
r, slow, _ = router_with(cache, english=Slow(delay=0.5))
t0 = time.perf_counter()
concurrently(4, lambda: r.predict("state %s" % threading.get_ident(), Q))
check("coalesce/distinct requests all compute", len(slow.calls), 4)
# Serialised, four 0.5 s passes take 2 s; together, about 0.5 s. 1.5 s leaves room for a busy runner.
check_true("coalesce/distinct requests are not serialised", time.perf_counter() - t0 < 1.5,
           "%.2fs" % (time.perf_counter() - t0))

# a batch holding the same request twice runs it once and does not wait on itself
cache = DecisionCache()
r, en, _ = router_with(cache)
out = r.predict_batch([{"state": "dup", "questions": Q, "model": "english"}] * 3)
check("coalesce/a batch never waits on its own requests", len(out), 3)
check("coalesce/nothing claimed after the batch", cache._inflight, {})

# a failed forward pass releases its waiters, which then compute for themselves
class Failing(Slow):
    def __init__(self):
        super().__init__(delay=0.2)
        self.failed = False

    def predict_batch(self, states, questions, batch_size=None, **overrides):
        with self.mutex:
            first, self.failed = not self.failed, True
        if first:
            time.sleep(self.delay)
            raise RuntimeError("out of memory")
        return super().predict_batch(states, questions, batch_size, **overrides)


def attempt():
    try:
        return r.predict(STATE, Q)
    except RuntimeError as exc:
        return exc


cache = DecisionCache()
r, failing, _ = router_with(cache, english=Failing())
t0 = time.perf_counter()
outs = concurrently(4, attempt)
check("coalesce/one caller sees the failure", sum(isinstance(o, RuntimeError) for o in outs), 1)
check("coalesce/the rest still get a decision", sum(isinstance(o, dict) for o in outs), 3)
# The wait bound is 300 s; released at once takes well under a second, so 30 s is still decisive.
check_true("coalesce/waiters are released at once, not at the wait bound", time.perf_counter() - t0 < 30,
           "%.2fs" % (time.perf_counter() - t0))
check("coalesce/the failure leaves no claim", cache._inflight, {})

# hooks that could be waited on in return never wait
serial = DecisionCache()
r, slow, _ = router_with(serial, english=Slow(delay=0.2))
r.hooks_concurrent = False
r._hooks_lock = threading.Lock()
outs = concurrently(3, lambda: r.predict(STATE, Q))
check("coalesce/under a hooks lock nothing is claimed", serial._inflight, {})
check("coalesce/under a hooks lock every caller answers", sum(isinstance(o, dict) for o in outs), 3)
timed = DecisionCache()
r, slow, _ = router_with(timed, english=Slow(delay=0.2))
r.hooks_timeout = 10
outs = concurrently(3, lambda: r.predict(STATE, Q))
check("coalesce/under hooks_timeout nothing waits", timed.cache_info()["coalesced"], 0)
check("coalesce/under hooks_timeout every caller answers", sum(isinstance(o, dict) for o in outs), 3)

# a stuck owner holds its waiters only until the wait bound, then its claim is dropped
cache = DecisionCache()
stuck = threading.Event()
cache._inflight[b"k" * 16] = (-1, stuck)
saved, consistency._COALESCE_WAIT = consistency._COALESCE_WAIT, 0.05
try:
    filled, owned = cache._claim([b"k" * 16], [None])
finally:
    consistency._COALESCE_WAIT = saved
check("coalesce/a stuck owner's claim is taken over", [key for key, _ in owned], [b"k" * 16])
check("coalesce/and nothing was filled", filled, 0)
cache._release(owned)
check("coalesce/the take-over releases too", cache._inflight, {})

# --------------------------------------------------------------- one cache for fp32, int8 and ONNX
class Int8(CountingAgent):
    """The same checkpoint quantized: another class, another dtype, the same fingerprint inputs."""
    dtype = "qint8"
    amp_enabled = False


class OnnxLike(CountingAgent):
    backend = "onnxruntime"
    precision = "int8"


cache = DecisionCache()
keys = [cache._keys(laya.PredictContext(states=[STATE], questions=Q, model="english", agent=agent_cls()),
                    [STATE], False)[0] for agent_cls in (CountingAgent, Int8, OnnxLike)]
check("backends/fp32, int8 and ONNX of one checkpoint share a key", len(set(keys)), 1)
r_fp32, fp32, _ = router_with(cache, english=CountingAgent(0.7))
r_int8, int8, _ = router_with(cache, english=Int8(0.69))
r_onnx, onnx, _ = router_with(cache, english=OnnxLike(0.71))
answers = [rt.predict(STATE, Q)["answers"]["billing"]["noul"] for rt in (r_fp32, r_int8, r_onnx)]
check("backends/int8 and ONNX replay the fp32 decision", answers, [0.7, 0.7, 0.7])
check("backends/only the first backend ran", (len(fp32.calls), len(int8.calls), len(onnx.calls)), (1, 0, 0))

# --------------------------------------------------------------- SQLite store
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "decisions.sqlite")
    cache = DecisionCache(path, ttl=100)
    r, en, _ = router_with(cache)
    stored = r.predict(STATE, Q)
    cache.close()
    reopened = DecisionCache(path, ttl=100)
    r, en, _ = router_with(reopened)
    again = r.predict(STATE, Q)
    check("sqlite/decisions survive a restart", len(en.calls), 0)
    check("sqlite/the replay is the stored answer", again["answers"], stored["answers"])
    clock.t += 100
    r.predict(STATE, Q)
    check("sqlite/expired decisions are recomputed", len(en.calls), 1)
    clock.t += 50
    r.predict("fresh", Q)
    clock.t += 60
    check("sqlite/prune drops what is expired", reopened.prune(), 1)
    check("sqlite/prune keeps the rest", reopened.cache_info()["size"], 1)
    reopened.cache_clear()
    check("sqlite/clear", reopened.cache_info()["size"], 0)
    reopened.close()

    bounded = DecisionCache(os.path.join(tmp, "bounded.sqlite"), maxsize=3)
    r, en, _ = router_with(bounded)
    for i in range(10):
        clock.t += 1
        r.predict("state %d" % i, Q)
    check("sqlite/maxsize bound", bounded.cache_info()["size"], 3)
    r.predict("state 9", Q)
    r.predict("state 0", Q)
    check("sqlite/the oldest decisions left first", len(en.calls), 11)
    bounded.close()


# --------------------------------------------------------------- Agent path (real predict_batch)
def make_fake():
    """A real `predict_batch` whose tokenizer and forward pass are stubbed; answers name the state."""
    fake = Agent.__new__(Agent)
    fake.tok = type("Tok", (), {"pad_token_id": 0})()
    fake._to_internal = staticmethod(Agent._to_internal).__func__
    fake.encoded = []

    def _encode_state(state, ids, internal, **overrides):
        fake.encoded.append(state)
        # The per-row truncation report `predict_batch` reads (laya 0.3.29); nothing is cut here.
        stats = {"state_tokens": 3, "state_tokens_used": 3, "state_tokens_dropped": 0, "truncated": False}
        return [{"ids": [1, 2, 3], "markers": [0, 1], "qtype": 2, "state": state, "state_stats": stats}
                for _ in ids]

    def _forward(b):
        n = b["input_ids"].shape[0]
        return np.zeros((n, 2), dtype=np.float32), np.full((n, 2), 0.5, dtype=np.float32)

    def _decode_answers(logits, act, items, ids, internal, offset, **kw):
        return {qid: {"type": "noul", "noul": 0.5, "of": items[j]["state"]} for j, qid in enumerate(ids)}

    fake._encode_state = _encode_state
    fake._forward = _forward
    fake._decode_answers = _decode_answers
    return fake


seen_states = []
cache = DecisionCache()
agent = make_fake()
agent.hooks = [cache]
first = agent.predict_batch(["s0", "s1"], Q)
out = agent.predict_batch(["s0", "s2", "s1"], Q, on_predict_end=lambda c: seen_states.append(list(c.states)))
check("agent/only the missing state is tokenized", agent.encoded, ["s0", "s1", "s2"])
check("agent/results line up with the states", [o["answers"]["billing"]["of"] for o in out], ["s0", "s2", "s1"])
check("agent/replays equal the first answers", [out[0], out[2]], first)
check("agent/later end hooks see every state", seen_states, [["s0", "s2", "s1"]])
check("agent/usage covers every state", [o["usage"]["input_tokens"] for o in out], [3, 3, 3])
agent.encoded.clear()
again = agent.predict_batch(("s1", "s2"), Q)
check("agent/an all-hit batch skips inference", agent.encoded, [])
check("agent/an all-hit batch returns every replay", [o["answers"]["billing"]["of"] for o in again], ["s1", "s2"])
check("agent/system_one hits too", agent.system_one("s0", Q)["answers"], first[0]["answers"])
check("agent/still nothing tokenized", agent.encoded, [])

per_lang = make_fake()
per_lang.hooks = [DecisionCache()]
per_lang.lang_temperatures = {"de": {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}}
per_lang.predict_batch(["s0"], Q, lang="de")
per_lang.predict_batch(["s0"], Q, lang="fr")
check("agent/with per-language temperatures the cache stands aside", per_lang.encoded, ["s0", "s0"])

# --------------------------------------------------------------- abstention, predict_long and the cache
class CharTok:
    """A character tokenizer with `decode`, so `predict_long` can split a state into windows."""
    mask_token, mask_token_id, cls_token_id, sep_token_id, pad_token_id = "[MASK]", 1, 2, 3, 0

    def __call__(self, text, **kw):
        ids = [ord(c) + 4 for c in text]
        return {"input_ids": ids[:kw["max_length"]] if kw.get("truncation") else ids}

    def decode(self, ids, **kw):
        return "".join(chr(i - 4) for i in ids)


def make_scanner(hooks):
    """A real `predict_batch` and `predict_long` over the character tokenizer, forward pass stubbed."""
    a = Agent.__new__(Agent)
    a.tok, a.cfg, a.model_id = CharTok(), {"max_len": 512, "head_max_len": 192}, "stub/scanner"
    a.hooks = list(hooks)
    a.rows = []

    def forward(b):
        a.rows.append(b["input_ids"].shape[0])
        shape = b["marker_pos"].shape
        return np.zeros(shape, dtype=np.float32), np.full((shape[0], 2), 0.5, dtype=np.float32)

    def decode(logits, act, items, ids, internal, offset, **kw):
        return {q: {"type": "choice", "choice": "billing", "answer_confidence": 0.5, "confidence": 0.5,
                    "probabilities": {"billing": 0.6, "tech": 0.4}} for q in ids}

    a._forward, a._decode_answers = forward, decode
    return a


TEAM = {"team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "charges", "tech": "errors"}}}


def marked(result):
    return result["answers"]["team"].get("low_confidence")


# min_confidence marks answers for one call; the cache keeps them unmarked and they are marked
# again for each call that asks, replays included.
a = make_scanner([DecisionCache()])
a.predict_batch(["a"], TEAM)
got = a.predict_batch(["a", "b"], TEAM, min_confidence=0.9)
check("abstention/a replay in a partial hit is marked for this call", marked(got[0]), True)
check("abstention/the computed answer beside it too", marked(got[1]), True)
check("abstention/a replay for a call with no threshold is not marked", marked(a.predict_batch(["b"], TEAM)[0]), None)
check("abstention/nor for a call whose threshold it clears",
      marked(a.predict_batch(["b"], TEAM, min_confidence=0.2)[0]), None)
router = Router(hooks=[DecisionCache()])
router.attach("english", make_scanner([]))
router.predict(STATE, TEAM, model="english")
check("abstention/Router: a full hit is marked for this call",
      marked(router.predict(STATE, TEAM, model="english", min_confidence=0.9)), True)
batch = router.predict_batch([{"state": STATE, "questions": TEAM, "model": "english"},
                              {"state": STATE + " again", "questions": TEAM, "model": "english"}],
                             min_confidence=0.9)
check("abstention/Router: a batch marks replays and computed answers", [marked(b) for b in batch], [True, True])
check("abstention/Router: the stored decision carries no mark",
      marked(router.predict(STATE + " again", TEAM, model="english")), None)

# Agent.predict_long answers from windows; a cache that holds them, all or some, is still that scan.
DOC = " ".join("w%d" % i for i in range(120))
a = make_scanner([DecisionCache()])
first = a.predict_long(DOC, TEAM, window=80, stride=50)
rows = sum(a.rows)
second = a.predict_long(DOC, TEAM, window=80, stride=50)
check("long/every window kept: the document is answered again", second["answers"]["team"]["choice"],
      first["answers"]["team"]["choice"])
check("long/every window kept: the model does not run", sum(a.rows), rows)
check("long/every window kept: the deciding window is still named", second["answers"]["team"].get("window"),
      first["answers"]["team"].get("window"))
check("long/every window kept: no window was read", (first["usage"]["windows"] > 1, second["usage"]["windows"]),
      (True, 0))
third = a.predict_long(DOC + " and a tail the first document did not have", TEAM, window=80, stride=50)
check("long/some windows kept: the deciding window is still named", "window" in third["answers"]["team"], True)
check("long/some windows kept: only the windows read are counted",
      0 < third["usage"]["windows"] < third["answers"]["team"]["window"]["count"], True)


# ONNXAgent follows the same rules on its session path: replays are marked per call, and a cached
# scan is still that scan -- before, every window cached raised, and some cached lost the window.
def make_onnx_scanner(hooks):
    """A real ONNXAgent `predict_batch`/`predict_long` over the character tokenizer, session stubbed."""
    o = ONNXAgent.__new__(ONNXAgent)
    o.tok, o.cfg, o.model_id = CharTok(), {"max_len": 512, "head_max_len": 192}, "stub/onnx-scanner"
    o.temperature, o.temperature_by_options, o.lang_temperatures = [1.0, 1.0, 1.0], {}, {}
    o.hooks, o.hooks_raise, o.hooks_concurrent, o.hooks_timeout = tuple(hooks), True, True, None
    o._hooks_lock, o._hooks_mutex = None, threading.Lock()
    o.rows = []

    class Session:
        def run(self, names, inputs):
            shape = inputs["marker_pos"].shape
            o.rows.append(shape[0])
            return [np.zeros(shape, dtype=np.float32), np.full((shape[0], 2), 0.5, dtype=np.float32)]

    def decode(logits, act, items, ids, internal, offset, **kw):
        return {q: {"type": "choice", "choice": "billing", "answer_confidence": 0.5, "confidence": 0.5,
                    "probabilities": {"billing": 0.6, "tech": 0.4}} for q in ids}

    o.session, o._decode_answers = Session(), decode
    return o


o = make_onnx_scanner([DecisionCache()])
o.predict_batch(["a"], TEAM)
got = o.predict_batch(["a", "b"], TEAM, min_confidence=0.9)
check("onnx/abstention: a replay in a partial hit is marked for this call", marked(got[0]), True)
check("onnx/abstention: the computed answer beside it too", marked(got[1]), True)
check("onnx/abstention: system_one marks a full hit", marked(o.system_one("a", TEAM, min_confidence=0.9)), True)
check("onnx/abstention: a replay for a call with no threshold is not marked",
      marked(o.predict_batch(["b"], TEAM)[0]), None)
o = make_onnx_scanner([DecisionCache()])
first = o.predict_long(DOC, TEAM, window=80, stride=50)
rows = sum(o.rows)
with warnings.catch_warnings():
    warnings.simplefilter("error")  # a cached scan is not a hook answering the document
    second = o.predict_long(DOC, TEAM, window=80, stride=50)
check("onnx/long: every window kept: the document is answered again", second["answers"], first["answers"])
check("onnx/long: every window kept: the session does not run", sum(o.rows), rows)
check("onnx/long: every window kept: no window was read", (first["usage"]["windows"] > 1, second["usage"]["windows"]),
      (True, 0))
third = o.predict_long(DOC + " and a tail the first document did not have", TEAM, window=80, stride=50)
check("onnx/long: some windows kept: the deciding window is still named", "window" in third["answers"]["team"], True)
check("onnx/long: some windows kept: only the windows read are counted",
      0 < third["usage"]["windows"] < third["answers"]["team"]["window"]["count"], True)


def cache_attribution_scanner(factory, winner):
    scanner = factory([DecisionCache()])
    encode = scanner._encode_state

    def encode_with_state(state, *args, **kwargs):
        items = encode(state, *args, **kwargs)
        for item in items:
            item["scan_test_state"] = state
        return items

    def decode(logits, act, items, ids, internal, offset, **kwargs):
        state = items[0]["scan_test_state"]
        wins = len(state) == 64 if winner == "duplicate" else ord(state[0]) - 0x100 == winner * 32
        confidence = 0.95 if wins else 0.6
        return {qid: {"type": "choice", "choice": "billing", "answer_confidence": confidence,
                      "confidence": confidence, "probabilities": {"billing": confidence, "tech": 1 - confidence}}
                for qid in ids}

    scanner._encode_state, scanner._decode_answers = encode_with_state, decode
    return scanner


for backend, factory in (("torch", make_scanner), ("onnx", make_onnx_scanner)):
    unique_doc = "".join(chr(0x100 + i) for i in range(250))
    for winner in (4, 6):  # the winner is retained, then newly computed from a nonzero miss-local row
        scanner = cache_attribution_scanner(factory, winner)
        scanner.predict_batch([unique_doc[32:96], unique_doc[128:192]], TEAM)
        for phase in ("partial", "full"):
            result = scanner.predict_long(unique_doc, TEAM, window=64, stride=32,
                                          on_predict_end=lambda ctx: ctx.results.reverse())
            check("%s/cache attribution: %s winner %d keeps its original span after reordering" % (backend, phase, winner),
                  result["answers"]["team"].get("window"),
                  {"index": winner, "token_start": winner * 32, "token_end": min(winner * 32 + 64, 250), "count": 7})
            check("%s/cache attribution: %s counts inference windows only" % (backend, phase),
                  result["usage"]["windows"], 5 if phase == "partial" else 0)

        def clone(ctx):
            ctx.results = copy.deepcopy(ctx.results)
        result = scanner.predict_long(unique_doc, TEAM, window=64, stride=32, on_predict_end=clone)
        check("%s/cache attribution: a later hook's cloned replay names no span" % backend,
              "window" in result["answers"]["team"], False)

        def edit(ctx):
            answer = ctx.results[winner]["answers"]["team"]
            answer["probabilities"]["billing"] = 0.94
        result = scanner.predict_long(unique_doc, TEAM, window=64, stride=32, on_predict_end=edit)
        check("%s/cache attribution: a later hook's nested replay edit names no span" % backend,
              "window" in result["answers"]["team"], False)

    scanner = cache_attribution_scanner(factory, "duplicate")
    for phase in ("coalesced", "full"):
        result = scanner.predict_long("a" * 250, TEAM, window=64, stride=32,
                                      on_predict_end=lambda ctx: ctx.results.reverse())
        check("%s/cache attribution: %s duplicate windows keep their own restored origins" % (backend, phase),
              result["answers"]["team"].get("window"),
              {"index": 5, "token_start": 160, "token_end": 224, "count": 7})
        check("%s/cache attribution: %s duplicate inference count" % (backend, phase),
              result["usage"]["windows"], 2 if phase == "coalesced" else 0)

# Router.predict_long scans every window; Router.predict reads one. They must not share an entry.
cache = DecisionCache()
router = Router(hooks=[cache])
router.attach("english", make_scanner([]))
plain = router.predict(DOC, TEAM, model="english")
scan = router.predict_long(DOC, TEAM, model="english", window=80, stride=50)
check("long/Router: predict_long is not a replay of predict", ("windows" in plain["usage"], "windows" in scan["usage"]),
      (False, True))
check("long/Router: nor is predict a replay of predict_long",
      "windows" in router.predict(DOC, TEAM, model="english")["usage"], False)
check("long/Router: each replays its own", router.predict_long(DOC, TEAM, model="english", window=80, stride=50),
      scan)
check("long/Router: two hits, one for each", cache.cache_info()["hits"], 2)
router.predict_long(DOC, TEAM, model="english", window=60, stride=50)
check("long/Router: another window size is another request", cache.cache_info()["hits"], 2)

# --------------------------------------------------------------- retention policy
DAY = 24 * 3600
FRAUD = {"fraud": {"type": "noul", "instructions": "Is this fraud?"}}


def by_question_set(questions, result):
    return 7 * DAY if "fraud" in questions else 30 * DAY


cache = DecisionCache(ttl=by_question_set)
r, en, _ = router_with(cache)
r.predict(STATE, Q)
r.predict(STATE, FRAUD)
clock.t += 8 * DAY
r.predict(STATE, Q)
r.predict(STATE, FRAUD)
check("policy/ttl per question set: the 30-day set replays, the 7-day one expired",
      [c[0] for c in en.calls], [[STATE], [STATE], [STATE]])


def keep_close_calls_longer(questions, result):
    margins = decision_margins(result)
    return 90 * DAY if min(margins.values(), default=1.0) < 0.1 else DAY


cache = DecisionCache(ttl=keep_close_calls_longer)
close = CountingAgent(0.52)
r, _, _ = router_with(cache, english=close)
r.predict(STATE, Q)
clock.t += 2 * DAY
r.predict(STATE, Q)
check("policy/ttl by margin: a close call is kept longer", len(close.calls), 1)
clear = CountingAgent(0.95)
r, _, _ = router_with(DecisionCache(ttl=keep_close_calls_longer), english=clear)
r.predict(STATE, Q)
clock.t += 2 * DAY
r.predict(STATE, Q)
check("policy/ttl by margin: a clear answer expires after a day", len(clear.calls), 2)

seen = []
cache = DecisionCache(ttl=lambda questions, result: seen.append(sorted(result)) or 0)
r, en, _ = router_with(cache)
r.predict(STATE, Q)
r.predict(STATE, Q)
check("policy/a lifetime of 0 keeps nothing", (len(en.calls), cache.cache_info()["size"]), (2, 0))
check("policy/the function sees the payload without routing", seen[0], ["answers", "model", "usage"])

forever = DecisionCache(ttl=lambda questions, result: None)
r, en, _ = router_with(forever)
r.predict(STATE, Q)
clock.t += 10 ** 6
r.predict(STATE, Q)
check("policy/None keeps a decision for ever", len(en.calls), 1)

broken = DecisionCache(ttl=lambda questions, result: "a week")
r, _, _ = router_with(broken)
check_raises("policy/a lifetime that is not a number raises", TypeError, lambda: r.predict(STATE, Q))
check_raises("args/ttl is a positive number or a function", ValueError, lambda: DecisionCache(ttl="1d"))

# --------------------------------------------------------------- retention from last use
for label, make in (("memory", lambda tmp: DecisionCache(ttl=100, renew_on_hit=True)),
                    ("sqlite", lambda tmp: DecisionCache(os.path.join(tmp, "renew.sqlite"), ttl=100,
                                                         renew_on_hit=True))):
    with tempfile.TemporaryDirectory() as tmp:
        cache = make(tmp)
        r, en, _ = router_with(cache)
        r.predict(STATE, Q)
        for _ in range(5):
            clock.t += 90
            r.predict(STATE, Q)
        check("renew/%s: a decision in use outlives its ttl" % label, len(en.calls), 1)
        clock.t += 101
        r.predict(STATE, Q)
        check("renew/%s: it expires ttl after its last use" % label, len(en.calls), 2)
        cache.close()


class SpyStore:
    """A dict-backed DecisionStore that records calls: the protocol is all a store needs."""

    def __init__(self):
        self.data, self.calls = {}, []

    def get(self, key, now):
        self.calls.append("get")
        entry = self.data.get(key)
        return entry if entry is not None and entry[1] > now else None

    def add(self, items, now):
        self.calls.append("add")
        held = []
        for key, value, expires_at in items:
            if self.get(key, now) is None:
                self.data[key] = (value, expires_at)
            held.append(self.data[key][0])
        return held

    def renew(self, key, expires_at, now):
        self.calls.append("renew")
        if key in self.data:
            self.data[key] = (self.data[key][0], expires_at)

    def prune(self, now):
        expired = [k for k, (_, e) in self.data.items() if e <= now]
        for k in expired:
            del self.data[k]
        return len(expired)

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data.clear()

    def close(self):
        self.calls.append("close")


spy = SpyStore()
cache = DecisionCache(store=spy, ttl=160, renew_on_hit=True)
r, en, _ = router_with(cache)
r.predict(STATE, Q)
for _ in range(40):
    clock.t += 1
    r.predict(STATE, Q)
check("store/a custom store serves hits", len(en.calls), 1)
# ttl 160: renewed once more than 10 s (a 16th) of it has passed -- at 11, 22 and 33 s of 40 hits
check("store/renewals are written once a decision has aged a 16th of its ttl", spy.calls.count("renew"), 3)
check("store/cache_info reports no maxsize for a custom store", cache.cache_info()["maxsize"], None)
check("store/size comes from the store", cache.cache_info()["size"], 1)
clock.t += 200
check("store/prune goes to the store", cache.prune(), 1)
cache.close()
check("store/close goes to the store", spy.calls[-1], "close")

first = SpyStore()
cache_a, cache_b = DecisionCache(store=first), DecisionCache(store=first)
r_a, _, _ = router_with(cache_a, english=CountingAgent(0.51))
r_b, _, _ = router_with(cache_b, english=Racing(0.49, r_a))
check("store/first decision wins through a shared custom store",
      r_b.predict(STATE, Q)["answers"]["billing"]["noul"], 0.51)
check_raises("store/path and store together", ValueError, lambda: DecisionCache("x.sqlite", store=SpyStore()))
check_raises("store/maxsize with a custom store", ValueError, lambda: DecisionCache(store=SpyStore(), maxsize=5))
check_raises("store/an object without the protocol", TypeError, lambda: DecisionCache(store=object()))
check("store/maxsize=None with a custom store says the store bounds itself",
      DecisionCache(store=SpyStore(), maxsize=None).maxsize, None)


# --------------------------------------------------------------- a failing store never fails a prediction
class FailingStore(SpyStore):
    """Raises on the operations named, as a locked SQLite file or an unreachable Redis would."""

    def __init__(self, *fail):
        super().__init__()
        self.fail = set(fail)

    def get(self, key, now):
        if "get" in self.fail:
            raise OSError("store unreachable")
        return super().get(key, now)

    def add(self, items, now):
        if "add" in self.fail:
            raise OSError("database is locked")
        return super().add(items, now)

    def renew(self, key, expires_at, now):
        if "renew" in self.fail:
            raise OSError("database is locked")
        return super().renew(key, expires_at, now)


def quietly(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = call()
    return out, [str(w.message) for w in caught if "DecisionCache: could not" in str(w.message)]


for op in ("get", "add"):
    cache = DecisionCache(store=FailingStore(op))
    r, en, _ = router_with(cache)
    outs, warned = quietly(lambda: [r.predict(STATE, Q) for _ in range(3)])
    check("fail-open/%s raises: every request is still answered" % op,
          [o["answers"]["billing"]["noul"] for o in outs], [0.7] * 3)
    check("fail-open/%s raises: each request runs the model" % op, len(en.calls), 3)
    check("fail-open/%s raises: each failure is counted" % op, cache.cache_info()["errors"], 3)
    check("fail-open/%s raises: it warns once" % op, len(warned), 1)
    check("fail-open/%s raises: no claim outlives its request" % op, cache._inflight, {})

cache = DecisionCache(store=FailingStore("renew"), ttl=160, renew_on_hit=True)
r, en, _ = router_with(cache)
r.predict(STATE, Q)
clock.t += 20  # past a 16th of the ttl, so the hit tries to renew
out, _ = quietly(lambda: r.predict(STATE, Q))
check("fail-open/renew raises: the hit is still replayed", (len(en.calls), out["answers"]["billing"]["noul"]),
      (1, 0.7))
check("fail-open/renew raises: counted", cache.cache_info()["errors"], 1)

spy = SpyStore()
cache = DecisionCache(store=spy)
r, en, _ = router_with(cache)
r.predict(STATE, Q)
for key, (_, expires_at) in list(spy.data.items()):
    spy.data[key] = (b"z not zlib, not JSON", expires_at)
out, warned = quietly(lambda: r.predict(STATE, Q))
check("fail-open/an undecodable decision is computed again", (len(en.calls), out["answers"]["billing"]["noul"]),
      (2, 0.7))
check("fail-open/an undecodable decision is counted and warned about", (cache.cache_info()["errors"], len(warned)),
      (1, 1))
check("fail-open/the new decision replaces the undecodable one", r.predict(STATE, Q)["answers"]["billing"]["noul"],
      0.7)
check("fail-open/and is replayed from then on", len(en.calls), 2)

for malformed in (b"j[]", b"jnull", b"j42", b'j{"answers":[]}', b'j{"answers":{},"usage":[]}',
                  b'j{"answers":{},"usage":{"input_tokens":[1]}}',
                  b'j{"answers":{},"usage":{"output_tokens":"bad"}}',
                  b'j{"answers":{},"usage":{"input_tokens":NaN}}',
                  b'j{"answers":{"billing":[]}}', b'j{"answers":{"billing":null}}',
                  b'j{"answers":{"billing":{"probabilities":[]}}}',
                  b'j{"answers":{"billing":{"legend":[]}}}',
                  b'q{"answers":{},"usage":{}}'):
    spy = SpyStore()
    cache = DecisionCache(store=spy)
    r, en, _ = router_with(cache)
    r.predict(STATE, Q)
    for key, (_, expires_at) in list(spy.data.items()):
        spy.data[key] = (malformed, expires_at)
    out, warned = quietly(lambda: r.predict(STATE, Q, min_confidence=0.9))
    check("fail-open/foreign JSON %r is recomputed" % malformed,
          (len(en.calls), out["answers"]["billing"]["noul"]), (2, 0.7))
    check("fail-open/foreign JSON %r is counted once" % malformed, cache.cache_info()["errors"], 1)
    r.predict(STATE, Q)
    check("fail-open/foreign JSON %r is replaced" % malformed, len(en.calls), 2)

# --------------------------------------------------------------- SQLite layout
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "forever.sqlite")
    cache = DecisionCache(path)
    r, en, _ = router_with(cache)
    r.predict(STATE, Q)
    cache.close()
    clock.t += 10 ** 7
    cache = DecisionCache(path)
    r, en, _ = router_with(cache)
    r.predict(STATE, Q)
    check("sqlite/a decision without expiry survives a reopen", len(en.calls), 0)
    cache.close()

    import sqlite3
    old = os.path.join(tmp, "old.sqlite")
    db = sqlite3.connect(old)
    db.execute("CREATE TABLE laya_decisions (key BLOB PRIMARY KEY, stored_at REAL NOT NULL, value BLOB NOT NULL)")
    db.close()
    check_raises("sqlite/a file in an older layout is refused, not misread", ValueError, lambda: DecisionCache(old))

# --------------------------------------------------------------- memory store sweeps mixed lifetimes
store = consistency._MemoryStore(maxsize=None)
now = 1000.0
store.add([(b"%d" % i, b"v", now + (5 if i % 2 else 10 ** 6)) for i in range(10)], now)
check("memory/prune drops only the expired among mixed lifetimes", store.prune(now + 6), 5)
check("memory/the rest stay", len(store), 5)


# --------------------------------------------------------------- exports
for name in ("DecisionCache", "DecisionStore", "decision_margins"):
    check_true("export/%s is in __all__" % name, name in laya.__all__)

# --------------------------------------------------------------- the archived Feishu runs
# research/benchmarks/feishu_zh keeps three repeats of 64 cases for Jev and Laya. An answer that
# changed side between Jev's runs must sit inside a 0.1 margin in every run, while that band
# flags only a few percent of all answers; Laya's repeats must be identical.
ARCHIVE = os.path.join(ROOT, "research", "benchmarks", "feishu_zh", "results", "v1")
THRESHOLDS = {"related": 0.5, "action": 0.5, "urgent": 0.75, "value": 0.5}
for backend in ("jev", "laya"):
    with open(os.path.join(ARCHIVE, backend, "raw.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    groups = {}
    for row in rows:
        groups.setdefault((row["mode"], row["id"]), []).append(row)
    flagged = unstable = covered = 0
    identical = 0
    for reps in groups.values():
        margins = [decision_margins(rep["response"], THRESHOLDS) for rep in reps]
        flagged += sum(m < 0.1 for per in margins for m in per.values())
        identical += all(rep["response"]["answers"] == reps[0]["response"]["answers"] for rep in reps)
        answers = [rep["response"]["answers"] for rep in reps]
        for name in answers[0]:
            if "noul" in answers[0][name]:
                sides = {a[name]["noul"] >= THRESHOLDS[name] for a in answers}
            else:
                sides = {a[name]["choice"] for a in answers}
            if len(sides) > 1:
                unstable += len(reps)
                covered += sum(per[name] < 0.1 for per in margins)
    total = sum(len(rep["response"]["answers"]) for reps in groups.values() for rep in reps)
    if backend == "jev":
        # 3 answers changed side between runs; counted once per run, that is 9 readings.
        check("archive/jev readings of answers that changed between runs", unstable, 9)
        check("archive/a 0.1 margin flags each of them in every run", covered, unstable)
        check_true("archive/while flagging under 5%% of answers (%d/%d)" % (flagged, total),
                   flagged < 0.05 * total)
        # 38 of the 64 choice answers repeat their probabilities, but 9 of those still change
        # `confidence`, which Jev computes below the precision it prints probabilities at.
        check("archive/jev responses identical across three runs", identical, 29)
    else:
        check("archive/laya never changed an answer", unstable, 0)
        check("archive/laya responses identical across three runs", identical, len(groups))

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all consistency tests passed")
sys.exit(1 if FAIL else 0)
