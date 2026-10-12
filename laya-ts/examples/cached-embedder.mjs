// Build laya-ts first. Uses only an existing local split-ONNX export.
// node laya-ts/examples/cached-embedder.mjs --model-dir SPLIT_EXPORT --report cached.json
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { cpus, totalmem } from "node:os";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { parseArgs } from "node:util";
import { Agent, cached, embedFnFromAgent, shortlistChoice } from "../dist/index.js";

const { values } = parseArgs({ options: {
  "model-dir": { type: "string" }, report: { type: "string" },
  repeats: { type: "string", default: "3" },
  "checkpoint-revision": { type: "string" },
} });
assert(values["model-dir"] && values.report,
  "usage: cached-embedder.mjs --model-dir LOCAL_SPLIT_EXPORT --report JSON [--repeats 3]");
const modelDir = resolve(values["model-dir"]);
assert((await stat(modelDir)).isDirectory(), "--model-dir must be an existing local directory");
const reportPath = resolve(values.report);
const reportRelative = relative(modelDir, reportPath);
assert(isAbsolute(reportRelative) || reportRelative.startsWith(`..${sep}`),
  "--report must be outside the model directory");
const repeats = Number(values.repeats);
assert(Number.isSafeInteger(repeats) && repeats >= 3, "use at least three measured repeats");
if (values["checkpoint-revision"]) assert.match(values["checkpoint-revision"], /^[a-f0-9]{40}$/);
const files = ["encoder.onnx", "head.onnx", "rl_agent_config.json", "tokenizer.json"];
files.push(...(await readdir(modelDir)).filter(name => name.endsWith(".onnx.data")));
const hashes = {};
for (const file of files) {
  const digest = createHash("sha256");
  for await (const part of createReadStream(resolve(modelDir, file))) digest.update(part);
  hashes[file] = digest.digest("hex");
}
const source = JSON.parse(await readFile(new URL("../../laya-java/fixtures/tournament_golden.json", import.meta.url)));
const originalCriteria = source.cases[0].questions.intent.criteria;
const criteria = Array.isArray(originalCriteria) ? originalCriteria.slice(0, 20)
  : Object.fromEntries(Object.entries(originalCriteria).slice(0, 20));
assert.equal(Object.keys(criteria).length, 20);
const baseState = source.cases[0].state;
const agent = await Agent.load(modelDir, { device: "cpu", numThreads: 2, expectedSha256: hashes });
const raw = embedFnFromAgent(agent, 64, 32);
const count = { uncached: { calls: 0, rows: 0 }, cached: { calls: 0, rows: 0 } };
const tracked = kind => async texts => {
  count[kind].calls++;
  count[kind].rows += texts.length;
  return raw(texts);
};
const plain = tracked("uncached");
const retained = cached(tracked("cached"), 64);
const select = (state, embed) => shortlistChoice(state, criteria, embed, 5);
for (let i = 0; i < 2; i++) {
  const state = `${baseState}\nWarmup request ${i}.`;
  assert.deepEqual(await select(state, plain), await select(state, retained));
}
for (const value of Object.values(count)) { value.calls = 0; value.rows = 0; }
const samples = [];
for (let i = 0; i < repeats; i++) {
  const order = i % 2 === 0 ? ["uncached", "cached"] : ["cached", "uncached"];
  const states = Array.from({ length: 2 }, (_, j) => `${baseState}\nMeasured request ${i * 2 + j}.`);
  const row = { order, requests: states.length };
  const outputs = {};
  for (const kind of order) {
    const started = performance.now();
    outputs[kind] = [];
    for (const state of states) outputs[kind].push(await select(state, kind === "cached" ? retained : plain));
    row[`${kind}_repeat_ms`] = performance.now() - started;
    row[`${kind}_per_request_ms`] = row[`${kind}_repeat_ms`] / states.length;
  }
  assert.deepEqual(outputs.cached, outputs.uncached, "caching changed the shortlisted labels");
  samples.push({ ...row, labels: outputs.cached });
}
const median = values => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};
const summary = Object.fromEntries(["uncached", "cached"].map(kind => [kind, {
  ...count[kind], median_repeat_ms: median(samples.map(row => row[`${kind}_repeat_ms`])),
  median_per_request_ms: median(samples.map(row => row[`${kind}_per_request_ms`])),
}]));
assert.equal(summary.uncached.rows, repeats * 2 * 21);
assert.equal(summary.cached.rows, repeats * 2);
const report = {
  schema: "laya-ts-cached-embedder-result/1", passed: true,
  checkpoint_revision: values["checkpoint-revision"] ?? null, onnx_sha256: hashes,
  environment: { node: process.version, platform: process.platform, arch: process.arch,
    cpu_model: cpus()[0]?.model, logical_cpus: cpus().length,
    memory_gib: Math.round(totalmem() / 2 ** 30), device: "cpu", ort_threads_requested: 2 },
  settings: { labels: 20, shortlist_k: 5, encoder_max_length: 64, encoder_batch_size: 32,
    maxsize: 64, warmup_requests: 2, measured_repeats: repeats, requests_per_repeat: 2,
    ordering: "alternating uncached-first/cached-first", units: "milliseconds" },
  summary, cache_info: retained.cacheInfo(), samples,
  note: "Active workstation; small repeated-catalog panel. Each repeat runs two requests sequentially; per-request milliseconds divide that total by two. Row counts show avoided encoder work. Timings exclude model loading and hashing and do not establish general throughput or decision accuracy.",
};
await mkdir(dirname(reportPath), { recursive: true });
await writeFile(reportPath, JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ passed: true, summary }));
