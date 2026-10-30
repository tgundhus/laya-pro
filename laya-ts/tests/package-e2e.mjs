import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { access, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

async function run(command, args, cwd) {
  return execFileAsync(command, args, {
    cwd,
    env: {
      ...process.env,
      npm_config_audit: "false",
      npm_config_fund: "false",
    },
    maxBuffer: 10 * 1024 * 1024,
  });
}

async function runNpm(args, cwd) {
  const npmCli = process.env.npm_execpath
    ?? join(dirname(process.execPath), "node_modules", "npm", "bin", "npm-cli.js");
  try {
    await access(npmCli);
    return run(process.execPath, [npmCli, ...args], cwd);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  if (process.platform === "win32") {
    return run(process.env.ComSpec ?? "cmd.exe", ["/d", "/s", "/c", "npm.cmd", ...args], cwd);
  }
  return run("npm", args, cwd);
}

test("the packed module installs and works in an external ESM TypeScript project", async (t) => {
  const workDir = await mkdtemp(join(tmpdir(), "laya-ts-package-"));
  t.after(() => rm(workDir, { force: true, recursive: true, maxRetries: 3, retryDelay: 100 }));

  const packDir = join(workDir, "pack");
  const consumerDir = join(workDir, "consumer");
  await Promise.all([
    mkdir(packDir, { recursive: true }),
    mkdir(consumerDir, { recursive: true }),
  ]);
  await mkdir(join(packageRoot, "dist"), { recursive: true });
  await Promise.all([
    writeFile(join(packageRoot, "dist", "stale.js"), "export const stale = true;\n"),
    writeFile(join(packageRoot, "dist", "stale.d.ts"), "export declare const stale: true;\n"),
  ]);

  const { stdout } = await runNpm(
    ["pack", "--json", "--silent", "--pack-destination", packDir],
    packageRoot,
  );
  const packs = JSON.parse(stdout);
  assert.equal(packs.length, 1);

  const packed = packs[0];
  const paths = new Set(packed.files.map((file) => file.path));
  assert(!paths.has("dist/stale.js"));
  assert(!paths.has("dist/stale.d.ts"));
  for (const required of ["README.md", "LICENSE", "package.json", "dist/index.js", "dist/index.d.ts"]) {
    assert(paths.has(required), `packed artifact is missing ${required}`);
  }
  for (const path of paths) {
    assert.match(path, /^(README\.md|LICENSE|package\.json|dist\/|scripts\/)/);
    // `files` whitelists all of `scripts/`, so the `__pycache__` left by running
    // `scripts/export_onnx.py` used to ship. `laya-ts/.npmignore` keeps it out; this pins that.
    assert(!path.includes("__pycache__"), `packed artifact includes bytecode: ${path}`);
    assert(!path.endsWith(".pyc"), `packed artifact includes bytecode: ${path}`);
  }
  for (const path of paths) {
    if (path.startsWith("dist/") && path.endsWith(".js")) {
      assert(paths.has(path.replace(/\.js$/, ".d.ts")), `${path} has no declaration file`);
    }
  }

  const tarball = join(packDir, packed.filename);
  await writeFile(
    join(consumerDir, "package.json"),
    JSON.stringify({ name: "laya-ts-package-consumer", private: true, type: "module" }),
  );
  await runNpm(
    [
      "install",
      "--ignore-scripts",
      "--omit=optional",
      "--no-package-lock",
      tarball,
    ],
    consumerDir,
  );

  for (const dependency of ["onnxruntime-node", "onnxruntime-web"]) {
    await assert.rejects(
      access(join(consumerDir, "node_modules", dependency)),
      (error) => error?.code === "ENOENT",
      `${dependency} should be omitted from the consumer install`,
    );
  }

  await writeFile(
    join(consumerDir, "runtime.mjs"),
    `import assert from "node:assert/strict";
import { Agent, Router, VERSION, predictTournament, DEFAULT_TOURNAMENT_GROUP, cached, cachedEmbedFn,
  DEFAULT_SHORTLIST_MAXSIZE, decideBatch } from "laya-ts";

let encoderCalls = 0;
let headCalls = 0;
const provider = {
  async runEncoder() {
    encoderCalls += 1;
    return { lastHidden: [[[0]]] };
  },
  async runHead(_hidden, batch) {
    headCalls += 1;
    return { logits: batch.qtype.map(() => [4, 0]), act: batch.qtype.map(() => [3, 0]) };
  },
};
const agent = new Agent({ provider });
const result = await agent.predict("charged twice", {
  intent: {
    type: "choice",
    instructions: "What does the customer want?",
    criteria: { refund: "money back", other: "anything else" },
  },
});

assert.equal(VERSION, ${JSON.stringify(packed.version)});
assert.equal(result.answers.intent.choice, "refund");
assert.equal(result.answers.intent.probabilities.refund > 0.5, true);
assert.equal(result.usage.output_tokens, 0);
assert.equal(encoderCalls, 1);
assert.equal(headCalls, 1);
const schema = { type: "object", properties: { intent: { enum: ["refund", "other"] } } };
const details = await decideBatch(agent, ["one", "two"], schema, { returnDetails: true, batchSize: 2 });
assert.deepEqual(details.map(item => item.values.intent), ["refund", "refund"]);
assert.equal(details[0].answerConfidence.intent, 0.982);
const routed = await new Router({ loader: () => agent }).decideBatch(["one"], schema);
assert.deepEqual(routed, [{ intent: "refund" }]);
const reversed = await agent.predict("one", { intent: {
  type: "choice", instructions: "Pick", criteria: ["refund", "other"], option_order: [1, 0],
} });
assert.equal(reversed.answers.intent.choice, "other");
await assert.rejects(agent.predict("one", { intent: {
  type: "choice", instructions: "Pick", criteria: ["refund", "other"], option_order: [1, 0, null],
} }), /permutation/);
assert.equal(DEFAULT_TOURNAMENT_GROUP, 16);
const tournament = await predictTournament(agent, "charged twice", {
  intent: { type: "choice", instructions: "Pick a route", criteria: ["refund", "other"] },
});
assert.equal(tournament.answers.intent.choice, "refund");
assert.deepEqual(tournament.tournament.intent, { labels: ["refund", "other"], n: 2, rounds: 0 });
assert.equal(cached, cachedEmbedFn);
assert.equal(DEFAULT_SHORTLIST_MAXSIZE, 4096);
let embeddedRows = 0;
const embed = cached(texts => { embeddedRows += texts.length; return texts.map(text => [text.length, 1]); }, 2);
assert.deepEqual(await embed(["refund", "other", "refund"]), [[6, 1], [5, 1], [6, 1]]);
await embed(["refund"]);
assert.equal(embeddedRows, 2);
assert.deepEqual(embed.cacheInfo(), { size: 2, maxsize: 2, hits: 1, misses: 3 });
embed.cache_clear();
assert.equal(embed.cache_info().size, 0);
const checkpoints = [];
const router = new Router({ autoTaskDetection: true, loader(name) {
  return { async systemOne(_state, questions) {
    checkpoints.push(name);
    return { model: name, usage: {}, answers: Object.fromEntries(Object.entries(questions).map(([qid, q]) =>
      [qid, { type: "choice", choice: q.criteria[0], confidence: 1 }])) };
  } };
} });
const workflowQuestions = Object.fromEntries(["action", "category", "churn_risk", "needs_human", "urgency"].map(qid =>
  [qid, { type: "choice", instructions: "Pick a route", criteria: qid === "category"
    ? Array.from({ length: 20 }, (_, i) => String(i)) : ["only"] }]));
const workflowResult = await predictTournament(router, "I need help", workflowQuestions, 4);
assert.deepEqual(checkpoints, ["typed-decisions", "typed-decisions", "typed-decisions"]);
assert.equal(workflowResult.routing.workflow, "customer_service");
`,
  );
  await run(process.execPath, ["runtime.mjs"], consumerDir);

  await writeFile(
    join(consumerDir, "consumer.ts"),
    `import {
  Agent,
  Router,
  decideBatch,
  type DecideBatchOptions,
  type DecisionResult,
  VERSION,
  predictTournament,
  DEFAULT_TOURNAMENT_GROUP,
  cached,
  type CachedEmbedFn,
  type CacheInfo,
  type QuestionDef,
  type SessionProvider,
  type SystemOneResult,
  type TournamentMeta,
} from "laya-ts";

const provider: SessionProvider = {
  async runEncoder(batch) {
    return { lastHidden: batch.inputIds.map((row) => row.map(() => [0])) };
  },
  async runHead(_hidden, batch) {
    return {
      logits: batch.qtype.map(() => [4, 0]),
      act: batch.qtype.map(() => [3, 0]),
    };
  },
};
const questions: Record<string, QuestionDef> = {
  intent: {
    type: "choice",
    instructions: "What does the customer want?",
    criteria: { refund: "money back", other: "anything else" },
  },
};
const prediction: Promise<SystemOneResult> = new Agent({ provider }).predict(
  "charged twice",
  questions,
);

void VERSION;
void prediction;
const tournament: Promise<SystemOneResult & { tournament: Record<string, TournamentMeta> }> =
  predictTournament(new Agent({ provider }), "charged twice", questions, DEFAULT_TOURNAMENT_GROUP);
void tournament;
const embed: CachedEmbedFn = cached(texts => texts.map(text => [text.length]));
const info: CacheInfo = embed.cacheInfo();
const vectors: Promise<number[][]> = embed(["refund"]);
embed.cacheClear();
void info;
void vectors;
const batchOptions: DecideBatchOptions = { batchSize: 2, minConfidence: { "choice:2": 0.8 } };
const schema = { type: "object", properties: { flag: { type: "boolean" } } };
const batch: Promise<Record<string, unknown>[]> = new Agent({ provider }).decideBatch(["a"], schema, batchOptions);
const details: Promise<DecisionResult[]> = decideBatch(new Agent({ provider }), ["a"], schema, { returnDetails: true });
const routed: Promise<DecisionResult[]> = new Router().decideBatch(["a"], schema, { returnDetails: true });
void batch; void details; void routed;
`,
  );
  await writeFile(
    join(consumerDir, "tsconfig.json"),
    JSON.stringify({
      compilerOptions: {
        target: "ES2022",
        module: "NodeNext",
        moduleResolution: "NodeNext",
        strict: true,
        noEmit: true,
      },
      files: ["consumer.ts"],
    }),
  );
  await run(
    process.execPath,
    [join(packageRoot, "node_modules", "typescript", "bin", "tsc"), "--project", "tsconfig.json"],
    consumerDir,
  );
});
