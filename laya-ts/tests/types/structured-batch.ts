import {
  Agent, Router, decideBatch,
  type DecideBatchOptions, type DecisionResult, type DecideRunner, type BatchRequest,
} from "../../src/index.js";

declare const agent: Agent;
declare const router: Router;
declare const runner: DecideRunner;
const schema = { type: "object", properties: { flag: { type: "boolean" } } };
const options: DecideBatchOptions = { batchSize: 2, minConfidence: { "noul:2": 0.8 } };
const values: Promise<Record<string, unknown>[]> = agent.decideBatch(["a"], schema, options);
const details: Promise<DecisionResult[]> = router.decideBatch(["a"], schema, { returnDetails: true });
const free: Promise<DecisionResult[]> = decideBatch(runner, ["a"], schema, { returnDetails: true });
const request: BatchRequest = { state: "a", questions: {}, maxLen: 64, headMaxLen: 24 };
void values; void details; void free; void request;
