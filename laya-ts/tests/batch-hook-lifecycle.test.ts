import { describe, expect, it } from "vitest";
import { Agent, Router } from "../src/index.js";

const questions = { flag: { type: "noul", instructions: "Relevant?" } };
const provider = {
  async runEncoder(batch: any) { return { lastHidden: batch.inputIds.map(() => [[0]]) }; },
  async runHead(_hidden: any, batch: any) {
    return { logits: batch.qtype.map(() => [2, 0]), act: batch.qtype.map(() => [1, 0]) };
  },
};

describe("awaited Router batch lifecycle", () => {
  it.each([true, false])("awaits state and nested result writes with hooksRaise=%s", async hooksRaise => {
    const events: string[] = [];
    const router = new Router({ loader: () => new Agent({ provider }) });
    const results = await router.predictBatch(["a", "b"].map(state => ({ state, questions })), null, {
      hooksRaise,
      async onPredictStart(ctx) {
        await Promise.resolve();
        events.push(`start:${ctx.states[0]}`);
        ctx.maxLen = 40;
      },
      async onPredictEnd(ctx) {
        await Promise.resolve();
        events.push(`end:${ctx.states[0]}`);
        const answers = ctx.results![0].answers as Record<string, Record<string, unknown>>;
        answers.flag.noul = 0.75;
      },
    });
    expect(events).toEqual(["start:a", "start:b", "end:b", "end:a"]);
    expect(results.map(result => result.answers.flag)).toEqual([
      expect.objectContaining({ noul: 0.75 }), expect.objectContaining({ noul: 0.75 }),
    ]);
  });

  it("awaits error and end hooks while preserving the inference failure", async () => {
    const events: string[] = [];
    const router = new Router({ loader: () => ({ async systemOne() { throw new Error("inference failed"); } }) });
    await expect(router.predictBatch(["a", "b"].map(state => ({ state, questions })), null, { hooks: {
      async onError(ctx) {
        await Promise.resolve();
        events.push(`error:${ctx.states[0]}`);
        throw new Error("error hook failed");
      },
      async onPredictEnd(ctx) {
        await Promise.resolve();
        events.push(`end:${ctx.states[0]}`);
        throw new Error("end hook failed");
      },
    } })).rejects.toThrow("inference failed");
    expect(events).toEqual(["error:a", "error:b", "end:b", "end:a"]);
  });

  it("passes request budgets to Agent-like single-prediction fallbacks", async () => {
    const seen: unknown[] = [];
    const router = new Router({ loader: () => ({ async systemOne(_s: unknown, _q: unknown, opts: unknown) {
      seen.push(opts);
      return { model: "fake", answers: {}, usage: {} };
    } }) });
    await router.decideBatch(["a"], undefined, { questions, maxLen: 64, headMaxLen: 24 });
    expect(seen).toEqual([{ maxLen: 64, headMaxLen: 24 }]);
  });
});
