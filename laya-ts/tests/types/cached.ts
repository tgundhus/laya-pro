import { cached, cachedEmbedFn, DEFAULT_SHORTLIST_MAXSIZE,
  type CachedEmbedFn, type CacheInfo } from "../../src/index.js";

const embed: CachedEmbedFn = cached(texts => texts.map(text => [text.length, 1]), 64);
const alternate: CachedEmbedFn = cachedEmbedFn(async texts => texts.map(() => [1]));
const info: CacheInfo = embed.cacheInfo();
const pythonStyleInfo: CacheInfo = embed.cache_info();
const vectors: Promise<number[][]> = embed(["refund"]);
embed.cacheClear();
embed.cache_clear();
// @ts-expect-error cached embedders accept text arrays
embed("refund");
// @ts-expect-error cache capacity is numeric
cached(texts => texts.map(() => [1]), "64");
// @ts-expect-error cache counters are numeric
const invalid: string = info.hits;
void [alternate, pythonStyleInfo, vectors, DEFAULT_SHORTLIST_MAXSIZE, invalid];
