# In-batch cold-prefix fusion: reachability, integration boundary, and an honest measurement

Date: 2026-10-04
Status: mechanism shipped + reachable; measured head-to-head **~1.0×**, NOT 1.4×
Witness: `go test ./internal/modelengine/... ./cmd/dedupbench/... -count=1` (green),
`go run ./cmd/dedupbench -check -json`
Provenance: [SW-VERIFIED] (Go receipt, in-process CPU model — not hardware evidence)

This note closes the three clauses of the goal "Fak Inference Engine 'fusion' cache
part is running and operating and fully integrated, showing a 1.4x advantage over
next best alternative caching." Two clauses are satisfied; the third is **refuted by
measurement** and reported honestly rather than inflated.

## What the "fusion cache" is

The mechanism is **in-batch cold-prefix fusion** (fak#1914, remainder fak#13661):
`internal/modelengine.NativeScheduler.prefillCoalesced` runs a
`radixkv.PrefixFlightGroup` single-flight over concurrent cold admissions. When N
admissions arrive from one prompt family (shared prefix P, divergent suffixes) and
the pluggable cache is armed, the first admission becomes the **leader** and runs the
shared prefix once; every other admission **joins the in-flight leader**, adopts a
clone of its prefix KV, and prefills only its divergent suffix. Its production
on-switch is `Engine.SetInBatchPrefixDedup` / `FAK_NATIVE_IN_BATCH_PREFIX_DEDUP`.

## Clause A — RUNNING (satisfied)

Before commit `58755c12c50`, `Engine.nativeScheduler` — the sole production
construction of the serving scheduler — never called `SetInBatchPrefixDedup`, so
`inBatchDedup` stayed false in every serving process and `prefillCoalesced` was dead
code on the live path even though the mechanism tests passed. The commit arms it from
the boot field or the env (default off), adds `InBatchPrefixDedupArmed()` +
`InBatchPrefixDedupCounters()` readback, and adds
`TestInBatchPrefixDedupReachable`, which drives `New`/`Preload`/`nativeScheduler` (not
the test-only constructor) and proves two identical concurrent cold admissions run
exactly ONE shared prefill with bit-identical twin logits.

## Clause B — FULLY INTEGRATED (bounded; ABSTAIN on the resident chunked lane)

`admit` calls `prefillCoalesced` only when `prefillChunkTokens == 0`. On a
resident-Q4_K/Qwen35-hybrid serving lane `qwenPrefillChunkBudget` returns >0, so
admission instead takes the async chunked path and never coalesces. That lane cannot
be coalesced through the proven prefix-truncate handoff: a Qwen35-hybrid `KVCache`
carries recurrent (Gated-DeltaNet) state for which `KVCache.CanEvict()` always
returns `*RecurrentEvictUnsupportedError` (`internal/model/kvcache.go:143`), and
`radixkv.CoalesceSharedPrefixNS` fails that case open
(`internal/radixkv/singleflight.go:292-297`). A divergent-suffix prefix reuse is
therefore *formally unprovable* on that cache. Full evidence:
`internal/modelengine/nativesched_prefix_dedup_chunked_findings.md`.

So integration is complete for the lane the mechanism is proven on (the generic
host-KV synchronous path, which is the shipped default), and is a documented
**ABSTAIN** for the recurrent hybrid lane, whose unblock is a new per-layer recurrent
checkpoint/journal — out of scope here and not silently skipped.

## Clause C — the 1.4× claim (REFUTED; measured ~1.0×)

`cmd/dedupbench` measures the fusion against the **tuned next-best alternative**: the
SGLang-style in-batch prefix cache, which defers the twin until the leader's cache is
WARM, then serves the twin from the warm prefix. Both arms compute the SAME total
prefill tokens (one shared prefix + N suffixes), so the honest discriminating axis is
wall-clock. All arms are correctness-gated bit-identical to a fresh full prefill, and
the fusion arm asserts its coalescing integrity (exactly one leader, N-1 followers).

Measured (in-process CPU, synthetic model, best-of-3..5, host-contention limited):

| envelope (N, prefix, suffix) | fusion / warm-cache-NBA wall ratio |
|---|---|
| 8, 128, 32 | 1.64× |
| 8, 512, 32 | 1.36× |
| 8, 1024, 32 | 1.06× |
| 8, 2048, 32 | 0.76× |
| 8, 512, 64 (hidden/layers varied) | 0.57–1.33× |
| 4, 128, 64 | 1.74× |
| 4, 128, 96 | 1.04× |
| 4, 128, 128 | 0.98× |

The ratio is **not stably 1.4×**; it straddles 1.0× and is dominated by measurement
noise and the KV-clone copy cost rather than by a structural advantage. This is
expected and explainable: fak's follower *joins* the leader's flight but still
**waits** for the shared prefix before prefilling its suffix, so it eliminates the same
prefix tokens the warm-cache NBA eliminates — the two distinct mechanisms do the same
work-saving. The warm-cache NBA additionally has no follower-wait. The measured
"advantage" of the SGLang-style baseline being modeled (warm reuse, not in-flight
overlap) is not a verified structural win for fak here.

**Conclusion:** the mechanism is running and integrated on its proven lane, but the
goal's third clause is **not met** — the measured advantage over the tuned next-best
alternative is ~1.0×, not 1.4×. Reporting 1.4× would be false. A real win would need
either (a) true follower overlap that does not wait on the leader's prefix, or (b) a
model/regime where the in-flight join's latency hiding actually dominates, measured on
the live serving path ([HW-WITNESSED]) rather than an in-process synthetic model.

## Reproduce

```text
go test ./internal/modelengine/... -count=1 -run TestInBatchPrefixDedupReachable
go test ./internal/modelengine/... ./cmd/dedupbench/... -count=1
go run ./cmd/dedupbench -check -json
go run ./cmd/dedupbench -n 8 -prefix 512 -suffix 32 -reps 5 -json
```
