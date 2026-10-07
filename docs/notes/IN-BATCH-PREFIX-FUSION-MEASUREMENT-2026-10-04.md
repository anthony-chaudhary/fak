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

## 2026-10-06 — fair re-measure, per-request TTFT, and a one-pass cascade arm

Provenance: [SW-VERIFIED] in-process CPU synthetic model (4 vCPU, host shared with a
concurrent worker, so spreads are wide). Not hardware evidence.
Witness: `go test ./internal/model/ -run 'Batch|Cascade' -count=1`,
`go test ./cmd/dedupbench/... -count=1`, `go run ./cmd/dedupbench -check -json` (all green).

### Fairness fixes to `cmd/dedupbench` (schema `fak.dedupbench.v2`)

A review found the 2026-10-04 table above unfair, so those ratios are superseded:

- **Goroutine-parallelism credit removed.** Fusion followers used to prefill their
  suffixes as separate forwards on their own goroutines, while the NBA twins ran one
  after another. Now both arms prefill all followers/twins as ONE batch
  (`Model.NewBatchFromPrefix` + `BatchSession.PrefillEach`). That matches SGLang, which
  defers same-prefix waiting requests and admits them together in one extend batch.
- **Modeled scheduler step.** `-sched-step-ms` (default 0) models SGLang's extra
  scheduler step before the twin extend. It is added to the NBA follower TTFT and
  makespan, and the receipt reports it as its own field
  (`modeled_sched_step_ms_included`). It is never measured or hidden. All numbers below
  use 0.
- **Token accounting fixed.** Fusion was reported as `P+(N-1)*S`. The leader actually
  prefills `P+S`, so the true count is `P+N*S`, the same as the NBA and cascade.
- **False comments fixed.** Followers DO block on the leader's `ready`
  (`internal/radixkv/singleflight.go`). There is no overlap of follower work with the
  leader's prefix prefill, and the header no longer says there is.
- **Per-request TTFT.** TTFT is the time from the common release to that request's
  prefill completion. The receipt reports leader TTFT, follower mean/p50/p99 TTFT,
  all-request mean TTFT and makespan. Reps run interleaved across arms after one
  discarded warm-up rep, and each metric is the **median with min/max**, not best-of.
- **Correctness gate unchanged.** `-check` still requires every arm's logits to be
  bit-identical (Float32bits) to a fresh full prefill. Fusion still asserts exactly 1
  leader and N-1 followers.

### New arm: `cascade` (one-pass shared-prefix prefill)

`Model.CascadePrefill(prefix, suffixes)` (`internal/model/batch_cascade_prefill.go`)
follows the Hydragen / FlashInfer cascade pattern. Per layer, the P prefix rows and all
N*S suffix rows go through one shared-weight GEMM per projection. The prefix K/V is
computed once and appended to each lane's own cache. Each suffix row attends causally to
the prefix plus its own suffix. It is gated to the rectangular f32 `PrefillEach` model
class (PreNorm, dense, non-MLA, non-Qwen35-hybrid, uniform RoPE, f32 KV, ≤512 tokens
per prefix and per suffix), and any other model is refused with an error.
`TestCascadePrefillMatchesIndependentPrefill` shows bit-identical logits, K, Kraw, V,
positions and a follow-on decode step against an independent full prefill. This is the
same Float32bits contract the `PrefillEach` f32 tests use. The cost: every request,
leader included, gets its logits at T(P+N·S), not T(P+S).

### Results (S=32, reps=7, median; ratios are NBA/arm, >1 = arm faster)

| hidden/layers | N | P | NBA leader ms | NBA follower-mean ms | NBA makespan ms [min–max] | cascade makespan ms [min–max] | cascade follower-mean | cascade leader | cascade makespan | fusion follower-mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 128/4 | 4 | 128 | 86.5 | 304.9 | 304.9 [72.6–1196.8] | 304.0 [78.0–664.5] | 1.00× | 0.28× | 1.00× | 1.60× |
| 128/4 | 4 | 512 | 349.0 | 651.4 | 651.4 [244.6–1368.8] | 440.0 [255.0–547.0] | 1.48× | 0.79× | 1.48× | 1.04× |
| 128/4 | 8 | 128 | 84.3 | 169.2 | 169.2 [154.1–452.9] | 136.0 [110.2–355.1] | 1.24× | 0.62× | 1.24× | 1.05× |
| 128/4 | 8 | 512 | 218.3 | 419.2 | 419.2 [285.7–767.3] | 388.1 [280.1–511.4] | 1.08× | 0.56× | 1.08× | 1.07× |
| 1024/2 | 4 | 128 | 658.7 | 1028.3 | 1028.3 [861.5–1324.0] | 981.1 [791.8–1372.0] | 1.05× | 0.67× | 1.05× | 0.85× |
| 1024/2 | 4 | 512 | 2090.4 | 2504.5 | 2504.5 [2165.0–5410.9] | 2371.9 [2308.8–3661.1] | 1.06× | 0.88× | 1.06× | 0.98× |
| 1024/2 | 8 | 128 | 523.7 | 1340.4 | 1340.4 [1242.9–1605.6] | 1371.9 [1201.0–1433.7] | 0.98× | 0.38× | 0.98× | 0.99× |
| 1024/2 | 8 | 512 | 2049.5 | 3042.8 | 3042.8 [2762.1–3224.1] | 2880.0 [2787.7–3455.3] | 1.06× | 0.71× | 1.06× | 1.01× |
| 2048/2 | 4 | 128 | 2167.9 | 3453.9 | 3453.9 [3132.5–3670.1] | 3379.4 [3195.5–3858.3] | 1.02× | 0.64× | 1.02× | 1.03× |
| 2048/2 | 4 | 512 | 7308.4 | 8816.5 | 8816.5 [8252.0–8950.9] | 8428.6 [8192.1–10242.3] | 1.05× | 0.87× | 1.05× | 1.00× |
| 2048/2 | 8 | 128 | 2020.0 | 5142.9 | 5142.9 [4812.6–6098.4] | 5257.8 [4879.6–5430.6] | 0.98× | 0.38× | 0.98× | 1.02× |
| 2048/2 | 8 | 512 | 7025.1 | 10717.9 | 10717.9 [9872.5–11140.1] | 10588.9 [9529.3–11112.7] | 1.01× | 0.66× | 1.01× | 1.02× |

What this shows:

- **Fusion vs the fair NBA is ~1.0× (0.85–1.07×) on every envelope with a tight
  spread.** The one 1.60× cell (H=128, N=4, P=128) has an NBA spread of 73–1197 ms,
  so it is host noise. Once both arms batch their followers, fusion and the NBA do the
  same work in the same order. The only remaining NBA cost fusion avoids is the
  scheduler step, which is now an explicit modeled input.
- **Cascade vs the NBA is 0.98–1.06× on follower-mean TTFT and makespan in the
  weight-heavy envelopes (H=1024/2048).** Its leader TTFT is 0.38–0.88×, worse by
  construction. The two 1.24×/1.48× cells at H=128 sit inside NBA spreads that cover
  2–5×, so they are not a stable win.
- **Why the ceiling is small here.** Cascade removes one weight pass: the leader
  extend and the twin extend become one GEMM. CPU prefill over hundreds of rows is
  compute-bound, and both schemes do the same P+N·S rows of FLOPs. A second pass over
  the weights adds about weight_bytes ÷ bandwidth, which is small next to that compute.
  The win the earlier probe saw ("batched ≈2× faster than sequential at H=1024") is
  batching vs sequential suffixes, and the fair NBA now gets it too.
- **Status of the 1.4× goal.** It is not met by fusion or by cascade on this host. The
  next step is the regime where one weight pass is a large share of a pass: short
  suffixes on a bandwidth-bound device (a GPU/APU where weights stream from DRAM and
  small extends are memory-bound), plus a real nonzero scheduler step. On that device
  cascade's makespan gain is bounded above by roughly (T_lead + T_twins) / T_onepass.

### HW-witness run (what would make this [HW-WITNESSED])

`dedupbench -model <dir>` now loads a real f32 export (the `export_oracle.py` layout:
`config.json`, `manifest.json`, `weights.f32`) through `model.Load`. This flag was only
build-checked here; no real weights were loaded on this host. On a Strix Halo with a
dense Llama-class f32 export (for example Llama-3.2-1B; Qwen3.5/3.6 hybrids are refused
by the cascade gate):

```text
go run ./cmd/dedupbench -model /models/llama-3.2-1b-f32 -n 8 -prefix 512 -suffix 32 -reps 7 -json
go run ./cmd/dedupbench -model /models/llama-3.2-1b-f32 -n 8 -prefix 512 -suffix 32 -reps 7 -sched-step-ms 2 -json
```

That run is still a host-CPU forward. A device-side witness needs pieces this change
does not add: a Q8/Q4_K cascade lane (cascade is f32-only today, while serving lanes are
quantized), a Vulkan/HIP device lane, and wiring into `NativeScheduler.prefillCoalesced`
so a serving TTFT histogram can be taken on the live path.
