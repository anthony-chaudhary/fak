# Prefill throughput gap — B-001 / issue #9 — scaffold + named blocker

**Status:** host-tractable scaffold SHIPPED (CPU-correct, bit-exact witnessed).
**Blocked:** the wall-clock perf acceptance is deferred to a CUDA bench node (this host has no CUDA).

## What shipped (compute lane, `internal/compute/prefill.go` + tests)

The issue's four scope bullets, each in the honest form a CUDA-less host can witness:

| Scope bullet | Shipped here | Witnessed by |
|---|---|---|
| Profile current prefill bottleneck | `PrefillCostModel` / `Profile` — an analytic roofline: per-stage **exact** FLOPs + bytes moved + arithmetic intensity. No timer, no fabricated throughput. Surfaces the O(P²) attention term and the crossover (≈ `DFF`) where attention overtakes the FFN GEMMs. **`PrefillRoofline.PredictTime` / `StageCost.RooflineSeconds`** convert those exact counts into a per-stage time LOWER BOUND given two device-measured peaks (peak FLOP/s, peak bytes/s) — the bottleneck in the issue's own units (146ms vs 82.9ms). The TIME-dominant stage diverges from the FLOP-heaviest `Dominant` whenever the prefill is memory-bound, so the time view aims the kernel work at the right stage. **`WithinTarget`** is the 1.2× acceptance gate in code (device-fed). | `TestPrefillCostModelStructure`, `TestPrefillAttentionCrossover`, `TestStageRooflineSeconds`, `TestPredictTimeBottleneckDivergesFromFLOPs`, `TestWithinTargetGradesIssueNumbers` |
| Implement batched-GEMM prefill kernel | `PrefillGEMM` — a tiled panel×tile GEMM skeleton (the device-kernel blocking shape), **bit-exact** to `Backend.BatchedMatMul` (every cell is the same `fdot`; tiling moves no byte). | `TestPrefillGEMMBitExactToBatchedMatMul`, `TestPrefillGEMMEqualsFdot` |
| Optimize attention for long sequences | The cost model shows naive attention's intensity is a P-independent ~0.5 FLOP/byte (deeply memory-bound) — the structural motivation for fused/flash attention. The optimization itself is device work. | `TestPrefillCostModelStructure` (intensity == 0.5) |
| CUDA graph for prefill | `PrefillGraphCapturer` (pure-Go HAL seam, **always compiled**) + `CapturePrefillGraph`/`ResetPrefillGraph` guarded helpers. The CUDA backend already implements `GraphBegin/GraphEndLaunch/GraphReset` and satisfies the seam under `-tags cuda` (pinned by `prefill_cuda.go`'s compile-time assertion); a non-CUDA build has no capturer and takes the eager fallback. | `TestCapturePrefillGraph{Fallback,Captured,Declined}`; non-CUDA build compiles clean |

`go test ./internal/compute/...` is green (CPU path). `go build ./...` (default, non-CUDA) is clean.

## The named blocker (do NOT fake this on a CUDA-less host)

The acceptance bullets

- [ ] Prefill within **1.2× llama.cpp Q8_0**
- [ ] Sustained at **P=256, P=512, P=1024**

are a **wall-clock measurement on a CUDA device**. They cannot be witnessed on this host
(no CUDA). **No prefill throughput number was run, estimated, or fabricated here** — the
profiling deliverable is the analytic FLOP/byte roofline, which is exact and
hardware-independent; only the *timing* needs the device. The perf acceptance is **deferred
to a CUDA bench node** (run on a node that is NOT also running the agent — placement law).

Bit-exact acceptance (`Bit-exact results unchanged`) IS witnessed here, by construction:
`PrefillGEMM` is byte-identical to the shipped reference path.

## Next agent, on a CUDA node

1. Lower `PrefillGEMM`'s tiling into the device GEMM and wire `CapturePrefillGraph` around
   the prefill op-stream in the live forward (the seams are already in place).
2. Run the **timed** prefill bench at P=256/512/1024 against llama.cpp Q8_0; record lineage
   (version + UTC + commit + sanitized machine + harness). Grade the gate with
   `compute.WithinTarget(measured, llamaBaseline, 1.2)` — the 1.2× rule lives in one tested
   place, no re-deriving it by hand.
3. Feed the node's measured peak FLOP/s + bandwidth into `PrefillRoofline.PredictTime` to get
   the per-stage time FLOOR, then compare each stage's MEASURED time against `PerStage[i]`:
   the stage with the largest measured-over-floor headroom is where the attention/graph work
   pays off. Note the time-dominant stage is NOT in general the FLOP-heaviest `Dominant` —
   trust `PredictTime` for a memory-bound prefill, not the FLOP-only `Profile.Dominant`.
