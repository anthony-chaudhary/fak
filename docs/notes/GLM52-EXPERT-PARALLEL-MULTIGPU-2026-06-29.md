---
title: "GLM-5.2 multi-GPU: expert-parallel MoE sharding lands + the resident-fit benchmark (2026-06-29)"
description: "The expert-parallel (EP) MoE FFN decomposition for GLM-5.2 is now host-proven on trunk (f713391c); the resident-EP fit on the 8×80GB GPU server is feasible by VRAM (640 GiB > 434 GiB); the live resident-EP tok/s witness is not yet — it needs the device NCCL collective, EP live-wiring, and a multi-GPU binary on the box."
---

# GLM-5.2 multi-GPU: expert-parallel sharding + the resident-fit benchmark

_2026-06-29._ Companion to
[native-753b-track-staged-plan.md](native-753b-track-staged-plan.md) (the dependency
map) and [GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md](GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md)
(why the cpu-offload serve is host-bound). It records one increment that landed and the
benchmark question it makes answerable.

## What landed (SHIPPED, host-proven): expert-parallel MoE FFN sharding

`internal/model/expert_parallel.go` (+`expert_parallel_test.go`), commit `f713391c`. This is
the expert-parallel (EP) decomposition the live forward's `forwardTPSupported()` names but
fails closed on ("ForwardTP does not yet shard MoE FFN — expert-parallel is a separate
sub-lever"). It is the MoE counterpart of `tensor_parallel.go`'s `TensorParallelFFN`.

EP partitions the experts across ranks (`ExpertParallelPlan(numExperts, ranks)`, a named
`TPPlan` over `[0,NumExperts)`): each rank holds a contiguous band of experts resident — about
the model's expert bulk divided by the rank count — the router runs replicated and picks the
same top-k, and each rank contributes only the picks it owns. The per-rank `[H]` residual
partials are combined by exactly one `AllReduceSum`. Experts are independent (no shared
intermediate to gather, unlike the dense FFN's column/row split), so EP is the cheap, natural
multi-GPU decomposition for a MoE model — and it is the one the overnight GPU-server data
pointed to: move the expert GEMM off the host onto the idle GPUs.

The correctness gates mirror the dense row-parallel rung and run with **no multi-GPU hardware**
(`LocalCollective` is the single-box, bit-exact default):

| Gate | Result |
|---|---|
| `EP(ranks=1)` == routed monolith (`moeFFN`/`glmMoeFFN`) | **max\|Δ\|=0** (bit-exact) |
| `EP` via Collective == `EP` rank-order reference (ranks 1/2/4/8) | **max\|Δ\|=0** |
| `EP(ranks=N)` vs routed monolith | **cosine 1.0**, ~3.7e-9 drift (reassociation only) |
| load-imbalanced (all picks on one rank) ranks=2 == ranks=1 | max\|Δ\|=0 |
| fail-closed (`plan.Dim != NumExperts`, `ranks > experts`) | rejected |
| GLM shared-expert wrapper on a real `glm_moe_dsa` fixture | ranks=1 bit-exact |

Scope, kept honest: this is the **proven primitive**, the same posture every TP brick landed
in (`tensor_parallel.go`, `BackendCollective`, `DistComm`). It is **not yet wired into the
live `glmMoeFFN` forward**, and the Collective is cpu-ref.

## The benchmark this makes answerable: does GLM-5.2 fit resident across 8 GPUs?

The whole point of EP for GLM-5.2 is to escape the cpu-offload wall by holding the experts in
VRAM across many GPUs instead of in host RAM. So the first benchmark is a fit question, and on
the 8-GPU GPU server the VRAM clears it:

| quantity | value | source |
|---|---|---|
| per-GPU VRAM × count | **8 × 81920 MiB ≈ 640 GiB** | WITNESSED (`nvidia-smi`, 2026-06-29) |
| GLM-5.2 UD-Q4_K_M on disk | **433.82 GiB** (753.86 B params) | WITNESSED (llama-bench, 2026-06-28) |
| even per-GPU expert shard (434/8) | **~54 GiB** | COMPUTED |
| + replicated dense/attn/router + KV @ 4K | a few GiB + ~1 GiB | COMPUTED |
| per-GPU resident estimate | **~55–60 GiB** (< 80 GiB, ~20 GiB headroom) | COMPUTED |

So **resident EP is feasible by capacity** on this hardware — the model fits across the eight
GPUs with room for the KV cache and per-op scratch. (At session time GPU0 held two peer-owned
cpu-offload serves; GPUs 1–7 were idle. 882 GiB host RAM free.) The exact replicate-vs-shard
split of the non-expert tensors is the EP+TP plan detail; the floor above assumes the simplest
case and still fits.

This is the lever past the measured wall. The baseline EP would have to beat, both from the
overnight collection (`docs/nightrun/collected.jsonl`, 2026-06-28):

- fak GPU + `--cpu-offload-experts` steady-state TPOT: **0.2324 tok/s** (WITNESSED, fak kernel)
- llama.cpp CPU mmap baseline on the same model: **0.89 tok/s** decode (OBSERVED, third-party)

The cpu-offload path is host-expert-GEMM-bound; resident EP moves that GEMM onto the idle GPUs.

## Benchmark run this session (native GLM-DSA decode, host CPU)

To keep the benchmark thread honest with an actually-executed measurement (not only the
capacity computation above), `cmd/glmdsatput -backend legacy` was run on the agent-host CPU
this session. It drives fak's **native** glm_moe_dsa forward — the real MLA + DSA-indexer +
sparse-attend + dense-FFN kernels — over a synthetic, reduced-layer model. A 3-point sweep
(Q8_0, prompt=64, 16 decode steps, median of 3 reps):

| config (layers × hidden, heads, inter) | prefill tok/s | decode tok/s | ms/tok |
|---|---|---|---|
| 4 × 1024, h8, i4096 | 33.6 | **33.57** | 29.79 |
| 8 × 1024, h8, i4096 (2× depth) | 19.1 | **17.85** | 56.01 |
| 4 × 2048, h16, i8192 (2× width) | 12.5 | **9.31** | 107.42 |

Scaling is coherent: 2× depth → 1.88× slower decode (near-linear in layer count), 2× width
→ 3.6× slower (super-linear — attention + FFN GEMMs grow with hidden²). These are WITNESSED on
fak's own kernels, this session, on the **agent-host CPU** (a desktop, not a bench-node), and
carry the tool's own scope label: **synthetic weights, reduced layers, dense-FFN (no MoE expert
GEMMs), optimistic lower-bound, NOT the 753B**. They measure the native GLM-5.2-architecture
per-token cost on this CPU, not full-checkpoint serving throughput (that is the cpu-offload /
resident-EP number below). The EP decomposition this note lands does not change these single-box
figures — its win is multi-GPU residency, which needs the device collective.

### Decode-path lever benchmarks (executed this session, AMD Ryzen 9 9950X, Zen 5 / AVX-512-VNNI)

The host expert path EP shards is built on two shipped levers
([GLM52-DECODE-PATH-TO-10-TOKS](GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md)). Both were
re-measured on current trunk this session — they corroborate the decode-path doc's numbers:

| lever | benchmark | result |
|---|---|---|
| **2: batched expert dispatch** (`hostBatchedGLMExperts`, the path EP reuses) | `GLMExpertDispatch` Looped→Batched (8 experts, MI=1536, H=5120) | 12.93 ms → 6.27 ms = **2.06×** (8→1 allocs); doc estimated ~1.8× |
| **3: Q4_K int8 SIMD reducer** (gate/up experts) | `Q4KGEMV` f32→int8 | 19.42 ms → 1.65 ms = **11.79×** (matches the doc's ~11.9× VNNI) |
| **3: Q5_K int8 SIMD reducer** (down experts) | `Q5KGEMV` f32→int8 | 55.76 ms → 3.03 ms = **18.4×** |
| **3: Q6_K int8 SIMD reducer** | `Q6KGEMV` f32→int8 | 22.72 ms → 2.70 ms = **8.41×** |

These are the per-kernel wins under each expert GEMV; EP distributes those same GEMVs across
ranks, so the multi-GPU path inherits them. (Reproduce: `go test ./internal/model -run '^$'
-bench 'GLMExpertDispatch|Q4KInt8GEMV|Q4KF32GEMV|Q5KInt8GEMV|Q5KF32GEMV|Q6KInt8GEMV|Q6KF32GEMV'
-benchmem`.)

## What is `not yet` (the honest gap to a live 753B number)

A **live resident-EP tok/s witness** does not exist yet. Three things gate it, in order:

1. **A real cross-DEVICE collective.** The CUDA backend still pins `cudaSetDevice(0)`
   (`internal/compute/cuda_kernels.cu:55`) and reports `Caps().Collective=false`; the
   `compute.CollectiveBackend` seam has only the cpu-ref implementation. A device tensor must
   all-reduce across real GPUs (NCCL/RCCL) before "multi-GPU" may be claimed
   (native-753b-track P3, the documented honesty line).
2. **EP wired into the live `glmMoeFFN` forward**, selecting per-rank experts and reducing
   through the device collective — the host primitive landed today is the proven core of this.
3. **A multi-GPU-capable binary on the box.** `go` is absent on the GPU server and the staged
   `fak` binary is single-GPU; building the multi-device path on-box is a prerequisite
   (the box has `git` + internet, so installing Go on-box is the unblock, per the decode-path note).

Until those land, the resident-EP number is `not yet`, and the cpu-offload wall (0.2324 tok/s)
stands as the baseline. The capacity fit and the EP decomposition are real and on trunk; the
serving throughput is the multi-month residual.
