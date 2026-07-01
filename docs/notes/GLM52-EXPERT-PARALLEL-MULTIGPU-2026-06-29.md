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

## Update (later 2026-06-29): gates 1 & 2 closed at the seam — the residual narrows

Two of the three gates above have since landed on trunk (cpu-ref bit-exact; no multi-GPU
hardware was available, so the live witness is still the residual):

- **Gate 1 — a real cross-DEVICE collective: BUILT.** `internal/compute/cuda_collective.go`
  (+`cuda_nccl.cu`, `-tags cuda,nccl`) implements `compute.CollectiveBackend` over a real NCCL
  communicator (`ncclCommInitAll` single-process-multi-GPU): AllReduceSum / AllGather /
  ReduceScatter across distinct GPUs, with `Caps().Collective` advertised true only after
  `fcuda_nccl_init` succeeds over >1 device. (Item 1 above — "only the cpu-ref implementation" —
  is superseded; the device seam now exists, gated behind the NCCL build tag.) `AllToAll` still
  fails closed (grouped ncclSend/Recv is the follow-on; EP uses AllReduceSum, not AllToAll).
- **Gate 2 — EP wired into the live decode forward, reducing through the device collective: DONE.**
  `glmMoeEPFFN` (`internal/model/moe.go`) is dispatched by `ffnForLayer` when `epRanks>1` and now
  reduces the routed-expert partials through `m.expertParallelCollective()` — the `BackendCollective`
  the gateway sets over the device NCCL backend when `--expert-parallel N>1`, instead of the
  hardcoded single-box `LocalCollective` it carried before (commits `24071294` model + `191ae9d6`
  gateway). So a multi-GPU serve's DECODE now issues a genuine cross-GPU all-reduce per MoE layer —
  the first live multi-GPU decode path — rather than initializing a communicator it never used.
  Bit-exact on cpu-ref (`TestGlmMoeEPFFNReducesThroughDeviceCollective`, the EP-decode twin of
  `TestForwardTPViaBackendCollective`).

**What is STILL `not yet`** (the honest residual to a live tok/s number, none of it host-codeable):

1. **Per-rank resident expert COMPUTE.** `expertParallelPartials` still computes every band's
   `expertSwiGLU` on the single session kernel; only the REDUCTION is distributed, not the expert
   GEMMs or their weight residency. The multi-GPU compute win — each rank holding and running only
   its band's experts on its own GPU — is the next rung. This is what turns the cross-GPU all-reduce
   from "correct but no speedup" into the cpu-offload-wall escape the capacity table predicts.
2. **A multi-GPU `-tags cuda,nccl` binary built on the GPU server** (gate 3 above, unchanged — `go`
   is still absent on the box).

So the device line moved from "the seam is cpu-ref only" to "the seam is real and the decode
reduction flows through it"; the remaining residual is per-rank expert residency + the on-box
multi-GPU binary, then the live witness.

## Update (2026-06-30): per-rank expert RESIDENCY closed on the host multi-process rung

Residual item 1 above — "per-rank resident expert COMPUTE ... each rank holding and running only
its band's experts" — is now DONE on the **host multi-process (DistComm) topology**. A sharded EP
serve runs N separate `fak serve --gguf X --expert-parallel N` processes, each holding ONLY its
expert band and computing only that band, reducing its single `[H]` partial across the process group
through a real cross-process collective. No process holds the full GLM-5.2 expert set — the
residency the capacity table predicts.

What shipped (all bit-exact vs full-model EP, host-testable on one box over loopback):

- **Single-band forward + dispatch.** `Model.expertParallelRankPartial` (the one-band partial, fails
  closed on a missing band), `expertParallelRankLocalGLMMoEDelta` (the sharded `glmMoeFFN` twin —
  one local part reduced in rank order, the replicated shared expert added once post-reduce), and
  `Model.epRank/epRankSet` + `SetExpertParallelRank`. `glmMoeEPFFN.rankLocal` dispatches it and
  HARD-fails (no monolith fallback — a sharded rank lacks peer bands). Commit `0abd822e`.
- **Sharded load.** `fak serve` threads `ggufload.WithExpertShard(ExpertShardForRank(N,rank))` into
  the three resident-Q4K load arms so this process admits only `[Lo,Hi)`; a sharded load on a
  non-Q4K arm is refused. Rank identity is env-driven (`FAK_EP_RANK` / `FAK_EP_COORD_ADDR`, the
  torchrun/NCCL convention). Commit `63a7e162`.
- **Cross-process reduce.** `distCommCollective` adapts `model.DistComm` (each rank holds only its
  own part) to the model `Collective` seam; the serve joins the group after load and wires it. The
  gateway skips its device-collective wiring on a rank-local model (`IsExpertParallelRankLocal`) so
  it does not clobber the DistComm reduce. Commit `c1a924f0`.
- **Witnesses.** `TestRankLocalEPForwardMatchesFullEP` (per-rank band-only models through the LIVE
  dispatch over a real DistComm socket, bit-exact vs full-model EP) + a missing-band fail-closed
  witness; `TestQwen3MoEGGUFExpertShardLoadsCorrectBandBytes` (a sharded load admits the RIGHT
  band's BYTES, not just the right tensor count, on a distinct-per-expert non-zero Q4_K payload).

**Honesty boundary — this is NOT multi-GPU.** `DistComm` reduces **host** float32 across processes
(one box over loopback, or across boxes over TCP). It proves per-band residency + correct
distributed tokens above the device line. It does NOT reduce a DEVICE tensor across GPUs — that is
the separate device-NCCL rung (the existing `BackendCollective` / `SetExpertParallelDeviceCollective`
seam, gated by `Caps().Collective`), still `not yet` and still gated on the on-box `-tags cuda,nccl`
binary (residual item 2, unchanged). The remaining residual to a live multi-GPU tok/s number is: run
the sharded serve on the GPU box (host DistComm across the A100s' processes gets residency today),
then swap the DistComm reduce for the device-NCCL tensor reduce for the on-GPU expert GEMMs.

## Update (2026-06-30 continued): multi-process device-NCCL primitive built

The previous update's closing sentence named the exact next primitive: "swap the DistComm reduce
for the device-NCCL tensor reduce." That primitive is now built and wired, opt-in, on trunk:

- **`compute.ProcessGroupBackend`** (`internal/compute/compute.go`) — a new optional HAL seam,
  distinct from `CollectiveBackend`'s single-process `ncclCommInitAll` shape. It is the
  multi-**process** bootstrap (`ncclGetUniqueId`/`ncclCommInitRank`, the torchrun/MPI convention):
  `ProcessGroupUniqueID()` (rank 0 mints an ID), `InitProcessGroup(id, world, rank, device)` (every
  rank joins with the identical ID), `AllReduceSumPG(t Tensor)` (this process's single device
  tensor, all-reduced across the group), `DestroyProcessGroup()`.
- **`internal/compute/cuda_nccl_pg.cu` + `cuda_collective_pg.go`** (`-tags cuda,nccl`) — the C/cgo
  implementation of the seam above, with its own `g_pg_comm`/`g_pg_rank`/`g_pg_world` globals kept
  separate from `cuda_nccl.cu`'s single-process communicator state (the two collectives can coexist
  in one process without sharing state).
- **`model.DistComm.BroadcastFromRoot`** (`internal/model/dist_collective.go`) — the new primitive
  that distributes rank 0's NCCL unique ID (128 bytes) to every rank over the sharded serve's
  *existing* TCP group (no new transport). Bit-exact gates: `TestDistCommBroadcastFromRootMatches`
  (sizes 1/2/3/5), plus a fail-closed op-desync witness.
- **`model.NewDevicePGCollective`** (`internal/model/dist_device_collective.go`) — bridges
  `compute.ProcessGroupBackend` to the `model.Collective` seam the EP decode path consumes, the
  device-process-group twin of `BackendCollective`.
- **`cmd/fak/serve_ep.go`'s `joinDevicePGIfSupported`**, wired into `cmd/fak/serve.go`'s sharded-EP
  block — the opt-in upgrade. It type-asserts `chatBackend` for `compute.ProcessGroupBackend`; when
  absent (every build except `-tags cuda,nccl` with a real device), it returns `(nil, nil)` and the
  serve falls through unchanged to today's `NewDistCommCollective(group)` — zero behavior change on
  every existing path. When present, it performs the ID-broadcast rendezvous over the already-open
  DistComm group and sets `SetExpertParallelCollective` to the device-PG collective instead.

**What is verified here (Go-only, no CUDA toolchain needed):** `go test ./internal/model/...`
passes (the broadcast primitive + its fail-closed gate, plus the full existing DistComm suite);
`python tools/cuda_abi_parity.py --check` passes with the new `.cu`/`.go` files recognized in the
`KERNELS`/`BINDINGS` tuples; `go build ./cmd/fak/... ./internal/model/... ./internal/compute/...`
and `go vet` are clean, including the opt-in `serve.go`/`serve_ep.go` wiring.

**What is STILL `not yet` — none of it host-codeable:**

1. **The CUDA/NCCL compile itself.** `cuda_nccl_pg.cu` and `cuda_collective_pg.go` have never been
   built — no nvcc/CUDA/NCCL toolchain exists on this dev box. Next checkable step:
   `internal/compute/build_cuda.sh test` with `FAK_CUDA_NCCL=1` on a real CUDA+NCCL host.
2. **A live multi-process, multi-GPU witness.** No `devicePGCollective` has ever run against real
   GPUs — the rendezvous, the `ncclCommInitRank` join, and `AllReduceSumPG` are unverified beyond
   their Go-level shape/error-contract match to `distCommCollective`. Next checkable step: 2+
   `fak serve --expert-parallel N` processes on distinct GPUs of the GPU-server box, built with
   `-tags cuda,nccl`, producing a bit-exact-vs-cpu-ref (or cosine-1.0-Approx, matching
   `cuda_collective.go`'s documented device-reduction-order caveat) EP decode step.

So the residual named at the end of the previous update — "swap the DistComm reduce for the
device-NCCL tensor reduce" — now has a concrete, code-complete, opt-in swap on trunk; what remains
is exclusively the hardware-gated compile-and-run step, unchanged in kind from residual item 2
above (an on-box multi-GPU `-tags cuda,nccl` binary), now sharpened to a specific test command.
