---
title: "DeepSeek V4 Pro MoE dispatch + fused MegaMoE prior-art baseline"
description: "Design note and benchmark plan mapping DeepSeek V4 Pro's all-MoE dispatch requirements onto existing fak seams; no native MegaMoE kernel lands."
---

# DeepSeek V4 Pro MoE dispatch + fused MegaMoE prior-art baseline

**2026-07-08.** Issue **#3018**, parent epic **#3006** (native DeepSeek-V4 kernel track).
**Design note + benchmark plan only** — no native MegaMoE kernel lands here, and no claim
that fak beats SGLang/vLLM. Current-state claims are witnessed against the exact `path:line`
cited (read 2026-07-08 on `main`). Companion to the MoE row of
`docs/notes/CONCEPT-MODEL-PROGRESS-CACHING-TAXONOMY-2026-07-07.md` (proposed M11).

## Thesis — V4 Pro's FFN is all-MoE; a naive per-expert loop is the wrong shape

V4 Pro uses **MoE layers in every Transformer block**: **1 shared expert, 384 routed
experts, intermediate hidden 3072, 6 routed experts activated per token**. The report calls
out a **single fused kernel that overlaps compute, communication, and memory access**. fak
already has the *skeleton* of a real MoE dispatch path (router → top-k → per-expert SwiGLU,
expert-parallel sharding, CPU-offload split, a paged expert ring), but it has **no fused
MegaMoE kernel and no all-to-all expert-parallel dispatcher**. So the honest first step is a
dispatch design + a synthetic benchmark that compares *naive* vs *grouped/fused* scheduling
at V4's shape — before any native expert kernel.

## The V4 MoE facts that drive the design (from the issue grounding)

Source: DeepSeek V4 technical report, https://arxiv.org/html/2606.19348v1 (per #3018 Grounding).

| V4 Pro MoE fact | Value | Consequence for dispatch |
|---|---|---|
| Shared experts | 1 (always-on) | fires on **every** token — hit rate 1.0, must stay resident |
| Routed experts | 384 | too many for a naive per-expert loop; grouping/fusion required |
| Activated per token | 6 | top-6 routing shape; all-to-all traffic scales with 6, not 384 |
| Expert intermediate hidden | 3072 | per-expert GEMM size |
| Fused kernel | 1 kernel overlapping compute/comm/memory | the MegaMoE target |

## Seam map — V4 MoE requirement → fak seam (`path:line`) or proposed

| V4 requirement | Nearest fak seam (verified `path:line`) | Fit / gap |
|---|---|---|
| **Router → top-k selection** (top-6 of 384) | `internal/model/moe.go:277` `route`, `:272` `routePick`; GLM path `:586` `glmRoute` | **Fit** — top-k router with tie-break matching `torch.topk`; `NumExperts`/`NumExpertsPerTok` (`config.go:159`) carry 384/6 as config. |
| **Per-expert SwiGLU dispatch** | `internal/model/moe.go:378` `moeFFN.apply` | **Fit for the math**, but it is the **naive per-expert loop** — the baseline the benchmark measures *against*, not the fused target. |
| **Shared-expert scheduling** (always-on) | `internal/model/moe.go:694` `glmSharedExperts` | **Fit** — a distinct always-on path already exists; V4's 1 shared expert maps directly. (Note: `moe_offload.go:91` `isExpertWeight` currently offloads shared experts *identically* to routed — a known mis-fit for a residency cache, see M11 in the taxonomy note.) |
| **Expert-parallel (EP) sharding + reduce** | `internal/model/expert_parallel.go:68` `ExpertParallelPlan`, `:76` `expertParallelPartials` | **Strong fit** — experts tiled into contiguous bands across ranks, router replicated, one `AllReduceSum` of per-rank `[H]` partials; **bit-exact at ranks=1**. This is the multi-GPU decomposition V4 needs. |
| **All-to-all / DeepEP-style dispatcher** | *No seam yet* — EP today reduces partials, not an all-to-all token shuffle | **Gap.** DeepEP v2's all-to-all token dispatch is not in fak; fence it. |
| **Fused MegaMoE kernel** (overlap compute/comm/mem) | *No seam yet* — `moeFFN.apply` is an unfused loop | **Gap.** The single fused kernel is the explicit out-of-scope follow-on. |
| **Tiered expert residency** (GPU/pinned-CPU/SSD) | `internal/model/paging_ring.go:45` `pagedRing`, `:60` `newPagedRing`, `:97` `matMulStaged`; CPU split `internal/model/moe_offload.go:25` `splitKernel`, `:91` `isExpertWeight` | **Fit** — a bounded per-weight LRU GPU ring (Tier-1 of a 3-tier cache) and a `--n-cpu-moe`-style host/device split both exist; token-grouping feeds their residency. |
| **EP-aware memory accounting** (replicated vs routed shards) | `internal/ggufload/estimate.go:119`–`151` (replicated-by-dtype vs routed-expert-shard plans) | **Fit** — load-time accounting already splits replicated weights from per-rank routed-expert shards. |
| **EPLB placement / router-affinity routing** | *No seam yet* — proposed (M11 sub-row B3 in the taxonomy note) | **Gap.** Expert-load-balanced placement is unbuilt; fence it. |

## Prior-art matrix (the tuned baseline)

Per the issue grounding (SGLang V4 roadmap, https://github.com/sgl-project/sglang/issues/23602):

| Prior art | What it provides | Relation to fak |
|---|---|---|
| **SGLang MegaMoE** | The fused MoE kernel (overlaps compute/comm/mem) | The **target + parity oracle**; fak's `moeFFN.apply` is the naive baseline it beats. |
| **DeepEP v2 MoE dispatcher** | All-to-all expert-parallel token dispatch | The dispatcher fak's `expert_parallel.go` reduce path lacks. |
| **EPLB mapping (TopK paths)** | Expert-load-balanced placement | The placement layer above `ExpertParallelPlan`. |
| **SM90 FP8 MegaMoE / w4a4 MegaMoE** | Quantized fused MoE | Ties to #3019 (FP4 expert weights). |
| **CUTLASS / FlashInfer grouped-GEMM** | Grouped per-expert GEMM primitives | What a native fused kernel would call. |

`fak sota` / `internal/sotamatrix` is the live home for this matrix; this note records the
snapshot.

## Benchmark plan (the witness)

A **synthetic, weight-free** MoE benchmark that:
1. Drives `route`/`glmRoute` at V4 shape (384 experts, top-6, shared=1) to lock the
   **dispatch contract** and its failure modes (bad top-k width, missing router weight,
   shared-expert mis-offload).
2. Compares **naive per-expert dispatch** (`moeFFN.apply`) vs **grouped/fused scheduling**
   (token-grouping into per-expert batches) — timing only the scheduling, no V4 weights.
3. Reports the minimum metrics named in the issue: **expert load balance, tokens/sec, memory
   bandwidth, communication time, p50/p95 per layer**.

Acceptance gate is **open**: the issue asks whether this extends an existing MoE benchmark
fixture or is a new package. `internal/model` already carries `bench_workload.go`; the
recommendation is a new benchmark leaf that imports the router/EP seams above, flagged for
operator input rather than silently chosen.

## Honest fences (what is NOT decided or built)

- **No fused MegaMoE kernel** — `moeFFN.apply` is an unfused per-expert loop (the baseline).
- **No all-to-all dispatcher** — EP reduces partials, it does not shuffle tokens (DeepEP gap).
- **No EPLB placement / router-affinity routing** — proposed, unbuilt.
- **No V4-scale synthetic benchmark yet** — that is this ticket's witness, named above.
- **No claim fak beats SGLang/vLLM** — the matrix names them as the oracle.

## Next rungs (split the follow-on by backend, at creation time)

Per the issue's done-condition, the native follow-on is **not** one oversized "support V4"
task — it splits into:
1. **CPU/mock** grouped-dispatch fixture (the witness above).
2. **CUDA** fused MegaMoE (wrap CUTLASS/FlashInfer grouped-GEMM, or native).
3. **Metal** expert path.
4. **Multi-GPU** all-to-all dispatcher (DeepEP-style) atop `ExpertParallelPlan`.
Each is filed as its own leaf when the synthetic benchmark lands — not silently deferred.
