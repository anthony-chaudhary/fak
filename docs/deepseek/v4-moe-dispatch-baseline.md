---
title: "DeepSeek V4 Pro MoE expert-dispatch + fused-MegaMoE prior-art baseline"
description: "Design note + synthetic benchmark PLAN for V4 Pro's all-MoE FFN (1 shared + 384 routed experts, top-6/token, expert hidden 3072): top-k routing shape, shared+routed scheduling, TP/EP/PP split, all-to-all comm assumptions, batch-size / token-grouping sensitivity, and vLLM/SGLang as the tuned baseline. Maps every claim onto real fak seams (internal/model MoE + expert-parallel, internal/ggufload glm_moe_dsa tensor map, internal/gateway DeepSeek route). No throughput number is stated as measured."
---

# DeepSeek V4 Pro MoE expert-dispatch + fused-MegaMoE prior-art baseline

Issue **#3018**. Parent epic **#3006** (native DeepSeek-V4 kernel track). This is a
**DESIGN + BENCHMARK PLAN only** — no native MegaMoE kernel lands here, and it makes **no
claim that fak beats SGLang or vLLM**. Every current-state claim is witnessed against the
exact `path:line` cited (read 2026-07-09 on `main`).

This is the canonical home under `docs/deepseek/` and is a **sibling of the other
`docs/notes/DEEPSEEK-V4-*` plan notes**:

- `docs/notes/DEEPSEEK-V4-MOE-DISPATCH-BASELINE-2026-07-08.md` — the dated first draft this
  note supersedes/canonicalizes.
- `docs/notes/DEEPSEEK-V4-ATTENTION-SEAM-MAP-2026-07-08.md` — the MLA + DSA attention half.
- `docs/notes/DEEPSEEK-V4-KV-LAYOUT-PLAN-2026-07-08.md` — the KV/latent layout half.
- `docs/notes/DEEPSEEK-V4-FP4-QUANT-PLAN-2026-07-08.md` — the FP4/FP8 expert-weight half (#3019).

MoE is the FFN axis; those siblings own attention, KV, and quantization. This note owns the
router → expert dispatch → parallelism → comm path only.

## Thesis — V4 Pro's FFN is all-MoE, and fak's live dispatch is the *naive* baseline

V4 Pro uses an **MoE FFN in (nearly) every Transformer block**: **1 always-on shared expert,
384 routed experts, per-expert intermediate hidden 3072, top-6 routed experts activated per
token**. The V4 report describes a **single fused kernel that overlaps compute, communication,
and memory access** (the MegaMoE target).

fak already carries the *skeleton* of a real MoE dispatch path — a DeepSeek-family sigmoid /
group-limited top-k router, per-expert SwiGLU, an always-on shared expert, expert-parallel
sharding with a bit-exact reduce, a `--n-cpu-moe`-style host/device split, and a bounded GPU
weight ring — **but it has no fused MegaMoE kernel and no all-to-all expert-parallel token
dispatcher**. The honest first step is therefore a dispatch *design* plus a *synthetic,
weight-free* benchmark that measures **naive per-expert dispatch vs grouped/fused scheduling**
at V4's shape, before any native expert kernel is written. The grouped side is not invented —
it is the already-in-tree `batchedExpertDelta` primitive (`moe_host_batch.go`), which is
bit-identical to the loop and collapses ~3·K dispatches per layer to 3.

## V4 Pro MoE facts that drive the design

Source: DeepSeek-V4 technical report, https://arxiv.org/abs/2606.19348 (arXiv 2606.19348v1),
per #3018 grounding. All architecture numbers below are **PROVIDER/PAPER-STATED**, not
fak-measured.

| V4 Pro MoE fact | Value | Consequence for dispatch |
|---|---|---|
| Shared experts | 1 (always-on) | fires on **every** token — hit rate 1.0; must stay resident, never paged like a routed expert |
| Routed experts | 384 | far too many for a naive per-expert loop; grouping/fusion is required, not optional |
| Activated per token | 6 | top-6 routing; all-to-all traffic scales with **6**, not 384 |
| Per-expert intermediate hidden | 3072 | per-expert GEMM inner width (`expertIntermediate()`) |
| Fused kernel | 1 kernel overlapping compute/comm/memory | the MegaMoE target — **not built in fak** |

## Top-k routing shape for 6 active experts / token

V4 Pro's router is the **DeepSeek-family sigmoid + group-limited (`noaux_tc`) top-k**, not the
Mixtral softmax router. fak already has BOTH, and the DeepSeek one is the live GLM-MoE-DSA
path:

- `internal/model/moe.go:289` `route` — the **Mixtral / Qwen3-MoE** softmax router:
  `softmax(router·xn)` over all experts → `torch.topk` (stable, low-index tie-break) → optional
  renorm. This is the wrong family for V4 but documents the exact accumulation-order contract.
- `internal/model/moe.go:586` `glmRoute` — the **DeepSeek-family** router that V4 Pro maps
  onto: per-expert **sigmoid** gate, an **`e_score_correction_bias`** added before selection
  (`mlp.gate.e_score_correction_bias`), **group-limited** selection via `NGroup` / `TopKGroup`
  (`sumTopK(group, 2)` group score, keep top groups, then top-`K` within allowed groups), raw
  sigmoid weights, optional `NormTopKProb` renorm, and a final `RoutedScalingFactor` scale.
  This IS the V4 top-k shape; the only config change is `NumExperts=384`, `NumExpertsPerTok=6`,
  and V4's group counts.
- `routePick` (`moe.go:272`) is the `(expert, weight)` pair; `glmRoute` returns picks sorted
  **expert-ascending**, which is load-bearing for the EP bit-exactness below.

Config carriers already exist (`internal/model/config.go`): `NumExperts` (`:158`),
`NumExpertsPerTok` (`:159`, the top-k = 6), `MoEIntermediateSize` (`:161`, = 3072),
`NSharedExperts` (`:162`), `FirstKDenseReplace` (`:163`, leading dense layers), `NGroup` /
`TopKGroup` (`:165`/`:166`), `RoutedScalingFactor` (`:167`). `IsMoE()` (`:817`) and
`isGLMMoeDsa()` (`:724`) are the family gates. **No new config field is required to express
the V4 shape** — this is the single strongest reuse in the plan.

## Shared + routed expert scheduling

- **Routed (top-6):** `internal/model/moe.go:378` `moeFFN.apply` runs the picked experts'
  SwiGLU and accumulates the gate-weighted sum. Its default per-expert loop (`:409`) is the
  **naive baseline** the benchmark measures against. `glmMoeFFN.apply` (`:494`) is the GLM/
  DeepSeek twin with the always-on shared add.
- **Shared (always-on):** `internal/model/moe.go:694` `glmSharedExperts` — the un-gated PLURAL
  `mlp.shared_experts.*` expert added to the routed delta every token (V4's 1 shared expert
  maps directly). Distinct from the qwen3.5 SINGULAR sigmoid-gated `mlp.shared_expert.*`
  (`moe.go:452`, `qwen35SharedExpert`) — V4 uses the un-gated plural form.
- **Known scheduling mis-fit (fence, not fix here):** `internal/model/moe_offload.go:91`
  `isExpertWeight` treats the shared expert **identically to routed** for the `--n-cpu-moe`
  split. Because the shared expert has hit-rate 1.0 it should be pinned resident, not offloaded
  like a cold routed expert. Naming this as a distinct follow-on, not repairing it in a plan.

## Grouped / fused scheduling — the real in-tree comparator

The benchmark's "grouped/fused" arm is **not hypothetical**. `internal/model/moe_host_batch.go`
already collapses the per-expert dispatch:

- `batchExpertRows` (`:22`) runs one `parFor` over the flattened `[0, K·rowsPer)` (expert, row)
  space, splitting work chunks on expert boundaries.
- `batchedExpertDelta` (`:103`) batches gate+up (shared activation) then down across all K
  picked experts — **3 dispatches per layer instead of ~3·K** — and is **bit-identical** to the
  loop (`TestBatchedExpertDeltaMatchesLoop`).
- `hostBatchedGLMExperts` (`:142`) is the Model-bound host entry; `batchedMetalExperts` (`:180`)
  is the Metal-decode twin (`q4kFusedMLPBatch`, one command buffer per layer's top-k).

This is a genuine **naive-loop vs grouped-batch** seam already living in the repo — the synthetic
benchmark drives exactly this pair at V4's 384/top-6 shape. It is **grouping**, not the paper's
**fused compute/comm/memory** kernel; that fusion remains a named gap.

## Tensor / expert / pipeline parallel split

| Axis | fak seam (verified `path:line`) | Fit for V4 |
|---|---|---|
| **Expert parallel (EP)** | `internal/model/expert_parallel.go:68` `ExpertParallelPlan`, `:85` `expertParallelPartials`, `:178` `ExpertParallelDelta`, `:130` `expertParallelRankPartial` (sharded), `:255` `expertParallelRankLocalGLMMoEDelta` | **Primary fit.** Experts tiled into contiguous ascending bands across ranks; router **replicated** (every rank picks the same top-6); each rank contributes only owned picks; one `AllReduceSum` of per-rank `[H]` partials. **Bit-exact at ranks=1** (host-witnessable, no GPU). This is the right decomposition because a 384-expert model's params are dominated by the routed experts, and MLA's shared latent KV makes head-parallel TP wrong for attention. |
| **Tensor parallel (TP)** | `internal/model/tensor_parallel.go:70` `NewTPPlan`, `:317` `TensorParallelFFN`, `:389` `TensorParallelFFNReference` | **Partial fit** — row-parallel dense FFN + reference oracle exist for the DENSE projections and the leading `FirstKDenseReplace` layers. TP of a SINGLE 3072-wide expert is wasteful (collective on a matmul that fits one device); EP is preferred for the routed body. |
| **Pipeline parallel (PP)** | *No dedicated MoE seam* | **Gap / proposed.** Layer-stage pipelining across the ~all-MoE stack is unbuilt; name it as its own rung, do not model it. |
| **Collective substrate** | `internal/model/collective_bridge.go:38` `BackendCollective` over `compute.CollectiveBackend`; `LocalCollective` default | **Fit for the seam, host-gated for the win.** Only the **cpu-ref** `CollectiveBackend` exists today (bit-exact, single box). A real cross-GPU reduce needs an NCCL/RCCL backend — **host-gated**; ranks>1 carry a multi-GPU claim only once that lands. |

Numeric contract (from `expert_parallel.go` header): EP is **bit-exact vs the monolith at
ranks=1** and matches within AllReduce reassociation round-off (~1e-6) across ranks — witnessable
on CPU with no multi-GPU hardware.

## All-to-all / expert-parallel comm assumptions

- **What fak has:** a **reduce**, not a shuffle. EP replicates the router and sums per-rank `[H]`
  residual partials via one `AllReduceSum` (`ExpertParallelDelta`, `expert_parallel.go:178`).
  This is correct and cheap because experts are independent (no shared intermediate to gather),
  but it assumes every rank can compute a partial for any pick it owns from the token's `xn`.
- **What V4/DeepEP assume and fak lacks:** a **DeepEP-style all-to-all token dispatch** — tokens
  are routed to the ranks owning their top-6 experts, computed there, then combined back. Traffic
  scales with **6 active experts × token count**, not 384. fak's reduce path does **not** shuffle
  tokens across ranks. **Named gap** (see gap table). The comm-time metric in the benchmark is
  therefore **MODELED** from the dispatch/combine byte volume, never measured on a fabric.
- **Batch-size coupling:** all-to-all efficiency is dominated by per-rank token *grouping* (see
  next section); the comm assumption is only favorable at batch sizes large enough to amortize
  the dispatch latency — which the synthetic benchmark sweeps rather than asserts.

## Batch-size sensitivity and token-grouping

The dispatch cost model has a strong batch-size knee, and the benchmark's job is to *find* it,
not claim it:

- At **batch 1 / decode**, each token fires 6 tiny expert GEMVs; the per-dispatch overhead
  (`parFor` mutex + worker wake + drain) dominates — exactly the regime `batchedExpertDelta`
  targets (3 dispatches/layer vs ~3·K).
- At **larger batches**, tokens are **grouped by destination expert** so each expert runs one
  larger GEMM over all its assigned tokens (the grouped-GEMM shape MegaMoE / CUTLASS want).
  Grouping quality depends on **expert load balance**, which the group-limited router
  (`glmRoute`) and a future EPLB placement layer control.
- **Tiered residency feeds grouping:** `internal/model/paging_ring.go:45/60/97` (`pagedRing`,
  `newPagedRing`, `matMulStaged`) is a bounded per-weight LRU GPU ring — Tier-1 of a residency
  cache whose hit-rate depends on how tightly tokens group onto a hot expert set.
- **EP memory accounting already splits replicated vs routed:** `internal/ggufload/estimate.go:100`–`155`
  plans replicated-by-dtype weights separately from per-rank **routed-expert shards**, so the
  batch/rank sizing has a real load-time accounting seam.

## Fallback to vLLM / SGLang as the tuned baseline

Per the issue's acceptance, **no tokens/sec is claimed without a named tuned baseline**. The
tuned baseline is an **external engine**, and fak already speaks to DeepSeek as a provider and
as a self-hosted route:

- **Provider route (Anthropic-compatible):** `internal/gateway/deepseek_anthropic.go` — the
  DeepSeek V4 route profile + compatibility fences (#3010); `deepseek-v4-pro` / `deepseek-v4-flash`
  model ids (`DeepSeekAnthropicModelDirect`). Usage/pricing counters are **PROVIDER-OBSERVED**
  (`deepseek_pricing.go`), never fak-authored.
- **Self-hosted baseline route (host-gated live):** `internal/gateway/deepseek_selfhost_smoke_test.go`
  — the optional live self-host smoke/readiness/streaming rungs (#3013), skipped unless the
  operator sets the live env. This is where a **vLLM/SGLang-served DeepSeek V4** is the tuned
  oracle.
- **Reasoning-content preservation / conformance:** `internal/agent/reasoning.go` `splitReasoning`
  + `deepseek_reasoning_test.go` / `deepseek_conformance_test.go` — V4's `reasoning_content` is
  preserved, the in-kernel equivalent of vLLM `--reasoning-parser`.

**Baseline naming contract (mandatory before any tokens/sec):** hardware (GPU model/count),
engine + version (vLLM or SGLang MegaMoE), precision (FP8/FP4/BF16), context length, and
concurrency. Any number lacking all five is inadmissible.

## Synthetic MoE benchmark PLAN (the witness) — WITHOUT V4 weights

A **synthetic, weight-free** benchmark, driven off the existing constructors so no V4 download
is needed:

- **Fixtures:** `internal/model/synthetic.go:154` `NewSyntheticMoE` (router + E expert SwiGLU
  triples) and `:226` `NewSyntheticGLMDsa` (the DeepSeek-family MLA+DSA+MoE layout). Configure
  `NumExperts=384`, `NumExpertsPerTok=6`, `MoEIntermediateSize=3072`, `NSharedExperts=1`, plus
  V4 group counts — deterministic LCG weights, meaningless logits, faithful **shape**.
- **Harness shape:** reuse the `BenchWorkload` / `BenchCase` record (`internal/model/bench_workload.go`)
  so token/turn provenance is consistent with the other compute benchmarks.
- **Arms compared (all weight-free):**
  1. **Dispatch-contract lock** — drive `glmRoute` at V4 shape; assert top-6 width, group-limit
     honored, `e_score_correction_bias` applied, shared-expert always-on, and the failure modes
     (bad top-k width, missing router weight, shared mis-offload).
  2. **Naive per-expert loop** (`moeFFN.apply` loop path, `moe.go:409`) vs **grouped batch**
     (`batchedExpertDelta`, `moe_host_batch.go:103`) — time the SCHEDULING only.
  3. **Batch-size / token-grouping sweep** — vary batch and expert load skew; report the knee.
- **Metrics (labels are load-bearing):** expert load balance (measured on synthetic routing),
  dispatch count, and **MODELED** per-layer p50/p95, memory-bandwidth, and comm-time (from byte
  volume, not a fabric). Any tokens/sec stays **absent** until an external tuned baseline is
  named per the contract above.
- **Placement (operator-gated):** whether this **extends** `bench_workload.go`'s harness or
  lands a **new benchmark leaf** importing the router/EP seams is flagged for operator input,
  not silently chosen. Recommendation: a new leaf so the naive-vs-grouped comparator is a first-
  class artifact.

## Prior-art matrix (the tuned baseline + MegaMoE targets)

Per #3018 grounding — SGLang V4 roadmap (https://github.com/sgl-project/sglang/issues/23602),
vLLM (https://github.com/vllm-project/vllm):

| Prior art | What it provides | Relation to fak |
|---|---|---|
| **SGLang MegaMoE** | Single fused MoE kernel overlapping compute/comm/memory | The **target + parity oracle**; fak's `moeFFN.apply` loop is the naive baseline it beats |
| **DeepEP v2 MoE dispatcher** | All-to-all expert-parallel token dispatch | The token-shuffle fak's EP reduce path lacks (`expert_parallel.go`) |
| **EPLB mapping (TopK paths)** | Expert-load-balanced placement | The placement layer above `ExpertParallelPlan` |
| **SM90 FP8 MegaMoE / w4a4** | Quantized fused MoE (grouped-GEMM) | Ties to #3019 (FP4/FP8 expert weights) |
| **CUTLASS / FlashInfer grouped-GEMM** | Grouped per-expert GEMM primitive | What a native fused kernel would call |
| **vLLM (tuned serve)** | Reference throughput baseline on real hardware | The named tuned baseline for any tokens/sec claim |

`fak sota` / `internal/sotamatrix` is the live home for this matrix; this note records the
snapshot.

## Missing seams (each named distinctly — the follow-on split)

1. **Fused MegaMoE kernel** — single kernel overlapping compute/comm/memory. `moeFFN.apply` is
   an **unfused** per-expert loop; `batchedExpertDelta` is **grouped, not fused**. *(host-gated;
   CUDA/Metal.)*
2. **All-to-all / DeepEP v2 token dispatcher** — EP reduces `[H]` partials, it does not shuffle
   tokens to expert-owning ranks. *(host-gated multi-GPU.)*
3. **EPLB expert-load-balanced placement** — no placement layer above `ExpertParallelPlan`'s
   contiguous banding; token-grouping quality is unmanaged.
4. **SM90 FP8 / FP4 grouped-GEMM fused path** — quantized fused MoE; ties to #3019. *(host-gated.)*
5. **V4-scale synthetic naive-vs-grouped benchmark** — the constructors + grouped primitive
   exist, but no driver runs them at 384/top-6 and emits the metric table. **This ticket's
   witness.**
6. **Shared-expert residency policy** — `isExpertWeight` (`moe_offload.go:91`) offloads the
   always-on shared expert like a cold routed expert; it should pin resident.
7. **Cross-GPU `CollectiveBackend`** — only cpu-ref exists; NCCL/RCCL reduce is host-gated.

## Acceptance mapping

Bullet-by-bullet against #3018's acceptance / witness criteria:

- **"top-k routing shape for 6 active experts/token"** → *Top-k routing shape* section:
  `glmRoute` (`moe.go:586`) is the DeepSeek sigmoid + group-limited top-k; `NumExpertsPerTok=6`
  (`config.go:159`). **Satisfied (design, existing seam).**
- **"shared + routed expert scheduling"** → *Shared + routed expert scheduling* section:
  routed `moeFFN.apply` (`:378`) + always-on `glmSharedExperts` (`:694`); mis-fit fenced.
  **Satisfied.**
- **"tensor/expert/pipeline parallel split"** → *TP/EP/PP split* table: EP (`expert_parallel.go:68`,
  bit-exact ranks=1) primary, TP (`tensor_parallel.go:317`) for dense, PP named as a gap.
  **Satisfied (EP/TP built, PP proposed).**
- **"all-to-all / expert-parallel comm assumptions"** → *All-to-all comm* section: fak has a
  reduce, not a shuffle; DeepEP all-to-all is a named gap; comm-time is MODELED. **Satisfied
  (assumptions + gap stated).**
- **"batch-size sensitivity and token-grouping"** → *Batch-size sensitivity* section: decode vs
  batched knee, grouped-GEMM, `pagedRing` residency, EP memory accounting. **Satisfied (plan).**
- **"fallback to vLLM/SGLang as the tuned baseline"** → *Fallback* section + prior-art matrix:
  self-host route (#3013), provider route (#3010), five-field baseline naming contract.
  **Satisfied.**
- **"SYNTHETIC MoE benchmark comparing naive vs grouped/fused WITHOUT V4 weights"** → *Synthetic
  benchmark PLAN* section: `NewSyntheticMoE` / `NewSyntheticGLMDsa` fixtures, naive loop vs
  `batchedExpertDelta`, no V4 weights. **Satisfied (plan; driver is the named follow-on witness).**
- **"NO tokens/sec claim without a tuned baseline named (hardware, engine, precision, context,
  concurrency)"** → enforced throughout: every perf figure is MODELED or host-gated, and the
  five-field naming contract gates any throughput number. **Satisfied.**
- **"dispatch design + parallelism split + synthetic benchmark plan"** (the three deliverables)
  → this document. **Satisfied.**

## Honest fences (what is NOT decided or built)

- No fused MegaMoE kernel — `moeFFN.apply` is an unfused loop; `batchedExpertDelta` is grouped,
  not fused.
- No all-to-all dispatcher — EP reduces partials, it does not shuffle tokens.
- No EPLB placement / router-affinity routing — proposed, unbuilt.
- No pipeline-parallel MoE seam — proposed.
- No cross-GPU collective — cpu-ref only; NCCL/RCCL is host-gated.
- No V4-scale synthetic benchmark **driver** yet — that is this ticket's witness.
- No claim fak beats SGLang/vLLM — the matrix names them as the oracle.
- No tokens/sec anywhere in this document — every perf figure is MODELED or host-gated.

## Next rungs (split the follow-on by backend at creation time)

Not one oversized "support V4" task — split, each filed as its own leaf when the synthetic
benchmark driver lands:

1. **CPU/mock** grouped-dispatch fixture at V4 shape (the witness above).
2. **CUDA** fused MegaMoE (wrap CUTLASS / FlashInfer grouped-GEMM, or native) — host-gated.
3. **Metal** expert path (extends `batchedMetalExperts`) — host-gated.
4. **Multi-GPU** all-to-all dispatcher (DeepEP-style) atop `ExpertParallelPlan` + NCCL/RCCL
   `CollectiveBackend` — host-gated.
5. **EPLB placement** layer above `ExpertParallelPlan`.
6. **Shared-expert residency** fix in `moe_offload.go`.

## Sources (researched July 2026)

- DeepSeek-V4 technical report — https://arxiv.org/abs/2606.19348 (arXiv 2606.19348v1). MoE
  shape: 1 shared + 384 routed experts, expert hidden 3072, top-6/token; single fused
  compute/comm/memory kernel.
- DeepSeek Anthropic-compatible API guide — https://api-docs.deepseek.com/guides/anthropic_api
  (route profile + compatibility fences).
- SGLang V4 roadmap — https://github.com/sgl-project/sglang/issues/23602 (MegaMoE, DeepEP v2
  dispatcher, EPLB TopK mapping, SM90 FP8 MegaMoE).
- vLLM — https://github.com/vllm-project/vllm (tuned-serve baseline oracle).
</content>
</invoke>
