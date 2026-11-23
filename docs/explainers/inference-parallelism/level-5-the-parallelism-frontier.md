---
title: "The parallelism frontier: what is witnessed and what is not"
description: "The honest fence on inference parallelism: what each test pins, what is not witnessed (no measured speedup, no device collective), and the open edges."
slug: the-parallelism-frontier
keywords:
  - parallelism frontier
  - what is witnessed
  - pipeline parallelism open problems
  - tensor parallelism limits
  - expert parallelism limits
  - honest fences
  - fak inference ladder
date: 2026-11-18
---

# The parallelism frontier

The parallelism frontier is the boundary between what fak's inference parallelism tests actually prove and what they do not. This page tabulates what each test pins, states plainly what is not witnessed (no measured speedup, no device collective), and names the open edges as questions.

*You are on **Level 5 of 5** of the [fak inference ladder](README.md).*

> **Audience.** You have finished [Levels 1–4](level-4-tensor-and-expert-parallelism.md).
> You know how a forward pass works, how the layer stack splits, and how a single layer
> splits. Now you want the honest answer to a different question: *what is actually
> witnessed, what is not, and where are the edges?* This rung is the fence page. It is
> deliberately more careful about what it does **not** claim than about what it does.

> **Short answer.** The repo witnesses **equality**: every parallel path is pinned to the
> single-worker answer, bit-exact at one rank and within a documented floating-point
> tolerance above one rank. It does **not** witness **speedup**. There is no measured
> pipeline-parallel or tensor-parallel throughput number in this repo, and the
> cross-process transport is witnessed for *equality with the local path*, not for
> throughput. The device collective that would make a real multi-GPU serve claimable is
> hardware-gated. The open edges — how to choose cut points, what happens when a stage
> fails mid-generation, how the pipeline bubble is filled — are named here as questions,
> not promises.

## What is witnessed today?

Every row below is a real test in the repo. The third column is the part that matters: it
says what "matches" *means* in that row, because "matches" is not one thing.

| Mechanism | Test that pins it | What the test actually asserts |
|---|---|---|
| Pipeline band split | `TestForwardBandTwoWayMatchesMonolithic` (`internal/model/forward_band_test.go:17`) | a two-way band split equals the monolithic forward — **bit-exact** |
| Separately-loaded stages | `TestPipelineSeparatelyLoadedStagesMatchMonolithic` (`internal/model/pipeline_test.go:27`) | stages loaded as separate models equal the monolith — **bit-exact** |
| Incremental decode | `TestPipelineIncrementalMatchesMonolithicSession` (`internal/model/pipeline_test.go:118`) | stepwise decode equals a monolithic session |
| Per-step logits | `TestPipelineIncrementalLogitsBitExactPerStep` (`internal/model/pipeline_test.go:159`) | each decode step's logits are **bit-exact** |
| Shared boundary refused | `TestPipelineIncrementalRejectsSharedBoundary` (`internal/model/pipeline_test.go:223`) | a boundary that shares state is refused, not silently served |
| Generate loop | `TestPipelineGenerateMatchesMonolithicSession` (`internal/model/pipeline_test.go:363`) | the full generate loop equals a monolithic session |
| Cross-process transport | `TestTCPTransportMatchesLocal` (`internal/model/pipeline_test.go:534`) | `TCPTransport` equals `LocalTransport` — **equality, not throughput** |
| Stage worker | `TestPipelineEngineAdmitRunsServeBandWorker` (`internal/modelengine/pipeline_test.go:19`) | the engine admits and runs a band worker |
| TP end to end | `TestForwardTPMatchesForward` (`internal/model/tensor_parallel_forward_test.go:364`) | **bit-exact** at ranks=1; same argmax at every position across rank combos |
| TP fails closed | `TestForwardTPFailsClosed` (`:399`) | unsupported architectures and a missing f32 tensor are refused |
| TP rank clamp | `TestForwardTPRankCountClamped` (`:449`) | over-large rank counts clamp, not error |
| TP cross-process | `TestForwardTPDistCommRanksMatchLocalForwardTP` (`internal/model/tensor_parallel_dist_test.go:90`) | independent per-rank models match a two-rank local `ForwardTP` — **bit-exact** |
| Vocab-parallel tail | `TestForwardTPVocabRanksMatchesRanks1AndForward` (`internal/model/tensor_parallel_vocab_test.go:144`) | logits and every hidden layer **bit-exact** at ranks 2/3/5/8 |
| HAL collective bridge | `TestForwardTPViaBackendCollective` (`internal/model/collective_bridge_test.go:107`) | the backend collective is **bit-equal** to `LocalCollective` |
| EP end to end | `TestForwardEPMatchesForward` (`internal/model/expert_parallel_forward_test.go:23`) | **bit-exact** at ranks=1; argmax-stable at ranks=2 |
| EP ranks=1 | `TestExpertParallelRanks1MatchesMonolith` (`internal/model/expert_parallel_test.go:62`) | routed delta **byte-for-byte** equal to the monolith |
| EP reduce order | `TestExpertParallelCollectiveMatchesRankOrderRef` (`:106`) | `AllReduceSum` == rank-order reference at max\|Δ\|=0, ranks 1/2/4/8 |
| EP sharding invariant | `TestExpertParallelShardingInvariant` (`:142`) | cosine ≥ 0.99999 vs monolith across ranks — **reassociated** |
| EP load imbalance | `TestExpertParallelLoadImbalance` (`:178`) | a zero-partial rank still gives the exact ranks=1 result |
| EP fails closed | `TestExpertParallelFailsClosed` (`:218`) | wrong-dim plan, too many ranks, non-MoE count refused |
| EP shared expert | `TestExpertParallelGLMSharedExpert` (`:268`) | the replicated shared term is added once, after the reduce |

Read the third column as a spectrum, not a yes/no:

```text
  BIT-EXACT (max|Δ| = 0)          REASSOCIATED (~1e-6 tolerance)
  ────────────────────────        ────────────────────────────────
  ranks=1 for every path          ranks>1 for the row-parallel blocks
  column-parallel matmul          row-parallel matmul
  vocab-parallel tail             EP across expert bands
  sequence-parallel               TP/EP end-to-end above one rank
  pipeline band split
  the HAL collective bridge
  ────────────────────────        ────────────────────────────────
  "the split adds NOTHING"        "the split adds ONLY float round-off"
```

The `~1e-6` figure is a **tolerance the tests assert**, not a measured result. It is the
size of the floating-point reassociation drift, and the summation order is pinned to rank
order so the result is reproducible.

## What is NOT witnessed?

This is the part most pages skip. Be specific.

**There is no measured pipeline-parallelism speedup in this repo.** Pipeline parallelism
buys **fit** — it lets a model too big for one worker run across several. It does not, by
itself, make anything faster, and the **bubble** is real: while stage 0 works on
microbatch 2, stage 1 may be idle waiting for microbatch 1. The repo witnesses that a cut
stack gives the *same answer*; it does not witness that the cut stack is *faster*.

**There is no measured tensor-parallel speedup claim here either.** TP shards a matmul
across ranks, but every sharded block pays a collective — an `AllGather` or an
`AllReduceSum` — and on real hardware that collective is a network round trip. Whether the
sharded layer is faster than the single-worker layer depends on the compute-to-communication
ratio, which is a hardware property this repo does not measure.

**The `TCPTransport` is witnessed for equality with the local path, not for throughput.**
`TestTCPTransportMatchesLocal` (`internal/model/pipeline_test.go:534`) asserts that the
socket transport produces the same bytes as the in-process transport. It says nothing about
how fast the socket is.

**The device collective is hardware-gated.** `LocalCollective` is the single-box, bit-exact
default. A real cross-device reduction needs an NCCL/RCCL `CollectiveBackend` — a backend
that implements the collective methods behind the seam. The seam is proven (the HAL bridge
test shows a backend collective moves no number), but the *device* backend and a live
multi-GPU run are not shipped here. The serve flags `--tensor-parallel` and
`--expert-parallel` require an initialized non-cpu-ref collective backend; a CUDA build
provides that only with the NCCL build tag on a box with enough visible GPUs.

**No performance numbers are claimed anywhere on this ladder.** If you see a number in
this rung, it is a tolerance or a count, never a measurement.

## What are the open edges?

Named as questions, because that is what they are.

**How do you choose the cut points?** `NewPartitionPlan(cfg, cuts)`
(`internal/model/pipeline.go:68`) takes the cut points as **input**. Nothing in this repo
*searches* for a good cut. A cut that balances layers evenly is not the same as a cut that
balances *work* evenly, and a cut that balances work is not the same as one that minimizes
the bubble. The choice is the operator's, and the repo does not yet have an opinion.

**What happens when a stage fails mid-generation?** A pipeline stage is a worker. If it
dies while a generation is in flight, the hidden state in transit is lost. The repo has
`Validate` (`internal/model/pipeline.go:101`) for a malformed plan and fail-closed guards
for unsupported architectures, but the *runtime* failure of a live stage mid-generation is
an open edge, not a solved problem.

**How is the bubble filled in practice?** The standard answer is micro-batching: split a
batch into microbatches and pipeline them so no stage is idle. The repo has the incremental
decode machinery (`StepPipelineDecode`, `internal/model/pipeline.go:474`) and the generate
loop (`PipelineGenerateIncremental`, `:514`), but the *scheduling* that fills the bubble —
how many microbatches, in what order — is not a witnessed end-to-end path here.

**How do PP and TP compose?** They are orthogonal axes and the repo has both. A real plan
is a grid: pipeline stages down one axis, tensor ranks across the other. But the
*composition* is not a single witnessed end-to-end path. Each axis is proven against the
monolith on its own; the grid is the design, not yet one test.

```text
  THE TWO ORTHOGONAL AXES  (a real plan is a grid)
  ─────────────────────────────────────────────────────────────────────
                    TP rank 0      TP rank 1      TP rank 2
                  ┌─────────────┬─────────────┬─────────────┐
   PP stage 0     │  layers     │  layers     │  layers     │   each cell
   (layers 0-15)  │  0-15,      │  0-15,      │  0-15,      │   holds a
                  │  shard 0    │  shard 1    │  shard 2    │   SLICE of
                  ├─────────────┼─────────────┼─────────────┤   the layer
   PP stage 1     │  layers     │  layers     │  layers     │
   (layers 16-31) │  16-31,     │  16-31,     │  16-31,     │
                  │  shard 0    │  shard 1    │  shard 2    │
                  └─────────────┴─────────────┴─────────────┘
                    ▲                                        ▲
                    │  PP crosses a hidden state BETWEEN stages (down)
                    │  TP crosses a collective INSIDE a layer (across)
  ─────────────────────────────────────────────────────────────────────
  Each axis is proven against the monolith alone. The GRID is the design;
  the composed end-to-end path is an open edge.
```

**What about the architectures TP does not yet shard?** `forwardTPSupported`
(`internal/model/tensor_parallel_forward.go:125`) fails closed on MLA+DSA, MoE, DenseMLP,
ParallelResidual, and linear attention. Each is a named later sub-lever, not a silent gap.
The MLA case is instructive: its latent KV is shared across heads, so head-parallel TP is
simply the *wrong* decomposition for it — a different one is needed.

## Where do I go next?

- [Multi-GPU tensor parallelism in fak](../multi-gpu-tensor-parallelism.md) — the full
  architecture reference for the tensor axis: the API, the four-collective seam, the
  cross-process transport, and the exact NCCL/RCCL swap-in point. This ladder introduces
  the idea; that page is the setup document.
- [The fak caching ladder](../caching/README.md) — the sibling ladder. Caching is about
  *not recomputing* work you have already done; inference is about *doing* the work. They
  meet at the KV cache.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.
- [Addressable KV cache](../addressable-kv-cache.md) · [Context](../context.md) — the
  cache and context layers the serving path sits on top of.

## What would you check?

Three questions you can now answer from source. Each names the file to look in.

1. **A row-parallel matmul is not bit-exact against the monolith. What *is* it bit-exact
   against, and why does that reference exist?** Look in
   `internal/model/tensor_parallel.go` at `RowParallelReference` (`:286`) and read the
   comment on `sumPartialsRankOrder` (`:370`). The answer is about what a loose bound
   cannot see.

2. **If you added the GLM shared expert to each rank's partial *before* the reduce, by what
   factor would the layer output be wrong?** Look in `internal/model/expert_parallel.go` at
   `addGLMSharedExpertsOnce` (`:167`) and `expertParallelGLMMoEDelta` (`:153`). The answer
   is `(N-1)·shared`, where `N` is the rank count.

3. **`ForwardTP` and `ForwardEP` are different decompositions. What single function must
   they both end in, and why does that matter?** Look in `internal/model/forward.go` at
   `logitsFromHidden` (`:559`) and the comment on `fillSeqLogits` (`:544`). The answer is
   what makes the whole ladder a story about *the same computation, cut differently*.

## Try it

```bash
# The pipeline equality witness: a cut stack gives the same answer as the monolith.
go test ./internal/model/ -run TestPipelineSeparatelyLoadedStagesMatchMonolithic -v

# The cross-process transport: equality with the local path (not throughput).
go test ./internal/model/ -run TestTCPTransportMatchesLocal -v

# The HAL collective bridge: a backend collective moves no number.
go test ./internal/model/ -run TestForwardTPViaBackendCollective -v
```

## See also

- [Level 4 — Tensor and expert parallelism](level-4-tensor-and-expert-parallelism.md): the
  mechanisms this rung fences.
- [Level 3 — Pipeline parallelism](level-3-pipeline-parallelism.md): the axis the grid
  composes with.
- [Multi-GPU tensor parallelism in fak](../multi-gpu-tensor-parallelism.md) — the
  architecture reference, including the hardware-gated residual.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---
*[Full ladder (README)](README.md)*
