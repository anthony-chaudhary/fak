---
title: "Tensor and expert parallelism: splitting inside a layer"
description: "When one layer is still too big, cut inside it: tensor parallelism shards a matmul across workers, and expert parallelism routes MoE experts to ranks."
slug: tensor-and-expert-parallelism
keywords:
  - tensor parallelism explained
  - column parallel row parallel matmul
  - Megatron sharding
  - expert parallelism MoE
  - AllReduce AllGather collective
  - bit-exact sharding
  - fak inference ladder
date: 2026-11-18
---

# Tensor and expert parallelism

Tensor parallelism is splitting one matrix multiply across workers, and expert parallelism is routing a mixture-of-experts model's experts to different ranks. Both cut inside a layer, where pipeline parallelism cuts between layers.

*You are on **Level 4 of 5** of the [fak inference ladder](README.md).*

> **Audience.** You have read [Level 3](level-3-pipeline-parallelism.md) and you can cut a
> model's layer stack across workers. Now you ask the harder question: *pipeline
> parallelism splits the stack — but what if a single layer is still too big, or I want
> the layer itself to go faster?* You are comfortable with a matrix multiply and with the
> idea of a reduction (a sum over many terms).

> **Short answer.** Pipeline parallelism crosses a **hidden-state tensor** *between* two
> layers. Tensor parallelism crosses a **collective** *inside* one layer: you split a
> matmul's weights across ranks and recombine the pieces with an `AllGather` (concatenate)
> or an `AllReduceSum` (add). Expert parallelism is the same idea for a mixture-of-experts
> layer: instead of splitting a matmul, you split the *experts* across ranks and add their
> outputs. Every one of these paths is pinned to the single-worker answer — bit-exact at
> one rank, and within a documented floating-point tolerance above one rank — so a
> regression is a red test, not a slow drift.

## What is the difference from Level 3?

One sentence, and we will repeat it until it is boring:

> **Pipeline parallelism crosses a hidden-state tensor *between* layers; tensor
> parallelism crosses a collective *inside* a layer.**

In [Level 3](level-3-pipeline-parallelism.md) the unit of work was a *band of layers*
(`ForwardBand`, `internal/model/partition.go:122`). Worker 0 ran layers 0–15, handed the
hidden state to worker 1, which ran layers 16–31. The thing that crossed the wire was a
`[seq][hidden]` tensor — the same shape that flows through the whole stack.

Tensor parallelism (TP) does not move the cut. It keeps *one* layer on *several* workers
at once. The thing that crosses the wire is not a hidden state but a **partial result of a
matmul** — a slice of the output, or a partial sum — and the operation that recombines it
is a **collective**: a primitive where every rank contributes and every rank (or one rank)
receives a combined answer. The seam is declared in `internal/model/tensor_parallel.go:140`
as the `Collective` interface.

The two axes are orthogonal. A real serving plan is a **grid**: pipeline stages down one
axis, tensor ranks across the other. We will draw that grid in
[Level 5](level-5-the-parallelism-frontier.md).

## How do you split a matmul?

A matmul is `y = x·Wᵀ`, where `W` is a weight matrix. There are exactly two ways to cut
it, and they are not symmetric — one is bit-exact and one is not. This asymmetry is the
whole story of this rung.

### Column-parallel: shard the output features

Split `W` into row-bands `[W₀; W₁; …]`. Rank `r` computes `y_r = x·W_rᵀ` over *its* band
of output features. Then an **`AllGather`** concatenates the bands in rank order into the
full `y`.

```text
  COLUMN-PARALLEL  (shard the OUTPUT features)          y = x · Wᵀ
  ─────────────────────────────────────────────────────────────────────
        W  =  [ W₀ ]        rank 0:  y₀ = x · W₀ᵀ   ─┐
              [ W₁ ]        rank 1:  y₁ = x · W₁ᵀ    ├─ AllGather ─▶  y = y₀‖y₁‖y₂‖y₃
              [ W₂ ]        rank 2:  y₂ = x · W₂ᵀ    │   (concatenate, rank order)
              [ W₃ ]        rank 3:  y₃ = x · W₃ᵀ   ─┘
  ─────────────────────────────────────────────────────────────────────
  Each output element y[o] is still  fdot(W[o], x)  computed by exactly ONE
  rank, in the SAME inner order as the single-worker matmul.
  ⇒ BIT-EXACT.  max|Δ| = 0.
```

Why is it bit-exact? Because nothing about the *arithmetic* changed. Output element
`y[o]` is a dot product of one weight row against `x`. In the single-worker version, some
loop computes that dot product. In the sharded version, the rank that owns row `o`
computes the *same* dot product, over the *same* numbers, in the *same* order. Only the
question "which rank owns row `o`" is different. Concatenating the bands back in rank
order restores the original vector exactly. This is `ColumnParallelMatMul`
(`internal/model/tensor_parallel.go:225`), and the shard is a zero-copy view of the
weight rows (`shardWeightRows`, `internal/model/tensor_parallel.go:202`).

### Row-parallel: shard the contraction dimension

Split `W` into column-bands and `x` into matching segments. Rank `r` computes a **partial**
`y` over its slice of the contraction, and an **`AllReduceSum`** adds the partials.

```text
  ROW-PARALLEL  (shard the CONTRACTION dim)             y = x · Wᵀ
  ─────────────────────────────────────────────────────────────────────
        W = [ W₀ │ W₁ │ W₂ │ W₃ ]     x = [ x₀ │ x₁ │ x₂ │ x₃ ]
             rank 0:  p₀ = x₀ · W₀ᵀ   ─┐
             rank 1:  p₁ = x₁ · W₁ᵀ    ├─ AllReduceSum ─▶  y = p₀ + p₁ + p₂ + p₃
             rank 2:  p₂ = x₂ · W₂ᵀ    │   (add, rank order)
             rank 3:  p₃ = x₃ · W₃ᵀ   ─┘
  ─────────────────────────────────────────────────────────────────────
  The single-worker matmul computes ONE dot product over the whole row.
  The sharded version computes FOUR partial dot products and ADDS them.
  Floating-point addition is not associative: (a+b)+c ≠ a+(b+c) in general.
  ⇒ REASSOCIATED.  Drifts ~1e-6 from the monolith — a tolerance, not a bug.
```

This is the honest part. Row-parallel is **not** bit-exact against the single-worker
matmul, because it changes the *order of the additions*. The single-worker version sums
`in` products left to right; the sharded version sums four sub-sums. Those two orders give
answers that differ in the last few bits of a float32. The drift is around `1e-6` — the
same non-associativity the repo already documents for its dot product. This is
`RowParallelMatMul` (`internal/model/tensor_parallel.go:256`).

But row-parallel *is* bit-exact against a **shard-grouped reference**: the sum, in rank
order, of each shard's partial dot product. That reference is `RowParallelReference`
(`internal/model/tensor_parallel.go:286`), and it is what the gate pins. The point of the
reference is subtle and worth stating: it reproduces the reassociated reduction *on
purpose*, so the test proves the sharded path adds **nothing beyond the unavoidable
round-off** — not a logic error, not a dropped rank, not a double-counted one. A loose
"close to the monolith" bound cannot see those bugs, because they stay inside the
round-off. A bit-exact reference can.

The row-parallel shard is a real copy, not a view, because a contraction-dim band is
strided across the original rows (`shardWeightColumns`, `internal/model/tensor_parallel.go:210`).

## What is a plan?

A **plan** is a validated tiling of one dimension into contiguous, non-overlapping,
complete shards. It is dimension-agnostic: the same plan over the *output* dimension
drives a column-parallel matmul, and over the *input* dimension drives a row-parallel one.

- `TPShard` (`internal/model/tensor_parallel.go:49`) is one half-open band `[Lo,Hi)`.
- `Width` (`internal/model/tensor_parallel.go:55`) is `Hi - Lo` — how many indices this
  shard owns.
- `TPPlan` (`internal/model/tensor_parallel.go:61`) is the whole tiling: a `Dim` and a
  slice of shards.
- `NewTPPlan(dim, ranks)` (`internal/model/tensor_parallel.go:70`) builds a near-even
  tiling: the first `dim % ranks` shards get one extra index, so widths differ by at most
  one.
- `Validate` (`internal/model/tensor_parallel.go:103`) fails closed on any malformed
  tiling — a gap, an overlap, an empty shard, a mislabelled rank.

**Worked example.** Hidden size 4096, four ranks:

```text
  NewTPPlan(4096, 4)
    base  = 4096 / 4 = 1024
    extra = 4096 % 4 = 0
    shard 0: [   0, 1024)   width 1024
    shard 1: [1024, 2048)   width 1024
    shard 2: [2048, 3072)   width 1024
    shard 3: [3072, 4096)   width 1024
  ⇒ each rank holds 1024 columns. 4 × 1024 = 4096. ✓
```

Now the uneven case, 4096 across three ranks:

```text
  NewTPPlan(4096, 3)
    base  = 4096 / 3 = 1365   (integer division)
    extra = 4096 % 3 = 1
    shard 0: [   0, 1366)   width 1366   ← gets the extra index
    shard 1: [1366, 2731)   width 1365
    shard 2: [2731, 4096)   width 1365
  ⇒ 1366 + 1365 + 1365 = 4096. ✓  Widths differ by at most 1.
```

`NewTPPlan` refuses `ranks > dim` (`internal/model/tensor_parallel.go:77`): a rank with no
work is a rank the collective cannot place. `Validate` also insists that shard `i` is
labelled `Rank == i`, because rank order is what makes the `AllGather` concatenation
unambiguous (`internal/model/tensor_parallel.go:111`).

## What is the collective seam?

The `Collective` interface (`internal/model/tensor_parallel.go:140`) has exactly two
methods, one per matmul split:

- `AllGather(parts, plan)` — concatenate the per-rank output bands in rank order. Used to
  recombine a column-parallel matmul.
- `AllReduceSum(parts)` — add equal-length per-rank partials in rank order. Used to
  recombine a row-parallel matmul.

`LocalCollective` (`internal/model/tensor_parallel.go:156`) is the single-box default: it
does both in-process, exactly. `AllGather` (`:161`) checks each rank's width against the
plan before concatenating, so a mis-sized rank output is rejected at the boundary instead
of silently shifting every downstream feature. `AllReduceSum` (`:181`) adds in a **fixed
rank order** — `acc = parts[0]; acc += parts[r]` — which is what makes the result
deterministic and bit-identical to the shard-grouped reference.

The same two operations exist one layer down, over device tensors, in
`internal/compute/collective.go`. There are **four** real collectives there, and they are
the vocabulary of every parallel scheme in the field:

- `AllReduceSum` (`internal/compute/collective.go:115`) — sum the per-rank partials.
- `ReduceScatter` (`internal/compute/collective.go:132`) — sum, then give each rank its
  own contiguous `1/P` slice of the result.
- `AllGather` (`internal/compute/collective.go:153`) — concatenate the per-rank shards.
- `AllToAll` (`internal/compute/collective.go:191`) — the transpose collective: move a
  *different* shard to each peer.

They share three helpers: `collectF32` (`:35`) validates the parts (no parts, an unready
part, a non-F32 part, or a part owned by a different backend is refused at the boundary);
`reduceSumRankOrder` (`:68`) defines the rank-order sum **once**, so `AllReduceSum` and
`ReduceScatter` reduce byte-identically; and `evenShard` / `evenShardOutputs` (`:90`,
`:103`) enforce the divisibility a real communicator requires.

Two identities are pinned, and they are the reason a real NCCL/RDMA collective can be
swapped in without moving a number:

```text
  IDENTITY 1   AllGather(ReduceScatter(parts)) == AllReduceSum(parts)   byte-for-byte
               Both consume the SAME rank-order sum, so the round trip is exact.

  IDENTITY 2   AllToAll is an INVOLUTION:  AllToAll(AllToAll(parts)) == parts
               and summing AllToAll's per-rank shards == ReduceScatter.
               (NCCL's own construction of reduce-scatter.)
```

A real device collective is correct **only if it reproduces these bytes**. That is the
contract the seam enforces.

## What does the whole forward look like?

A plan and a collective are the pieces. `ForwardTP` (`internal/model/tensor_parallel_forward.go:548`)
is the assembled machine: a full prefill whose attention and dense FFN sublayers are
sharded across ranks and recombined through the collective, with every live feature
(RoPE, qk-norm, the attention gate, softcap, biases) re-entered.

`TPConfig` (`internal/model/tensor_parallel_forward.go:59`) is the knob set:

- `AttnRanks` — shard attention over the KV-head groups.
- `FFNRanks` — shard the FFN over the intermediate dimension.
- `VocabRanks` — shard the embedding table and the LM head over the vocabulary.
- `SequenceRanks` — shard the residual stream along the sequence dimension.
- `Coll` — the collective seam (nil → `LocalCollective`).
- `SharedAttentionGeometry` — consult the shared-attention shape contract before dispatch.

The attention and FFN ranks may differ; a real plan often shards attention by head and the
FFN by intermediate width independently.

The per-layer work is factored into **partials** — each rank's contribution before the
reduce:

- `tpAttnLayerPartials` (`:308`) — each rank's `[seq][hidden]` output-projection partial
  for its head band.
- `tpFFNLayerPartials` (`:378`) — each rank's `[seq][hidden]` down-projection partial for
  its intermediate band.

And the reduce helpers that combine them:

- `tpReduceWithBias` (`:463`) — reduce the partials position-by-position through the
  collective, then add the bias **once, after the reduce**.
- `tpReduceRefWithBias` (`:486`) — the bit-exact reference: rank-order summation
  (`sumPartialsRankOrder`, `internal/model/tensor_parallel.go:370`) instead of the
  collective, same once-after-reduce bias.

The bias rule is worth pausing on. A row-parallel output projection has a bias vector. If
each rank added the bias to its partial *before* the reduce, the `AllReduceSum` would add
the bias `N` times. So the bias is added exactly once, to the reduced sum. The same rule
appears for the FFN's down-projection bias.

Four guards keep the path honest:

- `clampRanks` (`:97`) — bound a requested rank count to `[1, dim]`; asking for more ranks
  than the dimension has indices uses the dimension.
- `forwardTPSupported` (`:125`) — fail closed on any architecture whose TP decomposition
  is not implemented (MLA+DSA, MoE, DenseMLP, ParallelResidual, linear attention).
- `requireTPResidentF32` (`:159`) — refuse unless the per-layer tensors are raw f32,
  because the sharding slices raw weight rows and a quantized layout cannot be addressed
  that way.
- `guardedSublayer` (`:520`) and `zeroRows` (`:504`) — the latch-and-fall-back protocol:
  the first sublayer error is latched, and that call (and every later one) yields a
  shape-safe slab of zeros so the residual add never indexes a nil row before the caller
  surfaces the error.

## What is expert parallelism?

A mixture-of-experts (MoE) layer does not have one big FFN. It has many small FFNs called
**experts**, plus a **router** that picks a few experts per token and weights their
outputs. Expert parallelism (EP) is the MoE counterpart of TP: instead of splitting a
matmul, you split the **experts** across ranks.

```text
  EXPERT PARALLELISM  (shard the EXPERTS, replicate the router)
  ─────────────────────────────────────────────────────────────────────
    8 experts, 2 ranks, top-2 routing

    rank 0 owns experts [0,4)          rank 1 owns experts [4,8)
    ┌───────────────────────┐          ┌───────────────────────┐
    │  router (REPLICATED)  │          │  router (REPLICATED)  │
    │   picks {1, 5}        │          │   picks {1, 5}        │
    │   expert 1 → local ✓  │          │   expert 1 → not mine │
    │   expert 5 → not mine │          │   expert 5 → local ✓  │
    │   partial p₀ = w₁·E₁  │          │   partial p₁ = w₅·E₅  │
    └───────────┬───────────┘          └───────────┬───────────┘
                └────────── AllReduceSum ──────────┘
                            delta = p₀ + p₁
  ─────────────────────────────────────────────────────────────────────
  The router runs on EVERY rank (replicated) and picks the SAME experts.
  Each rank computes only the picks whose expert it owns; the rest are zero.
  One AllReduceSum of the per-rank [H] partials is the whole combine.
```

The pieces:

- `ExpertParallelPlan(numExperts, ranks)` (`internal/model/expert_parallel.go:38`) tiles
  the experts into contiguous ascending bands. It is literally `NewTPPlan` over the expert
  count — the same tiling, a different dimension.
- `expertParallelPartials` (`:49`) computes each rank's `[H]` gate-weighted residual
  partial: rank `r` sums `w·expertSwiGLU(e)` over the picks whose expert falls in its band.
- `expertParallelRankPartial` (`:83`) computes only *one* rank's partial — the primitive a
  sharded multi-process serve needs, where this process holds only its own band's experts.
- `ExpertParallelDelta` (`:128`) reduces the partials through the collective.
- `expertParallelReference` (`:142`) is the rank-order bit-exact oracle.
- `ForwardEP` (`internal/model/expert_parallel_forward.go:117`) is the assembled forward,
  parameterized by `EPConfig` (`:54`).

Three facts make EP work, and each is load-bearing:

1. **Experts are assigned in contiguous ascending bands.** This is not cosmetic. The
   router returns picks in expert-ascending order, so a contiguous band means the
   reduction is a *pure associative reorder* of the monolith's sum. That is exactly what
   makes the one-rank case bit-exact.
2. **The router is replicated on every rank.** Every rank runs the same router and gets
   the same picks. A rank simply ignores the picks it does not own.
3. **The combine is one `AllReduceSum` of per-rank `[H]` partials.** Far less traffic than
   a per-expert all-to-all, because experts are independent — you are summing outputs, not
   routing tokens.

### The shared-expert trap

Some MoE models (GLM among them) have a **shared expert**: an expert that *every* token
uses, in addition to the routed ones. The shared expert is **replicated** on every rank,
not sharded. That creates a trap:

```text
  SHARED EXPERT: add ONCE, after the reduce
  ─────────────────────────────────────────────────────────────────────
    WRONG: add shared per-rank, then reduce
      rank 0:  p₀ + shared   ─┐
      rank 1:  p₁ + shared    ├─ AllReduceSum ─▶  p₀ + p₁ + 2·shared   ✗
      rank 2:  p₂ + shared    │                    (shared counted N times!)
      rank 3:  p₃ + shared   ─┘

    RIGHT: reduce the routed partials, THEN add shared once
      rank 0:  p₀   ─┐
      rank 1:  p₁    ├─ AllReduceSum ─▶  p₀+p₁+p₂+p₃  ──▶  + shared   ✓
      rank 2:  p₂    │                    (routed sum)      (once)
      rank 3:  p₃   ─┘
  ─────────────────────────────────────────────────────────────────────
```

If you added the shared expert per-rank, the `AllReduceSum` would sum it `N` times and the
layer output would be wrong by `(N-1)·shared`. The fix is `addGLMSharedExpertsOnce`
(`internal/model/expert_parallel.go:167`), which adds the replicated term **once, after the
reduce**, in place. `expertParallelGLMMoEDelta` (`:153`) is the full twin that does
routed-reduce-then-shared; `expertParallelRankLocalGLMMoEDelta` (`:184`) is the sharded
multi-process version of the same order. The comment on `addGLMSharedExpertsOnce` states
the rule in one line: "the shared expert is replicated on every rank, so adding it before
the reduce would sum it N times."

## The bit-exact algebra

This is the spine of the whole ladder. State it honestly:

```text
  ForwardTP(ranks=1)  ==  Forward                    max|Δ| = 0   BIT-EXACT
  ForwardTP(ranks=N)  ==  ForwardTP(ranks=1)         ~1e-6        reassociated
  ForwardBand split   ==  Forward                    max|Δ| = 0   BIT-EXACT (Level 3)
  Forward             ==  embedBand(ids) then ForwardBand(x, 0, NumLayers, true)
  ForwardEP(ranks=1)  ==  Forward                    max|Δ| = 0   BIT-EXACT
  ForwardEP(ranks=N)  ==  ForwardEP(ranks=1)         ~1e-6        reassociated
```

The `~1e-6` is a **tolerance the tests assert**, not a measured performance number. It is
the size of the floating-point reassociation drift, and the summation order is *pinned* to
rank order so the result is reproducible run to run.

Why is the one-rank case bit-exact? Because a one-shard plan makes every `AllGather` a
no-op concatenation and every `AllReduceSum` a single partial — the identity. The
row-parallel reassociation, the only non-bit-exact step, vanishes. And the per-rank
projection slices the weight's *output rows*, so it is the same dot-product-per-row as the
monolith.

All twins end in the same tail: `logitsFromHidden` (`internal/model/forward.go:559`) — the
final norm, the LM head matmul, the logit scale. The doc comment on `fillSeqLogits`
(`internal/model/forward.go:544`) says it plainly: `Forward` and its parallel twins
`ForwardTP`/`ForwardEP` must **end here identically**, so the tail is one function rather
than one copy per twin that could drift a norm, a head name, or a scale apart.

The tests that pin each leg:

| Claim | Test | What it asserts |
|---|---|---|
| `ForwardTP(ranks=1) == Forward` | `TestForwardTPMatchesForward` (`internal/model/tensor_parallel_forward_test.go:364`) | bit-identical logits at ranks=1; same argmax at every position across rank combos |
| `ForwardTP` fails closed | `TestForwardTPFailsClosed` (`:399`) | MoE, DenseMLP, ParallelResidual, linear-attn, MLA+DSA, and a missing f32 tensor are all refused |
| rank count clamps | `TestForwardTPRankCountClamped` (`:449`) | over-large ranks clamp, not error |
| cross-process ranks match local | `TestForwardTPDistCommRanksMatchLocalForwardTP` (`internal/model/tensor_parallel_dist_test.go:90`) | each rank owns an independent model; logits bit-equal to a two-rank `LocalCollective` `ForwardTP` |
| vocab tail is exact | `TestForwardTPVocabRanksMatchesRanks1AndForward` (`internal/model/tensor_parallel_vocab_test.go:144`) | vocab-parallel logits and every hidden layer bit-equal at ranks 2/3/5/8 |
| the HAL bridge moves no number | `TestForwardTPViaBackendCollective` (`internal/model/collective_bridge_test.go:107`) | `ForwardTP` via the backend collective is bit-equal to via `LocalCollective` |
| `ForwardEP(ranks=1) == Forward` | `TestForwardEPMatchesForward` (`internal/model/expert_parallel_forward_test.go:23`) | bit-identical logits at ranks=1; argmax-stable at ranks=2 |
| EP ranks=1 == monolith | `TestExpertParallelRanks1MatchesMonolith` (`internal/model/expert_parallel_test.go:62`) | routed delta byte-for-byte equal to the monolith |
| EP collective reduces in rank order | `TestExpertParallelCollectiveMatchesRankOrderRef` (`:106`) | `AllReduceSum` == rank-order reference at max\|Δ\|=0, ranks 1/2/4/8 |
| EP is sharding-invariant | `TestExpertParallelShardingInvariant` (`:142`) | cosine ≥ 0.99999 vs monolith across ranks |
| EP handles a zero-partial rank | `TestExpertParallelLoadImbalance` (`:178`) | all picks on one rank still gives the exact ranks=1 result |
| EP fails closed | `TestExpertParallelFailsClosed` (`:218`) | wrong-dim plan, too many ranks, non-MoE count all refused |
| shared expert added once | `TestExpertParallelGLMSharedExpert` (`:268`) | EP delta with the shared term equals the monolith |

## Why does bit-exactness matter?

It is tempting to read "bit-exact" as a curiosity — a nice property, not a design idea. It
is the design idea.

Because the parallel path is bit-exact at one rank, you can **assert the split equals the
monolith**. A regression — a dropped rank, a double-counted bias, a shared expert added
`N` times — becomes a **red test** instead of a slow drift you notice three months later
in a quality metric. The `~1e-6` reassociation is the *only* difference the tests tolerate,
and even that is pinned to a bit-exact reference so the tolerance cannot hide a logic bug.

This is why the repo has both a loose bound (vs the monolith) *and* a tight reference (vs
the rank-order sum). The loose bound catches "garbage"; the tight reference catches
"subtly wrong". You need both, and you can only have the tight one because the arithmetic
is pinned.

## Try it

```bash
# The end-to-end TP gate: bit-exact at ranks=1, argmax-stable across rank combos.
go test ./internal/model/ -run TestForwardTPMatchesForward -v

# The EP gate: one rank owns all experts, so the delta is byte-for-byte the monolith.
go test ./internal/model/ -run TestExpertParallelRanks1MatchesMonolith -v

# The shared-expert trap: the replicated term is added once, after the reduce.
go test ./internal/model/ -run TestExpertParallelGLMSharedExpert -v

# The live serve flags (both default to 1 = the unchanged monolith forward).
fak serve --tensor-parallel 4 --expert-parallel 4
```

## See also

- [Level 3 — Pipeline parallelism](level-3-pipeline-parallelism.md): the axis this rung
  cuts *inside* of.
- [Level 5 — The parallelism frontier](level-5-the-parallelism-frontier.md): how the axes
  compose, and what is witnessed versus still hardware-gated.
- [Multi-GPU tensor parallelism in fak](../multi-gpu-tensor-parallelism.md) — the full
  architecture reference for the tensor axis: the API, the collective seam, the
  cross-process transport, and the exact NCCL/RCCL swap-in point.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---
*[Full ladder (README)](README.md) · next → [Level 5 — The parallelism frontier](level-5-the-parallelism-frontier.md)*
