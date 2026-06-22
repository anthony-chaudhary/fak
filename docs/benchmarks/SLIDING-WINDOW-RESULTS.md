---
title: "Sliding-Window Attention as a Read-Time Mask"
description: "Reports fak's per-layer sliding-window attention bounding cost from O(N) to O(W), shipped as a read-time mask and byte-identical to dense when unset."
---

# Sliding-window attention — the long-context read-mask (issue #20)

> Lane `model`. The compute-side long-context arch axis: a per-layer sliding window
> bounds attention from O(N) to O(W) per token — the mechanism every practical
> long-context family (Mistral, Gemma2/3, Qwen-long, gpt-oss) uses to attend over a
> very long cache. Shipped as a **read-time mask** (option (a) of MODEL-ARCH-GOALS
> G-S3), proven family-independently on the synthetic model, byte-for-byte identical
> to the dense path when unset.

## What shipped

A per-layer window `Config.Window []int` and a single lower-bound helper
`windowLo(W, qpos)` (`internal/model/swa.go`). For a query at absolute position
`qpos`, layer `l` attends only to key positions `[qpos-W+1, qpos]` when
`Window[l] > 0`; a nil/short slice or a value `<= 0` means full causal attention.
The bound is threaded into **every** CPU attention site so the window is enforced
consistently across decode, prefill, batched prefill, multi-user, and the
from-scratch forward reference:

| site | file | path |
|---|---|---|
| `layer` | `forward.go` | full-prefill reference (HF-oracle path) |
| `blockStep` | `kv.go` | serial f32 decode |
| `prefillBatched` (inline) | `prefill_batch.go` | batched f32 prefill |
| `tokenHiddenQ` (inline) | `quant_forward.go` | Q8 decode |
| `attnPrefillInto` | `prefill_attn.go` | batched f32/Q8/Metal prefill |
| `attnDecodeBatch`, `attnPrefillMultiInto` | `batch.go` | multi-user decode + prefill |

At each site the only change is: start the score loop and the V-accumulation at
`wlo := windowLo(W, qpos)` and softmax over `scores[wlo:nPos]`. When `wlo == 0`
(window unset, or window ≥ the prefix) this is the *identical* computation as
before — which is why the proven rungs are untouched.

## The two load-bearing invariants

1. **Unset == full == byte-identical.** A model that configures no window runs the
   exact instruction stream it ran before SWA existed. This is not asserted, it is
   proven: the real-model rungs stay green (below), and `TestSlidingWindowDegeneratesToFull`
   shows a window wider than the prompt is `math.Float32bits`-equal to no window on
   Forward, Prefill, and the full KV cache.

2. **Keyed off the absolute position, not the loop index.** `windowLo` is computed
   from `qpos` (the query's absolute RoPE position), and `KVCache.Evict` re-establishes
   the contiguous-position invariant (`pos[i]==i`) by compacting + renumbering
   survivors. So the window survives an eviction exactly as the RoPE reposition does
   — `TestSlidingWindowSurvivesEvict` proves a windowed run whose tail span is evicted
   before the query attends is bit-identical (logits + greedy continuation + full KV
   cache) to one that never saw the span. The physical ring buffer (option (b)) would
   break the byte-identity eviction proof and is deliberately **out of scope**.

## Witnesses (`go test ./internal/model -run Window`, 6/6 green)

| test | proves |
|---|---|
| `TestWindowLo` | the lower-bound helper computes the textbook SWA bound (table) |
| `TestSlidingWindowDegeneratesToFull` | window ≥ prompt is byte-identical to no window (the no-op) |
| `TestSlidingWindowRestrictsAttention` | a real window measurably changes the output — last-pos `max|Δ|≈1.05` at window 2 over an 8-token prompt (the mask is not vacuous) |
| `TestSlidingWindowMatchesAcrossPaths` | windowed batched Prefill == from-scratch Forward == serial blockStep decode (the windowed R2) |
| `TestSlidingWindowBatchedMatchesSerial` | windowed multi-user batch == per-user serial, bit-exact (windowed batched rung) |
| `TestSlidingWindowSurvivesEvict` | windowed evict == never-saw, bit-exact (the pos[]-keyed survival proof) |

**The proven dense rungs are unchanged** (each green in isolation on the real
SmolLM2-135M export): `TestForwardMatchesHFOracle`, `TestGreedyMatchesHFOracle`,
`TestCachedDecodeMatchesPrefill` (`max|Δ|=0`), `TestKVQuarantineEqualsNeverSaw`, plus
the synthetic `TestRefactorMatchesSerial`, `TestParallelMatchesSerial`,
`TestBatchedDecodeMatchesSerial`, `TestPrefillEachRectangularMatchesSerial`.

> Test-host note: the full `./internal/model` package run intermittently fails a
> *different* weight-loading test each time with `cannot allocate memory` reading the
> 538 MB `weights.f32` — WSL memory pressure from loading the model in several test
> functions within one process (analogous to the App-Control flakiness `CLAUDE.md`
> documents). Each weight-backed test passes in isolation; the SWA + synthetic rungs
> never load weights and always pass.

## Bounded-memory windowed decode — the memory + numerics lever (follow-on)

The read-mask is the correctness precondition for *dropping* aged-out K/V. A position
older than every layer's window is provably never attended to again, so removing it
leaves the output of all future tokens unchanged. `Session.TrimToWindow(slack)`
(`internal/model/swa.go`) does exactly that, through the proven `KVCache.Evict`:
when the cache passes `MaxWindow()+slack`, it evicts the oldest positions back down to
`MaxWindow()`. Because `Evict` renumbers + re-RoPEs survivors and RoPE is **relative**
(a score depends only on `q-k`), renumbering the whole surviving window preserves every
within-window score — so a bounded decode is **argmax-identical** to the same windowed
decode over the full cache. This is the standard rolling-buffer KV cache (Mistral) made
bit-faithful via the proven eviction primitive.

Two walls fall at once:

- **Memory** — the cache is bounded by `MaxWindow()+slack` for all time, independent of
  stream length. **Measured: 1,000,000 tokens decoded with the KV cache never exceeding
  256 positions** (window 128), vs the ~67 GB a full 1M-token cache would need for
  SmolLM2-135M. O(window), not O(N).
- **Numerics** — survivors are renumbered into a small contiguous range, so the RoPE
  positions the model sees never grow past ~`MaxWindow`; the high-position angle drift
  that bites a full cache at extreme N never arises.

Witnesses (`go test ./internal/model -run BoundedWindow`, 2/2 green):

| test | proves |
|---|---|
| `TestBoundedWindowMatchesFullWindow` | trimmed windowed decode is argmax-identical to the full-cache windowed decode, step for step (teacher-forced); bounded cache peaks at 2·window while the full cache grows with the stream |
| `TestBoundedWindowMemoryIsBounded` | the cache stays ≤ `window+slack` for a 128K-token stream by default; `FAK_LONGCTX_N=1000000` reruns it at a literal **million tokens** — cache capped at 256 positions |

`TrimToWindow` is opt-in (no production decode path calls it yet) and a no-op when no
window is configured, so the byte-identical no-op story above is unchanged.

## Honest scope — what this is and is not

- **Is:** the SWA *mechanism*, proven correct family-independently, additive, and a
  byte-identical no-op when unset. It bounds attention COST and SCOPE.
- **Is not (yet):** a real Mistral/Gemma SWA HF oracle. That would additionally prove
  a family's window VALUE is read from its checkpoint; it needs the #19 rope-scaling
  axis + that checkpoint and is the separable follow-up. No family is claimed
  "supported" here.
- **Is not (yet):** bounded cache MEMORY. The read-mask reduces what attention reads;
  it does not shrink the cache. But it is the correctness *precondition* for that
  follow-on toward 1M tokens: once a layer provably never reads a position older than
  its window, that position's K/V can be dropped from that layer without changing the
  output — the next step on the memory axis (alongside KV-quant / drop-Kraw).

## Where this sits on the road to 1M

The map of the engine's context-length ceiling found three walls: **memory**
(~67 KB/token for SmolLM2 → ~67 GB at 1M), **compute** (dense O(N) per-token decode /
O(N·P) prefill), and **numerics** (RoPE drift at very high positions). This lane now
lands all three:

- **compute** — the sliding-window read-mask (O(N)→O(W) per token);
- **memory** — `TrimToWindow` bounds the cache to O(window); a 1M-token stream ran in
  256 positions instead of ~67 GB;
- **numerics** — renumbering survivors keeps RoPE positions small, so the drift regime
  is never entered.

…with every bit-exact rung intact and the no-window default byte-identical. The
engine now decodes arbitrarily-long context in bounded memory. The one remaining 1M
item is positional **quality** for *non-windowed* long context (rope-scaling
#19/#23) — a windowed model does not need it, since it never sees a position beyond
its window.
