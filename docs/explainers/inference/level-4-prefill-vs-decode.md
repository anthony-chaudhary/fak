---
title: "Prefill vs decode: why one model has two speeds"
description: "Why the first token is slow and the rest are fast: prefill is one wide compute-bound pass, decode is many narrow memory-bound passes."
slug: prefill-vs-decode
keywords:
  - prefill vs decode
  - time to first token
  - inter-token latency
  - why is decode memory bound
  - arithmetic intensity inference
  - KV cache size calculation
  - chunked prefill
  - speculative decoding explained
date: 2026-11-18
---

# Prefill vs decode: why one model has two speeds

*You are on **Level 4 of 5** of the [inference ladder](README.md).*

> **In one breath:** Inference is a two-phase process. Prefill reads the whole prompt
> at once. Decode then makes one token at a time. The first token is slow because
> prefill does far more work.

**One line:** Prefill processes every prompt position in a single wide pass, so the
arithmetic units are busy and the cost grows with prompt length; decode processes one
new position per pass but must still read every weight, so the memory bus is the limit
and the cost is roughly flat per token. The KV cache is what keeps decode linear.

> **Audience.** A performance engineer. You can read Go, and you care about latency
> and throughput. Levels 1–3 already gave you the pipeline, the `[][]float32` residual
> stream, and the block (norm → attention → add → norm → MLP → add), including the fact
> that a second, one-token form of the block exists (`blockStep`). This rung explains
> why those two forms have completely different performance profiles.

> **Short answer.** The model does *not* do the same work per token. **Prefill** is one
> big matrix multiply over the whole prompt: many positions, one weight read, high
> arithmetic intensity, compute-bound. **Decode** is a matrix-*vector* multiply for one
> position: one position, one weight read, low arithmetic intensity, memory-bound. That
> single asymmetry explains time-to-first-token, why batching helps decode and not
> prefill, and why quantization buys decode speed but not prefill speed.

## Why is the first token so much slower than the rest?

**Question: a 2,000-token prompt takes roughly 400 ms to produce its first token, and
then roughly 15 ms for each token after that. If the model does the same work per
token, why the asymmetry?**

It does not do the same work per token. The first token is the *last* token of a
2,000-token pass. The tokens after it are each the *only* token of a one-token pass.

Here is the shape of a single request, as an illustration rather than a measurement:

```
latency timeline (one request, 2000-token prompt, 200-token answer)

  t=0        t=400ms                                    t=3.4s
  |          |                                          |
  |--prefill-|--tok--tok--tok--tok-- ... --tok--tok------|
  |  ~400 ms |  15 ms each, 200 tokens = ~3.0 s          |
             ^
             first token appears here  (TTFT)

  TTFT        = prefill time, grows with PROMPT length
  inter-token = decode step, roughly flat, grows with MODEL size
```

Two numbers fall out of that picture, and they are the two a serving engineer tunes:

- **Time to first token (TTFT)** — how long until the user sees anything. Dominated by
  prefill, so it grows with the length of the *prompt*.
- **Inter-token latency** — the gap between tokens once streaming has started. That is
  one decode step, so it grows with the size of the *model*, not the prompt.

The rest of this page is why those two numbers have different shapes.

![Prefill versus decode, side by side: prefill processes every prompt token at once as a wide matrix multiply and is compute-bound with high arithmetic intensity, producing the first token; decode processes one token at a time as a matrix-vector product and is memory-bound with low arithmetic intensity because it must read the whole KV cache each step](diagrams/prefill-vs-decode.svg)

## What are the two phases actually doing?

**Question: what is different about the pass that produces the first token?**

**Prefill** ingests the whole prompt in one pass. Every position is computed. The
entry point is `Prefill` (`internal/model/kv.go:1093`):

```go
func (s *Session) Prefill(ids []int) []float32 {
```

Its doc comment says exactly what it returns: "the logits of its LAST token (the
distribution over the first generated token)". So prefill is not "the first token" —
it is *all* the prompt tokens, and the first generated token is just the last row of
its output.

**Decode** ingests one already-chosen token and returns the next-token logits. The
entry point is `Step` (`internal/model/kv.go:1508`):

```go
func (s *Session) Step(id int) []float32 {
```

Both are thin wrappers. `Prefill` calls `prefillOwned` (`internal/model/kv.go:1129`)
and `Step` calls `stepOwned` (`internal/model/kv.go:1535`). The wrappers bracket the
call with backend-execution scope and completion bookkeeping; the work is in the
`*Owned` functions. So the two phases are the same *kind* of code with the same
ownership rules — they differ in how much data they push through, not in what they are
allowed to do.

The loop that ties them together is `Generate` (`internal/model/kv.go:1601`): prefill
once, then step repeatedly. One prefill, then N steps — the whole asymmetry in five
lines.

```go
logits := s.Prefill(prompt)
for i := 0; i < n; i++ {
	next := argmaxF32(logits)
	out = append(out, next)
	logits = s.Step(next)
}
```

```
PREFILL — one pass over the whole prompt (P positions at once)

  prompt:  [t0][t1][t2][t3] ... [t1999]
              \   \   \   \        /
               \   \   \   \      /
            +--------------------------+
            |  one wide matrix multiply |   weights read ONCE
            |  P positions in parallel  |   P columns of work
            +--------------------------+
                        |
                    logits for t1999  ->  first generated token

DECODE — one pass per new token (1 position at a time)

  step 1:  [t2000]  -> [ weights ] -> token 2001
  step 2:  [t2001]  -> [ weights ] -> token 2002
  step 3:  [t2002]  -> [ weights ] -> token 2003
  ...      weights read AGAIN, every step, for ONE column of work
```

![The KV cache as a growing table: rows are sequence positions and columns are the K and V vectors per layer; each decode step appends one row, and the new query attends over every cached row; the memory cost is two tensors times layers times heads times head dimension times positions times bytes per element](diagrams/kv-cache.svg)

## What is the KV cache, and why does it have to exist?

**Question: decode processes one token. How does that token attend to the 2,000
tokens before it?**

It reads their keys and values from a cache. Without that cache, every decode step
would recompute K and V for every previous position, and generation would be quadratic
in the number of tokens produced. The cache is what makes decode linear.

The cache is a first-class object, `KVCache` (`internal/model/kvcache.go:11`). Its doc
comment is short and carries the design, so here it is in full:

```go
// KVCache is the first-class, kernel-OWNED attention state. This is the object the
// whole fusion is about: because it lives in the Go kernel's address space (not
// behind an HTTP boundary in a vLLM process), the context-MMU can EVICT a poisoned
// token span from it (so the model physically cannot attend to that span), and the
// vDSO can reuse a cached prefix — real operations on real K/V tensors, not
// metaphors. Each layer holds K and V as flat row-major [pos * (NumKVHeads*HeadDim)]
// slices; pos[] records each entry's absolute RoPE position so eviction can compact
// the cache and relabel subsequent positions to equal a run that never saw the span.
```

Read the fields and the two-copy design becomes concrete:

```go
K       [][]float32 // [layer] -> flat, NumKVHeads*HeadDim per cached position (post-RoPE)
Kraw    [][]float32 // [layer] -> the SAME entries pre-RoPE, so a span can be repositioned
V       [][]float32
pos     []int // absolute position of each cached entry (shared across layers)
```

**Why two copies of K?** Because a key is not just a vector — it is a vector *at a
position*. RoPE rotates the key by its absolute position, so the stored post-RoPE key
bakes the position in, and you cannot un-rotate it cleanly. So the cache also keeps
`Kraw`, the same key *before* rotation. To reposition an entry you re-rotate `Kraw`
once at the new position, which is exact. The comment on `NewKVCache` states the reason
plainly: one clean rotation is "bit-exact to a fresh prefill", whereas composing two
rotations "drifts ~1e-6 — enough to flip a greedy token". `V` needs no such twin,
because values are never rotated.

The cache grows by exactly one row per decode step, per layer:

```
KV cache, one row appended per step (per layer)

  after prefill (P=4):        after decode step 1:      after decode step 2:

  pos  K row   V row          pos  K row   V row        pos  K row   V row
   0   [....]  [....]          0   [....]  [....]        0   [....]  [....]
   1   [....]  [....]          1   [....]  [....]        1   [....]  [....]
   2   [....]  [....]          2   [....]  [....]        2   [....]  [....]
   3   [....]  [....]          3   [....]  [....]        3   [....]  [....]
                               4   [....]  [....]        4   [....]  [....]
                                                         5   [....]  [....]
  pos[] = {0,1,2,3}           pos[] = {0,1,2,3,4}       pos[] = {0,1,2,3,4,5}
```

`Len` (`internal/model/kvcache.go:118`) is just `len(c.pos)`, and `pos` is shared
across layers, so the cache has one length, not one per layer.

## How does a row get into the cache?

**Question: what is the append path, and what does it record?**

There are two append entry points, and they differ in *where* the rotation happens.

`AppendKV` (`internal/model/paged_hal.go:78`) takes both the pre-RoPE key and the
post-RoPE key, because the caller already rotated:

```go
func (k *pagedHALKV) AppendKV(layer int, kRaw, kRoPE, v compute.Tensor, pos int) {
```

`AppendKVRoPE` (`internal/model/hal.go:883`) takes only the pre-RoPE key and rotates
on the device:

```go
AppendKVRoPE(layer int, kRaw, val compute.Tensor, pos, nHeads, headDim int, theta float64)
```

The device implementation documents the contract: it "stores the pre-RoPE key (so
Evict can reposition it), rotates it in place to form the post-RoPE key, and stores
that and the value row". Both paths end with the same three rows — `Kraw`, `K`, `V` —
and both record the position.

Two details are load-bearing:

- **The cache is per-layer.** Every append takes a `layer` index. A 32-layer model has
  32 independent K/V row-sets, and the position list is the only thing shared.
- **`pos[]` records the absolute position of each entry.** Not the index — the
  position. That is what lets the cache be compacted later and still describe a run
  that never saw the removed span. The append sites only push to `pos` when
  `layer == 0`, which is how one shared list stays in step with 32 per-layer row-sets.

On the in-kernel path the append is `appendKV` (`internal/model/kvcache_q8.go:423`),
and the pre-RoPE stash happens in `ropeRowQK` (`internal/model/kv.go:1070`), which
appends to `Kraw` *before* rotating `k` in place.

## Why is decode memory-bound? The arithmetic-intensity argument

**Question: decode reads the whole model to produce one token. How much arithmetic
does it do per byte it reads?**

This is the core insight of the page, and it is a division.

For one decode step:

- **Bytes read** ≈ (number of weights) × (bytes per weight), plus the KV cache.
- **FLOPs** ≈ 2 × (number of weights) × 1 token. The factor 2 is one multiply and one
  add per weight.

Divide them and the token count cancels:

```
arithmetic intensity = FLOPs / bytes
                     = (2 x weights x 1) / (weights x bytes_per_weight)
                     = 2 / bytes_per_weight
```

So the ratio does not depend on the model size at all. It depends only on how many
bytes each weight occupies.

**Worked example: a 7B model at 4 bits per weight.**

```
weights            = 7 x 10^9 parameters
bytes per weight   = 0.5            (4 bits)
weight bytes       = 7 x 10^9 x 0.5 = 3.5 x 10^9 bytes  = 3.5 GB

FLOPs per step     = 2 x 7 x 10^9 x 1 = 1.4 x 10^10 FLOP
bytes per step     = 3.5 x 10^9 bytes

arithmetic intensity = 1.4 x 10^10 / 3.5 x 10^9 = 4 FLOP/byte
```

Four FLOP per byte. At 8 bits per weight the same formula gives 2 FLOP per byte. Both
are single digits.

A modern accelerator needs on the order of 100 FLOP per byte to be compute-bound — the
ratio of its arithmetic throughput to its memory bandwidth. At 4 FLOP per byte we are
roughly 25 times below that balance point. The arithmetic units finish long before the
memory bus does, and then they wait.

```
COMPUTE-BOUND (prefill)                 MEMORY-BOUND (decode)

  weights:  [========]  read once         weights:  [========]  read once
  work:     [========]  P positions       work:     [=]        1 position
            ^^^^^^^^^^                      ^^^^^^^^^^
            many FLOPs per byte             almost no FLOPs per byte

  the arithmetic units are the limit      the memory bus is the limit
  -> add more positions: still busy       -> add more tokens: bus is shared
```

**The floor that follows.** If decode is limited by reading weights, then its time is
approximately weights-bytes divided by memory bandwidth:

```
3.5 x 10^9 bytes / 1 x 10^12 bytes per second = 3.5 x 10^-3 s = 3.5 ms
```

So roughly 3.5 ms per token is the floor for a 7B model at 4 bits on a machine with
1 TB/s of memory bandwidth. That is an estimate of the *floor*, not a benchmark: real
decode also reads the KV cache, pays attention, and runs the sampling head, so the
observed number is higher. But the floor is the part that does not move when you make
the arithmetic faster.

**What batching does.** Serve 32 requests at once and the same weight read serves 32
tokens:

```
FLOPs per step  = 2 x 7 x 10^9 x 32 = 4.48 x 10^11 FLOP
bytes per step  = 3.5 x 10^9 bytes   (the weights are read once, not 32 times)

arithmetic intensity = 4.48 x 10^11 / 3.5 x 10^9 = 128 FLOP/byte
```

The ratio rose 32 times, from 4 to 128, and crossed the balance point. Now the
arithmetic units are the limit, and the 3.5 ms of weight-reading is amortized over 32
tokens instead of one. That is why serving systems batch: batching does not make the
memory faster, it makes each byte of memory serve more work.

## Why is prefill compute-bound?

**Question: prefill reads the same weights. Why is it not memory-bound too?**

Because it multiplies those weights against many positions instead of one. The bytes
read are the same; the FLOPs are multiplied by the prompt length.

```
prefill, P = 2000 positions:
FLOPs per pass  = 2 x 7 x 10^9 x 2000 = 2.8 x 10^13 FLOP
bytes per pass  = 3.5 x 10^9 bytes

arithmetic intensity = 2.8 x 10^13 / 3.5 x 10^9 = 8000 FLOP/byte
```

Eight thousand FLOP per byte, against a balance point of about 100. Prefill is
compute-bound by a wide margin, and it stays compute-bound for any prompt longer than a
handful of tokens.

This is also why batching does not help prefill much. Prefill is already saturated on
the arithmetic units; adding another request's prompt does not make the existing work
cheaper, it just adds more work. The lever for prefill is a faster matrix multiply, not
a bigger batch.

The code reflects this. The batched prefill lane exists precisely to turn the
per-token loop into one wide multiply. `prefillBatched`
(`internal/model/prefill_batch.go:35`) processes all P prompt tokens' projections as
one GEMM, and its header says why: "each weight row read once and reused across all P
tokens" instead of "GEMV-per-token (re-streaming all 537 MB of weights P times)". The
per-token fallback is `prefillTokenLoop` (`internal/model/kv.go:1500`), which the
architecture gate `prefillNeedsTokenLoop` selects for shapes the batched lane does not
cover.

## What decisions does this change?

**Question: given all that, what should I actually do differently?**

Four decisions follow directly.

**1. Quantization speeds up decode roughly proportionally, and prefill much less.**
Decode time is weights-bytes divided by bandwidth. Halve the bytes per weight and you
roughly halve the decode floor. Prefill is compute-bound, so fewer bytes per weight
buys much less — the arithmetic is the same either way. If your workload is
output-heavy, quantization is the lever; if it is prompt-heavy, it is not.

**2. Long prompts hurt TTFT linearly; long outputs hurt total latency linearly in
tokens.** Prefill cost is roughly linear in prompt length, so a 4,000-token prompt
costs about twice a 2,000-token one before the first token appears. Decode cost is
roughly flat per token, so a 400-token answer costs about twice a 200-token one in
total time. These are two different budgets, tuned separately.

**3. Chunked prefill exists to stop a long prompt from blocking everyone else.** If one
request's prefill occupies the machine for 400 ms, every other request's decode steps
stall behind it. Chunked prefill splits a long prompt into pieces and interleaves them
with other requests' decode steps, so a long prompt costs the fleet latency in slices
rather than in one block. The trade is that the long prompt's own TTFT goes up.

**4. The KV cache becomes the memory problem at long context.** Its size is:

```
KV bytes = 2 x layers x NumKVHeads x HeadDim x positions x bytes_per_element
```

The leading 2 is K and V. `NumKVHeads` and `HeadDim` are the per-position row width,
which the code calls the stride: `kvStride` (`internal/model/kvcache.go:121`) is
`NumKVHeads * HeadDim`. `GroupSize` (`internal/model/config.go:1692`) is the other half
of that story — it is `NumHeads / NumKVHeads`, the number of query heads that share one
KV head, which is why `NumKVHeads` is smaller than `NumHeads` and why the cache is
smaller than a naive per-head cache would be.

**Worked example: a concrete config.**

```
layers          = 32
NumKVHeads      = 8
HeadDim         = 128
positions       = 32768
bytes/element   = 2      (16-bit)

KV bytes = 2 x 32 x 8 x 128 x 32768 x 2
         = 2 x 32        = 64
         = 64 x 8        = 512
         = 512 x 128     = 65536
         = 65536 x 32768 = 2,147,483,648
         = 2,147,483,648 x 2 = 4,294,967,296 bytes
         = 4.29 GB
```

4.29 GB of cache, against 3.5 GB of 4-bit weights for the 7B model above. At 32k
context the cache is *larger* than the weights. That is the whole reason paged and
evictable caches exist.

fak's own accounting is more conservative still. `internal/model/cache_geometry.go:191` charges three
f32 rows per position — post-RoPE K, pre-RoPE `Kraw`, and V:

```go
g.KVBytes = int64(s.M.Cfg.NumLayers) * int64(capTokens) * int64(s.Cache.kvStride()) * 3 * 4
```

For the same config that is 32 × 32768 × 1024 × 3 × 4 = 12.9 GB, because it keeps the
`Kraw` twin and stores everything at 32 bits. That is the honest cost of the
repositioning guarantee from the `KVCache` comment, and it is why the cache has a
quantized tier: `KVPrecisionQ8_0` packs the attended K and V rows while leaving `Kraw`
in f32, so the exactness property survives the compression.

## Why are there two block implementations?

**Question: Level 3 mentioned `blockStep`. Why does the same block exist twice?**

Because the two phases read their K and V from different places.

`attnSeq` (`internal/model/forward.go:621`) computes attention over a whole sequence.
It materializes all K and V for the sequence in local slices and attends across them:

```go
func (m *Model) attnSeq(l int, xn [][]float32, rp rope) [][]float32 {
```

Its doc comment says it computes "causal GQA attention over a whole sequence of
already-normalized inputs". This is the prefill path. It is cacheless — the comment
notes "positions are 0..seq-1 (no eviction) and the index IS the absolute position".

`blockStep` (`internal/model/kv.go:917`) computes one position and reads K and V from
the cache:

```go
func (s *Session) blockStep(l, qpos int, x, cos, sin []float32, mat matKernel) []float32 {
```

Its doc comment calls it "the single-position decoder block: pre-attn norm, q/k/v,
RoPE, cache append, causal GQA, output projection, then SwiGLU MLP". This is the decode
path.

Same math, different data source. `attnSeq` builds K and V in registers; `blockStep`
appends its own row to the cache and then reads the whole cache back. The attention
arithmetic is identical; only where the keys and values come from differs.

The code is insistent that the two must agree. `blockStep`'s comment says the block
orchestration "exists in exactly one place" and that the f32 path "stays bit-exact".
The test `TestBlockTopologyDiffersFromPreNorm` (`internal/model/arch_test.go:730`)
asserts that "the cacheless Forward path and the cached Session/blockStep path AGREE on
the same topology". That is a correctness property, not a performance one: the two
implementations are the same function. The performance difference is a consequence of
the data source, not a licence to compute something different.

## What is speculative decoding, briefly?

**Question: if decode is memory-bound, can we get more than one token per weight
read?**

Yes, by guessing. Speculative decoding drafts several tokens cheaply, verifies them in
one batched pass, keeps the accepted prefix, and rolls back the rest. The verification
pass is a prefill-shaped pass over the drafted tokens, so it reads the weights once for
several positions instead of once per position — the same amortization batching uses.

The rollback is cheap because the cache is a set of slices. `RollbackSpeculative`
(`internal/model/kv.go:1622`) is three lines of intent:

```go
// RollbackSpeculative rolls back the session's KV cache by nTokens.
// It commits accepted positions 0..R-1 and rolls back speculative branches R..K-1
// via O(1) pointer adjustments without memory copying.
```

It calls `Truncate` (`internal/model/kvcache.go:430`), which is the same O(1)
slice-header adjustment. No rows are copied; the cache simply stops claiming the
rejected positions. Level 5 covers the full trade — what the draft model costs, and
when the acceptance rate makes it worth it.

## See also

- [Level 3 — Inside one decoder block](level-3-inside-a-block.md): the block this rung
  splits into two performance profiles.
- [Level 5 — What changes at scale](level-5-at-scale.md): batching, quantization, MoE,
  and speculative decoding, each as a trade against the memory-bound fact.
- [The kernel-owned KV cache](../caching/level-4-kernel-kv-cache.md): the cache as a
  kernel object, with bit-exact span eviction.
- [Addressable KV cache](../addressable-kv-cache.md) — the paged, evictable cache in
  depth, and [how the KV cache changes as agentic context grows](../kv-cache-agentic-context.md).
- [What is a CUDA kernel?](../what-is-a-cuda-kernel.md) — "kernel" means three different
  things in this repo. [The fak/DOS glossary](../glossary.md) defines the rest.

---

*[Full ladder (README)](README.md) · prev ← [Level 3 — Inside one decoder block](level-3-inside-a-block.md) · next → [Level 5 — What changes at scale](level-5-at-scale.md)*
