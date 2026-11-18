---
title: "What changes at scale: batching, quantization, MoE, and speculative decoding"
description: "Level 5 of the inference ladder. Continuous batching, paged KV, quantization, mixture-of-experts, speculative decoding, sampling at scale, and tensor vs pipeline parallelism — each as what it changes, what it costs, and what it does not change. Ends with the honest open edges."
slug: inference-at-scale
keywords:
  - LLM inference at scale
  - continuous batching explained
  - paged KV cache
  - mixture of experts inference
  - speculative decoding
  - quantization for inference
  - tensor parallelism vs pipeline parallelism
  - top-k top-p sampling at scale
date: 2026-11-18
---

# What changes at scale

*You are on **Level 5 of 5** of the [inference ladder](README.md).*

> **In one breath:** At scale, one model serves many users at once. Batching,
> quantization, and expert routing each change one thing. Each trades memory,
> capacity, or latency for throughput. None of them change what the model
> computes.

**One line:** Every technique on this rung is a trade against the single fact that
decode is memory-bandwidth-bound — batching amortizes one weight read over many
sequences, quantization shrinks the bytes read, expert routing reads fewer weights
per token, and speculative decoding amortizes one read over several tokens — and
the honest question is always what each one leaves unchanged.

> **Audience.** A researcher or systems lead. You already know the pipeline, one
> decoder block, prefill vs decode, the KV cache, and arithmetic intensity from
> Levels 1–4. This rung assumes all of it and does not re-teach it.

> **Short answer.** Every technique here is a trade, not a free win. Each changes
> *one* thing — throughput, memory, capacity, latency — and leaves the rest alone.
> The useful question is never "is this faster?" but "faster at what, paid for with
> what, and what does it leave unchanged?" The last section names what is genuinely
> unsolved, because a frontier page that only lists wins is marketing.

## The shape of every answer on this page

Level 4 gave you the one fact that makes all of this legible: **decode is
memory-bandwidth-bound.** To produce one token you must read the model's weights
and the whole KV cache, and you do almost no arithmetic per byte read. So the cost
of a decode step is roughly *bytes you must read*, not *operations you must do*.

That single fact explains why batching helps (Level 4), why quantization helps
(fewer bytes), why MoE helps (read fewer weights), and why speculative decoding
helps (amortize one weight read over several tokens). Each section below answers
the same three questions: **what it changes**, **what it costs**, and **what it
does NOT change**.

---

## 1. Continuous batching — what if we never wait for a batch to finish?

**The question.** Level 4 showed that batching raises arithmetic intensity: run 32
requests together and each weight read serves 32 tokens instead of one. But a
naive batcher waits for the *whole* batch to finish before starting the next one.
Requests finish at different times. What happens to the freed capacity?

**What it changes.** Continuous batching (also called *in-flight batching*) is a
scheduling idea, not a math idea. Do not wait for the batch. The moment one
sequence emits its end-of-sequence token and frees its slot, admit a new request
into that slot on the very next step. The batch is a *set of slots*, refilled
continuously, not a fixed cohort.

```
time →    0     1     2     3     4     5     6     7     8
slot 0  [===== A =====][== D ==]                          A done@3 → D admitted@3
slot 1  [======= B =======][=== E ===]                    B done@5 → E admitted@5
slot 2  [========= C =========][== F ==]                  C done@7 → F admitted@7
```

A static batch would leave slots 0–2 idle from t=3 until C finishes at t=7,
because the batch is not "done" until its slowest member is. Continuous batching
reclaims that idle time, so throughput goes up.

**What it costs.** Two things. First, **per-request latency can rise** — a request
admitted into a busy batch shares bandwidth, so its own tokens come out slower
than if it had the machine to itself. Throughput and latency are in tension;
continuous batching buys the former with the latter. Second, **memory for the
largest concurrent KV set.** Every admitted sequence holds a KV cache, and the
scheduler must budget for the worst case it allows — that budget, not compute,
usually caps how many sequences run at once.

**What it does NOT change.** The arithmetic. Each token is still computed by the
same forward pass. Continuous batching changes *when* work is admitted, never
*what* the work is.

---

## 2. Paged KV — why is the cache a paged allocator and not one big buffer?

**The question.** If each sequence needs a KV cache that grows one token at a
time, why not just give each sequence one contiguous buffer sized for the maximum
length?

**What it changes.** Because that wastes enormous memory. A sequence that stops
after 200 tokens still reserved room for 8,000, and a sequence that grows past its
reservation must be copied somewhere bigger. **Paging** fixes both: the cache is
divided into fixed-size blocks, and a sequence's blocks are scattered wherever
free memory happens to be, linked by a table. Growth means allocating one more
block, not copying the whole thing. Two sequences that share a prefix can point at
the *same* physical blocks.

fak's cache is a first-class kernel-owned object, and its shape reflects this. The
struct is at `internal/model/kvcache.go:11`:

```go
type KVCache struct {
	cfg     Config
	K       [][]float32 // [layer] -> flat, NumKVHeads*HeadDim per cached position (post-RoPE)
	Kraw    [][]float32 // [layer] -> the SAME entries pre-RoPE, so a span can be repositioned
	V       [][]float32
	pos     []int // absolute position of each cached entry (shared across layers)
	...
}
```

Two fields are worth teaching. The `pos []int` field (`internal/model/kvcache.go:16`)
records each entry's **absolute position** — its place in the original sequence —
shared across all layers. That is what makes the cache *addressable*: a span of
tokens can be named by position and removed, and the survivors relabelled so the
run looks exactly like one that never saw the removed span.

The second is the `mutGen` counter (`internal/model/kvcache.go:40`). Its doc
comment is the systems detail worth internalizing:

```go
// mutGen counts every NON-append mutation of the token rows: Truncate, Evict's
// compaction and survivor re-RoPE, tree compaction, RestoreSpan and precision
// conversion. Appends never bump it, so a device mirror keyed on (this cache's
// identity, mutGen, mirrored row count) stays valid across appends and is
// invalidated in O(1) by anything that rewrites or removes rows it already holds.
```

A device (GPU) holds a mirror of the cache, and the host must know when the mirror
is stale. The naive answer is "compare the whole cache" — expensive. The actual
answer: appends are the common case and they *cannot* invalidate the mirror, so
they do not bump the counter; only the rare rewrite-and-remove operations do. The
mirror is valid across the whole decode loop and invalidated in O(1) the moment
something destructive happens — a per-token cost turned into a per-eviction cost.

**What it costs.** Indirection. A paged cache needs a block table, and every
attention read follows it. A badly tuned block size either wastes memory (blocks
too big) or multiplies table lookups (blocks too small).

**What it does NOT change.** The attention math. Paging changes *where* the K and
V rows live, never *what* attention computes over them.

---

## 3. Quantization — how few bytes per weight can we get away with?

**The question.** Level 4 said fewer bytes per weight buys decode speed roughly
proportionally, because decode is bandwidth-bound. So why not store everything in
4 bits and be done?

**What it changes.** Quantization stores each weight in fewer bits — 8, 4, even
fewer — instead of 16 or 32. A 4-bit weight is a quarter the bytes of a 16-bit
one, so a decode step reads a quarter as much and, if bandwidth-bound, runs
roughly four times faster. That is the win, and it is real.

**What it costs.** Two things, and the second is the one people forget. First,
**accuracy.** Rounding a weight to 4 bits loses information; the model's outputs
drift, and how far is an empirical question that is not uniform — some layers
tolerate aggressive quantization, others do not. Second, **a dequantization step.**
The weights are stored packed, but the matmul needs real numbers, so you unpack on
the fly, which costs compute and, if done carelessly, memory traffic that eats the
savings.

fak's KV cache is a worked example of a quantization decision that is deliberately
*not* free. The cache can itself be quantized, and the code keeps a full-precision
copy of one specific thing on purpose. The `prec KVPrecision` field
(`internal/model/kvcache.go:29`) selects the storage tier, and its doc comment
explains the split:

```go
// prec is the realized storage tier of the two ATTENDED rows (post-RoPE K, V).
// Its zero value is KVPrecisionFP32, which keeps K/V as the f32 [][]float32 above
// and is byte-for-byte the historical cache. KVPrecisionQ8_0 instead stores K and
// V packed in kQ8/vQ8 (int8 codes + f32 per-32-block scale) and leaves Kraw f32,
// matching the compute.KVPrecisionQ8 planner tier and keeping Evict's pre-RoPE
// re-positioning exact.
```

The packed rows are `kQ8 []kvPackedRow` and `vQ8 []kvPackedRow`
(`internal/model/kvcache.go:32`). The attended K and V — the rows attention
actually reads — are quantized to 8 bits. But `Kraw`, the pre-rotation copy of K,
stays in f32. Why? Because eviction re-derives a survivor's post-rotation K at its
new position, and doing that from a quantized copy would accumulate error. The
comment on `NewKVCache` states the stake plainly: composing two rotations instead
of one "drifts ~1e-6 — enough to flip a greedy token." So the design quantizes the
hot path (attention reads, every token) and keeps full precision on the cold path
(eviction, rare) — quantize what you read constantly, keep precision where a small
error compounds.

**What it does NOT change.** The model's architecture, the number of parameters,
or the sequence of operations. Quantization changes the *representation* of the
weights, not the computation over them.

---

## 4. Mixture of experts — how do you grow capacity without growing decode cost?

**The question.** A bigger model is more capable. But a bigger model is slower to
decode, because decode reads all the weights. Can you have more parameters without
reading more of them per token?

**What it changes.** Yes — by not using all the parameters on every token. A
**mixture of experts** (MoE) replaces one big feed-forward network (FFN) per layer
with *many* smaller expert FFNs plus a **router**. For each token, the router
scores all experts and picks the top few (commonly 2 of 8, or 2 of 64). Only those
experts run. Total parameters go up; *active* parameters per token stay flat.

fak's dispatch is `ffnForLayer` (`internal/model/moe.go:56`), and the file's
opening comment states the scope of the change precisely
(`internal/model/moe.go:14`):

```go
// The FFN sub-layer is the ONLY thing MoE changes. It is KV-orthogonal: the
// attention block, the KV cache (Cache.K/V/Kraw), Evict, and Clone are untouched,
// because every cache append lives in the attention section and the FFN writes
// only the residual delta.
```

That is the "what it does NOT change" answer, stated by the code itself. The
per-layer call site is `mlpSeq` (`internal/model/forward.go:793`), whose comment
(`internal/model/forward.go:791`) says it "computes the SwiGLU MLP over a whole
sequence of normalized inputs and returns the per-position down-projection results
(pre residual)." MoE swaps what happens inside that call; it never touches the
attention path or the cache.

```
   token x
      │
      ▼
  ┌────────┐  gate = softmax(W_router · x)  →  scores over ALL N experts
  │ router │  pick top-2 of N=8
  └───┬────┘
      ├──────────────┬──────────────┐
      ▼              ▼              ▼
  ┌───────┐     ┌───────┐     ┌───────┐
  │  E0   │     │  E3   │     │  E7   │  (E1,E2,E4,E5,E6 idle this token)
  └───┬───┘     └───┬───┘     └───────┘
      │ g0          │ g3          (not read)
      └──────┬──────┘
             ▼
   delta = g0·E0(x) + g3·E3(x)   →   x ← x + delta
```

### Worked example: total vs active parameters

*Illustration, not a benchmark.* Take a 32-layer model where each layer's dense
FFN would be 0.5B parameters, and give each layer 8 experts.

| Quantity | Value (illustration) |
|---|---|
| Expert params per layer | 8 × 0.5B = 4B |
| Expert params, all 32 layers | 32 × 4B = **128B** |
| Attention + embeddings | ~12B |
| **Total parameters** | **~140B** |
| Experts active per token | 2 of 8 |
| Active expert params per token | 32 × 2 × 0.5B = **32B** |
| **Active parameters per token** | 32B + 12B = **~44B** |

Now the decode cost. Decode reads the weights it uses — 44B of parameters, not
140B. At 2 bytes per parameter that is ~88 GB read per token, versus ~280 GB if
every expert were read. At an illustrative 1 TB/s of memory bandwidth, that is
~88 ms/token (~11 tokens/s) versus ~280 ms/token (~3.6 tokens/s) — roughly a 3×
decode speedup at the same total capacity. The ratio 140B / 44B ≈ 3.2× is the
whole point: **decode cost tracks the active count, not the total.**

**What it costs.** Three things, and the third is the one that bites.

1. **Routing overhead.** Every token runs a router matmul and a top-k selection.
   Small, but not zero.
2. **Load imbalance.** If the router sends most tokens to a few experts, those
   experts become the bottleneck while the rest idle. Balancing is an active
   research problem.
3. **All experts must be resident in memory.** This is the honest cost. You read
   only 2 of 8 experts per token, but you must *hold* all 8, because you do not
   know which 2 the next token needs. So MoE trades a **compute/bandwidth** problem
   for a **memory capacity** problem. The 140B model above needs ~280 GB resident
   at 2 bytes/param, even though each token touches ~88 GB of it. MoE makes decode
   fast and makes the model *big*.

**What it does NOT change.** Attention, the KV cache, and the residual stream
shape. MoE changes only the FFN's contribution to the residual delta.

---

## 5. Speculative decoding — can a cheap model do the expensive model's work?

**The question.** Decode is bandwidth-bound: one token costs one full read of the
weights. That read is the same whether you produce one token or several. So can we
produce several tokens per read?

**What it changes.** Yes, with a **draft model** — a small, cheap model that
proposes the next few tokens. The **target model** (the big one) then verifies all
of them in a *single batched forward pass*. Because that pass is memory-bound, it
costs about the same as producing one token. If the draft was right, you got
several tokens for the price of one. The mechanism: the draft proposes `k` tokens;
the target runs one forward pass over the prompt plus all `k` drafts, yielding its
own next-token distribution at each position; the longest prefix where the draft
matches the target is accepted; the rest is rolled back.

```
draft model (cheap) proposes k=4 tokens:
   prompt ──► [d1] [d2] [d3] [d4]

target model (expensive) verifies all 4 in ONE batched forward:
   prompt  d1   d2   d3   d4
     │     │    │    │    │
     ▼     ▼    ▼    ▼    ▼
    p1    p2   p3   p4   p5   ← target's own next-token distribution per position

accept the longest correct prefix:
   d1 == argmax(p1)? YES → accept      d3 == argmax(p3)? YES → accept
   d2 == argmax(p2)? YES → accept      d4 == argmax(p4)? NO  → reject, roll back

result: 3 accepted + 1 free token from p4 = 4 tokens from ONE target pass
rollback: RollbackSpeculative(1) — O(1) pointer adjustment, no copy
```

The rollback is where the cache design pays off. `RollbackSpeculative` is at
`internal/model/kv.go:1622`, and its doc comment (`internal/model/kv.go:1619`) is
the claim worth quoting:

```go
// RollbackSpeculative rolls back the session's KV cache by nTokens.
// It commits accepted positions 0..R-1 and rolls back speculative branches R..K-1
// via O(1) pointer adjustments without memory copying.
```

O(1) pointer adjustments, no copying. The rejected branch's KV rows are simply
dropped by moving a length pointer — the same `Truncate` the cache already
supports. If rollback required copying the cache, speculative decoding would lose
most of its win to bookkeeping.

### Worked example: the acceptance-rate economics

*Illustration.* Draft 4 tokens, 3 accepted. You produced 4 tokens (3 accepted + 1
free) from one target forward pass. Without speculation, 4 tokens would need 4
target passes. But you also paid for the draft. Let `c_d` be the cost of one draft
forward per token, `c_t` the cost of one target forward, `k` the tokens drafted per
round, and `a` the average accepted per round:

```
cost per round   = k·c_d + c_t
tokens per round = a + 1
cost per token   = (k·c_d + c_t) / (a + 1)      baseline = c_t
win  ⟺  (k·c_d + c_t) / (a + 1) < c_t   ⟺   a > k · (c_d / c_t)
```

With `k = 4` and a draft 10× cheaper than the target (`c_d/c_t = 0.1`), you need
`a > 0.4` — an average of just over 0.4 accepted tokens per round. Our example
accepts 3, far above the bar. **It is a clear win.**

**The condition under which it is a loss.** If the draft is expensive
(`c_d/c_t` near 1) or usually wrong (`a` near 0), the inequality fails. A draft
that is nearly as costly as the target, or that is rejected most of the time,
means you pay for the draft *and* the verification and get almost nothing. The
worst case — 0 accepted — produces 1 token per round while paying `k·c_d + c_t`,
strictly worse than just running the target.

**What it costs.** Draft compute, verification compute, and the engineering of
keeping two models in memory and in sync.

**What it does NOT change.** The output distribution. A correct speculative
decoder produces *exactly* the same distribution as the target model alone — the
verification step guarantees it. Speculative decoding is a **latency** win, not a
throughput win: it makes a single sequence finish faster, but it does not increase
how many tokens per second the machine produces under load. Under a full batch,
the target is already saturated, and there is no idle bandwidth for the draft to
exploit.

---

## 6. Sampling at scale — how do you pick one token from 128,000?

**The question.** The model outputs a score (a **logit**) for every token in its
vocabulary — often 128,000 or more. To sample, you must turn that into a
probability distribution and truncate it (top-k, top-p). That is real work over a
huge array, every single token. How is it done without becoming the bottleneck?

**What it changes.** The work is split by *where the data lives*. The logits are
produced on the device (GPU); the sampler runs on the host (CPU). Shipping 128,000
floats per token across the bus is expensive, so the code transports only a
**bounded candidate set** — the top few logits — and *certifies* it before
trusting it.

The bound is explicit. `SamplingCandidateMaxK` is at
`internal/compute/sampling_candidates.go:6`:

```go
// SamplingCandidateMaxK bounds host transport, not the vocabulary on the device.
const SamplingCandidateMaxK = 64
```

Read that comment carefully: the bound is on *transport*, not on the vocabulary.
The device still inspects the whole row; it just sends back 64 entries. The
container is `SamplingCandidatePacket` (`internal/compute/sampling_candidates.go:12`),
and the check is `Certify` (`internal/compute/sampling_candidates.go:36`), which
validates the packet before the sampler trusts it: the count is exactly `k+1`,
every ID is in range, no ID repeats, no value is NaN or infinite, and the `k`-th
and `(k+1)`-th weights are strictly ordered (so the truncation boundary is
unambiguous). If any check fails, it returns false and the caller must fall back
to reading the full row. The device-side extraction is `ExtractTopK`
(`internal/compute/topk_greedy.go:105`), which maps local indices to global token
IDs via `vocabOffset` and breaks ties toward the lower ID.

The dispatch seam is `selectSamplingTruncationPath`
(`internal/agent/inkernel_sampling.go:36`), and its doc comment states the
fail-closed rule:

```go
// selectSamplingTruncationPath is the fail-closed dispatch seam for the
// cluster-launched top-k/top-p mechanism. It selects the cluster path only when a
// truncation is requested, the vocabulary is wide enough to justify partitioning,
// and the backend explicitly proves both cluster launch support and device
// residency. Selection is not execution: callers without a device implementation
// continue through applyCPUSamplingTruncation as the correctness oracle/fallback.
```

The "wide enough" threshold is `clusterSamplingMinVocabulary`
(`internal/agent/inkernel_sampling.go:18`), set to `1 << 16` — 65,536. Below that,
partitioning the vocabulary across a thread-block cluster is not worth the
overhead, so the CPU path runs. The host-side sampler itself is `SampleLogits`
(`internal/agent/inkernel_sampling.go:14`) and its full-featured sibling
`sampleLogits` (`internal/agent/inkernel_sampling.go:175`), which takes `topP` and
`topK` as well as temperature.

**Why a bounded transport plus a certificate is the right shape.** Two reasons.
First, **the bus is the bottleneck, not the arithmetic** — sending 64 entries
instead of 128,000 is a ~2,000× reduction in bytes moved per token, and the device
does the wide scan where the data already is. Second, **a bounded transport is only
safe if you can prove it is complete.** If the device silently dropped the true
maximum, the host would sample from a wrong distribution and never know. `Certify`
is the proof obligation: the packet must be internally consistent and correctly
ordered, or it is refused. The comment on `Certify` is explicit that this check
"cannot establish global completeness: that is the producer's obligation" — the
certificate covers the *shape* of the packet, the producer contract covers the
*content*. Splitting the obligation that way is what makes a bounded transport
trustworthy.

**What it costs.** A producer contract that must be honored (inspect the whole
row, preserve raw values, return the true top `k+1`), and a fallback path for when
the certificate fails.

**What it does NOT change.** The sampling distribution. A certified packet yields
the same draw as sampling over the full row.

---

## 7. Parallelism across devices — how do you split a model that does not fit?

**The question.** A model too large for one device must be split. There are two
fundamentally different ways to split it. What are they, and what does each cost?

**What it changes.** Two axes:

- **Tensor parallelism (TP)** splits each *weight matrix* across devices. Every
  device works on every token, computing a slice of each matmul, and the partial
  results are combined with an **all-reduce** (a collective that sums across
  devices). The entry point is `ForwardTP`
  (`internal/model/tensor_parallel_forward.go:548`).
- **Pipeline parallelism (PP)** splits the *layers* across devices. Each device
  owns a contiguous band of layers, and the hidden state is handed forward from
  one band to the next. The unit is `ForwardBand`
  (`internal/model/partition.go:122`), and the first band's entry is `embedBand`
  (`internal/model/partition.go:91`).

```
TENSOR PARALLEL (split each weight matrix)      PIPELINE PARALLEL (split the layers)

  device 0        device 1                        device 0        device 1
  ┌─────────┐    ┌─────────┐                      ┌─────────┐    ┌─────────┐
  │ W[:,0:h]│    │ W[:,h:] │                      │ layers  │    │ layers  │
  └────┬────┘    └────┬────┘                      │  0..15  │    │ 16..31  │
       │              │                           └────┬────┘    └────┬────┘
       └── all-reduce ──┘                              │  hidden x    │
              ▼                                        └─────────────►│
        full result                                                    ▼
                                                                   logits
  every device sees every token                   each device owns a band of layers
  cost: one collective PER LAYER                  cost: bubble unless micro-batches fill it
```

The pipeline split carries a strong, checkable claim. The `embedBand` doc comment
(`internal/model/partition.go:87`) says:

```go
// embedBand produces the initial hidden state for a pipeline's FIRST band: the
// token embedding lookup (with any arch embed scaling), the same as the head of
// Model.Forward. Later bands receive the hidden state from the previous worker
// instead of calling this.
```

And the `ForwardBand` doc comment (`internal/model/partition.go:103`) states the
exactness property: `Forward` is exactly `embedBand(ids)` followed by
`ForwardBand(x, 0, NumLayers, true)`, so splitting the model into bands and
threading the hidden state between them is **bit-exact** — each band runs the
identical per-layer instruction stream over the identical hidden state. The test
that proves it is `TestForwardBandTwoWayMatchesMonolithic`
(`internal/model/forward_band_test.go:17`). This is the teaching moment: the code
is *designed* so the split cannot change the answer. A parallelism scheme that
changes the answer is a bug; this one is constructed so it cannot.

For MoE models there is a third axis, **expert parallelism** (EP), which shards
the *experts* across devices. Its entry point is `ForwardEP`
(`internal/model/expert_parallel_forward.go:117`), and it fails closed on
architectures whose expert decomposition it does not implement.

**What it costs.**

- **Tensor parallelism** needs a collective *per layer*. Every layer's matmul ends
  in an all-reduce, and collectives are latency-bound — the more devices, the more
  the communication dominates. TP works best within a single node, where the
  interconnect is fast.
- **Pipeline parallelism** needs enough in-flight **micro-batches** to fill the
  **bubble**. The bubble is the idle time at the start and end of the pipeline:
  device 0 works while device 1 waits for its first input, and at the end device 1
  works while device 0 sits idle. Splitting the batch into micro-batches keeps
  every device busy, but the bubble never fully disappears.

**What it does NOT change.** For pipeline parallelism, nothing — the output is
bit-exact. For tensor parallelism, the output matches within the documented
round-off of the all-reduce, because floating-point addition is not associative and
summing across devices in a different order gives a slightly different result.

---

## 8. The honest open edges

Everything above is a solved-enough trade. This section is the opposite: what is
genuinely unsolved or actively contested. Each is a question, with what would
settle it.

**How do you keep KV memory bounded at very long context without losing recall?**
The KV cache grows linearly with sequence length; at 100k+ tokens it dominates
memory. Eviction, compression, and summarization all reduce it — and all risk
dropping the one token the answer depended on. *What would settle it:* a
task-success benchmark at long context showing a bounded-memory policy matching
full-cache quality on tasks that require recalling a specific early detail.

**Do sparse and linear attention hybrids preserve quality at scale?** Attention is
quadratic in sequence length; sparse and linear variants make it cheaper by
attending to less. Whether they preserve quality at frontier scale — or only at
the sizes where they have been tested — is contested. *What would settle it:* a
controlled comparison at matched parameter count and matched training compute,
evaluated on long-context recall, not just perplexity.

**How do you schedule a mixed prefill/decode workload optimally?** Prefill is
compute-bound; decode is bandwidth-bound. A real server runs both at once. The
optimal split of hardware between them, and the optimal admission order, is not
solved — it depends on the traffic mix, which changes. *What would settle it:* a
scheduler whose policy is provably near-optimal across a range of measured traffic
mixes, not tuned to one.

**How far can quantization go before quality breaks?** 8-bit is broadly safe;
4-bit is usually fine. Below that, results diverge by model and by layer, and there
is no reliable predictor of which layer will break first. *What would settle it:* a
per-layer sensitivity metric that predicts quality loss before you run the
evaluation, validated across architectures.

**Speculative decoding's win depends on a draft model you may not have.** The
economics in Section 5 require a draft that is both cheap and usually right. For a
model with no small sibling — or one whose distribution is too different — there is
no good draft, and the technique does not apply. Self-speculation (drafting with
early layers of the target) is one answer, but its acceptance rate is lower.
*What would settle it:* a draft-construction method that works without a
pre-trained small model and reaches a useful acceptance rate.

**And one meta-edge.** Every number on this page is an illustration. Real
throughput depends on the model, the hardware, the batch, and the traffic. The
techniques are real; the multipliers are not portable. Measure on your own
workload.

## See also

- [Level 4 — Prefill vs decode](level-4-prefill-vs-decode.md): the memory-bound
  fact this whole page builds on.
- [Multi-GPU tensor parallelism](../multi-gpu-tensor-parallelism.md) — the
  tensor-parallel axis in depth.
- [SOTA serving optimizations](../sota-optimizations.md) — the bundle of
  optimizations a tuned serving stack applies.
- [Addressable KV cache](../addressable-kv-cache.md) — the paged, evictable cache
  Section 2 describes.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---

*[Full ladder (README)](README.md) · prev ← [Level 4 — Prefill vs decode](level-4-prefill-vs-decode.md)*
