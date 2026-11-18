---
title: "Inside one decoder block: the ~10 operations, and where each one lives"
description: "Level 3 of the inference ladder. Level 2 showed the pipeline and said a block is 'normalize, attention, add, normalize, MLP, add'. This rung opens the block: the residual stream, RMSNorm, Q/K/V projection, GQA, RoPE, the causal mask and softmax, the output projection, the SwiGLU MLP, and the decode-path twin — each pinned to the real function in this repository, with a hand-computable attention step."
slug: inside-a-decoder-block
keywords:
  - transformer decoder block explained
  - residual stream transformer
  - RMSNorm vs LayerNorm
  - grouped query attention GQA
  - rotary position embedding RoPE
  - causal mask softmax attention
  - SwiGLU MLP explained
  - prefill vs decode block
date: 2026-11-18
---

# Inside one decoder block

*You are on **Level 3 of 5** of the [inference ladder](README.md).*

> **In one breath:** A decoder block is a vector per position. Two sublayers each
> compute a change and add it back. The first is attention, where positions share
> information. The second is the feed-forward network, where each position works
> alone.

**One line:** The block is a residual stream mutated in place by two sublayers —
attention, which projects each position to query, key, and value, rotates them by
position, masks the future, softmaxes the scores, and takes the weighted sum of
the values; and the feed-forward network, which expands, gates, and contracts each
position independently — with the norm placement and residual wiring chosen by
`BlockTopology`.

> **Audience.** A working engineer who can read Go and wants the real code. Level 2
> already gave you the pipeline — `Encode → embedBand → forwardHiddenRows (layer loop)
> → fillSeqLogits → logitsFromHidden → pick` — and told you `x` is `[][]float32`, one
> row per position, `HiddenSize` wide. It told you a block is "normalize → attention →
> add → normalize → MLP → add". This rung opens that block and names every operation.

> **Short answer.** A decoder block is a **residual stream** — one vector per position,
> mutated in place — plus two sublayers that each compute a *delta* and add it back.
> Sublayer one is **attention**: project each position to a query, key, and value;
> rotate Q and K by position (RoPE); score every query against the keys it is allowed
> to see (a causal mask); softmax the scores into weights; take the weighted sum of the
> values; project back out. Sublayer two is the **MLP**: per position, independently,
> expand to a wider vector, gate it, contract back. The same block exists twice in this
> repo — once for whole-sequence prefill (`composeSeqBlock`) and once for one-token
> decode (`blockStep`) — and the two must compute the same function.

## The residual stream, made precise

**Question: Level 2 said "add the delta back". Add it back to what, exactly?**

To a vector that never leaves. Call it the **residual stream**: `x`, one row per
position, `HiddenSize` floats wide. It is the block's only long-lived state. Every
sublayer *reads* it, computes something, and *adds* the result back into it. Nothing
overwrites it; everything accumulates.

The block body is `composeSeqBlock` (`internal/model/forward.go:826`). It does not
contain the attention math or the MLP math. It contains the *wiring*: which norm runs
where, and how the two deltas land in the stream. The actual sublayer bodies are passed
in as closures — `attnSub` and `mlpSub` — so the same wiring serves every architecture.

The per-sublayer wiring is `composeSeqSublayer` (`internal/model/forward.go:847`). It is
short, and it carries the design, so here it is in full:

```go
func composeSeqSublayer(t BlockTopology, x [][]float32, n normWeights, eps float32, cfg Config, body seqSublayer) {
	H := len(x[0])
	switch t {
	case PostNorm:
		addPostNormed(x, body(x), n, eps, cfg, H) // sub-layer reads the RAW residual stream
	case SandwichNorm:
		addPostNormed(x, body(normSeq(x, n, eps, cfg)), n, eps, cfg, H)
	default: // PreNorm — verbatim Llama
		out := body(normSeq(x, n, eps, cfg))
		for tt := range x {
			for i := 0; i < H; i++ {
				x[tt][i] += out[tt][i]
			}
		}
	}
}
```

Read the three cases as three sentences:

- **PreNorm** (the default, and what Llama, Qwen, and Mistral use): normalize the
  stream, run the body on the normalized copy, add the result back. `x += body(norm(x))`.
- **PostNorm** (OLMo2): run the body on the *raw* stream, normalize the body's output,
  add that back. `x += norm(body(x))`.
- **SandwichNorm** (Gemma2-style): normalize before *and* after the body.
  `x += post(body(pre(x)))`.

The fourth wiring is genuinely different, so it is handled one level up, in
`composeSeqBlock` itself. **ParallelResidual** (GPT-NeoX, Cohere) does not run attention
then MLP. It runs both *at the same time*, each reading the original stream, and adds
both deltas in one step:

```go
if topo == ParallelResidual {
	mlpNorm := m.parallelMLPNorms(l, attnNorm)
	o := attnSub(normSeq(x, attnNorm, eps, cfg))
	d := mlpSub(normSeq(x, mlpNorm, eps, cfg))
	for t := range x {
		for i := range x[t] {
			x[t][i] += o[t][i] + d[t][i]
		}
	}
	return
}
```

That is the interesting part: **one block body, four wirings.** The topology is a config
field (`BlockTopology`, `internal/model/arch.go:691`), and the PreNorm value lowers to
the verbatim Llama instruction stream, so the default path is bit-for-bit unchanged by
the existence of the other three.

```
   PreNorm (Llama)              ParallelResidual (GPT-NeoX)
   ───────────────              ───────────────────────────
   x ──┬──► norm ──► attn ──┐   x ──┬──► norm ──► attn ──┐
       │                    │       │                   │
       └──────────────(+)◄──┘       ├──► norm ──► mlp ──┤
       │                            │                   │
       ├──► norm ──► mlp ──┐        └───────────(+)◄────┘
       │                   │                    │
       └─────────────(+)◄──┘                    ▼
                    │                          x'
                    ▼
                    x'
```

The stream is the vertical spine. Each `(+)` is a residual add. In PreNorm the second
norm reads the *already-updated* stream; in ParallelResidual both norms read the
*original* one.

## RMSNorm: the cheapest thing in the block

**Question: what does "normalize" actually compute, and why not LayerNorm?**

**RMSNorm** rescales a vector so its root-mean-square is 1, then applies a learned
per-element weight. It is `internal/model/norm/scalar.go:23`:

```go
func SerialRMSNorm(x, w []float32, eps float32) []float32 {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv * w[i]
	}
	return out
}
```

In one formula: `out[i] = x[i] / sqrt(mean(x²) + eps) * w[i]`.

**Why it is cheaper than LayerNorm.** LayerNorm subtracts the mean first, then divides
by the standard deviation. That is two passes over the vector (one for the mean, one for
the variance) plus a subtraction on every element. RMSNorm skips the mean entirely — it
only needs the sum of squares — so it is one pass and no subtraction. The empirical
finding behind it is that the mean-centering was doing almost no work; the *scale* was
doing all of it. Dropping the mean costs nothing measurable and saves a pass.

**A 4-element example you can check by hand.** Let `x = [1, 2, 3, 4]`, `eps = 0`, and
`w = [1, 1, 1, 1]`:

```
mean(x²) = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
sqrt(7.5) = 2.7386
inv       = 1 / 2.7386 = 0.3651
out       = [1, 2, 3, 4] * 0.3651 = [0.3651, 0.7303, 1.0954, 1.4606]
```

The output has RMS 1 by construction. The weight `w` then lets the model re-scale each
dimension independently — that is the only learned part.

**Where it lives in the block.** `normSeq` (`internal/model/forward.go:810`) applies the
norm to every position's row, and each row goes through `normCfg`
(`internal/model/arch.go:615`), which is the dispatcher: RMSNorm by default, LayerNorm if
the config asks, and a `(1 + w)` gain variant for Gemma. The Llama defaults lower to
`SerialRMSNorm` bit-for-bit.

## Q/K/V projection: three matrix multiplies

**Question: how does a position's vector become a query, a key, and a value?**

By three matrix multiplies. `attnSeq` (`internal/model/forward.go:621`) is the
whole-sequence attention body. For each position `t` it does:

```go
q[t] = mat.mul(p("self_attn.q_proj.weight"), xp, nH*hd, H)
k[t] = mat.mul(p("self_attn.k_proj.weight"), xp, nKV*hd, H)
v[t] = mat.mul(p("self_attn.v_proj.weight"), xp, nKV*hd, H)
```

`mat.mul(weight, x, out, in)` computes `y = W·x` where `W` is `out × in`. So the shapes
are:

| Projection | Weight shape | Meaning |
|---|---|---|
| `q_proj` | `HiddenSize → NumHeads * HeadDim` | one query vector per head |
| `k_proj` | `HiddenSize → NumKVHeads * HeadDim` | one key vector per **KV** head |
| `v_proj` | `HiddenSize → NumKVHeads * HeadDim` | one value vector per **KV** head |

Note the asymmetry already: Q is `NumHeads` wide, K and V are `NumKVHeads` wide. That is
the next section.

The three names — query, key, value — come from information retrieval. A **query** is
what this position is looking for. A **key** is what each position advertises about
itself. A **value** is what each position will hand over if it is selected. Attention
computes, for each query, how well it matches each key, and uses those match scores as
weights to average the values.

## GQA: why K and V are narrower

**Question: why is `NumKVHeads` smaller than `NumHeads`?**

Because several query heads *share* one key/value head. This is **Grouped-Query
Attention** (GQA). The sharing factor is one line, `internal/model/config.go:1692`:

```go
func (c Config) GroupSize() int { return c.NumHeads / c.NumKVHeads }
```

For a model with 9 query heads and 3 KV heads, `GroupSize()` is 3: heads 0, 1, 2 all
read KV head 0; heads 3, 4, 5 read KV head 1; heads 6, 7, 8 read KV head 2. In `attnSeq`
that is the single line `kvh := h / grp`.

```
   query heads        KV heads
   ───────────        ────────
   q0 ─┐
   q1 ─┼────────────► k0 / v0
   q2 ─┘
   q3 ─┐
   q4 ─┼────────────► k1 / v1
   q5 ─┘
   q6 ─┐
   q7 ─┼────────────► k2 / v2
   q8 ─┘
```

**The memory win, in one sentence.** During decode the KV cache stores one key and one
value per position per KV head, so shrinking KV heads from 9 to 3 shrinks the cache — the
thing that grows with context length and dominates decode memory — by a factor of 3,
exactly the ratio of KV heads, while the queries (which are recomputed every step and
never cached) stay full width.

The two extremes have names. `NumKVHeads == NumHeads` is plain **multi-head attention**
(MHA). `NumKVHeads == 1` is **multi-query attention** (MQA). GQA is the middle, and it is
what almost every modern model uses.

## RoPE: how position gets in

**Question: attention is a dot product. A dot product does not care about order. So how
does the model know that token 5 comes after token 2?**

By rotating. **Rotary Position Embedding** (RoPE) takes each query and key vector and
rotates it by an angle proportional to its position. Two vectors at positions `i` and
`j` then have a dot product that depends on `i − j` — the *relative* distance — even
though the dot product itself is order-blind.

The rotation is built once per forward by `newRopeForLayer`
(`internal/model/forward.go:442`), which precomputes a `cos` and `sin` table for every
`(position, frequency)` pair. The rotation itself is `rope.apply`, and its doc comment
states the exact formula (`internal/model/forward.go:452`):

```go
// apply rotates one head vector (len head_dim) in place at position p, using HF's
// non-interleaved "rotate_half" convention:
//
//	out[j]      = x[j]*cos - x[j+half]*sin
//	out[j+half] = x[j+half]*cos + x[j]*sin
func (r rope) apply(hv []float32, p int) {
	applyRopeRow(hv, r.cos[p], r.sin[p])
}
```

The vector is split in half. The first half and the second half are treated as the real
and imaginary parts of a complex number, and the whole thing is multiplied by
`e^{iθ}` where `θ` depends on position. That is exactly what the two lines compute.

**A tiny numeric rotation.** Take `HeadDim = 2`, so there is one frequency and
`half = 1`. With the standard base `theta = 10000`, the inverse frequency for the first
pair is `1 / 10000^0 = 1`, so the angle at position `p` is just `p` radians. At `p = 1`:
`cos(1) = 0.5403`, `sin(1) = 0.8415`. Rotate `hv = [1, 0]`:

```
out[0] = 1*0.5403 - 0*0.8415 = 0.5403
out[1] = 0*0.5403 + 1*0.8415 = 0.8415
```

The vector `[1, 0]` became `[0.5403, 0.8415]` — rotated by one radian. At position 2 it
would rotate by two radians, and so on. The frequencies are built by `invFreq`
(`internal/model/rope.go:14`); the per-position table by `ropeRowFromInvScaled`
(`internal/model/rope.go:178`).

## The score, the mask, and softmax

**Question: given queries and keys, how do we turn them into a weighted average of
values?**

Three steps: score, mask, softmax. Then a fourth: weight the values.

**Step 1 — score.** For query head `h` at position `t` and key head `kvh` at position
`j`, the score is a scaled dot product. In `attnSeq`:

```go
scores[j-lo] = dot(qh, kh)*scale + cfg.alibiScoreBias(h, j, seq)
```

`dot` (`internal/model/forward.go:912`) is the plain sum of elementwise products. The
`scale` is `attnScale` (`internal/model/arch.go:411`), which is `1/sqrt(HeadDim)` by
default. Why divide by `sqrt(HeadDim)`? Because a dot product of two `d`-dimensional
vectors with unit-variance entries has variance `d`; without the scale, longer head
dimensions would push scores into a range where softmax saturates and gradients vanish.

**Step 2 — mask.** Position `t` must not see the future. The loop runs `j` from `lo` to
`t` only — never past `t`. That is the **causal mask**, and here it is implemented as a
loop bound rather than by writing `-inf` into a matrix. The effect is identical: the
future simply is not scored. (The `lo` bound is the sliding-window case; with the default
window `W = -1`, `lo` is 0 and the range is the full causal prefix `0..t`.)

```
   key position j →
        0    1    2    3
   q0 [ ✓    ·    ·    · ]     ✓ = scored
   q1 [ ✓    ✓    ·    · ]     · = masked (future)
   q2 [ ✓    ✓    ✓    · ]
   q3 [ ✓    ✓    ✓    ✓ ]
```

**Step 3 — softmax.** The scores become weights that are positive and sum to 1.
`softmaxInPlace` (`internal/model/forward.go:924`) delegates to `softmax.InPlace`
(`internal/model/softmax/scalar.go:21`), which subtracts the row max (for numerical
stability), exponentiates, sums, and divides. The attention call site is
`softmaxAttentionScores` (`internal/model/arch.go:447`), which also handles the
attention-sink variant some models use. Before softmax, `softcapInPlace`
(`internal/model/arch.go:491`) may squash the scores with `cap*tanh(z/cap)` — a no-op
unless the config sets a cap (Gemma2).

**Step 4 — weight the values.** Each value vector is scaled by its weight and summed.
That is the `saxpy` call (`internal/model/prefill_attn.go:259`): `out += weight * value`.

```
   q ──► Q·K ──► mask ──► softmax ──► weights ──► Σ wᵢ·Vᵢ ──► head output
         (score)  (causal)  (Σ=1)                  (weighted sum)
```

### A worked attention step (illustrative numbers, real shapes)

Three positions, `HeadDim = 2`, one head. The numbers are chosen to be hand-checkable;
the shapes are the real ones.

```
q0 = [1, 0]   k0 = [1, 0]   v0 = [1, 0]
q1 = [0, 1]   k1 = [0, 1]   v1 = [0, 1]
q2 = [1, 2]   k2 = [1, 1]   v2 = [1, 1]

scale = 1/sqrt(HeadDim) = 1/sqrt(2) = 0.7071
```

Take the last position, `t = 2`. It may see keys 0, 1, 2 (the causal mask allows all
three). The three dot products:

```
dot(q2, k0) = 1*1 + 2*0 = 1   →  score = 1 * 0.7071 = 0.7071
dot(q2, k1) = 1*0 + 2*1 = 2   →  score = 2 * 0.7071 = 1.4142
dot(q2, k2) = 1*1 + 2*1 = 3   →  score = 3 * 0.7071 = 2.1213
```

Softmax over `[0.7071, 1.4142, 2.1213]`. Subtract the max (2.1213), exponentiate, sum:

```
e^(0.7071 - 2.1213) = e^(-1.4142) = 0.2431
e^(1.4142 - 2.1213) = e^(-0.7071) = 0.4931
e^(2.1213 - 2.1213) = e^(0)      = 1.0000
sum = 0.2431 + 0.4931 + 1.0000   = 1.7362

weights = [0.2431, 0.4931, 1.0000] / 1.7362
        = [0.1400, 0.2840, 0.5760]        (sums to 1.0000 ✓)
```

Weighted sum of the values:

```
out = 0.1400*[1, 0] + 0.2840*[0, 1] + 0.5760*[1, 1]
    = [0.1400 + 0 + 0.5760,  0 + 0.2840 + 0.5760]
    = [0.7160, 0.8600]
```

So position 2's attention output is `[0.7160, 0.8600]`. It is mostly value 2 (weight
0.576), which is what the largest score earned. That is the whole mechanism: **score,
mask, normalize, average.**

## The output projection

**Question: the heads produced separate vectors. How do they become one vector again?**

Concatenate and project. The per-head outputs are packed into one `NumHeads * HeadDim`
vector, multiplied by `o_proj` back down to `HiddenSize`, and added to the residual. In
`attnSeq`:

```go
attnOut[t] = mat.mul(p("self_attn.o_proj.weight"), mat.prep(attnOut[t]), H, nH*hd)
```

`o_proj` is `NumHeads * HeadDim → HiddenSize`. This is the step that lets the heads
*combine* — up to here each head has been independent, and the output projection is the
first place their results mix. The result is the attention sublayer's delta, which
`composeSeqSublayer` adds into the stream.

## The MLP: the position-independent half

**Question: attention mixes positions. What does the other sublayer do?**

It does not mix positions at all. The **MLP** (multi-layer perceptron) runs on each
position *independently* — position 5's output depends only on position 5's input. That
contrast is the whole point of having two sublayers: attention is the only place
information moves *between* positions; the MLP is where each position *thinks* about
what it has gathered.

`mlpSeq` (`internal/model/forward.go:793`) is the whole-sequence body. It loops over
positions and calls the FFN for each:

```go
for t := range xn {
	out[t] = ffn.apply(m, l, mat.prep(xn[t]), mat)
}
```

The FFN form is chosen by `ffnForLayer` (`internal/model/moe.go:56`). The default is
`denseSwiGLU` (`internal/model/moe.go:140`), which is three matrix multiplies and a
nonlinearity:

```
   xn (HiddenSize)
     │
     ├──► gate_proj ──► g (IntermediateSize) ──► silu(g) ──┐
     │                                                     ├──► (*) ──► down_proj ──► delta (HiddenSize)
     └──► up_proj   ──► u (IntermediateSize) ──────────────┘
```

In words: `g = gate(xn)`, `u = up(xn)`, `activated = silu(g) * u`, `delta = down(activated)`.
The elementwise multiply of the activated gate with the up projection is the **SwiGLU**
gating — the gate decides *how much* of each intermediate dimension passes through. The
`ffn.Gated` helper (`internal/model/ffn/gated.go:17`) is the shared implementation.

**The shape funnel.** `IntermediateSize` is typically 2.5 to 4 times `HiddenSize`. So the
MLP expands wide, applies a nonlinearity in the wide space, and contracts back:

```
   HiddenSize ──► IntermediateSize ──► HiddenSize
     (e.g. 2048)      (e.g. 8192)        (2048)
        │                 │                │
        └── expand ───────┴── contract ────┘
```

**Where the parameters live.** The MLP holds the majority of a transformer's weights.
Three matrices of `HiddenSize × IntermediateSize` (gate, up, down) dwarf the four
attention matrices, because `IntermediateSize` is several times `HiddenSize`. When you
hear "the model is mostly MLP", this is why.

**The MoE variant.** `ffnForLayer` can also return a mixture-of-experts FFN, where a
router picks a few experts per token and only those run. That changes *which* MLP runs,
never the attention path — the FFN is the only sublayer MoE touches. Level 5 covers it.

## The decode-path twin

**Question: prefill processes a whole sequence at once. Decode processes one token at a
time. Is the block the same code?**

No — and this is the part most explanations skip. There are **two implementations of one
block** in this repo:

- `composeSeqBlock` (`internal/model/forward.go:826`) — the **prefill** path. It takes
  `x` as `[][]float32`, one row per position, and processes the whole sequence.
- `blockStep` (`internal/model/kv.go:917`) — the **decode** path. It takes `x` as a
  single `[]float32` — one position — and processes exactly one token.

`blockStep` composes the *same* sublayers through `composeBlockAtLayer`
(`internal/model/arch.go:751`), the single-position twin of `composeSeqBlock`. The
wiring is shared; only the data shape differs.

**Why two implementations?** Because the two paths do genuinely different work. Prefill
has no cache yet — it computes K and V for every position and keeps them. Decode has a
cache — it computes K and V for *one* new position, appends them, and reads all the
*previous* K and V from the cache instead of recomputing them. That cache read is the
entire reason decode is fast, and it is why the two paths cannot be the same function.

```
   prefill (composeSeqBlock)          decode (blockStep)
   ─────────────────────────          ──────────────────
   x: [seq][HiddenSize]               x: [HiddenSize]  (one position)
   K/V: computed for all positions    K/V: one new pair, appended to cache
   attention: full causal matrix      attention: one query vs cached keys
   RoPE: table for all positions      RoPE: one row, via ropeRowQK
```

**What must stay identical between them.** Everything except the cache handling. The
RMSNorm arithmetic, the RoPE rotation, the GQA head sharing, the softmax, the residual
adds, and the SwiGLU must compute the *same function* in both paths — otherwise a token
generated during decode would not match the same token scored during prefill, and the
model would be inconsistent with itself. The repo enforces this: `blockStep`'s doc
comment says the block orchestration "exists in exactly one place", and the f32 path is
pinned bit-exact against the prefill path. The decode path's RoPE goes through
`ropeRowQK` (`internal/model/kv.go:1070`), the single-row rotate-and-stash that every
per-position site funnels through, so a RoPE-convention change lands in one place rather
than five hand-copies.

This is the seam Level 4 opens: *why* the two paths have different performance
characteristics, and what the cache does to the arithmetic.

![One decoder block: the residual stream runs left to right as a horizontal line, with RMSNorm, the Q/K/V projection, RoPE, attention (causal mask and softmax), and the output projection hanging off it, then a second RMSNorm and the SwiGLU MLP, each sublayer adding its result back into the stream](diagrams/decoder-block.svg)

## The block, operation by operation

| # | Operation | What it computes | Where |
|---|---|---|---|
| 1 | Residual stream | the accumulating per-position vector | `composeSeqBlock` `internal/model/forward.go:826` |
| 2 | Norm placement | PreNorm / PostNorm / Sandwich / Parallel | `composeSeqSublayer` `internal/model/forward.go:847` |
| 3 | RMSNorm | `x / sqrt(mean(x²)+eps) * w` | `SerialRMSNorm` `internal/model/norm/scalar.go:23` |
| 4 | Q/K/V projection | three matrix multiplies | `attnSeq` `internal/model/forward.go:621` |
| 5 | GQA sharing | `NumHeads / NumKVHeads` queries per KV head | `GroupSize` `internal/model/config.go:1692` |
| 6 | RoPE | rotate Q and K by position | `newRopeForLayer` `internal/model/forward.go:442` |
| 7 | Score + mask + softmax | `dot(q,k)*scale`, causal, normalize | `softmaxInPlace` `internal/model/forward.go:924` |
| 8 | Output projection | concat heads, `o_proj`, add | `attnSeq` `internal/model/forward.go:621` |
| 9 | MLP (SwiGLU) | gate, up, silu, multiply, down | `mlpSeq` `internal/model/forward.go:793` |
| 10 | Decode twin | the same block, one token, cache-backed | `blockStep` `internal/model/kv.go:917` |

## See also

- [Level 2 — The forward pass](level-2-the-forward-pass.md): the pipeline this rung
  opens, and the `x` shape this rung assumes.
- [Level 4 — Prefill vs decode](level-4-prefill-vs-decode.md): why the two block
  implementations have different performance profiles, and what the KV cache does.
- [What is a CUDA kernel?](../what-is-a-cuda-kernel.md): "kernel" means three different
  things in this repo; read it before the `mat.mul` calls confuse you.
- [The abstraction map](../the-abstraction-map.md): where the model sits in the stack.
- [The fak/DOS glossary](../glossary.md): one-line definitions of the vocabulary.

---

*[Full ladder (README)](README.md) · prev ← [Level 2 — The forward pass](level-2-the-forward-pass.md) · next → [Level 4 — Prefill vs decode](level-4-prefill-vs-decode.md)*
