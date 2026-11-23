---
title: "The forward pass, one token at a time"
description: "The forward pass end to end, with a tiny example to follow by hand: text to ids, ids to vectors, the layer loop, the LM head, and the pick."
slug: the-forward-pass
keywords:
  - forward pass explained
  - transformer forward pass
  - how does a transformer work
  - embedding lookup explained
  - residual stream explained
  - logits to next token
  - argmax vs sampling
  - decoder layer loop
date: 2026-11-18
---

# The forward pass, one token at a time

*You are on **Level 2 of 5** of the [inference ladder](README.md).*

> **In one breath:** The forward pass turns your text into a next token. It
> encodes the text, looks up one vector per token, and runs the layers. Then it
> scores every possible next token. It picks one.

**One line:** The pipeline is `Encode → embedBand → forwardHiddenRows (the layer
loop) → fillSeqLogits → logitsFromHidden → pick`; the hidden state is one
`HiddenSize`-wide row per sequence position, the layer loop mutates it in place,
and only the final position's row is projected to a score per vocabulary entry.

> **Audience.** A new grad who can read Go or Python and knows what a function and
> an array are. You do **not** need to know what attention is — this page names it
> and moves on; Level 3 opens it up. Prerequisites: [Level 1](level-1-what-is-a-token.md)
> (text becomes token ids; the model is a function from ids to a score per
> vocabulary entry, run in a loop). By the end you will be able to draw the entire
> data flow from a prompt string to a chosen next token, and name the function that
> does each step.

> **Short answer.** The forward pass is five moves in a fixed order: **encode** the
> text to ids, **look up** one vector per id, run those vectors through **N identical
> blocks** that keep the same shape the whole way, **turn the last position's vector
> into one score per vocabulary entry**, and **pick** the highest (or sample from
> them). Then append the pick and do it again. Every move is one function in this
> repository, and this page walks all of them.

## What does the whole pipeline look like?

Here is the entire machine on one screen. Read it top to bottom; each box is a
function you will meet below.

```text
  prompt text  "the cat sat"
       │
       ▼
   Encode ────────────────────────────────  internal/tokenizer/tokenizer.go:356
       │
       ▼
   ids []int   [1820, 4758, 3332]
       │
       ▼
   embedBand ─────────────────────────────  internal/model/partition.go:91
       │
       ▼
   x [][]float32   (one row per position, HiddenSize wide)
       │
       ▼
   forwardHiddenRows ─────────────────────  internal/model/forward.go:508
       │    for l in 0 .. NumLayers-1:
       │        layer(l, x, rope) ──────────  internal/model/forward.go:572
       ▼
   fillSeqLogits ─────────────────────────  internal/model/forward.go:544
       │
       ▼
   logitsFromHidden ──────────────────────  internal/model/forward.go:559
       │
       ▼
   logits []float32   (one score per vocabulary entry)
       │
       ▼
   argmaxF32 / SampleLogits
       │
       ▼
   next token id  ──►  append to ids, run the whole thing again
```

That is the whole of inference. Everything else in this ladder — the KV cache,
batching, quantization — is about making this loop fast without changing what it
computes.

## What shape is the data?

This is the single most useful thing to learn first. Almost every bug and every
optimization in inference code comes down to knowing the shape of the array you are
holding. The working array is `x`, of Go type `[][]float32`. It is a **grid**: one
**row per sequence position** (one row per token in the prompt so far), each row
**`HiddenSize` numbers wide**. `HiddenSize` is a property of the model — 4096 is a
common value, 8192 for larger models — and it is fixed for the whole model. Here is
a toy grid: 5 positions, 8 hidden numbers wide.

```text
            h0      h1      h2      h3      h4      h5      h6      h7
        ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
 pos 0  │ 0.11  │-0.42  │ 0.07  │ 0.90  │-0.15  │ 0.33  │ 0.02  │-0.71  │
        ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
 pos 1  │-0.20  │ 0.55  │ 0.14  │-0.08  │ 0.61  │-0.30  │ 0.44  │ 0.19  │
        ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
 pos 2  │ 0.03  │ 0.12  │-0.66  │ 0.28  │ 0.05  │ 0.77  │-0.21  │ 0.40  │
        ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
 pos 3  │ 0.48  │-0.09  │ 0.31  │ 0.22  │-0.53  │ 0.10  │ 0.66  │-0.14  │
        ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
 pos 4  │-0.37  │ 0.24  │ 0.09  │-0.45  │ 0.18  │ 0.52  │-0.03  │ 0.29  │
        └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
          ◄────────────────── HiddenSize = 8 ──────────────────►
```

**The width never changes.** Every function in the stack takes a grid of this shape
and returns a grid of this same shape. The numbers change; the shape does not. That
is what makes this array a **residual stream** — information flowing down the stack,
position by position, read and rewritten but never resized. The number of rows *does*
grow, but only in one place: appending a token to the prompt adds a row. Inside the
stack, the row count is fixed.

## How does text become ids?

**Question:** I have a string. Where does it turn into numbers?

`Encode` does it, at `internal/tokenizer/tokenizer.go:356`:

```go
func (t *Tokenizer) Encode(text string) ([]int, error) {
	return t.encode(text, nil)
}
```

It returns `[]int` — a flat list of token ids, one per token. Level 1 covered what a
token is; here you only need the type. A three-token prompt becomes three ints, for
example `[1820, 4758, 3332]` — indices into the model's vocabulary.

## How do ids become vectors?

**Question:** The model does arithmetic on numbers. How does an integer id become a
vector?

By **lookup**, not computation. `embedBand` at `internal/model/partition.go:91`:

```go
func (m *Model) embedBand(ids []int) [][]float32 {
	cfg := m.Cfg
	H := cfg.HiddenSize
	embed := m.embedRows()
	x := make([][]float32, len(ids))
	for t, id := range ids {
		x[t] = append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x[t], cfg)
	}
	return x
}
```

Read the indexing line carefully, because it is the whole idea:

```go
embed[id*H : (id+1)*H]
```

`embed` is one long flat array holding the **embedding table**: `vocabSize` rows,
each `HiddenSize` wide, laid end to end. Row `id` starts at offset `id*H` and ends
at `(id+1)*H`. So the slice `embed[id*H:(id+1)*H]` is exactly row `id`. The code
copies it out (the `append([]float32(nil), ...)` is a copy) so the model can modify
it without corrupting the table.

There is no matrix multiply here. Embedding is a **table read**. The table itself
was learned during training; at inference time you just fetch rows. The last line,
`scaleEmbedInPlace(x[t], cfg)`, multiplies the row by a per-model constant
(`internal/model/arch.go:512`). Some architectures (Gemma) need it; for Llama it is
a no-op — different model families were trained with different embedding
conventions, and the scale reconciles them.

After `embedBand`, `x` is a grid: `len(ids)` rows, each `HiddenSize` wide. That is
the grid from the previous section.

## What does `Forward` do?

**Question:** Where is the entry point that ties encode and embed together?

`Forward` at `internal/model/forward.go:483`. Here it is with the two
architecture-specific branches elided (they route DeepSeek and GLM variants to
their own forward paths; the shape of the story is identical):

```go
func (m *Model) Forward(ids []int) *Activations {
	// ... arch-specific branches elided ...
	// embedding lookup -> x[t] is the working hidden vector for position t
	// (the arch embed scale applied inside embedBand).
	return m.forwardHiddenRows(m.embedBand(ids))
}
```

Line by line: `m.embedBand(ids)` turns the ids into the grid `x`; then
`m.forwardHiddenRows(x)` runs the grid through the whole stack. The return value is
`*Activations`, a small struct (`internal/model/forward.go:427`) carrying the
sequence length, the hidden grid at every layer, and the logits. So `Forward` is two
calls, and the real work is in the second one.

## What does `forwardHiddenRows` do?

**Question:** What is "the stack"?

`forwardHiddenRows` at `internal/model/forward.go:508`. This function **is** the
forward pass. Read it:

```go
func (m *Model) forwardHiddenRows(x [][]float32) *Activations {
	cfg := m.Cfg
	seq := len(x)
	act := &Activations{Seq: seq, Hidden: [][]float32{flatten(x)}}
	// ... per-arch setup elided ...
	for l := 0; l < cfg.NumLayers; l++ {
		switch {
		case gemma4:
			m.layerGemma4(l, x, gemma4RopeFreqs)
		case cfg.usesMLAMoELayout():
			m.layerGLMDsa(l, x, newRopeForLayer(cfg, l, seq), &glmDsaSharedTopK)
		case cfg.isMiniMaxSparseAttn():
			m.layerMiniMax(l, x, newRopeForLayer(cfg, l, seq))
		default:
			m.layer(l, x, newRopeForLayer(cfg, l, seq))
		}
		act.Hidden = append(act.Hidden, flatten(x))
	}

	m.fillSeqLogits(act, x)
	return act
}
```

Line by line:

- `seq := len(x)` — how many positions we are processing.
- `act := &Activations{...}` — start recording. `flatten(x)` copies the grid into
  one flat slice for the record; this is bookkeeping, not computation.
- `for l := 0; l < cfg.NumLayers; l++` — **the layer loop.** This is the heart.
- The `switch` picks the right block implementation for the model family. For a
  plain Llama-style model it takes the `default` branch: `m.layer(l, x, ...)`.
- `newRopeForLayer(cfg, l, seq)` (`internal/model/forward.go:442`) precomputes the
  rotary position tables for this layer. RoPE is how the model encodes *where* each
  token sits; Level 3 covers it. For now: it is a per-layer input, computed once.
- `act.Hidden = append(act.Hidden, flatten(x))` — snapshot the grid after this layer.
- After the loop, `m.fillSeqLogits(act, x)` — the tail, next section.

**The loop runs the same code every time.** `m.layer` is one function. It is called
`NumLayers` times. A 32-layer model calls it 32 times. The code is identical; only
the **weights** differ — layer 0 reads layer-0 weights, layer 1 reads layer-1
weights, and so on. Picture 32 passes over the same grid, each pass reading a
different set of numbers out of the model file.

Note that `x` is passed by reference and mutated **in place**. The grid is not
rebuilt each layer; it is edited. That is the residual stream again: one buffer,
rewritten 32 times.

## What does one block do?

**Question:** What is inside `m.layer`?

`layer` at `internal/model/forward.go:572`. We will not go deep — that is Level 3 —
but you must leave knowing the shape of a block. A block is six steps: **normalize**,
**attention** (the one step where positions talk to each other — each position looks
at the positions before it and pulls in information), **add the result back**,
**normalize** again, **MLP** (a small per-position network; each position thinks
**alone** here), **add the result back**.

The two "add back" steps are the **residual connections**, and they are why the
shape never changes: each sublayer computes a *change* to add onto the stream, not a
replacement. Draw it as a vertical line with two side-branches:

```text
        x[t]   (the residual stream — one row per position)
          │
          ├──────► norm ──► attention ──► (+) ──┐
          │        (positions talk)            │
          │◄───────────────────────────────────┘   x += attention(norm(x))
          │
          ├──────► norm ──► MLP ──► (+) ──┐
          │        (each position alone)  │
          │◄───────────────────────────────┘       x += mlp(norm(x))
          │
          ▼
        x[t]   (same shape, richer content)
```

In code, that "add back" is the `x[tt][i] += out[tt][i]` loop in
`composeSeqSublayer` (`internal/model/forward.go:847`), which the block calls twice
— once for attention, once for the MLP. The two sublayers themselves are `attnSeq`
(`internal/model/forward.go:621`) and `mlpSeq` (`internal/model/forward.go:793`).

The mental model to keep: **attention mixes across positions; the MLP transforms
within a position; the residual stream carries everything forward.** Every modern
decoder is a stack of that same block.

## What is the tail?

**Question:** After 32 blocks I have a grid. How does a grid become a prediction?

Two functions. First `fillSeqLogits` at `internal/model/forward.go:544`:

```go
func (m *Model) fillSeqLogits(act *Activations, x [][]float32) {
	act.Logits = make([][]float32, act.Seq)
	for t := 0; t < act.Seq; t++ {
		act.Logits[t] = m.logitsFromHidden(x[t])
	}
}
```

It runs the tail **once per position**, producing one score row per position. Then
`logitsFromHidden` at `internal/model/forward.go:559` is the tail itself:

```go
func (m *Model) logitsFromHidden(x []float32) []float32 {
	cfg := m.Cfg
	mat := residentKernel{m}
	xf := m.finalNorm(x)
	logits := mat.mul(m.headName(), mat.prep(xf), cfg.VocabSize, cfg.HiddenSize)
	logitScaleInPlace(logits, cfg)
	return logits
}
```

Line by line:

- `m.finalNorm(x)` — one last normalize of the position's vector
  (`internal/model/weights.go:767`).
- `mat.mul(m.headName(), ..., cfg.VocabSize, cfg.HiddenSize)` — the **LM head**: a
  matrix multiply that takes a `HiddenSize`-wide vector and produces a
  `VocabSize`-wide vector. This is the one place the width changes.
- `logitScaleInPlace(logits, cfg)` — an architecture-specific rescale (a no-op for
  Llama).

Here is the shape change, drawn:

```text
   x[last]  ──►  finalNorm  ──►  LM head (matmul)  ──►  logits
  [HiddenSize]                  [VocabSize × HiddenSize]   [VocabSize]
   e.g. [4096]                                            e.g. [128256]
```

The output is `logits`: one raw score per vocabulary entry. A higher score means the
model thinks that token is a more likely next token. (They are called *logits*
because a softmax turns them into probabilities; the model itself only produces the
raw scores.)

**Only the last position's row is used to predict the next token.** In the simple
case you throw away every other row of `act.Logits` and keep `act.Logits[seq-1]`.
Why? Because the next token comes *after* the last one, and the last position is the
only one that has seen the whole prompt. This is exactly the fact that makes the KV
cache possible: if only the last row matters for the next token, you do not need to
recompute the earlier positions from scratch — you can cache their intermediate
state. That is Level 4.

## How is the next token picked?

**Question:** I have a score per vocabulary entry. Which one do I take?

Two ways, and the difference matters.

**Greedy** — take the highest score. `argmaxF32` at `internal/model/kv.go:1652`:

```go
func argmaxF32(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}
```

It walks the scores once, remembers the largest and its index, and returns the
index. That index is the next token id. Greedy is deterministic: the same prompt
always gives the same output.

**Sampling** — draw from the distribution instead of always taking the top.
`SampleLogits` at `internal/agent/inkernel_sampling.go:14`:

```go
func SampleLogits(logits []float32, temp float64, rng *rand.Rand) int {
	return sampleLogits(logits, temp, 0, 0, rng)
}
```

Its doc comment states the rule: **argmax when `temp <= 0`, else a
temperature-scaled softmax draw.** Temperature controls how adventurous the draw is
— low concentrates on the top scores, high spreads the probability out. Sampling
gives variety; greedy gives reproducibility.

## Worked example: three tokens, by hand

**Question:** Can I see the whole thing with real numbers?

Yes. The **shapes** below are the real ones; the **numbers** are illustrative, chosen
so you can follow them. Toy model: `HiddenSize = 4`, `VocabSize = 6`, `NumLayers = 2`
(real models use thousands and 32+; the arithmetic is identical).

**Step 1 — encode.** Prompt `"the cat sat"` becomes three ids:

```text
  "the cat sat"  ──►  ids = [2, 5, 1]
```

**Step 2 — embed.** `embedBand` copies row 2, row 5, and row 1 out of the table.
The grid is 3 rows × 4 wide:

```text
            h0      h1      h2      h3
        ┌───────┬───────┬───────┬───────┐
 pos 0  │ 0.10  │-0.20  │ 0.30  │ 0.00  │   <- table row 2
        ├───────┼───────┼───────┼───────┤
 pos 1  │ 0.50  │ 0.10  │-0.10  │ 0.20  │   <- table row 5
        ├───────┼───────┼───────┼───────┤
 pos 2  │-0.30  │ 0.40  │ 0.00  │ 0.10  │   <- table row 1
        └───────┴───────┴───────┴───────┘
```

**Step 3 — the layer loop.** Two passes of `layer`. The grid stays 3 by 4; the
numbers change. After both layers, say:

```text
            h0      h1      h2      h3
        ┌───────┬───────┬───────┬───────┐
 pos 0  │ 0.22  │-0.11  │ 0.41  │ 0.05  │
        ├───────┼───────┼───────┼───────┤
 pos 1  │ 0.61  │ 0.03  │-0.22  │ 0.34  │
        ├───────┼───────┼───────┼───────┤
 pos 2  │-0.18  │ 0.52  │ 0.09  │ 0.27  │   <- the row we will use
        └───────┴───────┴───────┴───────┘
```

**Step 4 — the tail.** Take only the last row, `[-0.18, 0.52, 0.09, 0.27]`, run
`finalNorm` then the LM head. The width changes from 4 to 6:

```text
  [-0.18, 0.52, 0.09, 0.27]  ──►  LM head  ──►  [ -1.20, 0.30, 2.10, -0.50, 1.40, -0.80 ]
     HiddenSize = 4                                VocabSize = 6
```

**Step 5 — the pick.** Six scores, one per vocabulary entry:

```text
 vocab id:    0       1       2       3       4       5
 score:    -1.20    0.30    2.10   -0.50    1.40   -0.80
                            ▲
                            └── argmaxF32 picks id 2 (score 2.10)
```

**Step 6 — the loop closes.** Append id 2. The prompt is now `[2, 5, 1, 2]`. Run
the whole pipeline again to get the token after that. Repeat until you have enough
tokens or the model emits its end-of-sequence token.

## How does the loop close?

**Question:** Where is the loop itself?

`Generate` at `internal/model/kv.go:1601`:

```go
func (s *Session) Generate(prompt []int, n int) []int {
	if len(prompt) == 0 {
		return nil
	}
	defer s.BeginGPUKeepAlive()()
	logits := s.Prefill(prompt)
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		next := argmaxF32(logits)
		out = append(out, next)
		if s.M.Cfg.IsEOS(next) {
			break
		}
		logits = s.Step(next)
	}
	return out
}
```

Level 1 showed you this as a black box. Now every line is a function you have met:

- `s.Prefill(prompt)` (`internal/model/kv.go:1093`) runs the forward pass over the
  whole prompt and returns the logits for the last position.
- `argmaxF32(logits)` picks the next token — the greedy pick from above.
- `s.M.Cfg.IsEOS(next)` (`internal/model/config.go:1654`) checks whether the pick is
  a stop token; if so, stop.
- `s.Step(next)` (`internal/model/kv.go:1508`) runs the forward pass for **one new
  token** and returns fresh logits — the decode step, where the KV cache does its
  work (Level 4).

The loop is: prefill once, then step once per token, picking each time.

## See also

- [Level 1 — What is a token, and what is a model?](level-1-what-is-a-token.md) —
  the plain-words version, if any of the above went by too fast.
- [Level 3 — Inside one decoder block](level-3-inside-a-block.md) — the ~10 real
  operations in `layer`, and where each one lives in this repository.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---

*[Full ladder (README)](README.md) · prev ← [Level 1 — What is a token?](level-1-what-is-a-token.md) · next → [Level 3 — Inside one decoder block](level-3-inside-a-block.md)*
