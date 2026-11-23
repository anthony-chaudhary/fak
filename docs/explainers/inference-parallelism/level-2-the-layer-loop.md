---
title: "The layer loop: what a model does between embedding and logits"
description: "A transformer is a stack of N same-shaped layers: the layer loop runs attention plus an MLP in each, wrapped in a pre-norm and a residual."
slug: the-layer-loop
keywords:
  - transformer layer explained
  - attention and MLP
  - residual stream
  - pre-norm residual
  - why layers are the split point
  - pipeline parallelism basics
  - fak inference ladder
date: 2026-11-18
---

# The layer loop

The layer loop is the part of a transformer between the embedding and the logits: a stack of N same-shaped layers, each an attention sublayer plus an MLP sublayer wrapped in a pre-norm and a residual. The residual stream is a shared bus every layer reads and writes.

*You are on **Level 2 of 5** of the [fak inference ladder](README.md).*

> **Audience.** You know what a forward pass is — a token id goes in, a score for every
> next token comes out ([Level 1](level-1-what-is-inference.md)). Now you want to know
> what happens in the middle. No attention math required; we care about the *shape* of
> the computation, not the algebra inside it.

> **Short answer.** The middle is a **stack of N layers of identical shape**. Each layer
> is two sublayers — attention, then a small feed-forward network — and each sublayer is
> wrapped in a normalization and a residual add. The residual add means every layer
> *reads* a shared vector and *writes back* a modified copy of it, so the vector flows
> through the stack like a bus with N stops. Because the layers are sequential, the same
> shape, and connected by a small vector, the boundary between two layers is the natural
> place to cut the model in half.

## What is between the embedding and the logits?

In [Level 1](level-1-what-is-inference.md) we skipped Step 2 with a hand-wave: "the
hidden state goes through a stack of layers and comes out the other side." Let us open
that box.

The box is a **loop**. In fak it is literally a `for` loop
(`internal/model/forward.go:508`):

```go
for l := 0; l < cfg.NumLayers; l++ {
    // ... run layer l over the hidden state x, in place ...
}
```

`cfg.NumLayers` is the depth of the model — 32, 80, 120, whatever the checkpoint says.
The loop runs the *same* code N times, once per layer. The only thing that changes
between iterations is which layer's weights get used.

## What is one layer?

Each layer is two sublayers in sequence:

1. **Attention** — lets each token look at the other tokens and pull in information from
   them. This is how the model relates "cat" to "sat".
2. **MLP** (multi-layer perceptron) — a small feed-forward network applied to each token
   independently. This is where most of the model's raw parameters live.

In fak, the per-layer dispatch inside `forwardHiddenRows`
(`internal/model/forward.go:508`) calls `layer`, which builds exactly these two as
`attnSub` and `mlpSub` and composes them. The comment on `layer` says it plainly:
"attention + MLP, each with a pre-norm and a residual."

## The two wrappers: pre-norm and residual

Each sublayer is not applied to the hidden state directly. It is wrapped:

```text
  sublayer output = x + Sub( Norm(x) )
                    ^     ^
                    |     +-- the sublayer reads a NORMALIZED copy of x
                    +-------- the result is ADDED back onto the original x
```

Two things are happening:

- **Pre-norm.** The sublayer reads `Norm(x)`, a rescaled version of the hidden state,
  not `x` itself. "Pre" because the norm comes *before* the sublayer. This keeps the
  numbers in a sane range as they pass through many layers.
- **Residual.** The sublayer's output is *added* to `x`, not substituted for it. The
  original `x` survives into the next layer untouched, with the sublayer's contribution
  layered on top.

The residual is the important one for this ladder. It means a layer can only **add to**
the hidden state. It cannot erase it. Whatever the embedding put there is still there,
in some form, at the very last layer.

## The residual stream: a bus with N stops

Because every layer adds to the same vector, picture the hidden state as a **bus route**
and the layers as stops along it:

```text
  embedding
     │
     ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  the residual stream: one vector, x, of shape [hidden]        │
  └──────────────────────────────────────────────────────────────┘
     │
     ├─ layer 0 ──►  x = x + Attn(Norm(x));  x = x + MLP(Norm(x))
     │
     ├─ layer 1 ──►  x = x + Attn(Norm(x));  x = x + MLP(Norm(x))
     │
     ├─ layer 2 ──►  x = x + Attn(Norm(x));  x = x + MLP(Norm(x))
     │
     │      ...  (N layers, all the same shape)  ...
     │
     ├─ layer N-1 ►  x = x + Attn(Norm(x));  x = x + MLP(Norm(x))
     │
     ▼
  final norm ──► LM head ──► logits
```

Every layer reads the bus and writes back to the bus. The bus is the *only* thing that
connects one layer to the next. Nothing else crosses a layer boundary.

This is why people call it a "stream": information is added to it at each stop and
carried forward, and the final answer is read off the end of the route.

## What shape is the hidden state at each boundary?

Here is the part that makes the whole ladder work. The hidden state has the **same
shape at every boundary** — before layer 0, between any two layers, and after the last
layer:

```text
  boundary            shape of x
  ------------------  ---------------------------
  after embedding     [seq][hidden]
  after layer 0       [seq][hidden]     <- same
  after layer 1       [seq][hidden]     <- same
  ...                 [seq][hidden]     <- same
  after layer N-1     [seq][hidden]     <- same
  after final norm    [seq][hidden]     <- same
  after LM head       [seq][vocab]      <- shape changes HERE, once, at the very end
```

`seq` is the number of tokens in the input; `hidden` is the width of the bus (4 in our
Level 1 napkin example, thousands in a real model). The shape only changes at the very
last step, when the LM head turns each hidden vector into a score per vocabulary entry
(`logitsFromHidden`, `internal/model/forward.go:559`).

So the tensor crossing a layer boundary is **small** — one `[seq][hidden]` block — and
it is the *same* block at every boundary. That is a very convenient fact, and the next
section is about why.

## Why is the layer boundary the natural place to cut?

Suppose the model is too big to fit on one worker, and you want to split it across two.
Where do you cut? Three properties of the layer stack make the boundary between two
layers the obvious seam:

1. **The layers are sequential.** Layer 5 cannot start until layer 4 has finished — it
   reads layer 4's output. So the work is already ordered; a cut just assigns the early
   layers to one worker and the late layers to another, in that order.
2. **The layers are the same shape.** Every layer has the same weights-shape and the
   same code. So "run layers 0..k" and "run layers k..N" are the *same operation* on
   different data — one function, two calls.
3. **The tensor crossing the boundary is small.** Only the `[seq][hidden]` bus crosses.
   The weights — the billions of numbers — stay put on whichever worker owns them. You
   are not shipping the model across the wire; you are shipping one vector.

fak makes this concrete with `ForwardBand` (`internal/model/partition.go:122`), which
runs a contiguous band of layers `[lo, hi)` over the hidden state. Its doc comment states
the identity that ties the whole ladder together:

> `Forward` is exactly `embedBand(ids)` followed by `ForwardBand(x, 0, NumLayers, true)`.

In other words: the monolithic forward pass *is* a single band covering every layer. Cut
the band in two and you have two workers; the answer does not change.

## How does fak decide which layers a worker keeps?

Two small pieces of bookkeeping make the cut real:

- `layerWindow` (`internal/model/partition.go:28`) is a half-open band `[lo, hi)` of
  layer indices.
- `keepsLayer` (`internal/model/partition.go:40`) answers "does this worker keep layer
  N?" — and, importantly, always keeps *layer-agnostic* weights (the embedding, the
  final norm, the LM head), because a worker that owns the head or the embedding needs
  them no matter which band it holds.
- `WithLayerWindow` (`internal/model/partition.go:61`) is the load option a worker passes
  to keep only its own band's weights, so it never loads the layers it does not run.

That is the whole idea of pipeline parallelism in three functions: a worker loads its
band, runs its band, and hands the bus to the next worker. [Level 3](level-3-pipeline-parallelism.md)
covers the handoff itself.

## Try it

```bash
go test ./internal/model/ -run 'TestForwardBandTwoWayMatchesMonolithic' -count=1 -v
```

You should see `PASS`. This test is the proof that the stack can be cut in half and still
produce the same answer: it runs the full `Forward`, then runs the same model as two
bands — `embedBand` + layers `[0, k)`, then layers `[k, N)` + the head — and asserts the
two sets of logits are **bit-for-bit identical** (`max|delta| = 0`). If the residual
stream and the layer boundary were not a clean seam, this test would fail.

## See also

- [Level 1 — What is inference?](level-1-what-is-inference.md): the token loop and the
  worked example this rung builds on.
- [Level 3 — Pipeline parallelism](level-3-pipeline-parallelism.md): what crosses the
  wire when the band lives on another worker.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the words used here.

---
*[Full ladder (README)](README.md) · next → [Level 3 — Pipeline parallelism](level-3-pipeline-parallelism.md)*
