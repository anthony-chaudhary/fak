---
title: "The fak inference ladder: five guides, five audiences"
description: "How a transformer produces a token, and how fak splits that inference work across many workers, explained five times from first question to frontier."
slug: inference-parallelism-ladder
keywords:
  - fak inference guide
  - how inference works
  - transformer forward pass explained
  - pipeline parallelism explained
  - layer loop
  - tensor parallelism
  - expert parallelism
  - inference for beginners
  - inference for new grads
date: 2026-11-18
---

# The fak inference ladder

> **What this is.** The same subject — how a language model turns your text into the
> next token, and how fak cuts that work into pieces that run on many workers at once —
> written **five times**, once for each kind of reader. Each rung assumes exactly what
> the rung below it taught. A newcomer who has only ever called an API is served at
> Level 1; a reader who wants the open problems of parallel serving is served at
> Level 5. Pick your rung, or climb the whole ladder.

Inference is one of those subjects where the beginner's page and the expert's page
share almost no sentences. "The model predicts the next word" is true and useless to
someone debugging a pipeline stall; "the hidden state crosses the stage boundary as a
12-byte header plus a row-major float32 band" is precise and useless to someone who has
never seen a forward pass. This ladder splits the difference.

## The spine

Every rung is a different zoom level on **one** path through the code. It is worth
seeing the whole path once, in order, before you pick a rung:

```text
  one token id
      │
      ▼
  embedBand(ids)                     internal/model/partition.go:91
      │   token id -> a row of numbers (the hidden state)
      ▼
  the layer loop                     internal/model/forward.go:508
      │   for l := 0; l < cfg.NumLayers; l++ { ... }
      ▼
  ForwardBand(x, lo, hi, isLast)     internal/model/partition.go:122
      │   the SPLITTABLE unit: a contiguous band of layers
      ▼
  PipelineStage + StageTransport     internal/model/pipeline.go:51, :225
      │   one band per worker; the hidden state crosses the wire
      ▼
  ForwardTP / ForwardEP              internal/model/tensor_parallel_forward.go:548
      │   inside a single layer: split the matmuls / the experts
      ▼
  logitsFromHidden                    internal/model/forward.go:559
      │   the shared tail: final norm -> LM head -> logit scale
      ▼
  next-token logits
```

The last line of that diagram is the load-bearing one. `Forward` and every parallel
twin of it — `ForwardBand`, `ForwardTP`, `ForwardEP` — are required to **end
identically** at `logitsFromHidden` (`internal/model/forward.go:559`). That is what
makes the whole ladder a story about *the same computation, cut differently*, rather
than a story about several different computations that happen to look alike. Each rung
below is one cut.

## The five rungs

| Level | Who it's for | What you'll get |
|---|---|---|
| **[1 · What is inference?](level-1-what-is-inference.md)** | A reader with no ML background at all — a new grad, a student, a backend engineer who has only ever called an API. | A model is a pile of numbers; inference runs them over your input; a token is a chunk of text; the model does one token at a time and feeds it back in. One analogy, one worked example you can do by hand. |
| **[2 · The layer loop](level-2-the-layer-loop.md)** | Someone who now knows what a forward pass is and wants to know what happens *between* the embedding and the logits. | The stack of N same-shaped layers; attention + MLP wrapped in pre-norm + residual; the residual stream as a shared bus; why the layer boundary is the natural place to cut. |
| **[3 · Pipeline parallelism](level-3-pipeline-parallelism.md)** | A competent engineer who wants to run a model too big for one worker. | `ForwardBand` as the unit, `PipelineStage` and `StageTransport` across workers, what actually crosses the wire, and the bit-exactness proof that a cut stack gives the same answer. |
| **[4 · Tensor and expert parallelism](level-4-tensor-and-expert-parallelism.md)** | A senior engineer who wants to split *inside* a layer, not just between layers. | `ForwardTP` (column/row sharding of the matmuls) and `ForwardEP` (routing experts to ranks), the collectives they need, and how both stay pinned to `Forward`. |
| **[5 · The parallelism frontier](level-5-the-parallelism-frontier.md)** | A researcher who wants the composition of the axes and the honest open edges. | Pipeline × tensor × expert as one grid, the shared tail that keeps them consistent, and what is witnessed versus what is still hardware-gated. |

## How to read it

- **Never seen a forward pass?** Read **Level 1**, then **Level 2**, and stop. You will
  be able to explain inference to someone else.
- **Need to serve a model that does not fit one worker?** **Level 3** is the pipeline
  axis; **Level 4** is the tensor and expert axes.
- **Designing the serving plan?** **Level 5** is where the axes compose and where the
  open problems live.

Each rung ends with a nav footer linking its neighbours and back here.

## See also

- [The inference ladder](../inference/README.md) — the **sibling** ladder, and the right
  place to start if you have not yet seen a forward pass. It is the broad survey: what a
  token is, the forward pass end to end, the ~10 operations inside one decoder block,
  prefill vs decode, and a short tour of what changes at scale. This ladder is the
  **deep dive on one axis of that survey** — the layer stack and how you cut it across
  workers. Read the sibling first for the whole picture; read this one when the question
  is "how do I split a model that does not fit on one worker?"
- [Multi-GPU tensor parallelism in fak](../multi-gpu-tensor-parallelism.md) — the
  architecture reference for the **tensor axis**. Level 4 introduces the idea; that page
  is the full setup/API/collective-seam document. This ladder cross-links to it rather
  than duplicating it.
- [The fak caching ladder](../caching/README.md) — the sibling ladder. Caching is about
  *not recomputing* work you have already done; inference is about *doing* the work.
  They meet at the KV cache, which Level 3 and Level 4 touch on.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.
- [Context](../context.md) · [Addressable KV cache](../addressable-kv-cache.md) — the
  context and cache layers the serving path sits on top of.
