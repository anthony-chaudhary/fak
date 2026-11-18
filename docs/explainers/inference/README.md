---
title: "How inference works: a five-rung ladder (fak as the worked example)"
description: "The same subject — how a language model turns your prompt into the next token — explained five times, each rung pitched at a different reader. Start at Level 1 (what is a token?) and climb to Level 5 (what changes at scale). Every claim is pinned to a real file in this repository."
slug: inference-ladder
keywords:
  - how inference works
  - forward pass explained
  - transformer forward pass
  - prefill vs decode
  - KV cache explained
  - logits and sampling
  - LLM inference for beginners
  - inference internals
date: 2026-11-18
---

# How inference works — the five-rung ladder

> **In one breath:** A language model turns your text into numbers. It then
> scores every word it could say next. It picks one and starts again. That loop
> is inference.

**One line:** The model is a function from a sequence of token ids to a score for
every entry in its vocabulary; generation is that function applied repeatedly,
and everything else in this ladder is about making the function fast without
changing what it computes.

> **Who this is for.** One subject — *what actually happens between your prompt and
> the next token* — written **five times**, once for each kind of reader. A
> newcomer with no math gets Level 1. A working engineer who wants the real code
> paths gets Level 3. A performance engineer gets Level 4. A researcher gets
> Level 5. Each rung assumes exactly what the rung below it taught, and nothing
> more.

Most explanations of inference fail in one of two ways. Either they stay at the
level of "the model predicts the next word" and never show you the machine, or
they open with matrix calculus and lose everyone who has not already seen it.
This ladder splits the difference: **every rung answers a question you can
actually ask**, shows a worked example you can check by hand, and cites the file
in this repository where the real thing lives.

## The one-sentence answer (all five rungs agree on this)

A language model is a function. You give it a sequence of **tokens**; it gives
you back a **probability for every token in its vocabulary** being the next one.
You pick one, append it, and run the function again. That loop is inference.

Everything else — attention, the KV cache, quantization, batching — is about
making that function *fast* and *cheap* without changing what it computes.

## The five rungs

| Level | Who it's for | The question it answers |
|---|---|---|
| **[1 · What is a token, and what is a model?](level-1-what-is-a-token.md)** | A total newcomer. No math, no code. | "I type a sentence. What does the machine actually receive, and what does it give back?" |
| **[2 · The forward pass, one token at a time](level-2-the-forward-pass.md)** | A new grad who can read Go or Python. | "Show me the whole pipeline, end to end, with a tiny example I can follow by hand." |
| **[3 · Inside one decoder block](level-3-inside-a-block.md)** | A working engineer who wants the real code. | "What are the ~10 operations in a block, and where is each one in this repo?" |
| **[4 · Prefill vs decode: why one model has two speeds](level-4-prefill-vs-decode.md)** | A performance engineer. | "Why is the first token slow and the rest fast? Why is decode memory-bound?" |
| **[5 · What changes at scale](level-5-at-scale.md)** | A researcher or systems lead. | "Batching, quantization, MoE, speculative decoding — what do they change, and what is still open?" |

## How to read it

- **Curious, not technical?** Read **Level 1** and stop. It is the whole story in
  plain words.
- **Learning to work on inference code?** Read **1 → 2 → 3**. Level 3 is where
  the repository's real functions appear.
- **Tuning or serving a model?** **Level 4** is the one that changes your
  decisions.
- **Doing research or setting direction?** **Level 5** names the frontier and,
  honestly, the parts nobody has solved.

Each rung ends with a nav footer linking its neighbours and back here.

## How to check a claim on any rung

Every rung cites files as `path:line`. Those paths are real and resolve in this
repository. To check one yourself:

```sh
sed -n '483,500p' internal/model/forward.go   # the Forward entry point
```

Run that from the repository root.

If a citation does not resolve, that is a bug in the page — the pages are
verified by a link-and-citation check, not by trust.

## See also

- [The caching ladder](../caching/README.md) — the same five-rung pattern applied
  to fak's caching stack. This ladder is its sibling.
- [What is a CUDA kernel?](../what-is-a-cuda-kernel.md) — the word "kernel" means
  three different things here; read this before Level 3 if that trips you.
- [The abstraction map](../the-abstraction-map.md) — where the model sits in the
  full stack, from your terminal to the silicon.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.
- [The learning path](../../../LEARNING-PATH.md) — the prerequisite-ordered catalog
  this ladder plugs into (courses FAK 521–530 cover the same ground, rigor-first).
