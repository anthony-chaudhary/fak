---
title: "How tokenization works: a five-rung ladder (fak as the worked example)"
description: "The same subject — how text becomes the numbers a model reads — explained five times, each rung pitched at a different reader. Start at Level 1 (what is a tokenizer?) and climb to Level 5 (the tokenizer frontier). Every claim is pinned to a real file in this repository."
slug: tokenization-ladder
keywords:
  - tokenization explained
  - what is a tokenizer
  - byte pair encoding
  - BPE explained
  - byte level BPE
  - how does tokenization work
  - tokenizer for beginners
  - tokenizer internals
date: 2026-11-18
---

# How tokenization works — the five-rung ladder

> **In one breath:** A tokenizer turns your text into whole numbers. It does that
> by counting which chunks of text appear together often. The chunks it learns
> become the model's vocabulary. Everything the model does starts here.

**One line:** Tokenization is three stages in a fixed order — split the text into
pieces, rewrite each piece as a printable spelling of its bytes, then merge
adjacent symbols by a rank table learned in training — and the merge stage is the
one with a real algorithm in it, a naive reference loop and an incremental
frontier that must agree bit for bit.

> **Who this is for.** One subject — *how a string of text becomes the list of
> numbers a model reads* — written **five times**, once for each kind of reader. A
> newcomer with no math gets Level 1. A new grad who can read Go gets Level 2. A
> working engineer who wants the real algorithm gets Level 3. A performance
> engineer gets Level 4. A researcher gets Level 5. Each rung assumes exactly what
> the rung below it taught, and nothing more.

The [inference ladder](../inference/README.md) opens by telling you that a model
never sees your text, only a list of token ids. It then moves on, because its
subject is what happens *after* the ids exist. This ladder is the page that was
missing: it stays on the step the inference ladder skips, and it is the only place
in this repository where the merge algorithm is written out in full.

That matters more than it sounds. Tokenization is the first thing that happens and
the thing every later number depends on. A wrong token boundary shifts a span, and
a shifted span evicts the wrong rows from the key/value cache. The tokenizer
package says so itself, in its own package doc: a wrong text-to-id boundary "can
shift a `Segment.Len` and evict the wrong K/V rows"
(`internal/tokenizer/doc.go`). Nothing downstream can repair a mistake made here.

## The one-sentence answer (all five rungs agree on this)

A tokenizer is a **lossless, reversible map** between text and a list of whole
numbers. It is lossless in both directions: `Encode` then `Decode` gives you back
exactly the bytes you started with, including the ones that are not valid text.

Everything else — the split rules, the byte spelling, the merge ranks, the heap —
is about making that map *fast* and *stable* without changing what it maps.

## The five rungs

| Level | Who it's for | The question it answers |
|---|---|---|
| **[1 · What is a tokenizer?](level-1-what-is-a-tokenizer.md)** | A total newcomer. No math, no code. | "I type a sentence. How does it become numbers, and why those numbers?" |
| **[2 · The three stages](level-2-the-three-stages.md)** | A new grad who can read Go or Python. | "Show me the whole pipeline, end to end, with an example I can follow by hand." |
| **[3 · The merge loop, by hand](level-3-the-merge-loop-by-hand.md)** | A working engineer who wants the real algorithm. | "What is the merge rule exactly, and how do I know two implementations agree?" |
| **[4 · Why the merge loop is the hot path](level-4-why-the-merge-loop-is-the-hot-path.md)** | A performance engineer. | "Why is the naive version quadratic, and what does the fast one do instead?" |
| **[5 · The tokenizer frontier](level-5-the-tokenizer-frontier.md)** | A researcher or systems lead. | "How do you prove a tokenizer is right, and what is still open?" |

## How to read it

- **Curious, not technical?** Read **Level 1** and stop. It is the whole idea in
  plain words.
- **Learning to work on tokenizer code?** Read **1 → 2 → 3**. Level 3 is where the
  repository's real algorithm appears.
- **Chasing a tokenization cost or a slow encode?** **Level 4** is the one that
  changes your decisions.
- **Doing research or setting direction?** **Level 5** names the frontier and,
  honestly, the parts nobody has solved.

Each rung ends with a nav footer linking its neighbours and back here.

## How to check a claim on any rung

Every rung cites files as `path:line`. Those paths are real and resolve in this
repository. To check one yourself:

```sh
sed -n '22,30p' internal/tokenizer/bpe_merge.go   # the merge entry point
```

Run that from the repository root.

If a citation does not resolve, that is a bug in the page — the pages are
verified by a link-and-citation check, not by trust.

## See also

- [The inference ladder](../inference/README.md) — the parent ladder. Its Level 1
  says the tokenizer decides the chunks by counting; this ladder is where that
  sentence is cashed out.
- [The caching ladder](../caching/README.md) — the same five-rung pattern applied
  to fak's caching stack.
- [The mixed-precision ladder](../mixed-precision/README.md) — what the numbers
  are stored in, once the ids exist.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.
- [The learning path](../../../LEARNING-PATH.md) — the prerequisite-ordered
  catalog this ladder plugs into. Course **FAK 522** covers the same ground
  rigor-first.
