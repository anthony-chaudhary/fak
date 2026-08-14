---
title: "The fak caching ladder: five guides, five audiences"
description: "One subject — fak's caching stack — explained five times, each rung pitched at a different reader and model/agent capability tier. Start at Level 1 (what is a cache?) and climb to Level 5 (the caching frontier)."
slug: caching-ladder
keywords:
  - fak caching guide
  - prompt cache explained
  - managed cache
  - KV cache
  - cache economics
  - caching for AI agents
  - beginner caching guide
  - advanced caching guide
date: 2026-07-10
---

# The fak caching ladder

> **What this is.** The same subject — how fak caches so a long, stop-start agent
> session stays cheap — written **five times**, once for each kind of reader. Each rung
> is pitched at both a human audience *and* a model/agent capability tier: a newcomer or
> a small 8B agent is served at Level 1; a frontier / `fable`-tier reader is served at
> Level 5. Pick your rung, or climb the whole ladder.

Caching in fak spans two very different layers — the **provider's prompt cache** that
fak *steers* on the API wire, and the **kernel-owned KV cache** fak *owns* when it runs
the model itself. A single flat page either loses the beginner or bores the expert. This
ladder splits the difference: each rung assumes exactly what the one below it taught.

## The five rungs

| Level | Who it's for | What you'll get |
|---|---|---|
| **[1 · What is caching?](level-1-what-is-caching.md)** | A newcomer, or a small/8B agent that needs the one-screen version. No jargon. | What a prompt cache is, why it matters, what to type, how to tell it helped. |
| **[2 · Managed cache in practice](level-2-managed-cache-in-practice.md)** | A competent dev or mid-tier agent running `fak manage -- claude`. | `--managed-cache on\|off\|auto`, the defaults, Pro/Max headroom, how to verify. |
| **[3 · Cache economics & the wire](level-3-cache-economics-and-the-wire.md)** | A senior engineer / capable agent who wants the mechanism and the money. | Sliding window vs 1h tier, the read/write multipliers, byte-exact prefix, provider matrix. |
| **[4 · The kernel-owned KV cache](level-4-kernel-kv-cache.md)** | A platform engineer / strong agent running the model in-kernel. | Addressable KV cache, bit-exact span eviction, prefix cloning, WITNESSED vs OBSERVED accounting. |
| **[5 · The caching frontier](level-5-the-caching-frontier.md)** | A `fable`-tier reader / researcher who wants the SOTA and the open problems. | vCache enablement, SOTA optimizations, cross-provider futures, the honest open edges. |

## How to read it

- **Just want it to work?** Read **Level 1**, then **Level 2**, and stop. That's the
  whole user-facing story.
- **Operating a fleet / paying the bill?** **Level 3** is where the economics live.
- **Running the model yourself?** **Level 4** is the kernel layer; **Level 5** is where
  it's going.

Each rung ends with a nav footer linking its neighbours and back here.

## See also

- [What is fak's managed cache?](../what-is-managed-cache.md) — the original plain-English
  managed-cache page; Level 2 is its canonical successor.
- [Addressable KV cache](../addressable-kv-cache.md) · [KV cache for agentic context](../kv-cache-agentic-context.md) — the kernel-owned layer Level 4 builds on.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.
