---
title: "What is caching? The one-screen version"
description: "A prompt cache lets the AI provider remember the part of your conversation it has already seen, so each new turn is faster and cheaper. Level 1 of the fak caching ladder: no jargon, one analogy, one command."
slug: what-is-caching
keywords:
  - what is caching
  - what is a prompt cache
  - prompt cache explained simply
  - caching for beginners
  - AI agent caching
  - fak caching ladder
date: 2026-07-10
---

# What is caching?

*You are on **Level 1 of 5** of the [fak caching ladder](README.md).*

> **Audience.** Complete beginners — no AI or systems background assumed. By the end
> you'll be able to say what a prompt cache is, why an agent gets slower and more
> expensive without one, and why fak treats keeping it warm as its job.

> **Short answer.** An AI agent forgets everything between turns, so every time you ask
> it something it re-sends the *whole conversation so far* just to ask the next thing.
> A **prompt cache** lets the provider say "I've already seen that part — no need to
> process it again from scratch," which makes each turn faster and cheaper. fak's job
> is to keep that cache working for you, automatically.

## Why does the agent re-send everything?

The model has no memory of its own. Each turn, your agent bundles up everything said so
far — your instructions, its answers, the files it read — and sends it all again, plus
your new question. The conversation grows, so each turn re-sends a little more than the
last. Without a cache, the provider does the full work of reading all of it, every time.

## The coat check

Think of a prompt cache as a **coat check that resets its closing time each time you
touch your coat**.

- The first time, you hang up the whole coat — that's the provider reading your full
  conversation once and keeping a copy.
- Every time you come back quickly, pickup is fast and cheap — and touching your coat
  buys you more time before the rack closes.
- But if you stay away too long, the rack is cleared. Next time you start over and hang
  everything up again.

So the cache rewards **staying active** and punishes **long pauses**. (How long is "too
long," and what fak does about it, is exactly what [Level 2](level-2-managed-cache-in-practice.md)
covers.)

## What do I type?

Nothing extra. You run your agent through fak the normal way:

```bash
fak manage claude
```

Caching is the provider's feature, and fak looks after it for you from there. There are
knobs (Level 2), but you don't need any of them to benefit.

## How can I tell it helped?

One command, after you've used a session for a bit:

```bash
fak cachevalue report --dev-sessions
```

It analyzes your own recent sessions and shows what caching saved. On a brand-new
machine it may read zero until you've run a few sessions — that's the ledger still
filling, not the coat check failing.

## Try it

```bash
fak manage claude                      # run your agent through fak, as usual
fak cachevalue report --dev-sessions   # later: see what caching saved in your sessions
```

## See also

- [Level 2 — Managed cache in practice](level-2-managed-cache-in-practice.md): the
  settings, the defaults, and what "managed cache" in the startup banner means.
- [What is fak's managed cache?](../what-is-managed-cache.md) — the original
  plain-English page this ladder grew from.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the words used here.

---

*[Full ladder (README)](README.md) · next → [Level 2 — Managed cache in practice](level-2-managed-cache-in-practice.md)*
