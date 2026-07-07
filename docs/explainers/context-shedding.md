---
title: "Context shedding: how fak trims a long session from the middle out"
description: "Every turn, an agent re-sends its whole transcript. fak sheds the stale middle turns while keeping the cached head and your recent work, so the trimmed history stops costing you every turn. What it saves, what it doesn't, and how to read the number honestly."
slug: context-shedding
keywords:
  - context shedding
  - history compaction
  - trim the middle out
  - agent token savings
  - long session cost
  - prompt cache preservation
  - middle-out compaction
  - context health
date: 2026-07-06
---

# Context shedding

*Who this is for: anyone running a long agent session (Claude Code, a coding
agent, any loop that keeps talking to a model) who wants to know what fak's
default "trim the middle" saving actually is — in plain terms, and honestly.*

## The problem, in one picture

Every turn, your agent sends the **entire conversation so far** to the model.
Turn 1's message, turn 2's, the tool output from turn 3, all of it — every time.
The model has no memory between calls, so the transcript *is* the memory, and it
gets re-sent from scratch on every turn.

So the cost of one turn is not "the new thing you just said." It's "everything
that has ever been said, again." A 200-turn session pays for its early turns two
hundred times.

The provider softens this with a **prompt cache**: if the front of the request
is byte-for-byte identical to last turn, the provider serves that prefix from its
cache at roughly a tenth of the price. That helps a lot — but only for the
*unchanged front*. The growing **middle** of the transcript keeps shifting as new
turns land, so it keeps missing the cache and keeps costing full price, every
turn, forever.

That middle is the money leak. Shedding is how fak plugs it.

## What shedding does

When the resident history grows past a budget (48,000 tokens by default), fak's
compaction lever trims it back down. It keeps three things and drops one:

- **Keeps the cached head** — the system prompt, the tools, the stable opening.
  This is the part the provider's cache is discounting, so fak keeps it
  **byte-for-byte identical**. Touch one byte and the discount collapses; fak
  doesn't touch it.
- **Keeps your recent turns** — the last several exchanges, the working set the
  model actually needs to keep going.
- **Drops the stale middle** — the aged turns in between, the ones far enough back
  that they're no longer load-bearing.
- Leaves a **small stub** where the middle was, plus a restore handle so the
  dropped span can be paged back in if a later turn turns out to need it.

That's the "trim from the middle out." The head stays cached, your recent context
stays intact, and the dead weight in the middle stops riding along on every future
turn.

## Why this beats summarizing

Most agent products "compact" by asking the model to **write a summary** of the
old turns and then throwing the originals away. That's lossy and self-certified:
the summary might drop the one file path, rejected approach, or "don't touch X"
constraint that matters ten turns later, and nothing can tell you it did. Worse,
the summary is *new bytes* in the middle of the prompt — which can bust the very
prompt cache you were trying to save. (The full case against summarize-and-discard
is in the [built-in compaction audit](../notes/BUILT-IN-COMPACTION-AUDIT-2026-07-06.md).)

Shedding is different in kind. It doesn't rewrite history into prose — it **drops
a span and remembers where it went**. The head the provider is caching is never
rewritten, so the discount survives. And because the dropped span is kept
addressable (not summarized into oblivion), it can be restored by reference rather
than reconstructed from a lossy recap. Summarizing is *guessing what mattered*;
shedding is *deferring the question* until something actually asks.

## What it saves — and how to read the number honestly

Here is the honest part, because it's easy to get wrong (we got it wrong once, in
public, and corrected it — see the retraction below).

**The real saving is per-turn recurrence.** A token you shed from the middle is a
token you would otherwise have re-sent and re-paid for on *every remaining turn*.
Drop it once, and every future turn is a little cheaper. Over a long session those
per-turn savings add up — that's the genuine win, and it grows with how long the
session runs and how much middle there is to trim.

**But "tokens shed" is not the same as "distinct tokens saved."** Here's the
subtlety. fak doesn't keep a compacted copy of the transcript — the client
(Claude Code) re-sends its *full, uncompacted* history every turn, so fak
re-trims from scratch each time compaction fires. If a session's compaction fires
seven times, the same aged middle turns get dropped — and counted — on all seven
fires. The running total of "tokens shed" therefore **re-counts the same tokens**
once per fire.

So a raw `compaction_shed_tokens` of, say, 747,000 on a single session does **not**
mean fak removed 747,000 distinct tokens — a session whose live context never
exceeds ~300,000 tokens *can't* have 747,000 distinct tokens to remove. It means
fak trimmed roughly 107,000 tokens per fire, seven times, as the middle kept
regrowing and getting re-trimmed.

**The honest way to state the win:**

- Cite the trim **per fire** (`compaction_shed_tokens ÷ compaction_fired`) — "about
  a third of a turn's worth of context, trimmed each time compaction fires" — not
  the session total.
- Don't turn the session total into a "share of savings" by dividing it against the
  provider's cache-read count. Those are two different running sums over overlapping
  content, in different currencies; the ratio isn't a like-for-like share of
  anything.
- fak's own authored slice is real but **smaller than the provider's cache
  discount** on the Claude Code route. Fleet-wide it runs roughly 0.3–16% of the
  total tokens saved (`fak cachevalue report`), and its job on top of that is to
  keep the provider's much larger discount *alive* as the session grows.
- Value a shed token at what the provider would actually have billed it, not at
  full input. On a **warm** fire the dropped tokens were already a provider
  cache-read (billed at ~0.1× base input), so shedding them saves that 0.1× read
  marginal — not the 1.0× of fresh input. Only an observed-**cold** fire, where
  the prefix's cache had expired, avoids a full-input billing and is worth 1.0×.
  Booking every warm shed at 1.0× is the same ~10× over-count the retraction below
  came from; the report now stamps each figure with its price basis
  (`CACHE_READ_MARGINAL` vs `FULL_INPUT`), so a shed number is never read without
  knowing how it was priced.

> **A retraction, on the record.** An earlier version of the README and benchmark
> authority claimed fak's own share "climbs from ~15% to ~75% as the session gets
> long," citing a session that reported 75% by exactly the mistaken math above. A
> forensic audit found the numerator double-counted the re-trimmed middle across
> fires. We corrected it the same day. The lesson is baked into this page: **shed
> per fire, never the session sum; never ratio a cumulative shed against a
> cumulative cache-read.** Provenance and the code pointers are in
> [BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md) (the compaction row).

## Was the shed content pure value, or did it hurt?

Shedding only saves money if the tokens you dropped were genuinely dead weight. If
a later turn reaches for something in the shed span — re-reads a file whose content
was dropped, cites a result that's gone — the model can get confused, and the
"saving" cost you a turn of thrashing instead.

This is the same shape as a classic operating-system question: **when is it safe
to free a page?** The answer there is reference counting — you free a page when
nothing live points at it anymore, and freeing one that's still referenced is a
use-after-free bug. Context shedding wants the same discipline: a span is safe to
shed when no later turn still references it, and shedding a still-referenced span is
the context equivalent of a use-after-free — the deterministic signature of the
model getting confused.

That reference-counting view of context health — turning "was the model getting
confused?" into a mechanical, checkable signal rather than a fuzzy judgment call —
is where fak's context work goes next. It's tracked as its own line of work; this
page is the *what and why* of the trim itself.

## Try it

```bash
fak guard -- claude          # compaction is on by default at a 48K budget
fak cachevalue report        # the owner-attribution + per-fire compaction health
```

Tune or disable the trim with `fak guard --compact-history-budget <tokens>` (`0`
turns it off). Every `fak info` line shows the live `provider X% + fak Y%` split,
so you can watch the two savings — the provider's cache discount and fak's own
trim — side by side, honestly, without either one claiming the other's number.

## See also

- [Long-session economics](long-session-economics.md) — why the transcript is
  re-sent and why the cache discount depends on a byte-identical prefix.
- [Built-in compaction audit](../notes/BUILT-IN-COMPACTION-AUDIT-2026-07-06.md) —
  why the summarize-and-discard compaction other products ship is weak, and what
  "good" looks like.
- [BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md) — the compaction row, with
  the per-fire framing and the double-counting fences.
