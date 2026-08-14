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
  - goal pin
  - originating-task tombstone
  - context restore
date: 2026-07-10
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

![One compaction fire on the outbound request body: the protected prefix (through the first cache breakpoint) is copied byte-for-byte so its cache discount survives, the stale middle is dropped to a one-message stub carrying a restore handle, and the recent window is untouched — the "after" bar is visibly shorter than "before".](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/73-compaction-wire.svg)

The same picture as a plain-text twin, so it reads even where the image doesn't
load:

```text
COMPACTION: ONE FIRE, ON THE WIRE   [outbound request body]
legend: █ protected prefix (cache hit ~0.1x)  ▓ stale middle (cache miss, full price)  █ recent window (working set)  ░ stub + restore handle

before │███████████████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██████████████████████│ ~62,000 tok  (over the 48K budget -> fires)
 after │███████████████░██████████████████████                              │ ~34,600 tok  (prefix + window untouched)
```

The prefix band is copied by a byte splice (a `memcpy`, never a re-serialize), so the
provider sees a prefix that is `bytes`-equal to last turn and keeps the cache discount
on it. The stale middle collapses to that one-message stub; the recent window rides
through unchanged.

### The guarantees

The trim is built to fail safe, because a botched rewrite of the request body is worse
than no trim at all:

- **Identity on any ambiguity.** If anything about the body is unexpected — no cache
  breakpoint to anchor on, too few messages, a shape the splicer can't prove it
  understands — the transform returns its input **unchanged** and records *why* it
  bailed (a closed vocabulary: `no_breakpoint`, `too_few_msgs`, `prefix_mismatch`, and
  a dozen more). It never emits a body it can't stand behind.
- **The splice is proven, not assumed.** After building the trimmed body, fak
  re-decodes it and checks that the protected prefix bytes survived verbatim
  (`prefix_mismatch`) and that no message was left empty or malformed
  (`malformed_body`) — the shapes the provider would reject with a 400. If that proof
  fails, it ships the original.
- **Tool pairs stay intact.** The kept window is chosen so a `tool_result` is never
  separated from the `tool_use` it answers — the classic way a naive trim produces a
  malformed request.
- **Request-side only.** Shedding rewrites the bytes fak *sends upstream* for this one
  call; it changes nothing the kernel adjudicates. The full history is still there to
  audit — the trim is a transport-layer economy, not a rewrite of the record.

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

## What survives a drop: the pin and the tombstone

"Drops a span and remembers where it went" is only honest if you can say *what* the
stub remembers. Two things ride in it, depending on what was in the middle:

```text
WHAT THE STUB CARRIES — the one message that stands in for the dropped middle

 a pinned goal is present    [fak:goal] keep the invoice parser backward-compatible   <- hoisted VERBATIM
 (the fidelity path)         [fak] compacted 12 earlier turn(s) ... detail omitted        (nothing lost)

 an unmarked first task      [fak] compacted 12 earlier turn(s) ... detail omitted
 (the oriented fallback)     [fak] originating task (compacted): id=9f3c...a1 "port the
                             Q4_K decode kernel to AVX2 and prove it bit-exact"
                                  |__ id is a callable sha256 handle:
                                      fak_context_restore(id=9f3c...a1) -> pages the full turn back, verbatim
                                      fak_context_spans()               -> lists what is still restorable
```

**The goal pin.** If a message is tagged `[fak:goal]` — a standing instruction like an
acceptance criterion, a "keep X backward-compatible", the actual task the loop is
working toward — and it happens to fall in the compactible middle, fak **hoists it out
verbatim** and sets it beside the stub instead of dropping it. A goal you'd lose to a
badly-timed compaction survives in full. On the pin path the stub carries no
tombstone: nothing was lost, so there's nothing to leave a marker for.

**The originating-task tombstone.** The common case has no pin — the first user turn
*is* the task, unmarked. Rather than launder it into a bare "compacted 12 turns," fak
leaves a **bounded excerpt** of that first turn (whitespace collapsed to one line,
capped so it stays a low-volume, cache-untouched addition) plus a **content-address
handle**. The excerpt says *what* was dropped; the handle says *how to get all of it*.
You are reading a live instance of this right now: the top of this very session shows a
`[fak] originating task (compacted): id=… "…"` line — that is the tombstone, doing its
job.

## Getting a dropped span back: restore

The handle in the tombstone is not decoration — it is callable. It is a sha256
content-address (the same addressing scheme fak uses across recall and its context
planner), and two MCP tools resolve it:

- `fak_context_restore(id)` — pages the **full dropped bytes** back into context,
  verbatim. You recover the original turn, not a summary of it. A successful restore is
  stamped `WITNESSED` — these are bytes fak actually held and handed back, not
  reconstructed.
- `fak_context_spans()` — lists what is still restorable in this session, so a resuming
  model can discover its handles instead of guessing them.

Restore is **trust-gated**, not a blanket undo. A span that an operator sealed
(quarantined) or explicitly tombstoned is **refused**, not resurrected; an unknown or
already-evicted handle returns a clean miss. So "get it back" never becomes a way to
launder poisoned or withdrawn context back into the prompt.

This is what turns the trim from a gamble into a *deferral*. You don't have to be right
about what was dead weight. Drop the middle; if a later turn reaches for something in
it, the handle is sitting right there in the stub, and you page the span back in
instead of thrashing on a lossy recap. The dropped context is *deferred*, not
destroyed.

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
  full input. Within a fire the **warm** portion of the shed — the tokens the
  provider was already serving from cache (`min(shed, cache_read)`, billed at ~0.1×
  base input) — saves only that 0.1× read marginal, while the **cold** remainder
  (shed beyond the witnessed warm prefix) avoids a full-input billing and is worth
  1.0×. Booking every warm shed at 1.0× was a ~10× over-count (the retraction
  below); the correction that followed then discounted a whole cold-dominant
  session to 0.1× on a single warm token — a ~10× under-count. The report now
  prices the shed as a proportional blend (`cacheprice.ShedTokenEquiv`) and stamps
  each figure with its basis — `CACHE_READ_MARGINAL` (wholly warm), `FULL_INPUT`
  (wholly cold), or `BLENDED_MARGINAL` (mixed) — so a shed number is never read
  without knowing how it was priced.

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
fak manage claude            # compaction is on by default at a 48K budget
fak cachevalue report        # the owner-attribution + per-fire compaction health
```

Tune or disable the trim with `fak manage --compact-history-budget <tokens>` (`0`
turns it off). Every `fak info` line shows the live `provider X% + fak Y%` split,
so you can watch the two savings — the provider's cache discount and fak's own
trim — side by side, honestly, without either one claiming the other's number.

To keep a standing instruction safe across compactions, start the message that states
it with `[fak:goal]` — the pin hoists it out verbatim rather than letting it fall into
a dropped middle. And if a later turn needs a span that was shed, its handle is in the
stub: call `fak_context_restore(id=…)` to page it back, or `fak_context_spans` to list
what is restorable.

## See also

- [Long-session economics](long-session-economics.md) — why the transcript is
  re-sent and why the cache discount depends on a byte-identical prefix.
- [You never manage the context window](you-never-manage-the-context-window.md) — the
  doctrine this trim serves: the window is managed *for* you, not by you.
- [Context-tape visuals](context-tape-visuals.md) — the visual language these bars are
  drawn in, and how to render one from your own session.
- [Built-in compaction audit](../notes/BUILT-IN-COMPACTION-AUDIT-2026-07-06.md) —
  why the summarize-and-discard compaction other products ship is weak, and what
  "good" looks like.
- [BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md) — the compaction row, with
  the per-fire framing and the double-counting fences.
