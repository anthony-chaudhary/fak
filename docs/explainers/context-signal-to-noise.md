---
title: "Context signal-to-noise: the number cache-hit % can't give you"
description: "Provider cache-hit % is a denominator artifact — it climbs toward 1.0 with session length alone, whether or not the resident context is any good (measured: 0.88 short → 0.99 long across 247 sessions, with 10× density spread at the same 99%). The number actually worth maximizing is whether the resident window equals the desired window. fak's ctxplan records the ground truth per turn (which resident spans the turn referenced vs left idle), and ComputeSignalNoise folds it into a token-weighted signal-to-noise ratio that is invariant to caching and to length — so a session can read cache-hit 0.99 and S/N 0.30 at once, finally making the bloat legible."
slug: context-signal-to-noise
keywords:
  - context signal-to-noise
  - prompt cache-hit rate
  - context window quality
  - agent context management
  - resident context budget
---

# Context signal-to-noise: the number cache-hit % can't give you

Cache-hit percentage is the metric everyone reaches for, and it is the wrong one for
judging context quality. Here is the trap, the math, and the metric that replaces it.

## Cache-hit % rises with length, mechanically, whether or not the context is any good

Provider cache-hit fraction is

```
cache_hit = cache_read_tokens / (cache_read_tokens + fresh_input_tokens)
```

In a long agent session you append to a stable prefix. Each turn the **cached prefix
grows** (everything before this turn is now cacheable), while **fresh input stays about
one turn's worth**. So the denominator's first term grows without bound and the second
stays flat — and the ratio climbs toward 1.0 **as a function of length alone**.

This is not a hypothesis. Measured on fak's own corpus of 247 Claude Code sessions
(`tools/session_audit.py audit`):

| Session length | Median cache-hit |
|---|---|
| < 50 turns (n=202) | 0.88 |
| 50–100 turns (n=19) | 0.98 |
| 150–200 turns (n=7) | 0.992 |

Pearson correlation of cache-hit to turn count is ≈ 0.48; to total context size ≈ 0.39.
And the tell: among sessions **all at ~99% cache-hit**, context density differs **10×**
(8.69 vs 2.81 turns per MB). Same headline cache-hit, wildly different efficiency. A
high cache-hit on a bloated window just means you are **re-reading the wrong thing
cheaply** — efficiently caching garbage.

So cache-hit % answers "how much of what's resident did I avoid re-paying full price
for?" It never answers the question that matters: **is what's resident the right size?**

## The thing actually worth maximizing: |resident| == |desired|

The goal is for the resident window to equal the window the task actually needs. Too
big wastes budget on idle context (and yes, caches it). Too small forces the work back
out of the store turn after turn. The target is **lean and sufficient**.

fak already records the ground truth for this, per turn, in a `ctxplan.Outcome`:

- **Hits** — resident spans the turn *referenced* → **signal**
- **Wasted** — resident spans the turn *never touched* → **noise**
- **Faults** — *elided* spans the turn had to demand-page back → **under-resident**

## The metric: token-weighted context signal-to-noise

`ctxplan.ComputeSignalNoise(plan, outcome)` folds those labels into a ratio:

```
signal_tokens = Σ cost(span) for resident spans referenced this turn, plus pins
noise_tokens  = Σ cost(span) for resident spans never touched this turn
S/N ratio     = signal_tokens / resident_tokens          (in [0,1])
```

Three properties make it the right number where cache-hit is the wrong one:

1. **Token-weighted, not span-counted.** One 9 000-token stale blob next to two
   100-token live spans scores ~2% signal, not 67%. The bloat weighs what it costs.

2. **Invariant to caching and to length.** Re-reading a `Wasted` span cheaply (cached)
   does not make it signal — it is still resident-but-untouched. So a session can report
   **cache-hit 0.99 and S/N 0.30 at the same time**. That pair — high cache-hit, low S/N
   — *is* the pathology, finally legible.

3. **The opposite failure is on its own axis.** Trimming a needed span out of the window
   doesn't raise S/N; it moves cost to `FaultTokens` (`FaultRatio`), graded **starving**.
   You cannot game the ratio up by starving the turn.

`Grade()` reads both axes into one word:

| Grade | Condition | Meaning |
|---|---|---|
| `lean` | ratio ≥ 0.8, fault ≤ 0.1 | resident ≈ desired — the goal |
| `ok` | in between | acceptable, not yet ideal |
| `bloated` | ratio < 0.5 | most of the window idled (the cache-hit trap, in the open) |
| `starving` | fault > 0.25 | trimmed so lean the turn keeps faulting |

`|resident| == |desired|` is just **ratio → 1.0** with faults near 0.

## Where the exact number lives, and where it can't

The exact, token-weighted S/N requires the per-span Hit/Waste labels in a
`ctxplan.Outcome`, so it lives in `internal/ctxplan` and is available to anything that
plans a view (`ctxplanbench`, the planner's own learning loop). A raw Claude Code
transcript carries no such labels — there is no record of *which* resident span a turn
referenced — so the session auditor can offer only a coarse density **proxy**
(output ÷ ingested, turns ÷ MB), never the real ratio. That boundary is deliberate and
mirrors fak's WITNESSED-vs-OBSERVED line: the measured S/N is witnessed from the
planner's own ground truth; a transcript proxy is a separate, clearly-labeled surface
that can flag a suspect session but not prove it.
