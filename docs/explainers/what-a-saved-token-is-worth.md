---
title: "What a saved token is worth: one multiplier, two axes, and its six names"
description: "Turning a token fak avoided paying for into an honest dollar comes down to one number — the 0.1x cache-read price. This is the plain-English account of that number: the two ways it saves you money, the identity that keeps the dollar figure honest, and why the same 0.1 shows up in the code under six different names."
slug: what-a-saved-token-is-worth
keywords:
  - saved token value
  - cache read multiplier
  - cache pricing
  - compaction shed value
  - valuation basis
  - cache savings accounting
  - token economics
  - honest attribution
date: 2026-07-06
---

# What a saved token is worth

*Who this is for: anyone reading a `fak cachevalue report` number — a "fak saved
you N tokens / $X" figure — who wants to know exactly what that dollar means, how
it's computed, and where it could quietly lie. The mechanism of saving is covered
in [context shedding](context-shedding.md) and [long-session
economics](long-session-economics.md); this page is about **putting a price on it**.*

## The one number

A token served from the provider's prompt cache costs about **a tenth** of a fresh
one. That's the whole foundation. Anthropic prices a cached-prefix read at `0.1x`
the base input rate; a fresh (uncached) input token is `1.0x`. So the discount of
"this token was already cached" is `0.9x` of base input — you pay `0.1`, you'd have
paid `1.0`, you keep `0.9`.

That single ratio — call it the **cache-read multiplier**, `0.1` — is the only
price fact you need to value almost everything fak does with your context. Every
"saved token" number is arithmetic on it. (Cache *writes* are the mirror image and
cost a premium — `1.25x` at the 5-minute TTL, `2.0x` at 1-hour — which is why
caching only pays off once enough reads accrue. That asymmetry lives in
`internal/gateway/cache_pricing.go`; it's not the subject of this page.)

## Two ways to save, one price

The same `0.1` shows up in two different savings stories, and it's easy to confuse
them:

1. **The rebate (the provider's cache doing its job).** A token that was written to
   the cache once and then read back is billed at `0.1x` instead of `1.0x`. The
   saving is the `0.9x` you didn't pay. This is the big number on the Claude Code
   route, and it's the *provider's* saving — fak's job is to keep the byte-identical
   prefix alive so the discount survives, not to author it.

2. **The shed (fak's compaction dropping a token).** When fak trims a stale token
   out of the middle of a long transcript, what did it save you? It depends on what
   that token *would have* cost next turn:
   - If the prefix was **warm** — the provider already had it cached — the token
     would have been billed as a `0.1x` cache read. Dropping it avoids `0.1x`, not
     `1.0x`.
   - If the prefix was **cold** — no cache behind it — the token would have been
     fresh `1.0x` input. Dropping it avoids the full `1.0x`.

So the very same `0.1` is both *the price you keep paying on a warm read* and *the
price you avoid by shedding a warm token*. They're the same physical quantity seen
from two sides, which is exactly why the code keeps them numerically equal.

## The identity that keeps it honest

Every fak-attributed dollar is a product of two independent axes:

```
fak_$  =  (how many tokens)  ×  (price per token)
             count axis            price axis
```

Inflate **either** axis and the dollar lies. Both axes have burned this project
once, in public, and both fixes are now baked into the code:

- **The price axis.** For a while, a compaction-shed token was valued at the full
  `1.0x` input rate even when the fire landed on a warm prefix — where the honest
  price is `0.1x`. That over-credited fak's compaction **tenfold** on every warm
  fire. The first correction over-shot in the other direction: it discounted a
  session's *entire* shed to `0.1x` the moment a single warm cache-read appeared,
  which **under**-credited a cold-dominant session just as badly. Both were step
  functions on a continuous quantity. The durable fix (#2794 / #2798 / #2796):
  price the shed as a **proportional blend** — the warm portion
  `min(shed, cache_read)` at the `0.1x` cache-read marginal, the cold remainder at
  `1.0x` — so the number no longer swings with a session's warm/cold fire mix
  (`cacheprice.ShedTokenEquiv`, the one source the report, the live guard split, and
  the sessionobs net-true ledger all price on).

- **The count axis.** fak doesn't keep a compacted copy of your transcript — the
  client re-sends its full history every turn, so fak re-trims from scratch each
  time compaction fires. Summing "tokens shed" across fires therefore **re-counts
  the same middle** once per fire. The honest unit is *shed per fire*, never the
  session total. (The full story, including the retracted-75% headline that came
  from exactly this mistake, is in [context shedding](context-shedding.md).)

Notice the two failure modes are *orthogonal* — one is a wrong price, one is a
wrong count — which is why the fix for one does nothing for the other, and why the
report has to get both right independently.

## Every fak dollar wears a price tag

Because "at what price?" is the question that let the `1.0x`-on-warm error slip
through, the report refuses to let a number travel without its basis. Each saved-$
row carries a **valuation basis** label (`shedValuationBasis` picks the first three
by the shed's warm/cold mix):

| Basis | Price | When it's honest |
|---|---|---|
| `FULL_INPUT` | `1.0x` | a wholly-**cold** shed (`cache_read == 0`) — no cache behind the dropped tokens |
| `CACHE_READ_MARGINAL` | `0.1x` | a wholly-**warm** shed (`cache_read ≥ shed`) — every dropped token would have been a cache read |
| `BLENDED_MARGINAL` | `min(shed, cache_read)·0.1x + remainder·1.0x` | a **mixed** shed (`0 < cache_read < shed`) — warm portion at the marginal, cold remainder at full input |
| `OBSERVED_NET` | read rebate − write premium | the provider prompt-cache row |

The renderer will **refuse to print an unlabeled fak dollar** rather than emit a
figure whose pricing you can't see. That's the structural guard: you can never read
a fak-savings number without also being told how it was priced.

## The decoder ring: one quantity, six names

Here's the part that trips up anyone reading the code for the first time. That one
`0.1` is declared **six-plus times**, under six different names, in six packages:

| Symbol | Package | Role |
|---|---|---|
| `CacheReadMultiplier` | `gateway` (`cache_pricing.go`) | **canonical source** — the cache-read price |
| `CacheReadMultiplier` | `resume` | same value, redeclared |
| `defaultCacheReadMult` | `agent` | the compaction fire-gate's shed price |
| `providerCacheReadMultiplier` | `cachevaluereport` | the rebate's cache-read price |
| `compactionShedMarginalMultiplier` | `cachevaluereport` | the warm-shed marginal |
| `compactionCacheReadMarginal` | `sessionobs` | the warm-shed marginal (net-true sensor) |

(Plus a couple of bare `0.1` literals in bench/observe scaffolding.)

This looks like sloppy duplication. It mostly isn't. fak's packages form a layered
DAG, and the layering rule **forbids the upward import** that sharing one constant
would require: `resume` and `sessionobs` are tier-1 foundation leaves that cannot
import the tier-4 `gateway` integrator, and `agent` can't import `gateway` without
creating a cycle. So the value is **redeclared, not shared** — and a test pins each
copy to the canonical `gateway.CacheReadMultiplier`, so a drift in one is caught,
not silently tolerated. The canonical `0.1` lives in `gateway/cache_pricing.go`;
everything else is a mirror that must agree with it.

If you're changing the number, change `gateway.CacheReadMultiplier` and let the pin
tests tell you which mirrors to follow.

## The one honest simplification still owed

The layering makes *most* of those copies mandatory — but not all. Inside the
`cachevaluereport` package, the two constants `providerCacheReadMultiplier` and
`compactionShedMarginalMultiplier` are the same `0.1` in the same file: one used as
the rebate's cache-read price, one as the warm-shed marginal. They're the same
number by necessity (a warm-shed token's avoided cost *is* the cache-read price), so
they can and should collapse into a single well-named local constant with a comment
explaining both uses — one local source of truth instead of two that a test has to
keep equal. That's a small, contained cleanup that realizes the single-source intent
of #2798 at the file level; it's noted here as known, actionable debt rather than
left as silent drift.

## Try it

```bash
fak cachevalue report        # per-mechanism savings, each row carrying its ValuationBasis
```

Read a fak-attributed dollar as `count × price`, check the basis label to see the
price, and read the shed count *per fire*, not as a session sum. Those three habits
are the whole discipline of this page.

## See also

- [Context shedding](context-shedding.md) — the count axis in full: what fak trims,
  why the honest number is per-fire, and the retracted-75% lesson.
- [Long-session economics](long-session-economics.md) — why the transcript is
  re-sent every turn and why the cache discount depends on a byte-identical prefix.
- [fak manage's share of cache value on real sessions](../notes/FAK-GUARD-CACHE-VALUE-SHARE-2026-07-01.md)
  — the measured fak-vs-provider split, the number this valuation feeds.
- `internal/gateway/cache_pricing.go` — the canonical `CacheReadMultiplier` and the
  write-premium economics, in code.
