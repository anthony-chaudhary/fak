---
title: "What is fak's managed cache? A plain-English guide"
description: "fak managed cache steers Anthropic's prompt cache so an idle session re-enters on a cheap read instead of re-paying. Auto-on for API-key billing, passive on Pro/Max."
slug: what-is-managed-cache
keywords:
  - fak managed cache
  - managed cache
  - FAK_MANAGED_CACHE
  - fak guard --managed-cache
  - prompt cache
  - Anthropic prompt cache
  - 1h TTL cache
  - cache_control
  - prompt caching cost
  - keep the prompt cache warm
  - long session cache
date: 2026-07-09
---

# What is fak's managed cache?

> **Short answer.** "Managed cache" is fak *actively steering* your model provider's
> prompt cache, instead of just passing your agent's cache settings through. When it is
> on, a long session that goes quiet for a few minutes comes back **cheap** (a ~0.1×
> cache read) instead of paying full price to rebuild the whole prompt. You turn it on
> with one flag — `fak guard --managed-cache on` — but on a Claude **Pro/Max** plan you
> usually don't need to: fak leaves it off there on purpose, and there's nothing to
> configure. It **never changes your model's answers**.

*Who this is for: anyone running `fak guard -- claude` (or another agent) who has seen
the words "managed cache" in the startup banner or the docs and wants to know, in plain
terms, what it does, whether they should turn it on, and how to check it's working.*

---

## First, what is a prompt cache? (30 seconds)

An AI agent has no memory between turns, so **every turn it re-sends the whole
conversation** just to ask the next thing. On a long session that gets expensive fast.

Providers soften this with a **prompt cache**. If the front of your prompt is *exactly*
what they saw last time, they charge a small fraction for that part — Anthropic bills a
cached read at about **one tenth** (`0.1×`) of a fresh token. The catch is the word
*exactly*: the discount only holds while the cached front stays **byte-for-byte
identical** to what you sent before, and only for a limited time — the default cache
window is **5 minutes**.

Think of it like a coat check. Drop your coat once (a "cache write"), and picking it up
later is quick and cheap (a "cache read"). But the coat check closes after 5 minutes; go
past that and you have to hang everything up again from scratch.

## So what does "managed cache" mean in fak?

There are really two different caches people mean:

1. **The provider's prompt cache** — Anthropic's own feature above. fak doesn't own it;
   it rides it.
2. **fak's managed cache** — fak *actively driving* that provider cache to save you more,
   instead of just forwarding your agent's own cache markers unchanged.

When managed cache is **active**, fak does one concrete thing on the wire: it upgrades
the cache window on the stable front of your prompt from the default **5 minutes** to
Anthropic's **1-hour** tier.

That's the whole trick. It buys your coat check a much longer closing time.

## What problem does that solve?

Long agent sessions **go idle** all the time — you step away, a tool takes a while, or a
rate limit stalls the run. If the idle gap crosses the 5-minute window, the next turn
finds a *cold* cache and has to **re-write the entire prompt prefix at full price** (a
`1.25×` cache write). Do that repeatedly and a long session keeps re-paying for the same
setup.

With managed cache active, the window is an hour, so those normal idle gaps land back on
a **warm** cache: the next turn re-enters on a `0.1×` cache **read** instead of a
full-price rewrite. Same work, much smaller bill on the turns after a pause.

## How do I turn it on?

One flag on `fak guard` (with `FAK_MANAGED_CACHE` as a fleet-launcher default — see below):

```bash
fak guard --managed-cache on  -- claude   # force it on (API-key sessions)
fak guard --managed-cache off -- claude   # force it off
fak guard --managed-cache auto -- claude  # the default: fak decides (see below)
```

| Mode | What it does |
|---|---|
| `auto` *(default)* | fak turns it **on only when it can see you're paying per token** (an API key). Otherwise it stays **passive** and does nothing on the wire. |
| `on` | Force the 1-hour upgrade on. Meant for API-key sessions — on a Pro/Max subscription it buys nothing (flat rate), still costs the `2×` write, and whether the 1-hour request even clears the subscription-OAuth wire is unproven. |
| `off` | Never touch the provider cache; just forward your agent's own cache markers. |

The default is `auto`. If you launch through one of fak's fleet tools — `fak accounts
launch`, `fak codex`, the dispatch worker, or the resume watchdog — the
`FAK_MANAGED_CACHE` environment variable sets the mode those tools hand to guard, so you
don't repeat the flag. A bare `fak guard`, though, reads only the `--managed-cache` flag;
exporting the variable does nothing for it.

## Do I need it? (Pro/Max vs API key)

**Probably not, if you're on a Claude Pro/Max subscription — and that's by design.**

Here's the honest reasoning fak uses:

- **On a subscription (Pro/Max), your token price is flat.** You're not billed per
  token, so making a cache read cheaper doesn't change your bill — and the provider's
  cache already works off your agent's own markers. There's nothing for fak to win by
  steering it, so `auto` stays **passive**. fak's rule is simple: *never speculate with
  billing it can't see.* Forcing `--managed-cache on` here is off-label: no dollar
  benefit, you still pay the `2×` write, and the 1-hour request's acceptance on the
  subscription-OAuth wire is unproven.
- **On an API key, every token is real dollars you opted into managing.** There `auto`
  **activates** the 1-hour upgrade, because that's exactly where the savings are real.

So on a normal `fak guard -- claude` Pro/Max session you'll see managed cache reported as
**passive**, and that's the correct, cheapest posture — you don't need to do anything.

## Is it free? What does the 1-hour tier cost?

Almost free, and it pays for itself in minutes. The 1-hour cache window costs a bit more
to *write* — `2×` once, versus `1.25×` for the 5-minute tier — but every later read is
still the cheap `0.1×`. The break-even is roughly **three requests**, which any real
wrapped session clears in its first few minutes. After that, every idle gap that would
otherwise have gone cold re-enters on the cheap read instead of a full-price rewrite —
that avoided rewrite is the saving managed cache adds.

And it's **byte-safe by construction**: it upgrades the cache marker on the stable
system/tools front of your prompt — placing one there first if the client sent none — and
never touches anything that changes turn-to-turn. On any doubt it does nothing and
forwards your prompt unchanged. It **cannot change your model's output** — it only affects
how the provider bills the unchanged prefix.

*(One small footnote: the 1-hour tier needs an extra Anthropic beta header, which fak
adds automatically when the upgrade fires. This is why the feature is scoped to the
API-key path — see [the 1h-TTL fix note](../notes/MANAGED-CACHE-1H-TTL-400-FIX-2026-07-09.md).)*

## How do I know it's working?

Three places tell you, from quickest to most detailed:

1. **The startup banner.** `fak guard` prints one line at launch, e.g.
   `managed cache — ACTIVE (…): stable-prefix cache_control upgraded to the 1h TTL tier…`
   or `managed cache — passive (subscription OAuth (flat-rate) …)`. That single line is
   the truth of your session's posture.
2. **The metric.** `/metrics` exposes `fak_gateway_cache_ttl_upgrade_total`, counting
   every upgrade attempt (labeled by outcome), so a zero panel while active means every
   request was ineligible — visible, not silent.
3. **The dollars.** `fak cachevalue report` shows the cache-effectiveness P&L for your
   sessions — what was actually saved, with each number carrying how it was priced.

## "Managed cache" vs the other cache words

fak uses "managed" and "cache" in a few nearby places. They're different things — this is
the quick map so you don't conflate them:

| Term | What it is |
|---|---|
| **Managed cache** *(this page)* | The posture: does fak actively upgrade the provider's prompt-cache window on the wire? (`--managed-cache`) |
| **The provider prompt cache** | Anthropic's own caching feature — the thing managed cache *steers*. fak preserves and relays it, never claims to author it. |
| **Context compaction / shedding** | A *separate*, always-on feature: fak drops stale middle turns from a long session while keeping the cached front byte-identical. See [long sessions, keep the cache hit](long-sessions-keep-the-cache-hit.md). |
| **Managed context** | The gateway's context program (bounded resident view). Same word "managed", different resource — that's tokens in your window, not the provider's cache. |
| **Managed-cache restart plan** | A different sense entirely: how `fak resume` prices restarting a *dormant* crashed session warm instead of cold. |

## Try it

```bash
fak guard --managed-cache on -- claude   # force the 1h upgrade (API-key sessions)
fak cachevalue report                     # see what caching actually saved
```

On a Pro/Max plan, just run `fak guard -- claude` as usual and leave managed cache on
`auto` — passive is the right, cheapest posture there.

## FAQ

**Is managed cache on by default?**
The *setting* defaults to `auto`. Whether it's actually *active* depends on how you're
billed: active on an API-key Anthropic session, passive on a Pro/Max subscription,
passive on non-Anthropic or local models.

**Will it save me money on Claude Pro/Max?**
No — and it doesn't try to. Your subscription price is flat, so there's nothing to save
by steering the cache, and fak stays passive. Your other savings (context compaction,
tool-floor pruning, the provider's own cache discount) are still on.

**Does it change my model's answers?**
No. It only changes how the provider *bills* the unchanged front of your prompt. It's
byte-safe and forwards your prompt untouched on any ambiguity.

**Can it break a request?**
No. It upgrades the stable-head cache marker (placing one first if none exists), refuses
volatile prompts, and fak automatically adds the extra header the 1-hour tier requires so
the request stays well-formed.

**Does it work with OpenAI / Codex / Cursor?**
Not yet — those wires don't carry `cache_control` today, so managed cache stays passive
there. The flag is accepted and forwarded so a future cache-capable wire lands managed.

**What's the difference between managed cache and context compaction?**
Managed cache changes the *price* of the cached prefix (a longer cache window).
Compaction changes the *size* of what you send (it drops stale middle turns). They're
independent and both help long sessions.

## See also

- [Long sessions: shed history, keep the cache hit](long-sessions-keep-the-cache-hit.md) — the always-on compaction that keeps the cached prefix byte-identical.
- [What a saved token is worth](what-a-saved-token-is-worth.md) — the `0.1×` cache-read price and how a saved token becomes an honest dollar.
- [Long-session economics](long-session-economics.md) — why a growing transcript is re-sent every turn and why the discount depends on a byte-identical prefix.
- [The fak/DOS glossary](glossary.md) · [concept glossary](../fak/concept-glossary.md) — the precise definitions, including the "managed" word family.
- [Cache-value roll-up](../cache-value-rollup.md) — the WITNESSED-reuse vs OBSERVED-dollars accounting behind `fak cachevalue report`.
