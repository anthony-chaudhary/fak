---
title: "Managed cache in practice: the flag, the default, the proof"
description: "The operator's view of fak's managed cache: what --managed-cache on|off|auto does, why it defaults on through fleet launchers, and the three ways to see it working. Level 2 of the fak caching ladder."
slug: level-2-managed-cache-in-practice
keywords:
  - fak managed cache
  - fak guard managed-cache
  - FAK_MANAGED_CACHE default
  - managed cache on off auto
  - fak_gateway_cache_ttl_upgrade_total
  - fak cachevalue report
  - Pro Max usage limit headroom
  - 1h prompt cache TTL
date: 2026-07-10
---

# Managed cache in practice

*You are on **Level 2 of 5** of the [fak caching ladder](README.md).*

> **Short answer.** [Level 1](level-1-what-is-caching.md) said the provider's cache
> rewards staying active and punishes long pauses. **Managed cache** is fak *actively
> steering* that cache so the pauses hurt less: it upgrades the stable front of your
> prompt from the provider's default 5-minute window to Anthropic's **1-hour** tier, so
> an idle gap of 5–60 minutes re-enters on a cheap cache read instead of a full-price
> rewrite. One flag controls it, the sensible default is already on for fleet-launched
> seats, and three separate readouts prove it's happening. It never changes the model's
> answers — only how the unchanged prompt is billed.

## The one flag: `--managed-cache on|off|auto`

`fak guard` takes a single flag, `--managed-cache`, with exactly three values:

| Value | What it does |
|---|---|
| `on` | Force the stable-prefix 1h-TTL upgrade **regardless of billing**. On an API key that saves dollars; on Pro/Max it saves usage-limit headroom (below). |
| `off` | Never touch the provider cache — just forward your agent's own `cache_control` markers unchanged. |
| `auto` | Activate **only when fak can prove the session bills an API key** on the Anthropic wire (via `--api-key-env`). A subscription-OAuth seat, a passthrough credential, a non-Anthropic provider, or a local in-kernel model all stay **passive**. |

The `auto` rule is deliberately cautious: the 1-hour tier costs `2×` to *write* once
(versus `1.25×` for the 5-minute tier), so `auto` only spends that premium where fak
knows the multipliers are the operator's own dollars. It never speculates with billing
it can't see.

## Two layers of default (this is the subtle part)

The value you get if you set *nothing* depends on **how you launched**:

- **Fleet launchers** — `fak accounts launch`, `fak codex`, the dispatch worker, the
  resume watchdog. Since **2026-07-10** an **unset** `FAK_MANAGED_CACHE` normalizes to
  **`on`** (operator policy: best-effort managed cache everywhere; all four share one
  resolver, `normalizeManagedCacheMode`). So a seat launched through any of them runs
  `--managed-cache on` unless you set `FAK_MANAGED_CACHE=off` (or `=auto`). **One
  caveat:** `fak codex` fronts the OpenAI wire, which has **no `cache_control` today**, so
  there `on` resolves but stays *passive* — the flag is carried for a future
  cache-capable wire. On the Anthropic-wire seats (`accounts launch`, the dispatch
  worker, the resume watchdog) `on` actually steers the cache, subscription included.
- **A bare `fak guard`** reads only the `--managed-cache` flag, whose own default is
  **`auto`** — active on an API-key Anthropic session, passive on a Pro/Max
  subscription. Exporting `FAK_MANAGED_CACHE` does nothing for a bare `fak guard`; it's
  a fleet-launcher knob.

So the same unset config resolves to `on` under a launcher and `auto` under a bare
guard. If you want one specific posture, pass the flag explicitly and stop guessing.

```bash
fak guard --managed-cache on   -- claude   # force the 1h upgrade (any Anthropic-wire seat)
fak guard --managed-cache off  -- claude   # never steer the provider cache
fak guard --managed-cache auto -- claude   # on for an API key, passive on Pro/Max
```

## On Pro/Max, the payoff is headroom — not a refund

On a subscription your per-token price is flat, so a cheaper cache read won't shrink a
dollar bill. What it protects is your **usage limit** — the rolling session and weekly
caps, which are compute-weighted, not a flat message count. A cache read costs roughly a
tenth of the compute of a fresh prefix, so every cold prefix rewrite you *avoid* is cap
you *keep*. Managed cache turns the 5-minute idle window into an hour, which is exactly
where those avoidable rewrites live. Net effect: **more work before you hit the wall.**

> **Honest caveat (witnessed vs. assumed).** That the 1-hour upgrade is *sent* correctly
> is proven — fak adds the Anthropic beta header the tier needs, so the request is
> well-formed. Whether a subscription-OAuth wire returns the read rebate *end-to-end* is
> being witnessed through `fak cachevalue report`, not asserted here. The posture is
> **best-effort**: on any seat where `on` misbehaves, `FAK_MANAGED_CACHE=off` is the
> express opt-out. On an API key none of this hedging is needed — every avoided rewrite
> is real dollars, and `auto` already activates there.

## Three ways to see it working

From quickest to most detailed:

1. **The startup banner.** `fak guard` prints one posture line at launch. Active seats
   read `managed cache — ACTIVE (...): stable-prefix cache_control upgraded to the 1h TTL
   tier ...`; a passive seat reads `managed cache — passive (...)` with the reason and
   the override. That single line is the truth of your session's posture.
2. **The metric.** `/metrics` exposes `fak_gateway_cache_ttl_upgrade_total`, counting
   every upgrade attempt by outcome. A zero panel while you're active means every request
   was ineligible — visible, not silent.
3. **`fak cachevalue report`.** The cache-effectiveness P&L for your sessions — what was
   actually saved, each number carrying how it was priced. On a subscription this is
   where the read rebate either shows up or doesn't.

## What managed cache does *not* do

It never changes the model's output. It only changes how the provider **bills and
stores** the unchanged front of your prompt — it upgrades an existing stable
system/tools-head breakpoint, refuses volatile heads, and returns your prompt untouched
on any ambiguity. It is also *not* the same thing as compaction (which changes the *size*
of what you send) or fak's kernel-owned KV cache (when fak runs the model itself). The
[managed-cache explainer](../what-is-managed-cache.md) has the full "which cache word is
which" map.

## Try it

```bash
fak guard --managed-cache on -- claude   # force the 1h upgrade (Anthropic-wire seat)
fak cachevalue report                     # later: see what caching actually saved
```

If you launch through a fleet tool (`fak accounts launch`, `fak codex`, the dispatch
worker, the resume watchdog), managed cache already resolves to **on** — you don't need
the flag. It steers the cache on the Anthropic-wire seats and is carried (passive) on the
OpenAI-wire `fak codex` seat. Set `FAK_MANAGED_CACHE=off` only on a seat where it
misbehaves.

## See also

- [Level 1 — What is caching?](level-1-what-is-caching.md): the one-screen version, if
  the coat-check analogy above went by too fast.
- [Level 3 — Cache economics & the wire](level-3-cache-economics-and-the-wire.md): the
  read/write multipliers, the sliding window vs the 1h tier, and how fak prices the
  realized saving.
- [What is fak's managed cache?](../what-is-managed-cache.md) — the full plain-English
  page this rung condenses, including the provider matrix and the FAQ.
- [Long sessions: shed history, keep the cache hit](../long-sessions-keep-the-cache-hit.md) —
  the always-on compaction that keeps the cached prefix byte-identical.

---

*[← Level 1](level-1-what-is-caching.md) · [full ladder](README.md) · next → [Level 3](level-3-cache-economics-and-the-wire.md)*
