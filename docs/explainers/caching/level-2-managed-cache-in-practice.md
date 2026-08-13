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

> **Audience.** Operators who run fak seats and want the cache working for them.
> Prerequisites: [Level 1](level-1-what-is-caching.md) (what a prompt cache is) and a
> working `fak guard` launch — no billing, provider-API, or kernel knowledge assumed. By
> the end you'll be able to choose the right `--managed-cache` value for your credential
> class, predict which default your launch path actually gives you, and prove from your
> own run whether the 1-hour TTL upgrade activated or silently stayed passive.

> **Short answer.** [Level 1](level-1-what-is-caching.md) said the provider's cache
> rewards staying active and punishes long pauses. **Managed cache** is fak working that
> cache so the pauses hurt less — and it is a **family of levers, not one feature**:
> riding the provider's 5-minute prompt cache on the client's own breakpoints,
> star-anchor breakpoint placement (`--vcache-anchor`, default-on), compaction shed
> (`--compact-history-budget`), tool-prune and defer-cold-tools, in-kernel KV-prefix
> reuse, volatile-head redaction, and — *one member among those* — the **1-hour TTL
> upgrade**, which stretches the stable front's cache window so an idle gap of 5–60
> minutes re-enters on a cheap cache read instead of a full-price rewrite. The
> `--managed-cache` flag governs that 1h lever specifically; the rest of the family does
> its caching work regardless of the flag's posture. None of it changes the model's
> answers — only how the unchanged prompt is billed and how much of it is re-sent.

## The one flag: `--managed-cache on|off|auto`

`fak guard` takes a single flag, `--managed-cache`, with exactly three values:

| Value | What it does |
|---|---|
| `on` | Ask for the stable-prefix 1h-TTL upgrade. On an API key it activates and saves dollars. On a **subscription-OAuth seat the guard refuses it and downgrades to passive**, with the reason printed in the banner: the provider rejects a `ttl:"1h"` body on that credential class with an HTTP 400 in practice (measured 2026-07-18), so forcing it would fail your turns rather than steer the cache. |
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
  worker, the resume watchdog) `on` arms the 1h upgrade **only on API-key billing**; on a
  subscription-OAuth seat the guard downgrades `on` to passive (see the table above), and
  the caching work there is done by the rest of the family — the provider 5m cache on the
  client's own breakpoints, star-anchor placement, compaction, and tool-prune.
- **A bare `fak guard`** reads only the `--managed-cache` flag, whose own default is
  **`auto`** — active on an API-key Anthropic session, passive on a Pro/Max
  subscription. Exporting `FAK_MANAGED_CACHE` does nothing for a bare `fak guard`; it's
  a fleet-launcher knob.

So the same unset config resolves to `on` under a launcher and `auto` under a bare
guard. If you want one specific posture, pass the flag explicitly and stop guessing.

```bash
fak manage --managed-cache on   -- claude  # request the 1h upgrade (activates on API-key billing; downgrades to passive on subscription-OAuth)
fak manage --managed-cache off  -- claude  # never steer the provider cache
fak manage --managed-cache auto -- claude  # on for an API key, passive on Pro/Max
```

## On Pro/Max, the payoff is headroom — and it comes from the rest of the family

On a subscription your per-token price is flat, so a cheaper cache read won't shrink a
dollar bill. What caching protects there is your **usage limit** — the rolling session and
weekly caps, which are compute-weighted, not a flat message count. A cache read costs
roughly a tenth of the compute of a fresh prefix, so every cold prefix rewrite you *avoid*
is cap you *keep*.

The nuance is **which lever does that work on a subscription seat**. The 1h-TTL upgrade —
the one lever `--managed-cache` arms — is **not what runs there**: the provider rejects a
`ttl:"1h"` body on a subscription-OAuth credential with an HTTP 400 in practice (measured
2026-07-18), so the guard refuses `on` on those seats and downgrades to passive with a
witnessed reason instead of failing your turns. The headroom win on Pro/Max comes from the
family members that *do* run: the provider's 5-minute prompt cache riding the client's own
breakpoints, fak's default-on star-anchor breakpoint placement, compaction shed, and
tool-prune. Subscription seats serve with large provider `cached_prompt_tokens` under
exactly that passive posture. If you want the 1h tier itself, the sanctioned path is
API-key billing (`--api-key-env`).

> **Honest caveat (measured, not absolute).** The 400 is real and re-witnessed
> (2026-07-18), but it is not proven universal — the fleet ledger holds at least one
> 2026-07-10 OAuth session that fired 19 upgrades and served fully (3.5M cached prompt
> tokens). So read it as: the 1h tier is rejected on subscription-OAuth **in practice**,
> passive is the safe default there, and API-key billing is the sanctioned way to reach
> 1h. On an API key none of this hedging is needed — every avoided rewrite is real
> dollars, and `auto` already activates there.

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
   actually saved, each number carrying how it was priced. On a subscription this shows
   the value the passive-side family delivers (the provider 5m cache plus fak's
   breakpoint placement and compaction) — the 1h tier does not fire there.

## What managed cache does *not* do

It never changes the model's output. The 1h lever only changes how the provider **bills
and stores** the unchanged front of your prompt — it upgrades an existing stable
system/tools-head breakpoint, refuses volatile heads, and returns your prompt untouched
on any ambiguity. The family's other levers stay distinct from it: compaction shed
changes the *size* of what you send, and the kernel-owned KV cache applies only when fak
runs the model itself. The
[managed-cache explainer](../what-is-managed-cache.md) has the full "which cache word is
which" map.

## Try it

```bash
fak manage --managed-cache on -- claude  # request the 1h upgrade (API-key seats activate; subscription-OAuth downgrades to passive)
fak cachevalue report                     # later: see what caching actually saved
```

If you launch through a fleet tool (`fak accounts launch`, `fak codex`, the dispatch
worker, the resume watchdog), managed cache already resolves to **on** — you don't need
the flag. That arms the 1h upgrade on Anthropic-wire API-key seats; a subscription-OAuth
seat downgrades to passive (the family's other levers keep caching), and the OpenAI-wire
`fak codex` seat carries the flag passive. Set `FAK_MANAGED_CACHE=off` only on a seat
where it misbehaves.

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
