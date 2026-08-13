---
title: "What is fak's managed cache? A plain-English guide"
description: "fak managed cache is a family of caching levers — provider 5m prompt cache, star-anchor breakpoint placement, compaction shed, tool-prune, and the 1h-TTL upgrade — that lets an idle session re-enter cheap instead of re-paying. On Pro/Max the payoff is usage-limit headroom, delivered by the family's passive-side levers (the 1h tier is API-key-only in practice)."
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
  - prompt caching usage limits
  - keep the prompt cache warm
  - long session cache
date: 2026-07-10
---

# What is fak's managed cache?

> **Short answer.** "Managed cache" is fak working your model provider's prompt cache on
> your behalf — and it is a **family of features, not one lever**: riding the provider's
> 5-minute prompt cache on the client's own breakpoints, star-anchor breakpoint placement
> (`--vcache-anchor`, default-on), compaction shed (`--compact-history-budget`),
> tool-prune and defer-cold-tools, in-kernel KV-prefix reuse, volatile-head redaction,
> and — one member among those — the **1-hour TTL upgrade** that the `--managed-cache`
> flag governs. With the family at work, a session that goes quiet for a few minutes
> comes back **cheap** (a ~0.1× cache read) instead of paying full price to rebuild the
> whole prompt. On an API key the payoff is **dollars**, and the 1h upgrade activates
> there; on a Pro/Max subscription the payoff is **usage-limit headroom**, delivered by
> the *other* family members — the 1h tier is rejected on subscription-OAuth in practice,
> so the guard keeps those seats passive. It **never changes your model's answers**.

*Who this is for: anyone running `fak guard -- claude` (or another agent) who has seen
the words "managed cache" in the startup banner or the docs and wants to know, in plain
terms, what it does, whether they should turn it on, and how to check it's working.*

---

## First, what is a prompt cache? (30 seconds)

An AI agent has no memory between turns, so **every turn it re-sends the whole
conversation** just to ask the next thing. On a long session that gets expensive fast.

Providers soften this with a **prompt cache**. If the front of your prompt is *exactly*
what they saw last time, they charge a small fraction for that part — Anthropic bills a
cached read at about **one tenth** (`0.1×`) of a fresh token. Two catches:

- **Exactly.** The discount only holds while the cached front stays **byte-for-byte
  identical** to what you sent before.
- **Only for a while.** The default cache window is **5 minutes** — but it's a **sliding**
  window: *every time you hit the cache, the 5-minute timer resets, for free.* So an
  actively-typing session (a turn every minute or two) stays warm **indefinitely** at no
  extra charge. The cache only goes cold after ~5 minutes of **inactivity**.

Think of it like a coat check that resets its closing time each time you touch your coat.
Drop it once (a "cache write"); every quick pickup (a "cache read") is cheap *and* buys
another 5 minutes. Walk away for 5 minutes straight, though, and the rack is cleared — you
hang everything up from scratch next time.

That sliding-window detail is the key to the whole feature: while you're active you never
pay to rebuild. The only time you go cold is an **idle gap** longer than the window.

## So what does "managed cache" mean in fak?

There are really two different caches people mean:

1. **The provider's prompt cache** — Anthropic's own feature above. fak doesn't own it;
   it rides it.
2. **fak's managed cache** — the *family* of things fak does so that provider cache (and
   your context spend generally) works as hard as possible: star-anchor breakpoint
   placement (`--vcache-anchor`, default-on, so even a client that sent no
   `cache_control` gets a stable cached front), compaction shed
   (`--compact-history-budget`), tool-prune and defer-cold-tools, in-kernel KV-prefix
   reuse when fak runs the model itself, volatile-head redaction, and the **1-hour TTL
   upgrade**.

The `--managed-cache` flag governs exactly one of those members. When it resolves
**active**, fak does one concrete thing on the wire: it upgrades the cache window on the
stable front of your prompt from the default **5 minutes** to Anthropic's **1-hour**
tier. The rest of the family does its work regardless of that flag's posture — a seat
where the 1h lever is unavailable is still cached by the provider 5m window on
well-placed breakpoints, still compacted, still tool-pruned.

Because the 5-minute window already stays warm for free while you're active, the 1-hour
upgrade has exactly one job: **surviving idle gaps of 5–60 minutes** — you step away, a
tool takes a while, a side-agent runs long, or a rate limit stalls the run. It's idle-gap
insurance, not a continuous-use win.

## What problem does that solve?

Cross the 5-minute idle window and the next turn finds a *cold* cache: it has to
**re-write the entire prompt prefix at full price** (a `1.25×` cache write) before it can
read anything. On a long, stop-start session — the normal shape of agent work — you keep
re-paying for the same setup every time you come back from a pause.

With managed cache active, the window is an hour, so those normal idle gaps land back on
a **warm** cache: the next turn re-enters on a `0.1×` cache **read** instead of a
full-price rewrite. Same work, much less spent on the turns after a pause — whether
"spent" means dollars (API key) or usage-cap headroom (Pro/Max; see below).

## How do I turn it on? (and what's the default now)

There are **two layers** of default, and they differ — this is the one subtle part:

- **Fleet launchers** — `fak accounts launch`, `fak codex`, the dispatch worker, the
  resume watchdog. Since **2026-07-10** these read `FAK_MANAGED_CACHE`, and an **unset**
  value now resolves to **`on`** (operator policy: best-effort managed cache everywhere).
  So a worker launched through any of them gets `--managed-cache on` — which **activates
  the 1h upgrade on API-key-billed Anthropic seats** and **downgrades to passive on a
  subscription-OAuth seat** (the guard refuses the 1h tier there, with the reason in the
  banner; see below) — unless you set `FAK_MANAGED_CACHE=off` (or `=auto`).
- **A bare `fak guard`** still reads only the `--managed-cache` flag, whose own default is
  **`auto`** (fak's billing-gated auto — passive on a subscription unless you pass `on`).
  Exporting `FAK_MANAGED_CACHE` does nothing for a bare `fak guard`.

```bash
fak manage --managed-cache on   -- claude  # request the 1h upgrade (API key: active; subscription-OAuth: downgrades to passive)
fak manage --managed-cache off  -- claude  # force it off
fak manage --managed-cache auto -- claude  # billing-gated: on for API key, passive on Pro/Max
```

| Mode | What it does |
|---|---|
| `on` *(fleet default since 2026-07-10)* | Ask for the 1-hour upgrade. On an API key it activates and saves dollars. On a subscription-OAuth seat the guard **refuses it and downgrades to passive** with a witnessed reason: the provider rejects `ttl:"1h"` on that credential class with an HTTP 400 in practice (measured 2026-07-18), so forcing it would fail turns, not save headroom. |
| `auto` *(bare-guard flag default)* | Turn it **on only when fak can see you're paying per token** (an API key). On a subscription-OAuth seat it stays **passive** — fak's older "never speculate with billing it can't see" rule. Now reachable only by asking for `auto` explicitly. |
| `off` | Never touch the provider cache; just forward your agent's own cache markers. |

## Do I need it on Pro/Max? Yes — for headroom, and the family delivers it

This is the part the older version of this page got wrong. Here's the honest reasoning.

On a **subscription (Pro/Max)** your token *price* is flat, so it's true that a cheaper
cache read won't lower a dollar bill. But dollars aren't your binding constraint on a
subscription — your **usage limit** is (the rolling session and weekly caps). And those
limits are **compute-weighted, not a flat message count**: an inefficient request drains
them faster than an efficient one. Two facts make caching matter there:

- A cache **read** costs roughly **a tenth** of the compute of a fresh prefix, and
  Anthropic's docs say **cache-read tokens don't count against the API rate limit** at
  all (only cache *writes* and *uncached* input do).
- A **cold prefix rewrite** — what you pay after every idle gap that went cold — is a
  full-price cache **write** across your entire stable head.

So every cold rewrite you *avoid* is usage-cap you *keep*, and on a subscription the net
effect of caching is **more work done before you hit the wall**.

The part to get right is **which lever delivers that on a subscription seat**. It is
*not* the 1h-TTL upgrade: the provider rejects a `ttl:"1h"` cache_control on a
subscription-OAuth credential with an HTTP 400 in practice (measured 2026-07-18, even
with the required beta header — see the
[measured finding](../notes/2026-07-18-subscription-oauth-400s-1h-ttl-upgrade-MEASURED.md)),
so the guard refuses `--managed-cache on` there and **downgrades to passive with a
witnessed reason in the banner** rather than fail your turns. The headroom on Pro/Max
comes from the family's other members, which run regardless: the provider's 5-minute
prompt cache riding well-placed breakpoints, fak's default-on star-anchor placement,
compaction shed, and tool-prune. Subscription seats serve with large provider
`cached_prompt_tokens` under exactly that passive posture. The sanctioned way to reach
the 1h tier itself is API-key billing (`--api-key-env`).

> **Honest caveat (measured, not absolute).** The reasoning above is Anthropic's
> documented rate-limit accounting; the exact way Pro/Max *subscription* usage is metered
> isn't fully published. And the OAuth 400 on the 1h tier, while real and re-witnessed
> (2026-07-18), is not proven universal — the fleet ledger holds at least one 2026-07-10
> OAuth session that fired 19 upgrades and served fully (3.5M cached prompt tokens). So
> the honest statement is: the 1h tier is rejected on subscription-OAuth *in practice*,
> passive is the safe default there, and the API-key path is how you reach 1h. On any
> seat where the posture misbehaves, `FAK_MANAGED_CACHE=off` is the express opt-out.

On an **API key**, none of the hedging is needed: every avoided rewrite is real dollars
you opted into managing, and `auto` already activates there on its own.

## What about OpenAI, Gemini, and other models?

Caching itself is now near-universal — but only Anthropic exposes a **steerable cache
window on the wire fak sits on**, so the managed-cache family's TTL lever (the 1-hour
upgrade) is Anthropic-only.
On the other providers there's simply nothing for fak to steer, so it stays **passive** —
and that's correct, because those providers cache *automatically* and you already get the
discount without any marker:

| Provider | How its cache works | Is there a knob for fak to steer? |
|---|---|---|
| **Anthropic** | Explicit `cache_control` breakpoints; 5-minute (sliding) or 1-hour window; ~0.1× reads, `1.25×`/`2×` writes. | **Yes** — the window/TTL. This is what managed cache upgrades. |
| **OpenAI** | Automatic on prompt prefixes ≥ ~1,024 tokens; discounted cached reads; short, uncontrollable TTL; no storage fee. | No — no wire marker; passive, discount already applies. |
| **Gemini** | *Implicit* auto-cache on 2.5+ models, plus *explicit* named cache objects with a configurable TTL (billed for storage). | Not on the pass-through wire fak fronts today; passive. |

So "managed cache stays passive" on a non-Anthropic wire is **not** "you lose caching" —
it's "the provider already caches automatically and there's no TTL for fak to extend." The
`--managed-cache` flag is still accepted and forwarded, so a future cache-steerable wire
lands managed with no config change.

## This is the API-side cache — fak has a deeper one too

Everything on this page is about the **provider's prompt cache** on the *API* wire: fak is
a proxy, so the best it can do is steer bytes and TTLs the provider exposes.

When fak instead **runs the model itself** (`fak serve --engine inkernel`, or a local
`--gguf`), there is no provider wire — fak owns the **KV cache directly**, and a different,
deeper layer applies: exact prefix reuse, span-level eviction, prefix cloning for fan-out,
and O(1) queryable context. That's kernel-owned reuse (WITNESSED), a separate thing from
the provider rebate (OBSERVED cost/latency telemetry) this page describes. See
[addressable KV cache](addressable-kv-cache.md) and the
[vCache default-enablement plan](../cache-frontier/DEFAULT-ENABLEMENT-NEXT-50.md).

## Is it free? What does the 1-hour tier cost?

Almost free, and it pays for itself in minutes. The 1-hour cache window costs a bit more
to *write* — `2×` once, versus `1.25×` for the 5-minute tier — but every later read is
still the cheap `0.1×`. The break-even is roughly **three requests** (versus two for the
5-minute tier), which any real wrapped session clears in its first few minutes. After
that, every idle gap that would otherwise have gone cold re-enters on the cheap read
instead of a full-price rewrite — that avoided rewrite is what managed cache adds.

And it's **byte-safe by construction**: it upgrades the cache marker on the stable
system/tools front of your prompt — placing one there first if the client sent none — and
never touches anything that changes turn-to-turn. On any doubt it does nothing and
forwards your prompt unchanged. It **cannot change your model's output** — it only affects
how the provider bills the unchanged prefix.

## How do I know it's working?

Three places tell you, from quickest to most detailed:

1. **The startup banner.** `fak guard` prints one line at launch. A fleet-launched
   API-key seat reads `managed cache — ACTIVE (forced by --managed-cache on):
   stable-prefix cache_control upgraded to the 1h TTL tier…`. A subscription-OAuth seat —
   even under `--managed-cache on` — reads `passive (…)` with the refusal reason (the
   provider rejects the 1h tier on that credential class) and the API-key activation
   path. That single line is the truth of your session's posture.
2. **The metric.** `/metrics` exposes `fak_gateway_cache_ttl_upgrade_total`, counting
   every upgrade attempt (labeled by outcome), so a zero panel while active means every
   request was ineligible — visible, not silent.
3. **The dollars (and the rebate).** `fak cachevalue report` shows the cache-effectiveness
   P&L for your sessions — what was actually saved, with each number carrying how it was
   priced. On a subscription this is where the read-rebate either shows up or doesn't.

## "Managed cache" vs the other cache words

fak uses "managed" and "cache" in a few nearby places. They're different things — this is
the quick map so you don't conflate them:

| Term | What it is |
|---|---|
| **Managed cache** *(this page)* | The family of levers fak runs on your caching's behalf (breakpoint placement, compaction, tool-prune, the 1h upgrade, …). The `--managed-cache` flag governs one member: whether fak upgrades the provider's prompt-cache window on the wire. |
| **The provider prompt cache** | Anthropic's (or OpenAI's / Gemini's) own caching feature — cost/latency telemetry that managed cache *steers*. fak preserves and relays it, never claims to author it. |
| **Kernel / KV cache** | When fak runs the model itself, the reuse it owns directly (prefix reuse, span eviction, O(1) context). Deeper than the API cache; see the note above. |
| **Context compaction / shedding** | A sibling lever in the managed-cache family, on by default: fak drops stale middle turns from a long session while keeping the cached front byte-identical. See [long sessions, keep the cache hit](long-sessions-keep-the-cache-hit.md). |
| **Managed context** | The gateway's context program (bounded resident view). Same word "managed", different resource — that's tokens in your window, not the provider's cache. |
| **Managed-cache restart plan** | A different sense entirely: how `fak resume` prices restarting a *dormant* crashed session warm instead of cold. |

## Try it

```bash
fak manage --managed-cache on -- claude  # request the 1h upgrade (API key: active; subscription-OAuth: passive)
fak cachevalue report                     # see what caching actually saved
```

If you launch through a fleet tool (`fak accounts launch`, `fak codex`, the dispatch
worker), managed cache is already **on** by default — you don't need the flag. On a
subscription-OAuth seat that resolves to a passive posture (the family's other levers
keep caching). Set `FAK_MANAGED_CACHE=off` only on a seat where it misbehaves.

## FAQ

**Is managed cache on by default?**
Through fak's fleet launchers, **yes** — since 2026-07-10 an unset `FAK_MANAGED_CACHE`
resolves to `on`. That arms the 1h upgrade on API-key seats; on a subscription-OAuth
seat the guard downgrades `on` to passive (the provider rejects the 1h tier on that
credential class), and the family's other levers do the caching. A bare `fak guard`
still defaults its own flag to `auto` (active on an API-key Anthropic session, passive
on a Pro/Max subscription).

**Will it save me money on Claude Pro/Max?**
Not in *dollars* — your subscription price is flat. What it saves is **usage-limit
headroom**: cache reads cost ~0.1× the compute of a fresh prefix and don't count against
the rate limit, so avoiding cold prefix rewrites lets you do more work before you hit your
cap. On those seats the headroom comes from the family's passive-side levers — the
provider 5m cache, star-anchor placement, compaction, tool-prune — not the 1h tier, which
is rejected on subscription-OAuth in practice (see the section above, including the
honest caveat about what's measured vs. absolute).

**Does it change my model's answers?**
No. It only changes how the provider *bills* the unchanged front of your prompt. It's
byte-safe and forwards your prompt untouched on any ambiguity.

**Can it break a request?**
Not anymore — two different HTTP 400s used to bite here, and both are handled. First, the
1-hour tier needs an extra Anthropic beta header; fak adds it automatically when the
upgrade fires. Second, the provider rejects the 1h tier itself on a subscription-OAuth
credential even with that header (measured 2026-07-18), so the guard refuses to arm the
upgrade on those seats and downgrades to passive with the reason in the banner — the
rejected body is never sent. It also upgrades only the stable-head marker, placing one
first if none exists, and refuses volatile prompts.

**Does it work with OpenAI / Codex / Cursor / Gemini?**
Those wires cache *automatically* and expose no `cache_control` TTL for fak to steer, so
managed cache stays **passive** there — you still get the provider's own cache discount,
fak just has nothing to extend. The flag is accepted and forwarded so a future
cache-steerable wire lands managed.

**Isn't the 5-minute cache enough while I'm working?**
Yes — that's the point. The 5-minute window *slides* (it resets free on every cache hit),
so continuous work never goes cold and the 1-hour upgrade adds nothing there. The upgrade
only earns its keep across **idle gaps of 5–60 minutes**, which is most of what stop-start
agent sessions actually hit.

**What's the difference between managed cache and context compaction?**
Compaction is itself a member of the managed-cache family. The distinction people
usually mean is with the family's TTL lever: the `--managed-cache` flag changes the
*price* of the cached prefix (a longer cache window), while compaction changes the
*size* of what you send (it drops stale middle turns). They're independent and both
help long sessions.

## See also

- [Long sessions: shed history, keep the cache hit](long-sessions-keep-the-cache-hit.md) — the always-on compaction that keeps the cached prefix byte-identical.
- [What a saved token is worth](what-a-saved-token-is-worth.md) — the `0.1×` cache-read price and how a saved token becomes an honest dollar.
- [Long-session economics](long-session-economics.md) — why a growing transcript is re-sent every turn and why the discount depends on a byte-identical prefix.
- [Addressable KV cache](addressable-kv-cache.md) · [vCache default-enablement plan](../cache-frontier/DEFAULT-ENABLEMENT-NEXT-50.md) — the deeper, kernel-owned cache layer fak drives when it runs the model itself.
- [The fak/DOS glossary](glossary.md) · [concept glossary](../fak/concept-glossary.md) — the precise definitions, including the "managed" word family.
- [Cache-value roll-up](../cache-value-rollup.md) — the WITNESSED-reuse vs OBSERVED-dollars accounting behind `fak cachevalue report`.
