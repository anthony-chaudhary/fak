---
title: "Cache economics and the wire: read/write multipliers, break-even, and the 1h tier"
description: "The money-and-bytes rung of the fak caching ladder: what a cache write vs read really costs, how many reuses pay a write back, the sliding TTL that resets on use, and how fak measures the realized saving."
slug: level-3-cache-economics-and-the-wire
keywords:
  - prompt cache economics
  - cache read vs write cost
  - cache write multiplier
  - 1 hour cache TTL
  - extended-cache-ttl beta
  - prompt cache break-even
  - cachevalue report
  - token-equivalent savings
date: 2026-07-10
---

# Cache economics and the wire

*You are on **Level 3 of 5** of the [fak caching ladder](README.md).*

> **Audience.** Readers comfortable with the ladder's earlier levels who want the money
> side. By the end you'll be able to price a cache write against its reads, work out the
> break-even reuse count for the 5-minute and 1-hour tiers, and read fak's realized-saving
> report without over-crediting it.

> **Short answer.** A prompt cache is not free: the first **write** of a prefix costs
> *more* than sending it uncached (1.25× base input at the 5-minute tier, 2.0× at the
> 1-hour tier), and each later **read** costs 0.1×. So caching only wins once enough
> reads accrue — two requests at 5m, three at 1h. fak steers the tier on the wire, keeps
> the cache warm by holding the prefix byte-identical, and reports the realized saving as
> the provider's own numbers, split from its own.

## The three multipliers (relative to base input = 1.0×)

Every cache decision comes down to three published multipliers, expressed relative to the
model's base input price. In this repo they are a single canonical constant set
(`internal/cacheprice/cacheprice.go`), pinned by test so every surface prices the same
token the same way:

| Axis | Multiplier | Meaning |
|---|---|---|
| Cache **read** (`cache_read_input_tokens`) | **0.1×** | a prefix the provider served from its own cache — a 10× discount |
| Cache **write**, 5-minute TTL | **1.25×** | writing a prefix into the default ephemeral tier |
| Cache **write**, 1-hour TTL | **2.0×** | writing into the extended tier |
| Uncached input | 1.0× | the baseline everything else is measured against |

The asymmetry is the whole story: the **first** write of a span costs *more* than just
sending it uncached (1.25× or 2.0× vs 1.0×), and you only recover that premium on later
reads, each of which saves 0.9× of base. A cold-write-only turn therefore has a
**negative** saving until the reads repay it — fak's ledger prices the write premium as a
negative saving rather than hiding it (`CachePricing.SavingsUSD`).

## Break-even: how many reuses pay a write back

Because a write costs more than an uncached send, caching is a bet that the prefix will be
read again soon. The break-even is small:

- **5-minute tier:** one write + one read is `1.25 + 0.1 = 1.35`, which beats `2.0` (two
  uncached sends). **Break-even at two requests.**
- **1-hour tier:** one write + two reads is `2.0 + 0.2 = 2.2`, which beats `3.0` (three
  uncached sends). **Break-even at three requests.**

A wrapped agent session clears either threshold in its first minutes, which is why fak is
comfortable arming the 1h tier when it owns the billing (below). This break-even is the
*write-pays-for-itself* case. A separate, larger break-even governs whether it is worth
**bursting** an already-warm cache to shed history — that formula lives in
[Level 2's source explainer](../long-sessions-keep-the-cache-hit.md) and depends on how
much warm suffix a drop would invalidate, not just on the write premium.

## The sliding window that resets on use

The 5-minute TTL is not a hard countdown from the write. It is a **sliding window that
refreshes on each hit** — every read pushes the expiry back out. Back-tested against this
machine's own history (~82,000 turn-to-turn boundaries), fak's warm-or-cold planner
matched the provider **97.7%** of the time, and its one systematic miss is in the safe
direction: it calls a 5-to-15-minute gap cold, but "Anthropic's 5-minute cache is a floor
that refreshes on use, so in practice it survives longer than the clock"
(`long-sessions-keep-the-cache-hit.md`). The planner errs toward declaring cold, which
never claims a warm cache that isn't there.

The problem the sliding window leaves open is the **long idle gap** — a human stepping
away, a slow tool, a rate-limit stall — that outlives even the refreshed 5m window. Cross
it and the next turn re-**writes** the whole prefix at 1.25× instead of re-**reading** it
at 0.1×. That is what the 1-hour tier buys back.

## The 1h tier, and what fak does on the wire

fak's managed-cache posture (`--managed-cache auto|on|off`, `cmd/fak/guard_managed_cache.go`)
decides whether this guard session actively manages the outbound Anthropic prompt-cache
beyond forwarding the client's own `cache_control` bytes. The flag governs one member of
the wider managed-cache family (star-anchor breakpoint placement, compaction shed,
tool-prune, and kernel KV reuse all run independently of it). When **active**, it arms exactly
one lever: it upgrades the *existing* stable system/tools breakpoint from the default 5m
tier to 1h, so an idle gap past 5 minutes re-enters on a 0.1× read instead of a 2.0× (once)
re-write. The upgrade is byte-safe by construction — it only touches an existing stable-head
breakpoint, refuses volatile heads, and returns identity on any ambiguity.

On the wire this is two coordinated changes:

- **Body:** the breakpoint grammar goes from the bare `{"type":"ephemeral"}` (5m) to
  `{"type":"ephemeral","ttl":"1h"}` (1h) — `internal/gateway/cache_pricing.go`,
  `internal/agent` `UpgradeAnthropicStableCacheTTL1h`.
- **Header:** Anthropic accepts `ttl:"1h"` **only** when the request also negotiates the
  `extended-cache-ttl-2025-04-11` beta in its `anthropic-beta` header. The wrapped `claude`
  CLI does not send it, so fak **unions** that beta into the forwarded set whenever the
  upgrade fires (omitting it 400'd the request — see the
  [1h-TTL 400 fix note](../../notes/MANAGED-CACHE-1H-TTL-400-FIX-2026-07-09.md)).

**Why `auto` is conservative.** The 1h tier *doubles* the write premium (2.0× vs 1.25×),
so `auto` activates only when fak knows the session bills an **API key** on the Anthropic
wire — where every multiplier is operator dollars fak owns. On a Pro/Max subscription the
marginal token price is flat and the provider cache already rides the client's own
breakpoints, so `auto` stays passive there — and the passivity is more than caution: the
provider rejects a `ttl:"1h"` body on a subscription-OAuth credential with an HTTP 400 in
practice (measured 2026-07-18, even with the beta union above), so even an explicit
`--managed-cache on` is refused on those seats and **downgrades to passive with a
witnessed reason** rather than fail turns. The caching those seats still get comes from
the rest of the managed-cache family (the provider 5m cache on well-placed breakpoints,
star-anchor placement, compaction, tool-prune); API-key billing is the sanctioned path to
the 1h tier. Pass `--managed-cache on` to request it, `off` to opt out.

## A worked turn (illustrative arithmetic)

On a session that has grown to a **100,000-token** prefix, one more turn (round rates,
substitute your model's real per-token price):

| One more turn on a 100k prefix | Billed at | Relative input cost |
|---|---|---|
| No cache (full re-send) | 100,000 × 1.0 | **100,000** |
| Summarize to 20k, uncached (prefix match broken) | 20,000 × 1.0 | **20,000** |
| Cache hit on full prefix (fak keeps it byte-identical) | 100,000 × 0.1 | **10,000** |

The summarize row looks cheapest until you remember it *broke* the prefix match: it pays
full price for its 20k **and** forfeits the discount on everything it kept. Keeping the
discount beats shrinking the prompt (`long-session-economics.md`).

## How fak measures the realized saving

fak never *claims* the discount — the provider decides whether it actually reuses the
cache. fak reports both numbers side by side, in the same input-token currency, and keeps
provider-owned economics separate from fak-authored work:

- On `/metrics`: `fak_gateway_compaction_shed_tokens_total` (**WITNESSED** — what fak sent)
  next to `fak_gateway_compaction_cache_read_tokens_total` (**OBSERVED** — the provider's
  relayed `cache_read`), plus `fak_gateway_cache_ttl_upgrade_total` (the count of 1h
  upgrades fired) and `fak_vcache_saved_token_equiv` (net realized provider-cache saving in
  token-equivalents, negative on a cold-write-dominated session until reads repay writes).
- `fak cachevalue report` folds two durable ledgers, **never blended** (`cmd/fak/cachevalue_report.go`,
  [cache-value roll-up](../../cache-value-rollup.md)): **Track 1** WITNESSED kernel reuse
  (`docs/nightrun/cache-value.jsonl`) and **Track 2** OBSERVED net dollars
  (`docs/nightrun/cache-savings.jsonl`), split by owner as `avoided=$X (provider $P + fak $F)`.

On the production ledger snapshot for **2026-07-01 → 2026-07-09** (8.33 days, 2,004
finished sessions), the report recorded **$8,558 of API cost avoided** — a **78.3% net
reduction** — while shedding **97.1M** context tokens. The honesty is in the split:
fak-authored work is the *smaller* slice (**9.4%** of the token-equivalent saved); the rest
is provider prompt-cache economics fak *preserves* rather than *creates*
(`long-session-economics.md`). Treat that as a dated snapshot of a growing ledger — run the
command with `--since` for the current fold.

## Try it

```bash
fak manage --managed-cache on -- claude         # force the 1h TTL upgrade on the wire (API-key billing)
fak manage --managed-cache off -- claude        # forward the client's own cache_control only
fak cachevalue report --since 2026-07-01         # two-track P&L: WITNESSED kernel vs OBSERVED $, split by owner
fak cachevalue report --since 2026-07-01 --json  # the same fold as structured JSON
```

The default compaction lever that keeps the prefix byte-identical is on at ~48k resident
tokens (`gateway.DefaultCompactHistoryBudget`); tighten or disable it with
`--compact-history-budget N` (pass `0` to turn it off).

## See also

- [Level 2 — Managed cache in practice](level-2-managed-cache-in-practice.md): the
  `--managed-cache` posture and defaults this rung prices out.
- [Level 4 — The kernel-owned KV cache](level-4-kernel-kv-cache.md): when fak owns the
  cache and eviction becomes a kernel operation instead of a re-prefill.
- [Long-session economics](../long-session-economics.md) and
  [Long sessions: shed history, keep the cache hit](../long-sessions-keep-the-cache-hit.md) —
  the source explainers this rung grew from.
- [O(1) context-window economics](../o1-context-window-economics.md) — the crossover where
  sending less beats caching more.
- [Cache-value roll-up](../../cache-value-rollup.md) — the two-track measurement contract behind `fak cachevalue report`.

---

*[← Level 2](level-2-managed-cache-in-practice.md) · [full ladder](README.md) · next → [Level 4](level-4-kernel-kv-cache.md)*
