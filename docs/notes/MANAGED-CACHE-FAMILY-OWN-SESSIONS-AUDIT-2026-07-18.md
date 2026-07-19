---
title: "Managed-cache family — own-sessions audit from the durable gateway-usage ledgers"
description: "Witnessed, quantitative per-feature audit of the whole managed-cache feature family (provider 5m cache, star-anchor, compaction shed, tool-prune, defer-cold-tools, KV-prefix, 1h-TTL upgrade, volatile-head redaction) on our own OAuth Claude Code sessions, from the fleet and local gateway-usage JSONL ledgers."
date: 2026-07-18
---

# Managed-cache family — own-sessions audit (2026-07-18)

Managed cache is a **family** of features, not one lever. This note audits each
family member against what our own sessions actually recorded in the durable
gateway-usage ledgers — which features fire by default, which are inert, and why.

## Sources, windows, and population

Two JSONL ledgers, schema `fak-gateway-usage-ledger/1`, one `kind:"exit"` row per
session, counters under `counters`:

| Ledger | Path | Rows | Exit-row span (UTC) |
|---|---|---:|---|
| FLEET (historical) | `docs/nightrun/gateway-usage.jsonl` | 3,353 | 2026-07-01 → 2026-07-11 |
| LOCAL (this machine, live) | `.fak/nightrun/gateway-usage.jsonl` | 1,296 at snapshot (2026-07-18 local; file still appending) | 2026-07-16 → 2026-07-19 |

The two ledgers do not overlap: FLEET covers 07-01..07-11, LOCAL covers
07-16..07-19Z. "Last 7 days" (07-11..07-18) is therefore the entire LOCAL ledger
plus the FLEET 07-11 slice (73 own sessions); there is a 07-12..07-15 gap in
these two files.

**Own OAuth seats** = `session_type:"guard"` with context `claude` /
`...\claude.exe` (Claude Code client on OAuth subscription seats):

| Ledger | Own sessions / exit rows | Other big buckets |
|---|---|---|
| FLEET | **1,383 / 3,353** | `serve`/`stdio` 1,893; `guard`/`codex` 41; `guard`/`opencode` 14 |
| LOCAL | **1,096 / 1,296** | `serve`/`stdio` 185; `guard`/`codex` 6 |

Session profile (OBSERVED): LOCAL own-session uptime median **320 s** (p25 286,
p75 403) — the ~5-minute short-session shape. FLEET own sessions include a large
probe tail: **590/1,383** recorded zero tokens in any direction (vs 12/1,096
LOCAL). "Substantial" below = ≥20k prompt-side tokens
(`input + cached_prompt + cache_creation`): FLEET 793/1,383, LOCAL 1,084/1,096.

**Counter-semantics caveats (respect these when reading every table):**

- `cache_ttl_upgrades_upgraded` counts fak **authoring** the 1h-TTL upgrade on
  the outbound body — **not** the provider accepting it.
- `cached_prompt_tokens` = provider cache **read**; `cache_creation_tokens` =
  provider cache **write**. Cache-read share = read / (read + write).
- The ledger has **no star-anchor placement counter** (verified: no counter key
  matching anchor/breakpoint/vcache in either ledger), so anchor fire-vs-bail is
  not directly witnessable here.

## Per-feature verdict table

| # | Family member | Configured for our seats? | Fires on our sessions? | Verdict |
|---|---|---|---|---|
| 1 | Provider 5m prompt cache (client's own `cache_control` breakpoints) | On (Claude Code places breakpoints) | **Yes, dominant.** Median cache-read share 84.8% FLEET / 78.4% LOCAL on substantial sessions; 0 substantial sessions with zero cache traffic | **EFFECTIVE by default** |
| 2 | Star-anchor breakpoint placement (`--vcache-anchor`, default-on) | On | **No ledger witness** (no placement counter). INFERRED inert for our seats: Claude Code sends its own breakpoints, so the anchor path bails `already_set` by design | **INERT for our seats (by design); unwitnessed in this ledger** |
| 3 | Compaction shed (`--compact-history-budget`; launchers now 96000) | On (LOCAL provenance: 1,048/1,096 rows at 96000) | **Partially.** Fires in 43.7% of own sessions FLEET / 27.4% LOCAL; per-attempt fire rate 43.1% FLEET vs 20.2% LOCAL; dominant bail = `under_budget` | **EFFECTIVE on long sessions only; 96K budget regime mostly bails on our ~5-min sessions** |
| 4 | Tool-prune | On | **Yes, near-universal.** 96.0% of own sessions FLEET, 99.5% LOCAL | **EFFECTIVE by default** |
| 5 | Defer-cold-tools (`--defer-cold-tools`) | Lever-gated, off | **No.** Zero `fak_defer_cold_*` counters in either ledger (key never even appears) | **INERT (off)** |
| 6 | KV-prefix reuse (in-kernel model only) | N/A on this wire | **No.** `kv_prefix_*` = 0 everywhere | **INERT (not our wire)** |
| 7 | 1h-TTL upgrade (`--managed-cache` / cacheTTL1H) | Was passive-by-default (env trap); flipped auto→on 2026-07-18 | **Authors but is not served.** 315 FLEET rows with `upgraded>0` (all 07-09/10 forced-on era, 314 zero-traffic); 5 LOCAL rows post-flip (07-18/19Z), **all zero-traffic** | **PROVIDER-BLOCKED on OAuth (400); counter = authoring, not acceptance** |
| 8 | Volatile-head redaction (`FAK_CACHEBP_REDACT`) | Default-off | **No counter in ledger**; the one FLEET reasons histogram shows `volatile_head` as the dominant upgrade-skip reason (45 of 46 skips) | **INERT (off); is the gate in front of #7** |

## Per-feature detail

### 1. Provider 5m prompt cache — the workhorse (OBSERVED)

Cache-read share = `cached_prompt_tokens / (cached_prompt_tokens + cache_creation_tokens)`,
substantial own sessions only:

| Window | n | p25 | median | p75 | Aggregate share |
|---|---:|---:|---:|---:|---:|
| FLEET overall (07-01..11) | 793 | 77.1% | **84.8%** | 94.0% | 88.0% (read 2,180.4M / write 296.3M; uncached input 11.3M) |
| FLEET last-7d slice (07-11 only) | 61 | — | **78.8%** | — | — |
| LOCAL overall = last-7d (07-16..19Z) | 1,084 | 72.4% | **78.4%** | 81.2% | 77.4% (read 544.8M / write 159.4M; uncached input 2.6M) |

Per-day LOCAL medians are stable: 77.5% (07-16), 80.0% (07-17), 77.8% (07-18),
73.9% (07-19Z, partial day). Every substantial own session had nonzero cache
traffic (0 zero-cache substantial sessions in either ledger). `cached_turns>0`
in 778/1,383 FLEET and 1,083/1,096 LOCAL own sessions.

This caching rides the client's own breakpoints — fak observes and accounts for
it, but Claude Code causes it. It is the reason "is caching working" = yes.

### 2. Star-anchor placement (OBSERVED absence + INFERRED behavior)

No placement/outcome counter exists in either ledger (checked all counter keys),
so fire-vs-bail cannot be witnessed from this data. Honest status: **unwitnessed
here**. INFERRED from the established mechanism: the anchor only places a
breakpoint when the caller sent none, and Claude Code always sends its own — so
on our seats the path bails `already_set` every turn. Wiring
`PlaceAnthropicCacheBreakpointWithOutcome` outcomes into the usage ledger is the
missing witness.

### 3. Compaction shed (OBSERVED)

| Window | fired | bailed | per-attempt fire rate | sessions with ≥1 fire | shed tokens | dropped turns |
|---|---:|---:|---:|---:|---:|---:|
| FLEET overall | 13,962 | 18,413 | 43.1% | 606/1,383 (43.7%) | 636.9M | 423,497 |
| FLEET 07-11 slice | 1,669 | 2,535 | 39.7% | — | — | — |
| LOCAL overall | 1,957 | 7,718 | 20.2% | 300/1,096 (27.4%) | 94.7M | 70,671 |

Bail reasons — FLEET: `under_budget` 10,678, `burst_unprofitable` 4,784,
`too_few_msgs` 2,900, `window_no_drop` 27, `cached_span` 24. LOCAL:
`under_budget` 6,281, `too_few_msgs` 1,122, `burst_unprofitable` 313.

The budget regime is visible in the LOCAL provenance split: sessions launched
with `compact_history_budget=48000` (n=48) fire at **55.8%** per attempt and
100% of them fire at least once; sessions at 96000 (n=1,049) fire at **17.7%**
and only 24% ever fire. The 96K default means most of our short (~5-min)
sessions stay `under_budget` — working as designed, not broken.

### 4. Tool-prune (OBSERVED)

| Window | sessions with prune>0 | tool_prune_turns (p25/med/p75) | tool_prune_count (p25/med/p75) | total pruned |
|---|---:|---|---|---:|
| FLEET overall | 1,327/1,383 (96.0%) | 1 / 5 / 34 | 13 / 60 / 543 | 469,787 |
| LOCAL overall | 1,090/1,096 (99.5%) | 7 / 7 / 8 | 78 / 91 / 104 | 112,982 |

Fires on essentially every session that carries traffic. Clearly
effective-by-default.

### 5. Defer-cold-tools (OBSERVED absence)

No `fak_defer_cold_*` counter key appears in any of the 4,649 rows across both
ledgers — not zero-valued, absent. The lever is off fleet-wide. Inert, as
expected.

### 6. KV-prefix reuse (OBSERVED zeros)

`kv_prefix_prompt_tokens` = `kv_prefix_reused_tokens` = 0 in every own-session
row in both ledgers. In-kernel-model-only counter; not our Anthropic wire.

### 7. 1h-TTL upgrade (OBSERVED + explicit caveat)

Counter caveat again: `cache_ttl_upgrades_upgraded` = fak **authored** the
upgrade on the outbound body. It says nothing about provider acceptance.

- **FLEET:** 315 own-session rows with `upgraded>0`, all on 2026-07-09 (2) and
  2026-07-10 (313) — the forced-on era. **314 of 315 recorded zero tokens in
  every direction** (in=read=write=out=0): the sessions authored upgrades and
  then recorded no served traffic at all, consistent with the known provider
  **400 on OAuth** rejecting the request outright (INFERRED cause; the
  zero-traffic rows are OBSERVED). Total authored upgrades: 392.
- **The one counterexample (OBSERVED):** 2026-07-10 21:15:38Z, ctx `claude`,
  `upgraded=19`, served fully (in 25,269 / out 103,821 / read 3,519,466 / write
  737,644). Its `cache_ttl_upgrade_reasons` histogram — the only one in either
  ledger — shows the skip-side gate: `volatile_head: 45`, `no_stable_breakpoint: 1`
  (46 turns skipped vs 19 upgraded).
- **After 07-10** every FLEET row has `upgraded=0` (passive default — the HKCU
  `FAK_MANAGED_CACHE=auto` env trap).
- **LOCAL, post auto→on flip (2026-07-18):** 5 rows with `upgraded>0`
  (07-18: 2, 07-19Z: 3; upg 3–6, total 21), on post-flip builds
  (`f62f96286125-dirty`, `92ef745d80aa-dirty`, `c65841306f08`), budget 96000.
  **All five are zero-traffic rows** (in=read=write=out=0, uptime 20–42 s,
  errored=0) while 1,000+ sibling sessions the same days served normally with
  `upgraded=0`. OBSERVED: authoring resumed after the flip and authored-upgrade
  sessions record no served tokens. INFERRED: the OAuth provider block still
  bites when the upgrade is actually authored; most sessions avoid authoring
  (volatile-head skip, per the one reasons histogram) and serve normally.

Net: one row in the family, still provider-blocked for our seats.

### 8. Volatile-head redaction (OBSERVED absence)

`FAK_CACHEBP_REDACT` is default-off and no redaction counter exists in the
ledger. Its relevance is indirect but load-bearing: `volatile_head` is 45/46 of
the only observed upgrade-skip histogram, i.e. redaction-off is what keeps the
1h-TTL authoring rate near zero even when the feature is armed.

## Headline

**On our own OAuth Claude Code sessions, the managed-cache family is carried by
exactly two members — the provider 5m prompt cache riding the client's own
breakpoints (median cache-read share 84.8% FLEET / 78.4% LOCAL on substantial
sessions, zero substantial sessions without cache traffic) and tool-prune
(96–99.5% of sessions) — while the rest of the family is inert on our seats:
star-anchor bails `already_set` (and has no ledger witness), compaction mostly
bails `under_budget` in the 96K-budget regime on ~5-minute sessions (fires in
only 20–43% of attempts), defer-cold-tools and KV-prefix are off/not-our-wire,
and the 1h-TTL upgrade remains provider-blocked on OAuth — every one of the 320
sessions that ever authored an upgrade except one (2026-07-10, upg=19) recorded
zero served tokens, including all 5 post-flip rows on 2026-07-18/19.**

## Residual opportunity

1. **Witness gap (cheapest):** wire star-anchor placement outcomes and a
   1h-TTL acceptance/rejection signal (provider response status) into the usage
   ledger. Today the ledger cannot distinguish "authored and accepted" from
   "authored and 400'd" — we infer it from zero-traffic rows.
2. **1h-TTL on OAuth:** blocked at the provider, not in fak. The volatile-head
   skip (redaction default-off) is currently *protecting* sessions from the
   400; if/when the provider accepts 1h TTL on OAuth, `FAK_CACHEBP_REDACT`
   becomes the effectiveness gate to revisit.
3. **Compaction on short sessions:** the 48K-budget cohort fires at 55.8% vs
   17.7% at 96K. If shed-on-short-sessions is wanted, the budget is the lever;
   if 96K is the deliberate interactive regime, current behavior is correct and
   `under_budget` bails are healthy.
4. **Cache-write share:** ~12% FLEET / ~23% LOCAL of cache-touched tokens are
   still writes. That is the provider-5m-TTL churn a working 1h TTL (or longer
   sessions) would recover — the concrete number the blocked feature is worth.

*All numbers OBSERVED from the two ledgers cited above unless marked INFERRED.
Analysis script: session scratchpad (`mcaudit.py`, `mcaudit2.py`), not in the
repo tree. LOCAL ledger is live; counts are the 2026-07-18 snapshot.*
