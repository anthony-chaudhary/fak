# Cache-Savings Multi-Turn Demo

`cachedemo` is a no-model, read-only fold over fak's durable, WITNESSED cache-value
ledgers. It tells the per-turn story of how one fak-guarded coding session earns cache
value across turns — the provider prompt cache reading a stable prefix back cheaply, and
fak's own authored slice (per-fire compaction trim + tool-result prune) holding the
re-sent window inside budget — and prints a cumulative owner split that never blends the
two.

## Prerequisites

Requires Go only. It runs offline over local JSONL ledgers and does not need a model, an
API key, a GPU, a database, or a network. It reads counts and token totals only — never a
prompt byte.

## Quick Start

From the repo root:

```bash
go run ./cmd/cachedemo                     # human multi-turn narrative over the default ledgers
go run ./cmd/cachedemo --since 2026-07-04  # fold only savings rows on or after this date
go run ./cmd/cachedemo --session-pid 53664 # pin the per-turn spine to one witnessed session
go run ./cmd/cachedemo --json              # emit the raw fleet-benefit report JSON
```

A run completes in well under a second, writes nothing, and returns exit code 0. It is a
deterministic read-only fold: the same ledgers always produce the same output, so a run is
reproducible for any fixed ledger state.

## What You See

The numbers below are an illustrative capture — they reflect the ledgers at capture time
and grow as new witnessed sessions land, but the shape is stable:

```text
fak — shared cache savings, multi-turn demo
═══════════════════════════════════════════

PER-TURN SPINE — one witnessed guard session (pid 53664, 2026-07-08T00:47:01Z, 82 min)
  turns with cache activity ......... 599
  Provider prompt cache (OBSERVED — the provider's own cache, across turns):
    cache_read  = 82,253,550 tokens  (stable prefix read back, not re-billed at full rate)
    read : write = 18.0 : 1  (each written token was reused ~18.0×)
  fak-authored slice (WITNESSED — fak trims the re-sent history to hold budget):
    compaction fired ......... 11 turns  (bailed 608 — bail is a positive: no profitable shed)
    shed PER FIRE ............ ~33,202 tokens  (365,223 cumulative ÷ 11 fires — the honest marginal)

CUMULATIVE OWNER SPLIT (savings rows since 2026-07-04)
  API cost avoided ......... provider $4794.11 (OBSERVED) + fak $477.70 (WITNESSED, price-honest) = $5271.82
  fak share (count-axis) ... 9.0% — an UPPER BOUND off the over-counted shed
  provenance: gateway-usage counters are WITNESSED; provider prompt-cache dollars are OBSERVED/provider-relayed
```

Read it in three parts:

- **Per-turn spine** — one real guard session, showing the provider cache reading the
  prefix back (`cache_read`, labeled OBSERVED) alongside fak's compaction firing to trim
  aged history (`shed PER FIRE`, labeled WITNESSED). The headline is shed *per fire*, not
  the session sum.
- **Cumulative owner split** — whose cache work earned the avoided dollars, provider vs
  fak, never blended: OBSERVED provider dollars stay attributed to the provider, WITNESSED
  fak dollars to fak.
- **Provenance line** — the label every figure carries, so a provider-heavy corpus can
  never read as a fak win.

## What This Does Not Claim

This demo does not hit a live model and does not settle a dollar figure. Specifically:

- The cumulative `compaction_shed` sum is **not** distinct tokens saved — fak re-trims the
  full re-sent history every fire, so the same aged middle is re-counted. The honest
  headline is shed **per fire**; the cumulative sum is an over-count ceiling (the distinct
  COUNT axis is open in epic #3095).
- The `fak share` percentage is an **upper bound** off that over-counted shed, not a
  settled owner share (`fak cachevalue report` gives the honest ~0.3–16% range).
- Provider prompt-cache dollars are **OBSERVED / provider-relayed** projections, not a fak
  saving; only the WITNESSED fak compaction slice is attributed to fak.

It proves the owner-split accounting and the per-fire honesty discipline over recorded
evidence — nothing about a live session or a projected future rate.

## Related Docs

- [Cache-value rollup](../../docs/cache-value-rollup.md) — WITNESSED reuse vs OBSERVED dollars
- [What a saved token is worth](../../docs/explainers/what-a-saved-token-is-worth.md)
- [Context shedding](../../docs/explainers/context-shedding.md) — the per-fire trim this demo folds
- Ledgers it reads (live runtime defaults `.fak/nightrun/{gateway-usage,cache-savings}.jsonl` since the #3209 migration; the tracked publication snapshots are linked here): [`gateway-usage.jsonl`](../../docs/nightrun/gateway-usage.jsonl), [`cache-savings.jsonl`](../../docs/nightrun/cache-savings.jsonl)
