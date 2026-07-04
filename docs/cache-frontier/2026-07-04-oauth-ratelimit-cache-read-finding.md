---
title: "Finding: does provider cache-read discount rate-limit burn on subscription seats?"
description: "Cache-frontier finding for #2183 (epic #1844 C6): the managed-cache AUTO passive-on-OAuth rule was assumed on the dollar axis and never measured on the rate-limit axis. This records the result of looking at that axis — the correlation is currently UNMEASURABLE because the two sides of the join are not co-persisted — and classifies the horizon."
---

# Finding: cache-read vs rate-limit burn on subscription (OAuth) seats

- Issue: [#2183](https://github.com/anthony-chaudhary/fak/issues/2183) (epic
  [#1844](https://github.com/anthony-chaudhary/fak/issues/1844) C6).
- Date: 2026-07-04.
- Verdict: **negative-by-missing-witness** — the measurement the issue asks for
  cannot be produced from existing fak telemetry, because the limit-burn side and
  the cache-read side of the required join are not co-persisted. The
  passive-on-OAuth AUTO rule therefore stays in place, but its provenance is
  upgraded from *silently assumed on an axis it never looked at* to *explicitly
  unmeasured on the rate-limit axis, with a named witness gap*.

## The question

`--managed-cache` AUTO stays passive on subscription (OAuth) seats because "on a
Pro/Max subscription the marginal token price is flat"
([`cmd/fak/guard_managed_cache.go:21-23`](../../cmd/fak/guard_managed_cache.go)).
That reasoning is denominated in **dollars**. Subscription seats do not die on
dollars — they die on **rate limits** (the whole reason `fak resume scan`
exists; the live-vs-post-mortem cap taxonomy is
[`cmd/fak/resume_scan.go:237`](../../cmd/fak/resume_scan.go) →
`resume.ClassifyLimitText`). #2183 asks the axis the dollar argument never looked
at: if the provider's rate-limit accounting discounts cache reads the way its
*billing* does (the `0.1x` provider cache-read multiplier is real and pinned at
[`internal/cachevaluereport/track2.go:46`](../../internal/cachevaluereport/track2.go)
`providerCacheReadMultiplier = 0.1`), then keeping an OAuth session cache-hot
would extend its runway before a limit crash — real managed-cache value on a
flat-rate seat, denominated in rate-limit headroom instead of dollars.

## What was checked

The issue asserts "fak already has both sides of the join in the gateway." That
is true at the **live observation seam** and false at the **durable-ledger
seam** — and the correlation the acceptance criteria want is a post-hoc join over
real sessions, which needs the durable seam.

- **Limit-burn side (account usage).** `internal/accountobs` folds the
  provider's `anthropic-ratelimit-unified-*` (subscription 5h/7d window
  utilization / status / reset) and `anthropic-ratelimit-<family>-*` /
  `x-ratelimit-*` headers off every upstream response
  ([`internal/gateway/upstream_observe.go`](../../internal/gateway/upstream_observe.go)
  → `internal/accountobs/accountobs.go`). But a `Tracker` is a **latest-value,
  in-memory** view: it is rendered only to (a) the guard exit banner
  (`Snapshot.Report`) and (b) the `/metrics` `fak_account_ratelimit_*` gauges
  (`Snapshot.PrometheusText`). Nothing writes a per-session utilization/burn row
  to any append-only ledger. The gauges are point-in-time scrape values, not an
  archived per-session series.
- **Cache-read side.** The durable per-session record is the cache-savings ledger
  `docs/nightrun/cache-savings.jsonl` (schema `fak-cache-savings-ledger/1`,
  `internal/cachevaluereport` `SavingsRow`). It carries `cache_read_tokens`,
  `input_tokens`, `cache_creation_tokens` — but **no** rate-limit / window /
  utilization / limit-remaining field.
- **The join.** There is no durable row, and no shared session key, that pairs a
  session's cache_read-vs-uncached-input ratio with that same session's observed
  unified-window utilization delta or limit-crash timing. `SavingsRow` and the
  `accountobs` snapshot are never co-recorded. So the correlation the issue
  requests has no data source to run over.

## Result

- Acceptance box 1 ("a written finding with **measured** limit-burn vs cache-read
  data from real subscription sessions") is **blocked on a missing witness** — the
  co-persisted join does not exist. This is not "we measured and found no
  discount"; it is "the measurement cannot yet be taken." Recording the difference
  honestly is the point: the AUTO rule is no longer resting on an unstated
  assumption.
- Acceptance box 2 is satisfied by its second branch — **a documented result cited
  from `guard_managed_cache.go`**: the passive-on-OAuth default is retained, and
  its banner reason ("subscription OAuth (flat-rate) — pass `--managed-cache on` to
  force", `resolveGuardManagedCache`) remains honest, because fak still cannot see
  the wire economics on that axis. The rule is not flipped speculatively — the
  never-speculate honesty in `guard_managed_cache.go:17-23` and the banner-reason
  contract in `bannerLine` both forbid activating on evidence fak does not have.

### Untested hypothesis (must not be read as measured)

A plausible *prior* — explicitly **not** a finding — is that the subscription
unified window meters **usage**, not dollars, so a cache read may count toward the
5h/7d window at (or near) the same weight as uncached input even though it is
billed at `0.1x`. If that holds, cache reads would **not** discount limit burn on
the token axis, and the passivity would be correct on the rate-limit axis too. The
opposite prior — that the provider applies the same `0.1x` weighting to unified
usage as to billing — is equally unproven. Both are hypotheses the missing witness
would decide. Nothing here should be cited as evidence for either.

## Generation classification

Per [`docs/generation.md`](../generation.md), the issue was `unclassified`. On the
evidence above it classifies as **`gen/future`** (research / long-horizon option):
it studies whether a standing assumption holds on an axis with no current-product
witness, and its promotion depends on a witness that does not exist yet. It is not
`gen/now` (no current-product readout to move today), and not `gen/next` (no
near-term gate or dogfood — the enabling schema change has to land first).

- **Promotion evidence** (what would move it toward `now`): a schema field that
  co-persists the per-session `accountobs` unified-window utilization delta (and
  429/crash timing) next to `cache_read_tokens` / `input_tokens` on the
  `cache-savings` (or a sibling) ledger, then N real OAuth sessions correlating
  limit-burn against cache-read ratio. That correlation is the promotion witness.
- **Demotion / retirement evidence**: if the provider documents (or a measured
  session shows) that unified-window usage weights cache reads the same as uncached
  input, the lever has no runway value on OAuth — retire the "activate AUTO on
  OAuth" bet and keep passivity as *measured*, not assumed.
- **Invalidating assumption**: this finding assumes the join must be reconstructed
  from fak-side telemetry. If the provider ever relays a per-response cache-read
  *usage* attribution in the unified-window headers, the correlation could be read
  directly off `accountobs` without a new ledger schema, and the "missing witness"
  framing here would be wrong.

## Smallest next step

Land the co-persistence witness: add the per-session unified-window
utilization/burn fields to the durable session ledger so a later worker can run the
correlation on real OAuth sessions. Until then, `--managed-cache` AUTO passivity on
OAuth is **evidence-tracked as unmeasured**, not silently assumed — which is the
state #2183 asked for.
