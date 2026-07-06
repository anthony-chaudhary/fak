---
title: "Per-tenant weighted fairness floors for KV eviction — design + staged plan (#2674)"
description: "Grounded design dossier for a two-level KV evictor that protects each tenant's fair-share floor of resident KV before value-ranking the surplus globally, so a quiet tenant is not starved by a noisy neighbor. Extends the pure cost-of-losing primitives in internal/compute (#2239/#2666)."
slug: kv-tenant-fairness-floors
keywords:
  - KV cache eviction
  - multi-tenant fairness
  - weighted GDSF
  - max-min fair share
  - noisy neighbor
  - fair-share floor
  - internal/compute
date: 2026-07-06
---

# Per-tenant weighted fairness floors for KV eviction (#2674)

Design + staged-implementation dossier for
[#2674](https://github.com/anthony-chaudhary/fak/issues/2674) (epic
[#2236](https://github.com/anthony-chaudhary/fak/issues/2236), M5 lifecycle row; extends
[#2239](https://github.com/anthony-chaudhary/fak/issues/2239) /
[#2666](https://github.com/anthony-chaudhary/fak/issues/2666)). This is a **design** — no
code has been written. Every claim about what the code does *today* cites a real
`file:line`; everything under "Proposed mechanism" and later is a proposal, not a shipped
fact.

**Grounding correction up front.** The ticket dispatch named the primary subsystem as
`internal/kvbm`. **That package does not exist in this tree** (`ls internal/kvbm` fails).
The KV eviction primitives the issue actually targets live in **`internal/compute`** (the
issue body itself says "a pure, deterministic primitive in `internal/compute`"), with the
consuming radix cache in `internal/radixkv`. This dossier is grounded against those real
packages. The fak-guard lease was taken on the *named* tree `internal/kvbm/**` (verdict
below); because that tree is empty it is trivially disjoint from all live work, but a real
implementation lease must be re-taken on `internal/compute/**` (see Risks).

## (a) Problem restated

Cost-aware eviction (#2239) ranks *all* resident spans on one global value scale —
`recomputeCost × reuseProbability ÷ bytes` — and evicts the globally cheapest
(`KVEvictionCost` at `internal/compute/kvcost.go:141`; picker at
`internal/compute/kvcost.go:162`). Under a shared budget with multiple tenants, a
noisy-neighbor tenant with high traffic keeps producing high-`Hits` spans, so a quiet
tenant's spans — valuable *to that tenant* but low-traffic in absolute terms — always
score as the globally cheapest victim and are evicted to zero. The quiet tenant then
cold-starts every turn while the loud one hoards the cache. Pure value-ranking has no
notion of *fair share*: it optimizes aggregate hit-rate at the cost of per-tenant
starvation.

The issue's claim: per-tenant **weights + guaranteed floors** turn one global scale into a
two-level decision — protect each tenant's fair-share floor of resident KV, then
value-rank the surplus globally — retaining a quiet tenant's proportional working set under
a noisy neighbor, trading a bounded aggregate-hit-rate cost for a tail-latency/fairness
win. With a single tenant (or equal weights and no floors) it must reduce exactly to
today's `PickEvictionVictim`.

## (b) Current seam — the exact code to extend (witnessed)

The `internal/compute` KV eviction family is a set of **pure, no-state, per-span** cost
functions, each a strict generalization of `KVEvictionCost` with a documented reduction:

| Element | Location | What it does today |
|---|---|---|
| `KVSpanStats` struct | `internal/compute/kvcost.go:30` | Per-resident-span telemetry: `Tokens`, `Bytes`, `Hits`, `LastUsed`, `Pinned`, `Leased`, `AgeStamp`, `PinBoost`, `Hint`, `Sharers`. **There is no `Tenant` field** (grep for `Tenant` in `internal/compute` returns nothing — confirmed net-new). |
| `KVEvictionCost(s)` | `internal/compute/kvcost.go:141` | The base value-of-losing score; `+Inf` fail-open on `Bytes <= 0`. |
| `PickEvictionVictim(spans)` | `internal/compute/kvcost.go:162` | Returns index of the lowest-cost evictable span; skips `Pinned`/`Leased`; `LastUsed` tie-break; `-1` when nothing evictable. |
| `KVEvictionCostAged` / `PickEvictionVictimAged` | `internal/compute/kvcost_aging.go:35,52` | Adds GDSF aging clock (`AgeStamp`); reduces to base at zero stamp. |
| `KVEvictionCostFanout` / `PickEvictionVictimFanout` | `internal/compute/kvcost_fanout.go:38,60` | Weights recompute by concurrent `Sharers`; reduces to base at `Sharers <= 1`. |
| `KVEvictionCostPinned` / `PickEvictionVictimPinned` | `internal/compute/kvcost_pin.go:78,142` | TTL-decaying pin economics + bounded agent hint; reduces to base with no pin/hint. |
| `saturatingMulInt64(...)` | `internal/compute/capacity.go:287` | Overflow-safe multiply used by the fan-out weighting — reusable for weight scaling. |

**The load-bearing structural fact.** Every existing `PickEvictionVictim*` is a
**single-span scalar scorer folded into one argmin loop**: it needs only the candidate
span in hand (`KVEvictionCostFanout(s)` at `internal/compute/kvcost_fanout.go:38` takes one
`KVSpanStats`). Fairness is **not** expressible that way. Deciding whether a span is
"above or below its tenant's floor" requires the *aggregate* resident bytes of that
tenant across the whole resident set — a property of the set, not of the span. So the
fair picker cannot be a drop-in `KVEvictionCostFair(s)` scalar; it must see the full
`[]KVSpanStats` and a policy. This is the genuine architectural departure the issue
implies with its proposed signature `PickEvictionVictimFair(spans, policy) int`.

The R2 consumer seam (witnessed): `internal/radixkv/radixkv.go:401`
`(*Tree).costAwareLeaf()` collects leaf `KVSpanStats` via `node.kvSpanStats()`
(`internal/radixkv/radixkv.go:435`) and calls `compute.PickEvictionVictim(spans)` at
`internal/radixkv/radixkv.go:425`. Today `kvSpanStats()` populates only
`Tokens/Bytes/Hits/LastUsed/Leased` (`:436-442`) — **no tenant is threaded**. That is the
exact function a follow-on must extend to supply `Tenant` (and it is where per-tenant
resident-byte bookkeeping would attach).

The replay witness harness (witnessed): `ReplayKVCache(events, budget, policy)` at
`internal/compute/kvcost.go:225` simulates a token-budgeted cache under `KVEvictLRU` vs
`KVEvictCostAware` (`internal/compute/kvcost.go:186-198`) and returns `(hitTokens,
accessTokens)`. `KVReplayEvent` (`internal/compute/kvcost.go:205`) carries only
`SpanID`/`Tokens` — **no tenant** — so witnessing floor protection needs a tenant-aware
replay variant (see Test plan).

## (c) Proposed mechanism + concrete interface (design)

Add one file, `internal/compute/kvcost_fairness.go`, following the existing family's
no-state discipline. Net-new surface:

```go
// Tenant is an opaque per-tenant/per-session id. The zero value ("") is "untenanted"
// and always fails open to the global picker.
type Tenant string

// Add to KVSpanStats (internal/compute/kvcost.go:30), after Sharers:
//   // Tenant owns this span for fair-share accounting (#2674). Zero value = untenanted,
//   // which makes the fair picker fall through to the global PickEvictionVictim.
//   Tenant Tenant

// KVFairnessPolicy carries per-tenant weights + guaranteed floors. Pure data.
type KVFairnessPolicy struct {
    // Weights: relative share of the budget each tenant is entitled to. Absent/zero
    // weight => unweighted (falls back to global ranking for that tenant's surplus).
    Weights map[Tenant]float64
    // FloorBytes: absolute guaranteed resident floor per tenant. A tenant whose current
    // resident bytes are <= its floor is protected: none of its spans may be evicted
    // while any above-floor (surplus) span anywhere is evictable.
    FloorBytes map[Tenant]int64
    // Base selects the per-span scorer applied WITHIN the surplus set, so fairness
    // composes with aging/fanout/pin (default: KVEvictionCost).
    Base func(KVSpanStats) float64
}

// PickEvictionVictimFair is the two-level decision. It never evicts a span whose tenant
// is at-or-below its floor while any surplus span is evictable; among surplus spans it
// applies policy.Base (default KVEvictionCost) exactly like PickEvictionVictim. Returns
// -1 when nothing is evictable. Fail-open: nil/empty policy, or all-untenanted spans,
// reduce EXACTLY to PickEvictionVictim (proven by the reduction test).
func PickEvictionVictimFair(spans []KVSpanStats, policy KVFairnessPolicy) int

// KVFairnessStats is per-tenant observability: resident bytes vs floor, the witness input.
type KVFairnessStats struct {
    Tenant        Tenant
    ResidentBytes int64
    FloorBytes    int64
    SpanCount     int
    BelowFloor    bool
}

// KVFairnessReport summarizes a resident set against a policy (for metrics + the witness).
func SummarizeKVFairness(spans []KVSpanStats, policy KVFairnessPolicy) []KVFairnessStats
```

**Algorithm (max-min-fair-share form, the issue's ranked target #2).**

1. First pass: sum `ResidentBytes` per `Tenant` over `spans` (skip `Pinned`/`Leased`,
   which are excluded from candidacy regardless — matching the base picker's hard
   exclusions at `internal/compute/kvcost.go:167`).
2. Classify each evictable candidate: a span is **protected** if its tenant's
   `ResidentBytes <= FloorBytes[tenant]`, else **surplus**. A floor is derived from
   `FloorBytes[tenant]` directly, or (per the issue's "floor-as-fraction") from
   `Weights[tenant] / Σweights × budget` when a budget is supplied — the weighted-fair
   form (ranked target #1).
3. If any surplus span is evictable, pick the argmin **among surplus spans only**, using
   `policy.Base` (default `KVEvictionCost`), with the same `LastUsed` oldest-first
   tie-break as `PickEvictionVictim` (`internal/compute/kvcost.go:176`).
4. **Only if no surplus span exists** (every evictable tenant already at/below floor —
   floors over-subscribe the budget) fall back to a global argmin over the protected set,
   so the picker never deadlocks and always frees memory when eviction is forced. This is
   the graceful-degradation floor: fairness yields to liveness, never the reverse.

**Reduction (must be proven).** With `policy.Weights`/`FloorBytes` empty (or all spans
`Tenant == ""`), step 2 classifies *everything* as surplus, step 3 runs `policy.Base` over
all spans, and with `Base == nil` defaulting to `KVEvictionCost` the result is
**bit-identical to `PickEvictionVictim`**. This is the same "strict generalization, never a
divergence" contract every sibling in the family carries
(`internal/compute/kvcost_fanout.go:33`, `internal/compute/kvcost_aging.go:29`).

**Composition.** Because `policy.Base` is an injected `func(KVSpanStats) float64`, a caller
can pass `KVEvictionCostAged`, `KVEvictionCostFanout`, or `KVEvictionCostPinned` to
value-rank the surplus by any composed cost — fairness is orthogonal to the per-span terms,
layering *on top of* them rather than replacing them.

## (d) Staged plan — smallest first slice, then follow-ons

**Slice 1 (SMALLEST, ships alone; R1) — the pure primitive + reduction.**
Add `internal/compute/kvcost_fairness.go` with the `Tenant` field on `KVSpanStats`,
`KVFairnessPolicy`, `PickEvictionVictimFair`, `KVFairnessStats`, `SummarizeKVFairness`, and
unit tests. No caller changes. The whole slice is inside `internal/compute` (single lane).
The witnessed win is the *unit-level* starvation test (below) — a resident set where the
global picker evicts the quiet tenant to zero but the fair picker holds its floor. Adding
the `Tenant` field is byte-safe for every existing caller: the field is unread by all
`PickEvictionVictim*` functions and the zero value is untenanted.

**Slice 2 (R2) — the replay witness.** Add tenant to the replay harness: either extend
`KVReplayEvent` (`internal/compute/kvcost.go:205`) with a `Tenant` field or add a parallel
`ReplayKVCacheFair`. Witness the aggregate cost/benefit: per-tenant retained working set
(fairness win) alongside the bounded aggregate hit-rate delta vs `KVEvictCostAware`. Still
entirely within `internal/compute`.

**Slice 3 (R2 wiring, cross-lane) — supply `Tenant` from the radix cache.** Extend
`node.kvSpanStats()` (`internal/radixkv/radixkv.go:435`) to populate `Tenant` from the
session/lease that owns the leaf, and switch `costAwareLeaf()`
(`internal/radixkv/radixkv.go:401`) to `PickEvictionVictimFair` behind a
`FAK_NATIVE_KV_FAIR` flag (matching the existing `FAK_NATIVE_KV_*` gating convention
referenced at `internal/compute/kvcost.go:161`). This touches `internal/radixkv` and the
gateway/session plumbing that knows tenant identity — a separate lease.

**Slice 4 (R3) — benchmark arm.** A fairness-vs-throughput arm in the #2244 benchmark on a
noisy-neighbor trace, quantifying the aggregate-hit-rate cost against the tail-latency win.

Slices 1 and 2 are self-contained in `internal/compute` and independently shippable. Slice
3 is the first that leaves the compute leaf.

## (e) Test plan — the specific Go tests that witness each slice

All in package `compute`, file `internal/compute/kvcost_fairness_test.go`, matching the
naming/structure of `internal/compute/kvcost_fanout_test.go` and
`internal/compute/kvcost_aging_test.go`:

- **`TestPickEvictionVictimFairReducesToGlobalOnNoPolicy`** — empty policy / all
  `Tenant == ""` ⇒ same victim index as `PickEvictionVictim` on the same input (the
  reduction; mirrors `TestPickEvictionVictimFanoutReducesToPickEvictionVictimOnUniformSharers`
  at `internal/compute/kvcost_fanout_test.go:64`). **Witnesses Slice 1's core contract.**
- **`TestPickEvictionVictimFairProtectsQuietTenantFloor`** — the noisy/quiet resident set
  from the issue's witness: a noisy tenant with many high-`Hits` spans and a quiet tenant
  with few low-`Hits` spans under a tight budget, quiet tenant at/below floor. Assert the
  fair picker returns a *noisy* (surplus) span while `PickEvictionVictim` on the same input
  returns the *quiet* span. **The starvation witness — Slice 1's headline.**
- **`TestPickEvictionVictimFairFallsBackWhenAllAtFloor`** — floors over-subscribe the
  budget (every tenant at/below floor): the picker still returns a valid victim (liveness
  over fairness), not `-1`.
- **`TestPickEvictionVictimFairRespectsPinsAndLeases`** — `Pinned`/`Leased` spans are never
  victims even when they are a tenant's only surplus (mirrors
  `TestPickEvictionVictimAgedRespectsPinsAndLeases` at
  `internal/compute/kvcost_aging_test.go:80`).
- **`TestPickEvictionVictimFairComposesWithBaseScorer`** — passing `Base:
  KVEvictionCostAged` (or Fanout) ranks the surplus by the composed cost.
- **`TestSummarizeKVFairnessReportsBelowFloor`** — the observability struct reports correct
  per-tenant `ResidentBytes`/`BelowFloor`.
- **Slice 2:** **`TestReplayKVCacheFairHoldsQuietTenantWorkingSet`** — over a replayed
  noisy-neighbor trace, per-tenant retained working set under the fair policy strictly
  exceeds the plain `KVEvictCostAware` retained set for the quiet tenant, and the aggregate
  hit-rate delta is bounded (reported, not asserted-zero).

## (f) Risks & collisions

- **Lease/tree mismatch (must fix before implementation).** The fak-guard lease was taken
  on `internal/kvbm/**`, which does not exist. The real implementation tree is
  `internal/compute/**` (Slices 1-2) and `internal/radixkv/**` + gateway (Slice 3). A real
  edit must re-arbitrate a lease on `internal/compute/**`. `internal/compute` is a hot,
  frequently-edited leaf (the whole `kvcost*` family and #2673/#2670/#2668 land there) — an
  implementation session should expect contention and take an **exclusive** lease on just
  `internal/compute/kvcost_fairness*.go` if possible, or coordinate with any in-flight
  `kvcost*` work.
- **Non-scalar departure.** Unlike every sibling, `PickEvictionVictimFair` cannot be a
  `func(KVSpanStats) float64` composed into the shared argmin — it needs the full set for
  per-tenant totals. This means it does **not** get to reuse the exact
  `PickEvictionVictim*` loop body; the reduction test is therefore load-bearing (it is the
  only guarantee the two-level path collapses to the one-level path).
- **Floor over-subscription.** If `Σ FloorBytes > budget`, floors are unsatisfiable; the
  Slice-1 fallback (step 4) prevents deadlock but means floors are *soft* under pressure —
  document this as designed behavior, not a bug. A weighted (fraction-of-budget) floor form
  is self-normalizing and avoids over-subscription; prefer it for the R2 wiring.
- **Tenant identity is not yet plumbed.** `node.kvSpanStats()`
  (`internal/radixkv/radixkv.go:435`) has no tenant to report today; Slice 3 depends on the
  gateway/session layer attaching a stable tenant id to radix leaves. If that identity does
  not exist, Slice 3 is blocked and must be reported as `not yet` with the missing wiring
  named — Slices 1-2 (pure primitive + replay witness) still ship and stand alone.
- **Aggregate hit-rate regression.** Protecting floors necessarily costs some global
  hit-rate; the Slice-2 replay must *quantify* this, not hide it. If the aggregate cost is
  large on realistic traces the mechanism should stay flag-gated (off by default) until R3
  benchmarks justify it.
