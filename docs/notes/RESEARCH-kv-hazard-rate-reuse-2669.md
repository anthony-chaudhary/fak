---
title: "design dossier #2669: age-conditioned hazard-rate reuse estimate for KV eviction (LHD/LRB-lite) — replace the memoryless `Hits + 1` reuse term in internal/compute's cost-of-losing function with `KVReuseEstimate(s, clock)`, a closed-form `(Hits+1) × decay(age, meanInterArrival)` that reduces EXACTLY to `Hits+1` when no age signal exists. Design only — no code shipped by this note. Seam verified: `reuseProbability := float64(s.Hits + 1)` at internal/compute/kvcost.go:145 (copied at kvcost_fanout.go:47 and kvcost_pin.go:88); telemetry source verified: radixkv Lookup freshens lastUsed BEFORE incrementing hits (radixkv.go:251-259), so gap capture must read the pre-freshen lastUsed. Smallest slice: a new kvcost_hazard.go variant following the kvcost_aging.go extension discipline (zero-value telemetry ⇒ byte-identical to today), witnessed by an equal-Hits fresh-vs-stale tiebreak test the raw count provably cannot make. (2026-07-06)"
description: "Staged-implementation dossier for issue #2669 (milestone 'The KV cache value is owned, observed & 2x', epic #2236 row M5, extends #2239/#2666). Everything in section (b) is WITNESSED against the tree at the cited lines; everything in sections (c)-(e) is DESIGN, not shipped. Arbitration: dos_arbitrate ADMIT, shared lease, lane 'kvbm' — with the honest caveat that internal/kvbm/ does not exist as a directory; the real edit trees are internal/compute/** (R1), internal/radixkv/** + internal/modelengine/** (R2), so an implementer must arbitrate THOSE lanes before editing."
---

# Design dossier — age-conditioned hazard-rate reuse estimate for KV eviction (#2669)

> **Status: design, not shipped.** No code changes accompany this note. Section (b)
> is witnessed against the tree as of 2026-07-06 (trunk `main`, HEAD ~37b86919);
> sections (c)–(e) are proposals. Issue: [#2669], extends [#2239]/[#2666] under epic
> [#2236] row **M5 eviction/lifecycle**, milestone "The KV cache value is owned,
> observed & 2x".

## (a) Problem restated from the issue

`KVEvictionCost` scores a resident KV span's value-of-keeping as
`recomputeCost × reuseProbability ÷ bytes`, and its reuse term is **raw
Laplace-smoothed frequency**: `reuseProbability = Hits + 1`
(`internal/compute/kvcost.go:145`, documented at `kvcost.go:123-127`). That is a
*count*, not a *probability*: it is **memoryless** in the age of the span. A span
hit 5 times an hour ago scores identically to one hit 5 times in the last second,
because nothing in the term conditions on how long the span has gone **un-hit
relative to its own inter-arrival pattern**. Real KV reuse is a hazard process — a
system-prompt prefix re-hit every turn has a short, tight inter-arrival; a
tool-result span hit once and abandoned has an expiring hazard — and a frequency
count cannot separate them once counts are equal.

The GDSF aging term already shipped for the *sibling* failure mode
(`kvcost_aging.go` / #2668: a stale-but-historically-hot span holds memory forever),
but that term is an **additive pool-clock inflation** driven by *evictions*, not a
reuse probability conditioned on the span's *own observed inter-arrival*. #2669 asks
for the LHD-style estimate: `reuseProbability` becomes a function of
`(Hits, age = clock − LastUsed, observed inter-arrival)`, with the hard constraint
that **no age signal ⇒ exact reduction to `Hits + 1`** (current behavior, existing
tests unbroken).

Parity targets ranked by the issue: LHD (Beckmann et al., NSDI'18) as the
probabilistic model, LRB (Song et al., NSDI'20) as the learned upper bound the
heuristic should approach (via the good-decision-ratio metric fak's replay harness
already computes), LRFU/hyperbolic as the decay-fusion family, and arXiv 2506.02634
as the empirical reuse-distance distribution to eventually fit against.

Explicitly **not** in scope (per the issue): token-level attention-mass eviction
(H2O/SnapKV/Quest/Ada-KV) — intra-span compression is a different axis (#1474/M7;
see also the EpiKV triage note). This ticket is **inter-span** prefix lifecycle.

## (b) Current seam — what the code does today (witnessed)

Every claim below was read from the tree at the cited line on 2026-07-06.

**The reuse term exists in THREE copies**, all identical `Hits + 1`:

- `internal/compute/kvcost.go:145` — `reuseProbability := float64(s.Hits + 1)` in
  `KVEvictionCost` (the base cost, `kvcost.go:141-148`).
- `internal/compute/kvcost_fanout.go:47` — same term in `KVEvictionCostFanout`
  (#2670's sharer-weighted variant).
- `internal/compute/kvcost_pin.go:88` — `float64(s.Hits+1) * hintReuseMultiplier(s.Hint)`
  in `KVEvictionCostPinned` (#2673's pin/hint variant).

**The input struct and its extension discipline.** `KVSpanStats`
(`kvcost.go:30-89`) is "pure data the pool already carries": `Tokens` (:34),
`Bytes` (:39), `Hits` (:43), `LastUsed` (:46, "the logical clock of the most recent
access — the LRU key"), plus four *later-added* fields that each document the same
contract this design must honor: **zero value ⇒ byte-identical to today** —
`AgeStamp` (:62), `PinBoost` (:72), `Hint` (:80), `Sharers` (:88). The precedent for
"a new cost dimension" is a **separate top-level variant function**, not an edit to
`KVEvictionCost`: `KVEvictionCostAged` (`kvcost_aging.go:35-37`) and
`PickEvictionVictimAged` (`kvcost_aging.go:52-77`) are exactly this shape.

**The victim picker.** `PickEvictionVictim` (`kvcost.go:162-183`): lowest cost wins
victimhood, `Pinned`/`Leased` are hard exclusions (:167), ties break to oldest
`LastUsed` (:176-179) so uniform cost reduces to pure LRU. This LRU tie-break is
precisely where today's function **mis-picks** the issue's headline case: two spans
with equal `Hits`/`Bytes`/`Tokens` tie on cost, and the decision falls to recency
alone with no notion of either span's own reuse rhythm.

**The replay witness harness.** `ReplayKVCache` (`kvcost.go:225-292`) simulates a
token-budgeted cache under `KVEvictPolicy` (`kvcost.go:186-198`: `KVEvictLRU`,
`KVEvictCostAware`). Its per-span resident state already tracks `tokens`, `hits`,
`lastUsed` (`kvcost.go:227-231`), and the hit path refreshes them at
`kvcost.go:270-276`. A near-duplicate of the same loop lives in
`ReplayKVCacheResult` (`kvreplay_oracle.go:26-89`) — **both** `victimUnderPolicy`
switches (`kvcost.go:238-262`, `kvreplay_oracle.go:37-61`) would need a hazard arm.
`ReplayKVCacheMulti` (`kvreplay_oracle.go:94-106`) already scores every policy
against `BeladyKVReplayOracle` (`kvreplay_oracle.go:113`, exact DP up to 63 distinct
spans per :142-144) and emits `GoodDecisionRatio` (`kvreplay_oracle.go:15`,
:233-241) — the LRB metric the issue's R3 names is **already plumbed**; a new
policy enum value inherits it for free. Trace corpora exist:
`GenerateKVReplaySyntheticTrace` (`kvreplay_trace.go:170-228`, a deterministic
Zipf/bimodal generator) and `GatewayUsageRowsToKVReplayTrace`
(`kvreplay_trace.go:108-165`, the #2244 durable-ledger reduction).

**The telemetry source for R2 wiring.** `internal/radixkv/radixkv.go:78-79` — each
radix node keeps `lastUsed uint64` and `hits int`. `Lookup`
(`radixkv.go:249-262`) is the hit-time seam: it freshens `lastUsed` along the
matched path at :251-253 and *then* increments `hits` at :255-259. **Ordering
subtlety (witnessed):** the freshen loop overwrites `p.lastUsed` *before* the hits
loop runs, so an inter-arrival gap (`newClock − oldLastUsed`) must be captured
**before** line 252 executes, or the two loops merged — the naive "record the gap
where `hits++` happens" reads an already-freshened `lastUsed` and observes gap 0
forever. `kvSpanStats()` (`radixkv.go:435-443`) is where node bookkeeping becomes
`compute.KVSpanStats` (it already populates `Hits`/`LastUsed`/`Leased`);
`costAwareLeaf` (`radixkv.go:401-433`) calls `compute.PickEvictionVictim` at :425
under the `EvictionCostAware` policy knob (`radixkv.go:85-95`, settable via
`NewWithEvictionPolicy`/`SetEvictionPolicy` at :133-143). The parallel engine-side
consumer is `internal/modelengine/nativesched_preempt.go:413`
(`compute.PickEvictionVictim` over scheduler-local spans), selected by
`FAK_NATIVE_KV_VICTIM_RULE` at `internal/modelengine/modelengine.go:258`.

**Name-collision check (witnessed):** no identifier `KVReuseEstimate`,
`IntervalSum`, `IntervalCount`, or `*Hazard*` exists anywhere under
`internal/compute` today (grep over `internal/` finds "hazard" only in
`internal/ailuminate` and unrelated prose). The names below are free.

**One naming caveat, stated plainly:** the ticket's "primary subsystem" label and
the lane taxonomy both say `internal/kvbm` (the lane resolver maps `internal/kvbm`
→ lane `kvbm`), but **no `internal/kvbm/` directory exists in the tree** — the KVBM
work to date lives in `internal/compute` (pure primitives), `internal/radixkv`, and
`internal/modelengine` (wiring), which matches the issue body's own "pure primitive
in `internal/compute` beside `kvcost.go`". Everything below targets those real
paths.

## (c) Proposed mechanism + concrete interfaces (design)

Follow the shipped extension discipline (`kvcost_aging.go`, `kvcost_fanout.go`,
`kvcost_pin.go`): a new file `internal/compute/kvcost_hazard.go`, pure and
stateless, plus two zero-default telemetry fields on `KVSpanStats`. Nothing about
`KVEvictionCost`'s signature or behavior changes.

**New `KVSpanStats` fields** (zero value ⇒ byte-identical to today, same contract
as `AgeStamp`/`PinBoost`/`Hint`/`Sharers`):

```go
// IntervalSum is the sum of observed inter-hit gaps for this span, in the resident
// pool's logical-clock units (radixkv: Lookups; the replay simulator: events). The
// pool records gap = clockAtHit − lastUsedBeforeFreshen on every subsequent hit.
IntervalSum uint64
// IntervalCount is the number of gaps folded into IntervalSum (the number of
// re-hits with a measurable gap). Zero means no age signal: KVReuseEstimate then
// reduces exactly to Hits+1.
IntervalCount int
```

**The estimate primitive** (the issue's named signature):

```go
// KVReuseEstimate is the age-conditioned reuse score replacing the memoryless
// Hits+1 count: (Hits+1) × decay(age, meanInterArrival), where
// age = clock − LastUsed and meanInterArrival = IntervalSum ÷ IntervalCount.
func KVReuseEstimate(s KVSpanStats, clock uint64) float64
```

Closed-form decay (the LHD-lite tractable core; a fitted-distribution variant can
later hide behind the same signature, per the issue):

```
age    := clock − s.LastUsed            // 0 if clock <= s.LastUsed (guard, no underflow)
if s.IntervalCount <= 0 { return float64(s.Hits + 1) }   // no age signal ⇒ EXACT reduction
meanIA := float64(s.IntervalSum) / float64(s.IntervalCount)
if meanIA < 1 { meanIA = 1 }            // gaps are >= 1 clock tick; guard div-by-zero
if float64(age) <= meanIA { return float64(s.Hits + 1) } // within its own window ⇒ no penalty
return float64(s.Hits+1) * math.Exp(-(float64(age)-meanIA)/meanIA)
```

Properties, each one a test in section (e): (1) **exact reduction** — a span with no
recorded intervals, or whose age is within its own mean inter-arrival, scores
exactly `float64(Hits+1)`, so on a uniform trace nothing changes; (2) **continuity**
at `age == meanIA` (`exp(0) = 1`); (3) **monotone decay** — each additional
inter-arrival window of silence multiplies the estimate by `e⁻¹`, so a
high-`Hits` span abandoned for many of its own windows stops out-ranking a fresh
span; (4) strictly positive — never divides experience out of the ranking, the same
guarantee the Laplace `+1` gives today.

**The cost variant and picker** (mirroring `kvcost_aging.go`'s pair):

```go
// KVEvictionCostHazard is KVEvictionCost with the reuse term replaced by the
// age-conditioned estimate: cost = Tokens × KVReuseEstimate(s, clock) ÷ Bytes.
// Inherits the Bytes <= 0 ⇒ +Inf fail-open unchanged.
func KVEvictionCostHazard(s KVSpanStats, clock uint64) float64

// PickEvictionVictimHazard is PickEvictionVictim scored by KVEvictionCostHazard:
// same Pinned/Leased hard exclusions, same oldest-LastUsed tie-break, same -1.
func PickEvictionVictimHazard(spans []KVSpanStats, clock uint64) int
```

Design decision, flagged for review: the `clock` argument makes these the first
cost functions in the file family that are not unary over `KVSpanStats`. The
alternative — a caller-stamped `Age uint64` field keeping the unary shape — was
considered and set aside because the issue names the two-arg signature explicitly
and because deriving `age` *inside* the primitive keeps the underflow guard and the
reduction rule in one tested place rather than in every caller. Either shape is
compatible with the eventual "pluggable reuse term" seam; see Risks for why that
refactor is deliberately deferred.

**Replay policy arm:** a new `KVEvictHazard` value on `KVEvictPolicy`
(`kvcost.go:186-198`); the resident structs in both replay loops gain
`intervalSum`/`intervalCount`, recorded on the hit path **before** `r.lastUsed`
is refreshed (gap = `clock − r.lastUsed` at `kvcost.go:270-276` and
`kvreplay_oracle.go:69-73`); both `victimUnderPolicy` switches gain the hazard
case building `KVSpanStats` with the two new fields and calling
`KVEvictionCostHazard(stats, clock)`.

**R2 wiring (radixkv/modelengine, later slices):** radix node gains
`intervalSum`/`intervalCount` beside `lastUsed`/`hits` (`radixkv.go:78-79`);
`Lookup` captures the gap **before** the freshen loop at `radixkv.go:251-253` (the
ordering subtlety witnessed in section (b)); `kvSpanStats()` (`radixkv.go:435-443`)
populates the new fields; a new `EvictionHazard` policy value beside
`EvictionCostAware` (`radixkv.go:85-95`) drives a `hazardLeaf` twin of
`costAwareLeaf` (`radixkv.go:401-433`) passing `t.clock`; engine side, a `hazard`
arm for `FAK_NATIVE_KV_VICTIM_RULE` (`modelengine.go:258`) routing
`nativesched_preempt.go:413` through the hazard picker.

## (d) Staged plan

**Slice 1 — the smallest shippable increment (R1, pure compute, no wiring):**
`internal/compute/kvcost_hazard.go` + `kvcost_hazard_test.go` only — the two
`KVSpanStats` fields, `KVReuseEstimate`, `KVEvictionCostHazard`,
`PickEvictionVictimHazard`, and the unit witnesses including the exact-reduction
proof. Touches exactly one existing declaration (the `KVSpanStats` struct literal
in `kvcost.go`); no existing function body changes, no existing test changes. This
is deliberately the same footprint #2668's aging slice shipped with.

**Slice 2 — replay witness arm (R1→R3 seed, still compute-only):** `KVEvictHazard`
policy enum + hazard arms in both replay loops + interval telemetry in both
resident structs; bimodal-trace witness (hazard ≥ cost-aware hit rate at fixed
budget, strict on the bimodal corpus) and uniform-trace no-regression witness;
`GoodDecisionRatio` for the hazard arm reported via the existing
`ReplayKVCacheMulti`/Belady plumbing — host-free, CI-stable (keep witness traces
under the 63-distinct-span exact-oracle bound, `kvreplay_oracle.go:142-144`).

**Slice 3 — R2 wiring (cross-lane, radixkv + modelengine):** gap capture in
`Lookup` (pre-freshen ordering), node fields, `kvSpanStats()` population,
`EvictionHazard` policy knob, `FAK_NATIVE_KV_VICTIM_RULE=hazard`; no-regression on
the existing `eviction_policy_test.go` and `nativesched_preempt_test.go` suites.

**Slice 4 — R3 on the #2244 corpus (follow-on):** run the hazard arm through
`ReplayKVTrace` over the committed gateway-ledger corpus; report good-decision-ratio
vs Belady alongside LRU and de-aged cost-aware. Fitted-distribution variant behind
`KVReuseEstimate`'s signature stays a filed follow-on, not this ticket's scope.

## (e) Test plan — the witnesses, named

Slice 1, package `internal/compute` (`kvcost_hazard_test.go`), modeled on the
naming in `kvcost_aging_test.go:13-100`:

- `TestKVReuseEstimateReducesToHitsPlusOneWithoutAgeSignal` — `IntervalCount == 0`
  ⇒ exactly `float64(Hits+1)` (exact float equality: it is the same expression, not
  an approximation). The issue's "reduction to Hits+1 proven" done-condition.
- `TestKVReuseEstimateDecaysPastInterArrivalWindow` — fixed `Hits`, increasing
  `age` beyond `meanIA`: strictly decreasing estimate; within the window: exactly
  `Hits+1`.
- `TestKVEvictionCostHazardBreaksEqualHitsTie` — the issue's headline witness: two
  spans, equal `Hits`/`Bytes`/`Tokens`, one freshly re-hit, one aged well past its
  own inter-arrival; assert `KVEvictionCost` ties them (equal costs — the
  memoryless failure, witnessed as such) while `PickEvictionVictimHazard` evicts
  the stale one and keeps the fresh one.
- `TestKVEvictionCostHazardReducesToKVEvictionCost` — table of zero-telemetry
  spans: `KVEvictionCostHazard(s, clock) == KVEvictionCost(s)` exactly.
- `TestPickEvictionVictimHazardRespectsPinsAndLeases` — exclusion parity with
  `TestPickEvictionVictimAgedRespectsPinsAndLeases` (`kvcost_aging_test.go:80`).
- `TestKVEvictionCostHazardPreservesUnknownBytesFailOpen` — parity with
  `TestKVEvictionCostAgedPreservesUnknownBytesFailOpen` (`kvcost_aging_test.go:100`).

Slice 2, package `internal/compute`:

- `TestReplayHazardBeatsCostAwareOnBimodalTrace` — hot recurring span vs one-shot
  cold spans at fixed budget (shape of `TestReplayCostAwareBeatsLRUOnHotSpanTrace`,
  `kvcost_test.go:128`): hazard hit-tokens strictly greater.
- `TestReplayHazardEqualsCostAwareOnUniformTrace` — the no-regression reduction at
  the policy level (R2's done-condition, provable host-free).
- `TestReplayKVCacheMultiReportsHazardGoodDecisionRatio` — hazard arm's
  `GoodDecisionRatio` vs the exact Belady oracle ≥ cost-aware's on the bimodal
  corpus (the LRB metric, via existing `kvreplay_oracle.go:94-106` plumbing).

Slice 3, packages `internal/radixkv` / `internal/modelengine`:

- `radixkv.TestLookupRecordsInterArrivalTelemetry` — the gap-ordering witness: two
  lookups N clock-ticks apart yield `IntervalSum == N`, `IntervalCount == 1`
  (fails if the gap is read after the freshen loop at `radixkv.go:251-253`).
- `radixkv.TestEvictToBudgetHazardKeepsFreshEqualHitSpan` — end-to-end policy-knob
  witness in the home of `eviction_policy_test.go`.
- `modelengine`: extend the `FAK_NATIVE_KV_VICTIM_RULE` matrix in
  `nativesched_preempt_test.go` (existing env-arm precedent at :376) with `hazard`.

## (f) Risks & collisions

- **Three live copies of the reuse term.** `Hits+1` is textually duplicated at
  `kvcost.go:145`, `kvcost_fanout.go:47`, `kvcost_pin.go:88`. Hazard adds a fourth
  cost variant to a family that is already combinatorial (aged × fanout × pinned ×
  hazard do not compose today; each is its own function). The issue's "internal
  seam so the reuse term is pluggable" is the right eventual refactor, but doing it
  in slice 1 means editing all three shipped cost functions at once, in a tree
  where #2668/#2670/#2673 landed within days of each other and sibling M2 tickets
  are in flight. Recommendation: variant-first (the shipped precedent), unify the
  reuse-term seam as its own follow-on once hazard has its witnesses.
- **`KVSpanStats` is a contended struct.** Four fields were appended to it by four
  recent tickets; concurrent M2 workers may be appending more. A struct-field
  addition is a textual merge hazard even when semantically disjoint. Mitigation:
  slice 1 touches the struct once, appends at the end, and changes no existing
  field or function.
- **Arbitration scope vs real edit tree.** `dos_arbitrate` ADMITted a shared lease
  on lane `kvbm`, tree `internal/kvbm/**` (verdict verbatim: "cluster lane 'kvbm'
  free — admitted") — but `internal/kvbm/` does not exist; the actual edit trees
  are `internal/compute/**` (lane `compute`) and, for slice 3, `internal/radixkv/**`
  + `internal/modelengine/**`. A design read collides with nothing, but an
  implementer must arbitrate the **compute** lane (and later radixkv/modelengine)
  before editing, not kvbm. A first bare-invocation arbitrate call in this session
  also auto-picked an unrelated lane (`adjudicator`) — name the lane explicitly.
- **Gap-capture ordering in `Lookup`.** Witnessed above: `radixkv.go:251-253`
  freshens `lastUsed` before `hits++` at :255-259. Recording the gap in the wrong
  loop silently observes gap ≈ 0 for every span, which makes `meanIA ≈ 1` and turns
  the hazard into an aggressive recency filter — a wrong-but-plausible failure that
  only the slice-3 ordering test catches.
- **Clock-unit semantics.** `age` and `meanIA` are in logical accesses (radixkv
  ticks per Lookup, `radixkv.go:254`; replay ticks per event), not wall time. Under
  multi-tenant load one tenant's burst inflates every other span's age uniformly —
  acceptable (it is the same clock LRU and GDSF aging already ride) but worth
  naming: the hazard conditions on *relative* rhythm in shared-clock units, and the
  issue's "hit 5 times an hour ago" motivation maps onto access-count age, not
  seconds.
- **Belady oracle exactness bound.** The exact DP caps at 63 distinct spans
  (`kvreplay_oracle.go:142-144`); larger witness traces silently degrade to the
  greedy approximation (`Exact=false`), which would make a good-decision-ratio
  assertion flaky in spirit. Keep CI witness corpora under the bound.
- **Float determinism.** `math.Exp` is deterministic per platform in Go; tests
  assert orderings and the exact-reduction branch (which returns `float64(Hits+1)`
  without touching `Exp`), never exact equality of decayed values.
- **What this design does NOT claim.** No serving-quality or end-to-end GLM-5.2
  performance claim is made or makeable from this dev box; the replay witnesses are
  host-free simulations over synthetic and ledger-derived corpora. The "strictly
  improves victim ranking" claim is witnessed only in the forms named in (e):
  tie-break on the bimodal case, no-regression on uniform, and good-decision-ratio
  vs a finite-trace oracle.

[#2669]: https://github.com/anthony-chaudhary/fak/issues/2669
[#2666]: https://github.com/anthony-chaudhary/fak/issues/2666
[#2239]: https://github.com/anthony-chaudhary/fak/issues/2239
[#2236]: https://github.com/anthony-chaudhary/fak/issues/2236
