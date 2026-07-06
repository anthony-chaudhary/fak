---
title: "Design dossier: generalize evict to a min-cost tier ACTION (keep/quantize/spill/evict) — demote-not-drop across the ladder (#2671). Verdict: the seam is REAL and already shaped for this — compute.PickEvictionVictim (kvcost.go:162) is the binary drop-only decision #2671 generalizes, the sibling planners kvprecision.go (quantize)/kvresidency.go (spill-host) already price the other fates, and the kvcost_* extension-file pattern (pin/fanout/aging, each a strict generalization with a reduction-to-base witness) is the exact template PlanKVDemotion should follow. ONE load-bearing discovery the issue does not mention: internal/cachemeta/placement.go PlanPlacement (:168) ALREADY makes a per-entry keep/promote/demote/spill/compress_demote/evict decision with tier profiles and retain-vs-recompute economics — #2671's primitive is the BATCH, deficit-clearing sibling (which SET of (span, action) pairs frees deficitBytes at least aggregate restore cost), not a duplicate, and the dossier draws that seam split explicitly to prevent two diverging cost models. ONE honest deviation from the issue text: the proposed PlanKVDemotion signature needs a recompute-unit bridge (recomputeCostPerToken) the issue omits, because evict is priced in tokens and spill in bytes — precedent cachemeta.PlacementRequest.PerTokenPrefillNanos (placement.go:82). Design only — no code shipped; dos_arbitrate shared/compute = ADMIT. (2026-07-06)"
description: "Grounded design + staged-implementation dossier for issue #2671 (epic #2236 rows M5 lifecycle + M3 hierarchical tiering; concretizes #2666 bullet 4; extends #2239). Every code claim cites the current tree at file:line. What is WITNESSED today vs what is PROPOSED is marked throughout."
---

# Design dossier — min-cost KV tier ACTION: demote-not-drop (#2671)

> Issue: [#2671](https://github.com/anthony-chaudhary/fak/issues/2671) — part of epic
> [#2236](https://github.com/anthony-chaudhary/fak/issues/2236), concretizes
> [#2666](https://github.com/anthony-chaudhary/fak/issues/2666)'s offload-before-preempt
> bullet, extends [#2239](https://github.com/anthony-chaudhary/fak/issues/2239).
> **Design only.** Nothing in this note ships code; sections marked *witnessed* describe
> the tree as read on 2026-07-06, sections marked *design* are proposal.
>
> fak-guard: `dos_arbitrate` (mode=shared, lane=`compute`, tree=`["internal/compute/**"]`)
> → **ADMIT** ("cluster lane 'compute' free — admitted", decision only, no lease
> journaled). Two earlier bare-tree calls (tree-only, no lane) were treated as bare
> auto-picks and granted the unrelated `adjudicator` lane — recorded here for honesty;
> the compute-lane verdict above is the one that covers this design's real tree. The
> dispatcher's ticket named `internal/kvbm` as the primary subsystem; **`internal/kvbm/`
> does not exist in the tree** (witnessed: no such directory). "kvbm" is the milestone's
> Dynamo-KVBM parity metaphor; the issue body itself lands the primitive in
> `internal/compute` beside `kvcost.go`, `kvresidency.go`, `kvprecision.go`, and that is
> where this dossier grounds.

## (a) Problem, restated from the issue

Under budget pressure fak's cost-aware evictor answers one question: *which span is
cheapest to lose?* — and then **drops** it. But dropping is only the most expensive rung
of a ladder fak already prices as pure planners:

- **quantize-in-place** — ~2x denser, evict-correct (the #1047/#1474 q8 tier);
- **spill-to-host** — cold span to the roomy host pool (#1048);
- **spill-to-disk / L4 object** — the #2169 tier ladder.

The crux the binary evictor gets wrong: **a demoted span's cost-to-lose is not full
re-prefill — it is restore latency from the tier it sits on.** Dropping a span that could
have spilled to host for a ranged-read restore throws away recompute the ladder would
have avoided. #2671 asks for a *pure min-cost placement decision*: per resident span,
per available tier, weigh `(bytesFreed, restoreCost)` and choose the action set that
clears a byte deficit at least aggregate cost. Drop is just the tier whose restore cost
is full re-prefill; with no lower tier configured the primitive must reduce exactly to
today's `PickEvictionVictim`. Parity frame (from the issue): Dynamo KVBM's cost-aware
lifecycle across HBM→host→SSD→network, SGLang HiCache placement, LMCache's
load-vs-recompute economics, vLLM's CPU-offload connector.

## (b) Current seam — what the tree does today (witnessed)

### The decision being generalized

- `internal/compute/kvcost.go:30` — `KVSpanStats`, the per-span telemetry (Tokens,
  Bytes, Hits, LastUsed, Pinned, Leased, plus the opt-in zero-default fields AgeStamp
  :62, PinBoost :72, Hint :80, Sharers :88 added by later extensions).
- `internal/compute/kvcost.go:141` — `KVEvictionCost(s KVSpanStats) float64`:
  `recomputeCost × reuseProbability ÷ bytes` (Tokens × (Hits+1) / Bytes), +Inf fail-open
  on non-positive Bytes (:142).
- `internal/compute/kvcost.go:162` — `PickEvictionVictim(spans []KVSpanStats) int`: the
  binary drop-only decision #2671 generalizes. Skips Pinned/Leased (:167), lowest-cost
  victim, LRU tie-break (:176), -1 when everything is locked. (The issue cites this at
  `kvcost.go:104`; the line has since moved — today it is :162. Same function.)

### The other fates, already priced as pure planners

- **Quantize**: `internal/compute/kvprecision.go:32` `KVPrecision`; :41 `KVPrecisionQ8`
  (attended rows q8_0, pre-RoPE K kept f32 so `Evict` stays bit-exact — ~1.96x denser,
  honestly below the naive 2x); :87 `perTokenPerLayerBytes` (unexported, same package —
  reusable directly as the issue asks); :121 `AutoSelectKVPrecision` (fail-open keeps
  f32 on incomplete geometry, :125-126).
- **Spill-host**: `internal/compute/kvresidency.go:58` `PlanKVResidency` — hot/cold
  split across device/host budgets; fail-open empty split on non-positive per-token
  cost or want (:59-64).
- **Byte arithmetic**: `internal/compute/capacity.go:106` `EstimateKVStoreBytes`;
  `internal/compute/compute.go:260` `KVConfig`.

### The extension-file pattern this design should follow

Each prior generalization of the cost function is its own file, a strict generalization
with a proven reduction to the base on zero-value inputs:

- `internal/compute/kvcost_pin.go:78` `KVEvictionCostPinned` (TTL pin economics + hints,
  #2673) — reduces to `KVEvictionCost` when PinBoost==0 and Hint==HintNone.
- `internal/compute/kvcost_fanout.go:38` `KVEvictionCostFanout` / :60
  `PickEvictionVictimFanout` (#2670) — reduces at Sharers<=1.
- `internal/compute/kvcost_aging.go:35` `KVEvictionCostAged` — reduces at AgeStamp==0.

### The consumers a demotion plan would eventually feed (R2 targets, witnessed)

- `internal/radixkv/radixkv.go:348` `(*Tree).evictToBudget` — unexported (the
  `kvcost.go:10-12` comment calls it "radixkv.EvictToBudget"; the actual seam is the
  method plus its victim choice): :362 `victimLeaf` dispatches to :401 `costAwareLeaf`,
  which builds `[]compute.KVSpanStats` via `(*node).kvSpanStats()` (:435) and calls
  `compute.PickEvictionVictim` at :425. A refs>0 leaf is never a victim (:77).
- `internal/modelengine/nativesched_preempt.go:413` — the native preemptor's cost-aware
  victim rule (`NativePreemptVictimCostAware`, :53) also calls
  `compute.PickEvictionVictim`. Both call sites today can only *drop*.

### Load-bearing prior art the issue does not mention (witnessed)

`internal/cachemeta` **already owns a per-entry tier-action decision**:

- `internal/cachemeta/placement.go:39-55` — `PlacementAction` closed vocabulary:
  `ActionKeep`/`ActionPromote`/`ActionDemote`/`ActionSpill`/`ActionCompressDemote`
  (#523)/`ActionEvict`.
- `internal/cachemeta/placement.go:168` — `PlanPlacement(req PlacementRequest)
  PlacementDecision`: promote-if-hot, else demote/spill to the coldest colder tier with
  room when retaining beats recompute (`RetainCheaperThanRecompute` :128, priced in
  nanoseconds via `stageNanos` :106 and `recomputeNanos` :116 against
  `PerTokenPrefillNanos` :82), else compress-and-demote gated on PROVEN quality, else
  evict.
- `internal/cachemeta/hardware.go:105` — `TierProfile` (latency, bandwidth, capacity,
  attendable-in-place); `internal/cachemeta/cachemeta.go:216-226` — `ResidencyTier`
  vocabulary (hbm/dram/disk/remote/provider/recompute).
- `internal/cachemeta/quantized_demote.go:37` — `QuantizedDemoteTarget` (#1474) ties the
  compute q8 tier into the compress-and-demote lever; admission requires
  `Quality.Acceptable()` (:48-50, :87 `ApplyTo`).
- Consumer: `internal/storedrv/placement.go:44` `Router.PutPlaced` acts on a
  `cachemeta.PlacementDecision`.

**The distinction that keeps #2671 from being a duplicate:** `PlanPlacement` answers a
*per-entry, lifecycle-driven* question ("this one entry is expiring/pressured — where
does it go?"), one decision at a time, in wall-clock units, over `ResidencyTier`s.
#2671 asks the *batch, deficit-driven* question the evictor faces ("the pool is
`deficitBytes` over budget — which SET of (span, action) pairs clears it at least
aggregate restore cost?"), over `KVSpanStats`, reducing to `PickEvictionVictim`. Neither
subsumes the other today; the risk of two diverging cost models is real and addressed
in §(f).

## (c) Proposed mechanism (design — none of this exists yet)

New file `internal/compute/kvdemote.go` (+ `kvdemote_test.go`), same pure-planner
discipline as kvcost/kvresidency/kvprecision: no allocation, no byte movement, no model
state. Names follow the issue text; the one naming deviation worth debating is flagged
in §(f).

```go
// KVTierAction is the closed vocabulary of fates a resident span can meet under
// budget pressure, ordered roughly hottest-kept to coldest-lost.
type KVTierAction uint8

const (
    ActionKeep        KVTierAction = iota // leave resident, untouched
    ActionQuantize                        // requantize in place to the denser tier (stays attendable)
    ActionSpillHost                       // relocate to host DRAM (restore = transfer back)
    ActionSpillDisk                       // relocate to local disk (restore = ranged read)
    ActionSpillObject                     // relocate to L4 object store (#2169; restore = ranged GET)
    ActionEvict                           // drop; restore = full re-prefill (top of the ladder)
)

func (a KVTierAction) String() string // "keep" | "quantize" | "spill-host" | ...

// KVTierProfile prices ONE rung of the ladder for the demotion planner.
type KVTierProfile struct {
    Action             KVTierAction
    Available          bool    // rung is configured AND (for quantize) quality-proven by the caller
    RestoreCostPerByte float64 // cost units per byte to bring a span back (0/negative = unpriced)
    Density            float64 // quantize only: resident-bytes ratio after the action
                               // (f32→q8 ≈ 0.51, from EstimateKVStoreBytes at both tiers)
}

// KVDemotionDecision is one (span, action) verdict in the plan.
type KVDemotionDecision struct {
    SpanIndex   int          // index into the spans slice handed in
    Action      KVTierAction // never ActionKeep (keep is the implicit default)
    BytesFreed  int64        // freed from the pressured pool by this action
    RestoreCost float64      // expected cost booked if/when the span is needed again —
                             // the number the cache-value ledger (#1072) records as OBSERVED provenance
}

// PlanKVDemotion picks the min-cost action set that frees deficitBytes.
//
// DEVIATION FROM THE ISSUE TEXT (deliberate, argued in the dossier §f):
// recomputeCostPerToken is the unit bridge the issue's signature omits — evict is
// priced in tokens of re-prefill, spill in bytes of transfer, and they cannot be
// compared without a common unit. Precedent: cachemeta.PlacementRequest.
// PerTokenPrefillNanos (placement.go:82). Non-positive => only same-unit rungs
// (evict alone) are comparable => reduces to drop-only.
func PlanKVDemotion(spans []KVSpanStats, deficitBytes int64, tiers []KVTierProfile,
    recomputeCostPerToken float64) []KVDemotionDecision
```

### Economics (design)

For span `s` and rung `t`, both sides scaled by the span's reuse probability
(`Hits+1`, the Laplace-smoothed signal `KVEvictionCost` already uses, kvcost.go:145):

- `expectedLoss(s, t) = (Hits+1) × restoreCost(s, t)`
  - evict: `Tokens × recomputeCostPerToken` (full re-prefill — the top rung)
  - spill-host/disk/object: `Bytes(after action) × RestoreCostPerByte`
  - quantize: 0 transfer (stays attendable in place); the *quality* cost is not
    priced here — it is gated, not priced (see fail-open below)
- `bytesFreed(s, t)`:
  - evict / spill: `s.Bytes` (the whole span leaves the pressured pool)
  - quantize: `s.Bytes − quantizedBytes` (density delta; ~49% for f32→q8)
- rank candidates by `expectedLoss / bytesFreed` ascending (cheapest expected loss per
  byte freed first); take greedily until the running `bytesFreed` sum ≥ `deficitBytes`;
  tie-break oldest `LastUsed` first (the kvcost.go:176 rule).

**Reduction (the invariant tests must prove):** with only the evict rung available,
`expectedLoss/bytesFreed = Tokens×(Hits+1)×recomputeCostPerToken / Bytes`, which is
`KVEvictionCost × recomputeCostPerToken` — a positive constant times the existing score
— so the greedy's first pick is exactly `PickEvictionVictim`'s victim, ties included.

**Honesty about optimality:** clearing a byte deficit at minimum aggregate cost is a
covering/knapsack-shaped problem; greedy by cost-per-byte is the standard approximation,
not a proven optimum. The witness tests fix *behavior* (demote-not-drop flips, the
reduction, fail-open); they do not claim global optimality, and neither should the doc
comment.

### Correctness floors and fail-open (design, mirroring witnessed contracts)

- **Pinned/Leased spans get no action at all** — not evict, not spill, not quantize. A
  span being served (refs>0, radixkv.go:77) cannot be moved or requantized mid-decode.
  Same exclusion as kvcost.go:167, extended to every rung.
- **Unpriced/unavailable rung never fires**: `Available == false` or
  `RestoreCostPerByte <= 0` (for the spill rungs) or `Density` outside (0,1) (for
  quantize) drops the rung from consideration — fak never demotes into a tier it cannot
  cost. This is the kvresidency.go:59-64 / kvprecision.go:125-126 fail-open contract.
- **Quantize is gated, not priced**: compute has no `QualityEvidence` (that gate lives
  in `cachemeta`, quantized_demote.go:48-50). The caller arms `Available=true` on the
  quantize rung only with proven quality; unproven ⇒ the rung is absent ⇒ the span
  evicts, exactly #1474's refuse behavior.
- **No double-quantize**: a span already at the dense tier must not free phantom bytes.
  Proposed: add `Precision KVPrecision` to `KVSpanStats` — zero value is
  `KVPrecisionF32` (kvprecision.go:30-31 documents zero=f32), so every existing caller
  is byte-identical, the same opt-in pattern AgeStamp/PinBoost/Hint/Sharers used
  (kvcost.go:54-88). A span with `Precision == KVPrecisionQ8` is not a quantize
  candidate.
- **Non-positive deficit ⇒ empty plan** (everything keeps): fail-open, nothing demotes
  when there is no pressure.

## (d) Staged plan

**Slice 1 — the smallest first shippable slice**: `internal/compute/kvdemote.go` with
the vocabulary (`KVTierAction` + `String()`), `KVTierProfile`, `KVDemotionDecision`, and
`PlanKVDemotion` over a **two-rung ladder: {spill-host, evict}** — no quantize, no
disk/object, no `KVSpanStats` change. This is already enough to witness the issue's
headline claim (a span that would drop instead spills when restore undercuts re-prefill
and the freed bytes clear the deficit), the reduction to `PickEvictionVictim`, and the
unpriced-tier refute guard. Pure compute-lane, zero consumers touched.

**Slice 2 — quantize rung**: `ActionQuantize` + the `KVSpanStats.Precision` field
(zero-default) + density derived from `EstimateKVStoreBytes` at f32 vs q8 (the honest
~1.96x, quantized_demote.go:64 precedent). Witness mirrors #1474's compress_demote:
quality-armed rung flips evict→quantize; unarmed rung still evicts; q8 span never
re-quantizes. Still compute-lane only.

**Slice 3 — full ladder**: `ActionSpillDisk` / `ActionSpillObject` rungs (pure profiles;
object rung stays `Available=false` until #2169's wiring exists) and the multi-decision
deficit-clearing behavior hardened (mixed action sets across many spans).

**Slice 4 — R2 wiring (cross-lane, NOT this ticket's lane)**: consult `PlanKVDemotion`
from `radixkv.evictToBudget`'s victim path (radixkv.go:348/:401) and the native
preemptor (nativesched_preempt.go:413) behind a `FAK_NATIVE_KV_*` flag, with byte
movement performed by the existing engine/kvmmu adapters and restore-on-access (#1469)
closing the loop with a no-bit-drift check. Touches `internal/radixkv` and
`internal/modelengine` lanes — needs its own arbitration.

**Slice 5 — R3**: offload-tier arm in the #2244 benchmark (host vs disk vs object) vs
Dynamo/SGLang/LMCache behavior; report effective-recompute-saved.

## (e) Test plan — the witnesses, named

All slice 1–3 tests in package `compute` (files `internal/compute/kvdemote_test.go`),
matching the existing kvcost/kvresidency test-naming grain:

Slice 1:
- `TestPlanKVDemotionSingleTierReducesToPickEvictionVictim` — evict-only ladder picks
  the identical victim (ties included) as `PickEvictionVictim`; the issue's reduction
  requirement.
- `TestPlanKVDemotionSpillsInsteadOfDroppingWhenRestoreUndercutsRecompute` — the
  demote-not-drop flip: same span, drop-only economics ⇒ `ActionEvict`; host rung priced
  below re-prefill ⇒ `ActionSpillHost`, deficit still cleared.
- `TestPlanKVDemotionUnpricedTierStillEvicts` — the refute guard: rung present but
  `RestoreCostPerByte<=0` or `Available=false` ⇒ never demotes into it; span evicts.
- `TestPlanKVDemotionSkipsPinnedAndLeased` — no action of any kind on pinned/leased
  spans; all-locked ⇒ empty plan (the -1 analogue).
- `TestPlanKVDemotionFreesAtLeastDeficit` — the plan's summed `BytesFreed` clears
  `deficitBytes` whenever any admissible plan can.
- `TestPlanKVDemotionEmptyOnNonPositiveDeficit` — fail-open keep-everything.

Slice 2:
- `TestPlanKVDemotionQuantizeFreesDensityDelta` — freed bytes equal the f32→q8 delta
  from `EstimateKVStoreBytes`, not the naive 4x.
- `TestPlanKVDemotionNeverRequantizesDenseSpan` — `Precision==KVPrecisionQ8` span is not
  a quantize candidate.
- `TestKVSpanStatsZeroPrecisionIsF32` — the zero-value reduction (existing callers
  byte-identical).

Slice 4 (cross-lane, named now so R2 has its witnesses spec'd; do not write in this
ticket's lane): `radixkv.TestEvictToBudgetConsultsDemotionPlan`,
`modelengine.TestNativePreemptSpillsBeforeDropWhenTierPriced`, plus the #1469
restore-on-access no-bit-drift check in the engine's live test.

## (f) Risks & collisions

1. **Two cost models in one tree.** `cachemeta.PlanPlacement` (placement.go:168) already
   prices demote-vs-recompute per entry in nanoseconds; `PlanKVDemotion` prices a batch
   deficit in abstract cost units. If their prices disagree, the sweep and the evictor
   could fight (sweep demotes what the evictor would have kept). Mitigation: document
   the seam split (§b), and long-term have ONE side own the ladder prices — the natural
   direction is `cachemeta.TierProfile` (hardware.go:105) feeding `KVTierProfile`
   construction, so the numbers have one home. Out of scope here; named so the R2 wiring
   ticket carries it.
2. **Name shadowing.** The issue's `ActionKeep`/`ActionEvict` in `compute` shadow
   `cachemeta.ActionKeep`/`ActionEvict` (placement.go:42-54). Go package qualification
   keeps it compiling, but grep and readers will collide. Open naming decision: follow
   the issue verbatim (this dossier's default) or prefix `KVAction*` in compute. Cheap
   to decide at review; expensive to rename after consumers land.
3. **Signature deviation.** The issue's `PlanKVDemotion(spans, deficitBytes, tiers)`
   omits the unit bridge; this design adds `recomputeCostPerToken` (precedent:
   placement.go:82 `PerTokenPrefillNanos`). Without it the evict rung and the spill
   rungs are incommensurable and the "restore undercuts re-prefill" comparison cannot be
   computed. Flagging as a deliberate, argued deviation — reviewer should confirm.
4. **Greedy ≠ optimal.** "Least aggregate cost" is knapsack-shaped; greedy
   cost-per-byte is an approximation. The doc comment must say so; the witnesses pin
   behavior, not optimality.
5. **Quality gate stays out of compute.** Quantize admission on proven quality is
   cachemeta's contract (#1474, quantized_demote.go:48-50). `Available` is the caller's
   attestation; compute must never look like it verified quality. The refute-guard
   witness makes this observable.
6. **Stale line citations in prior text.** The issue cites `PickEvictionVictim` at
   kvcost.go:104 (now :162), and kvcost.go:10-12's comment names
   "radixkv.EvictToBudget" while the method is unexported `evictToBudget`
   (radixkv.go:348). Neither invalidates the seam; both are worth a comment-tidy in
   slice 1's commit.
7. **Tree contention.** Arbitration on lane `compute` returned ADMIT (free, disjoint
   from live leases) at design time; the working tree snapshot for this session shows
   heavy in-flight edits elsewhere (internal/agent, internal/gateway, cmd/fak) but no
   `internal/compute/kv*` modifications. Slice 4 touches `radixkv`/`modelengine` lanes
   and MUST re-arbitrate — do not assume this ADMIT covers it. Note the ticket
   dispatcher's `internal/kvbm` subsystem tag names a directory that does not exist;
   filing a dispatcher-metadata correction is a follow-on, not this ticket.
