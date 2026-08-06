---
title: "MoE activated-expert offload - the promotion plan"
description: "fak had built nine witnessed pieces of an activated-expert offload machine and wired none of them to the serve path. This is the ordered promotion ladder that makes the activated working set a first-class, bounded, observable object: ring wiring (R0, landed), graded spill knob, pin-set, prefetch, evictor policy, checkpoint tier, operator surface."
---

# MoE activated-expert offload - the promotion plan

## The thesis

A frontier Mixture-of-Experts checkpoint activates a *small fraction* of its experts per
token. GLM-5.2 routes top-8 of 256 routed experts per layer (~3%); DeepSeek-V4, Qwen3-MoE,
gpt-oss and MiniMax are all shaped the same way. The parameter bulk is the experts, but the
*per-token* demand is a thin, router-selected slice of it.

That gap between **stored bytes** and **activated bytes** is the whole opportunity, and it is
a residency problem, not a math problem: "is this token fast?" reduces to "were its activated
experts resident?" Making that first-class means fak must be able to name, bound, observe and
schedule **the activated working set** as an object.

Today it cannot. fak has built the parts and wired none of them.

## What the serve path actually offers today (two placements, both wrong for this)

**1. All-host, forever.** `--cpu-offload-experts` selects `splitKernel`
(`internal/model/moe_offload.go:132-142`), which routes every weight matching
`isExpertWeight` to the host kernel. It is a *static predicate over weight names*: all expert
GEMMs run on host RAM for the life of the session, and no activated expert ever becomes
device-resident. The file says so itself - "the floor, not a cache". The knob is a single
`bool` (`cmd/fak/serve.go:221`); there is no graded form.

**2. All-device, forever-growing.** On the resident-quant device path, routed experts go
through `Session.expertSwiGLUHAL` (`internal/model/hal.go:248-284`) into
`Session.weightHALStaged` (`internal/model/hal.go:170-181`), which memoizes every uploaded
weight handle in `s.halW` and **never evicts**: the map is only torn down at session close
(`internal/model/hal.go:83-103`). So device residency is the *union of every expert activated
since the session started*, monotonically converging on the full expert bulk. It is a
memoizer, not a cache - no budget, no victim, no accounting.

Neither placement has a bounded activated working set. There was no seam where a residency
policy, a prefetch, a pin-set or a byte budget could attach on the live path.

**3. Bounded, opt-in (R0, landed).** `Session.ExpertRingBytes > 0` now stages *routed* expert
weights through the bounded ring instead of `halW`
(`internal/model/expert_ring_hal.go`, #5611): budgeted, LRU-evicted, page-in/hit/evict
accounted, with dense/attention/router/shared-expert weights keeping permanent residency. It
is the seam the rest of the ladder attaches to.

**4. Graded and bounded, opt-in (R1, landed).** `Session.ExpertSpillLayers` grades placement 1
by layer instead of by name, and the same resolve sizes placement 3's ring from the leftover
device bytes (`internal/model/expert_spill_placement.go`,
`internal/agent/inkernel_expert_spill.go`, #5612). The three now compose rather than compete:
the first N MoE layers' experts on host, the rest device-resident but *bounded* by the ring,
the dense base permanent. `FAK_N_CPU_MOE=auto|<N>` reaches it on a serve; `--n-cpu-moe` is not
wired yet, so the flag surface is still R1's open remainder.

## What is already built (and where it stops)

Nine witnessed capabilities exist. Column 3 is the load-bearing one.

| Capability | Where | Live-path status |
|---|---|---|
| Bounded per-weight device ring (byte budget, LRU victim, pinned-exempt, page-in/hit/evict counters) | `internal/model/paging_ring.go` | **Wired (R0, #5611).** `pagedRing.stage` + `internal/model/expert_ring_hal.go` put it under the session weight HAL for routed experts, opt-in per session via `Session.ExpertRingBytes`. Was off-path (sole constructor `v4_expert_runtime.go:67`, DeepSeek-V4 only). R1 sizes the budget, so `FAK_N_CPU_MOE` now reaches it. |
| Graded spill sizing + auto-fit to a measured device budget (`--n-cpu-moe N` semantics) | `internal/model/expert_spill_fit.go` (#5281, closed) | **Wired (R1, #5612).** `Model.ResolveExpertSpillPlacement` builds the budget from measured resident bytes and real MoE layer ordinals, `Session.ApplyExpertSpillPlacement` installs it, and `InKernelPlanner.SetExpertSpill` resolves it once per serve against `compute.DeviceMemoryInfo`. Was exported and unused outside tests. Flag surface still open (`FAK_N_CPU_MOE` only). |
| Value-aware evictor: routing-heat hysteresis + LFU-decay, scored against the Belady oracle | `internal/model/expert_residency_lfu.go` (#4357, closed) | **Ranking wired (R4, #5615); hysteresis still simulation only.** `internal/model/expert_ring_policy.go` gives the ring a victim-policy seam (`Session.ExpertRingEvict`), records the ordered trace the seam is judged on, and gates promotion on measured regret. The admission hysteresis stays off the live path — a bypassed weight would fall back to *permanent* `halW` residency, inverting its intent. Was simulation only, with no seam to move. |
| Cross-session expert usage histogram, warm-start pin selection, between-turn `RepinPass` | `internal/model/expert_warmpins.go` (#4358, closed) | **Wired (R2, #5613).** `internal/model/expert_ring_pins.go` gives `pagedRing.pins` its live consumer: `weightHALStagedBounded` computes the `pinned` bool from the durable set per routed expert, stagings feed a per-turn histogram, and `Session.ExpertRingEndTurn` repins + dumps. Was off-path - the set was never consulted and `matMulStaged` took a static bool per call. |
| Per-expert range read + `MADV_WILLNEED` readahead over a fused expert slab | `internal/model/expert_readahead.go` (#4359, closed) | **Off-path.** Nothing in the load/decode path calls `readExpertSlice` or `willneedExpertSlice`. |
| Lazy per-expert GGUF slab source (reads `stride`, not `E*stride`) | `internal/model/gguf_expert_source.go` | **Off-path.** No constructor on the loader path; `ggufload` still materializes the whole fused block. |
| Routed-expert route observer + access-trace replay corpus | `internal/model/expert_replay.go` | **Off-path.** `SetExpertRouteObserver` has no non-test caller. |
| Planned-vs-observed placement coverage/drift gauge | `internal/model/expert_placement_drift.go` (#3902) | **Off-path.** No caller. |
| Expert-residency eviction regret vs the KV Belady replay oracle | #4233 (closed), on `compute/kvreplay_oracle.go` | **Off-path.** Reachable only through the replay harness. |

The honest summary: **the invention is done; the promotion is not.** Every rung below is a
wiring-and-witness task on code that already exists and already has correctness tests, not a
new mechanism. That is why this is cheap relative to its leverage - and why it has stalled:
each piece was landed under a "generation frame" that explicitly deferred wiring, and no
issue owned the wiring itself. #2726 (which *did* own it) was closed COMPLETED on the
primitive while its own follow-on rung stayed open in the code comments.

**Status: two of the nine are now wired** (row 1, R0/#5611 - the bounded ring every other rung
attaches to; row 2, R1/#5612 - the graded spill that sizes its budget). The remaining seven are
still off-path, and both wired rungs are opt-in: on today they are reached by an environment
knob, not yet by a `fak serve` flag.

## The ladder

Tracked as epic **#5606** with children **#5611-#5618**.

Ordered by leverage-over-cost. R0 is the keystone: R2, R3, R4, R5, R7 all attach to the seam
it creates, and none of them can land without it.

### R0 - P0 - #5611 - Bound the activated set: wire the ring under the expert weight HAL - **LANDED**

`Session` has a budgeted expert ring (`ExpertRingBytes`, default 0 = off) and *routed-expert*
weight staging goes through it instead of the unbounded `halW` memoizer, keyed by the same
dtype-prefixed HAL key and gated by `isRoutedExpertWeight` so dense/attention/router/shared
weights keep their permanent residency.

- **Why first:** it converts "residency" from an emergent side effect of a memo map into a
  declared, bounded object with a budget and a victim policy. Every other rung is a policy
  or an observer *on that object*.
- **Reuse:** `pagedRing` already proves bit-equality between hit and miss paths against the
  cpu-ref backend, and inherits `polymodel.Pool`'s `used<=budget` and pinned-never-evicted
  invariants by construction. R0 added only `stage()` - `matMulStaged` with the GEMM removed,
  so the two share one lifecycle and the bit-equality transfers - plus `hold`/`release`, because
  one expert is three co-used weights and a tight budget must not evict `gate` to admit `down`.
- **Witness** (`internal/model/expert_ring_hal_test.go`, all green): (a) bit-exactness - six
  experts through a two-expert ring are byte-identical to the resident-HAL forward, with hits,
  page-ins and evictions all exercised; (b) boundedness - peak device weight bytes stay under
  budget and no routed expert reaches `halW`; (c) recovery - an evicted expert pages back in on
  its next activation, and `Close` pages the ring out; (d) co-residency - a two-weight budget
  holds one expert's projections together and falls the third back to permanent residency rather
  than freeing a handle in use; (e) default-unchanged - at `ExpertRingBytes == 0` no ring is
  built and every weight uploads exactly once, as before.
- **Not done here:** the budget has no CLI knob (R1), the ring is plain LRU and ignores the
  warm-start pin-set (since closed by R2; victim choice among unpinned residents is still LRU,
  R4), misses are synchronous (R3) and read the fully-resident slab (R5).

### R1 - P0 - #5612 - Ship the graded knob and admit on the working set - **PARTLY LANDED**

Expose the spill count that `expert_spill_fit.go` already computes (`--n-cpu-moe N`, plus
`auto` = `AutoFitExpertSpill` against measured device headroom), and teach preflight to admit
a model on **activated working set + ring budget** rather than the full expert bulk.

- **Why P0 alongside R0:** without it, R0's budget has no operator control and preflight still
  refuses models that the ring makes servable. `ggufload`'s
  `FitCPUOffloadExpertsOnDevice` was binary: all experts host, or refuse.
- **Landed - the graded placement** (`internal/model/expert_spill_placement.go`, `c0d5e12b41`).
  `splitKernel`'s predicate is graded by `Session.ExpertSpillLayers`: `MoEExpertLayers` reads
  the model's *real* routed-expert layer ordinals (so "first N MoE layers" is not "layers
  0..N-1" on a hybrid checkpoint with a dense prefix), `ExpertSpillBudgetFor` builds the
  sizing input from measured resident bytes with the per-layer cost rounded **up** so an uneven
  checkpoint over-spills rather than under-counting into an OOM, and `RingBytesAt` derives R0's
  ring budget from the same arithmetic. At `ExpertSpillLayers <= 0` the predicate *is*
  `isExpertWeight`, so the default path is byte-for-byte unchanged.
- **Landed - the served seam** (`internal/agent/inkernel_expert_spill.go`).
  `InKernelPlanner.SetExpertSpill` resolves once at setup (sizing walks every resident tensor
  name; the device path builds a session per request) against a device budget measured from
  `compute.DeviceMemoryInfo` *free* bytes less a 15% KV/activation reserve, and installs it on
  every session the planner builds. It refuses rather than degrades on an out-of-range N, on
  `auto` with no measurable budget, and on an explicit spill of a model with no routed-expert
  residency. `FAK_N_CPU_MOE=auto|<N>|off` is the operator door.
- **Witnessed:** `internal/model/expert_spill_placement_test.go` (spill order follows real MoE
  ordinals behind a dense prefix; the ungraded grade matches the legacy predicate name-for-name;
  a graded split routes only the first layers to the host kernel with prefill bit-identical
  across placements; sizing from resident bytes; out-of-range refusal) and
  `internal/agent/inkernel_expert_spill_test.go` (the plan reaches successive sessions; the
  ungraded default leaves placement untouched; each refusal fires; the env door grades the
  planner the real constructor builds).
- **Landed - the activated-working-set admission** (`internal/ggufload/expert_activated_fit.go`).
  Both prior admissions ask *do the weights fit*: `FitOnDevice` charges the whole checkpoint,
  `FitCPUOffloadExpertsOnDevice` charges the dense base and moves the *entire* expert band to
  host. Neither can express what the ring runs. `ActivatedExpertFit` names three levels of device
  demand from the header alone - **floor** (dense base + one MoE layer's K experts), **token**
  (dense base + K experts on every MoE layer), **band** (what today's admission demands) - and
  the refusal line moves to the floor, the honest one: below it no ring budget can assemble a
  single expert GEMM group, above it the model is servable but paging. `RingBytes` is whatever
  the budget leaves after the dense base, clamped to `[floor, band]`, and *is* the device-scoped
  expert demand of the emitted plan, so the refusal falls out of the plan rather than a second
  branch. `RoutedExpertActiveSet` gained `MoELayers` as the per-layer denominator - deliberately
  not `block_count`, since a hybrid checkpoint's dense prefix carries no routed experts.
- **Witnessed** (cont.): `internal/ggufload/expert_activated_fit_test.go` - at a budget holding
  the base plus one whole token's activation, `FitOnDevice` **refuses** the fixture and
  `FitActivatedExpertsOnDevice` **admits** it (same source, same backend, same headroom); one
  byte below the floor it still refuses with a typed `*compute.FitError`; ring + host always
  re-sum to the band; unknown K and K > E both fall back to the whole band rather than being
  admitted as a small activated set.
- **Not done:** `fak serve --n-cpu-moe` is not wired (`cmd/fak/serve.go`,
  `internal/gateway/gateway.go`), so the knob is env-only; and `refuseEPPlanIfUnfit` still calls
  the band-shaped `FitCPUOffloadExpertsOnDevice` rather than the activated form that now exists
  beside it. Both are blocked on peer hunks in those files, not on design - the flag is three
  mechanical touch-points and the swap is one line. **#5612 is closed, so the open remainder is
  tracked as #5628** rather than left inside a closed thread.
- **Remaining witness:** a checkpoint whose full expert bulk exceeds the device budget is
  admitted *through the serve flag* and decodes at a chosen N, with the plan JSON reporting the
  sized split.

### R2 - P1 - #5613 - Consult the pin-set and persist the histogram on the live ring - **LANDED**

Make `matMulStaged` consult `isExpertPinned` instead of the caller's static bool, dump the
usage histogram per turn, sum it at boot to warm-start pins, and run `RepinPass` between
turns.

- **Reuse:** all four pieces ship in `expert_warmpins.go`; this is the consult + lifecycle.
- **Landed** (`internal/model/expert_ring_pins.go`, `64e3f5a92a5b`) as three joins onto code
  that already existed. *Consult:* `weightHALStagedBounded` derives `(layer, expert)` from the
  canonical weight name (`routedExpertIdentity`) and passes `r.isExpertPinned`, so polymodel's
  pinned-never-evicted invariant protects the workload's hot set instead of nothing. *Observe:*
  every routed-expert staging folds a touch into the ring's per-turn histogram, so the prior is
  built from real routing rather than a profiling run. *Persist:* `Session.ExpertRingEndTurn`
  decays the standing heat, folds the turn in, repins under a bounded swap cap, fills any pin
  slot still free, and dumps crash-safely to `Session.ExpertUsagePath`. Two new session knobs
  (`ExpertPinBudget`, `ExpertUsagePath`); declaring neither is R0's plain-LRU ring
  byte-for-byte.
- **One piece of new policy, not a join.** `RepinPass` only ever *swaps* - it walks the pinned
  experts and exchanges the coldest for a hotter candidate - so a set that starts empty stays
  empty however much heat accumulates. Warm-starting fills the set from a prior, so #4358 never
  had to; but a cold run with no prior on disk would then serve its whole life on plain LRU and
  only begin pinning after a restart. `fillPins` closes that: filling a free slot displaces
  nothing, so it is bounded by the operator's budget rather than by the churn cap, and it is
  refused for an expert carrying no heat (an unobserved pin is a guess; LRU is a better one).
- **Witness:** turn-2 cold-start page-ins fall against turn-1 for a repeated workload, and the
  pinned set survives a process restart. Both hold in
  `internal/model/expert_ring_pins_test.go`, on a sweep longer than the ring - the window pure
  recency cannot cache, so run 1 scores 0 hits by construction and every hit in run 2 is
  attributable to the pin-set. A second process sharing only the dumped histogram pages in
  **30** routed weights against the first's **39**, and the identity it warm-starts pinned is
  the one the first process's routing selected, not a tie-break default. Three gates alongside:
  declaring no knobs is R0 exactly (no pin-set, no observation, no-op boundary), the identity
  parse refuses shared experts / the router / dense weights / malformed ordinals rather than
  crediting a silent `(0,0)` (which is a *real* identity), and a corrupt dump degrades to a cold
  start while still reporting itself once at the next turn boundary.
- **Not done:** the turn boundary has no *caller* on the serve path yet - `ExpertRingEndTurn` is
  a session method a decode loop must invoke at a quiescent point, and picking that point (plus
  the `decay`/`maxSwaps` defaults, and where the dump lives for a multi-session host) belongs
  with the operator surface in R6/#5617. Victim choice among unpinned residents defaulted to LRU
  here (since made a measured seam by R4/#5615) and a miss is still a synchronous upload
  (R3/#5614).

### R3 - P1 - #5614 - Router-lookahead prefetch (composes with #4300) - **SEAM LANDED**

The router for layer L+1 is computable before layer L's expert GEMMs retire. Issue the
page-in for the next layer's activated experts against the ring while the current layer
computes.

- **Relation to #4300:** #4300 owns the *predictive/speculative* policy layer (probability
  threshold, cross-token prediction). R3 is the deterministic floor beneath it - same-step,
  known-not-predicted routing - and it is what gives #4300 a place to attach. Land R3 first,
  then #4300 raises the lookahead horizon.
- **Reuse:** `willneedExpertSlice` (#4359) is the host-side hint half.
- **Witness:** fraction of activated-expert page-in latency overlapped with compute; decode
  step time with prefetch on vs off at a fixed ring budget.
- **Landed** (`aa78492e5fea`, `internal/model/expert_ring_prefetch.go`): the SAME-STEP floor.
  `prefetchActivatedExperts` stages the layer's own routed top-k at layer entry, in route
  order, from a seam in `moeFFN.apply` / `glmMoeFFN.apply`. Two rules keep it a prefetch
  rather than a thrash - **do not prefetch what cannot stay** (only the prefix of the set the
  budget holds; running past it evicts what was just fetched) and **a prefetch is a hint, not
  a demand** (it takes recency but earns no heat, and never enters R2's histogram or R4's
  trace, because a policy ranked on its own prefetcher's guesses is self-confirming). It never
  falls back to permanent `halW`, so a misconfigured budget cannot convert speculation into
  permanent residency.
- **Coverage meter.** `ExpertRingStats.ActivatedCovered/ActivatedExperts` answers the plan's
  own thesis question - "were this token's activated experts resident?" - directly. A ring too
  small to hold one layer's top-k reports honest misses forever, which a hit rate alone cannot
  distinguish from a cold start.
- **Not done - the stated witness is NOT produced.** "Fraction of page-in latency overlapped
  with compute" cannot be measured here: `compute.Backend` exposes `Upload` as the only
  host->device path and no stream or event handle, so there is nothing on cpu-ref to measure an
  overlap against. What landed is the ORDERING the overlap would rest on (every weight upload
  precedes the first expert GEMM) at zero extra page-ins and bit-identical output. Turning that
  into a real overlap number needs an async staging primitive on the `Backend` interface - the
  next checkable step for this rung, and a prerequisite the plan did not price. **#5614 is
  closed, so that prerequisite is tracked as #5627** (an optional `AsyncUploader`/`Fence`
  extension, shaped like the existing `RankUploader`). It gates the ladder's actual latency
  win: R0-R4 made expert residency bounded, pinned, policy-chosen and prefetched, but a ring
  miss is still a *synchronous* upload on the critical path.
- **L+1 lookahead stays with #4300**, and the reason is structural rather than schedule: layer
  L+1's router input is downstream of layer L's own expert output, so crossing the layer
  boundary requires a *predictor*, not just earlier issue. `prefetchActivatedExperts` takes a
  pick list, and a predictor is simply a different producer of one.

### R4 - P1 - #5615 - Promote the evictor on measured regret, not on faith - **LANDED**

Give the ring a policy seam and select the policy by replaying the real expert-access trace
through the Belady oracle. The literature is explicitly contested here: recency is a poor
prior for MoE access because routing is layer-sequential, so the LRU the ring inherits is a
*default*, not a finding.

- **Reuse:** the candidate policy (#4357) and the regret gauge (#4233) both ship; this wires
  the winner into the ring behind the gauge.
- **Witness:** `GoodDecisionRatio` (realized/oracle) of the shipped policy on a real routing
  trace, reported against LRU on the same trace. Promote only on a positive delta with no hit
  regression.
- **Landed** (`fcd61d86a059`, `internal/model/expert_ring_policy.go`) as three joins:
  *seam* - under `Session.ExpertRingEvict = ExpertRingEvictValueAware` the ring ranks victims
  by decaying heat (LRU tie-break) and evicts them *before* `polymodel.Admit`, so Admit fits
  without choosing; *trace* - the ring records the **ordered** access sequence its own staging
  produced (`Session.ExpertRingTrace`), because R2's usage histogram is aggregate and loses the
  order Belady needs; *gate* - `SelectExpertRingEvictPolicy` replays that trace under both
  policies against one oracle and promotes only on a strict eviction win with no hit
  regression.
- **The gate scores the policy the ring actually runs.** It deliberately does *not* call
  `ReplayExpertResidencyLFUDecay`, which scores #4357's full policy - ranking **plus** admission
  hysteresis. `simulateLFUDecayResidency` was split over a new
  `simulateHeatResidency(..., hysteresis bool)` so the gate can score the ranking alone; #4357's
  own report keeps the hysteresis variant unchanged. A gauge that scores a policy the ring does
  not run measures nothing.
- **Measured**, live ring, same budget, same jitter window: LRU 60 page-ins / 6 hits / 54
  evictions; value-aware 47 / 19 / 41. The gate, fed the trace that ring recorded, returns the
  same verdict independently (good-decision ratio 0.571 vs LRU 0.143). Default is off
  byte-for-byte: the zero value is LRU, allocates no heat map, and a session declaring it has a
  ledger identical to one that never mentions a policy.
- **Not done here:** hysteresis on the live path. In the simulation a bypass is free; in the
  ring `weightHALStagedBounded`'s fallback for a refused stage is *permanent* `halW` residency,
  which would promote the very expert the policy wanted transient and break R0's "no routed
  expert reaches `halW`" bound. Serving a bypassed weight transiently needs a staging contract
  the HAL does not have yet. No operator verb over the gate either - that is R6/#5617.

### R5 - P2 - #5616 - Put a checkpoint tier under a ring miss

A ring miss should read *that expert's* stride from the checkpoint, not force the whole fused
`[E, out, in]` block resident in host RAM. Wire `ggufExpertSource` / `readExpertSlice` under
the miss path to complete the device / pinned-host / checkpoint ladder.

- **Witness:** bytes read per decode step scale with k, not E, at a fixed ring budget; output
  is bit-identical to the fully-resident slab path.

### R6 - P2 - #5617 - Make it an operator-visible object

There is no `fak` verb for MoE residency today. Add the surface: activated-set size, ring
hit/page-in/evict counts, bytes per token, pinned set, and the coverage/drift gauge
(`ScoreExpertPlacement`) against observed routing - in serve telemetry and in a `fak` verb.

- **Why it matters here:** "first-class" in this repo means an operator can turn it on and
  inspect it. R0-R5 are invisible without this rung.
- **Witness:** the verb reports a live ring's counters for a decode window; the numbers
  reconcile with the ring's own accounting.

### R7 - P2 - #5618 - Cross-agent coalescing on a live ring (#5243 L2)

With B concurrent agents, per-step top-k selections union to far fewer distinct experts than
`B*k`, so one page-in serves many agents. Epic #5243 and its children own the model, the
telemetry and the bench; what they lack is a live shared ring to coalesce *onto*. R0 supplies
it.

### R8 - P3 - #5247 - Grouped GEMM over the coalesced activated union

Already owned by #5247. Listed here only to fix its position in the ladder: it is a compute
optimization *downstream* of a bounded, shared activated set, not a substitute for one.

## Non-goals and dedup

- **Not a new evictor invented here.** R4 promotes a measured winner; it does not design a
  third policy.
- **Not multi-GPU expert placement.** #3886 (load-aware EP placement) is a distinct axis -
  balancing experts *across ranks*, not bounding residency *within* one device.
- **Not the deferred-expert compute pipeline.** #5239 overlaps CPU expert latency with GPU
  work; it composes with R3 but is a separate mechanism (compute scheduling, not residency).
- **Not a learned routing-path predictor.** #4318 owns approximate-router prediction; R3 is
  the deterministic same-step floor.
- **No new hardware claim.** Every rung's witness above is a delta measured against fak's own
  current path on the same host, not a cross-engine performance claim.

## Reading order

`internal/model/moe_offload.go` (the static split) -> `internal/model/hal.go` (`weightHALStaged`,
the unbounded memoizer that is still the default) -> `internal/model/paging_ring.go`
(`stage`/`hold`, the bounded twin) -> `internal/model/expert_ring_hal.go` (R0: the wiring that
binds them for routed experts) -> `internal/model/expert_spill_fit.go` (the sizing math) ->
`internal/model/expert_spill_placement.go` (R1: what gives that math a caller and the split
kernel a grade) -> `internal/agent/inkernel_expert_spill.go` (R1: the resolve-once, install-per-
session seam a serve reaches through `FAK_N_CPU_MOE`) ->
`internal/ggufload/expert_activated_fit.go` (R1: the admission that stops charging a checkpoint
for the experts it will never hold at once) -> `internal/model/expert_warmpins.go` (the durable
usage histogram, warm-start selection and between-turns actuator, all built by #4358) ->
`internal/model/expert_ring_pins.go` (R2: the three joins that put that machinery on the live
staging path, plus the fill `RepinPass` never had) -> `internal/model/expert_residency_lfu.go`
(the value-aware policy #4357 simulated and measured, and the `hysteresis` axis R4 split out of
it) -> `internal/model/expert_ring_policy.go` (R4: the victim seam, the ordered trace, and the
gate that promotes only on measured regret) -> `internal/model/expert_ring_prefetch.go` (R3:
the same-step activated-set prefetch, the anti-thrash prefix rule, and the coverage meter).
