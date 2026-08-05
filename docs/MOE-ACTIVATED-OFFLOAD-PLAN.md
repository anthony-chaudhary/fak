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
is the seam the rest of the ladder attaches to. It is **not yet operator-reachable** - no CLI
flag sets the budget, so every session in the tree still runs placement 2 by default. Sizing
that budget from a flag is R1 (#5612).

## What is already built (and where it stops)

Nine witnessed capabilities exist. Column 3 is the load-bearing one.

| Capability | Where | Live-path status |
|---|---|---|
| Bounded per-weight device ring (byte budget, LRU victim, pinned-exempt, page-in/hit/evict counters) | `internal/model/paging_ring.go` | **Wired (R0, #5611).** `pagedRing.stage` + `internal/model/expert_ring_hal.go` put it under the session weight HAL for routed experts, opt-in per session via `Session.ExpertRingBytes`. Was off-path (sole constructor `v4_expert_runtime.go:67`, DeepSeek-V4 only). Still needs R1's flag to be operator-reachable. |
| Graded spill sizing + auto-fit to a measured device budget (`--n-cpu-moe N` semantics) | `internal/model/expert_spill_fit.go` (#5281, closed) | **No caller.** `ResolveExpertSpill` / `AutoFitExpertSpill` are exported and unused outside tests; no CLI flag exposes N. |
| Value-aware evictor: routing-heat hysteresis + LFU-decay, scored against the Belady oracle | `internal/model/expert_residency_lfu.go` (#4357, closed) | **Simulation only,** deliberately off the live path; the ring is still plain LRU via `polymodel.Pool`. |
| Cross-session expert usage histogram, warm-start pin selection, between-turn `RepinPass` | `internal/model/expert_warmpins.go` (#4358, closed) | **Off-path.** `pagedRing.pins` is never consulted - `matMulStaged` still takes a static `pinned` bool per call (`paging_ring.go:56-61`). |
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

**Status: one of the nine is now wired** (row 1, R0/#5611) - the bounded ring, which is the
one every other rung attaches to. The remaining eight are still off-path, and the ring itself
is still session-level opt-in with no flag behind it.

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
  warm-start pin-set (R2/R4), misses are synchronous (R3) and read the fully-resident slab (R5).

### R1 - P0 - #5612 - Ship the graded knob and admit on the working set

Expose the spill count that `expert_spill_fit.go` already computes (`--n-cpu-moe N`, plus
`auto` = `AutoFitExpertSpill` against measured device headroom), and teach preflight to admit
a model on **activated working set + ring budget** rather than the full expert bulk.

- **Why P0 alongside R0:** without it, R0's budget has no operator control and preflight still
  refuses models that the ring makes servable. `ggufload`'s
  `FitCPUOffloadExpertsOnDevice` is binary today: all experts host, or refuse.
- **Witness:** a checkpoint whose full expert bulk exceeds the device budget is admitted and
  decodes at a chosen N, with the plan JSON reporting the sized split; the refusal path still
  fires when even the dense base plus the minimum ring does not fit.

### R2 - P1 - #5613 - Consult the pin-set and persist the histogram on the live ring

Make `matMulStaged` consult `isExpertPinned` instead of the caller's static bool, dump the
usage histogram per turn, sum it at boot to warm-start pins, and run `RepinPass` between
turns.

- **Reuse:** all four pieces ship in `expert_warmpins.go`; this is the consult + lifecycle.
- **Witness:** turn-2 cold-start page-ins fall against turn-1 for a repeated workload, and the
  pinned set survives a process restart.

### R3 - P1 - #5614 - Router-lookahead prefetch (composes with #4300)

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

### R4 - P1 - #5615 - Promote the evictor on measured regret, not on faith

Give the ring a policy seam and select the policy by replaying the real expert-access trace
through the Belady oracle. The literature is explicitly contested here: recency is a poor
prior for MoE access because routing is layer-sequential, so the LRU the ring inherits is a
*default*, not a finding.

- **Reuse:** the candidate policy (#4357) and the regret gauge (#4233) both ship; this wires
  the winner into the ring behind the gauge.
- **Witness:** `GoodDecisionRatio` (realized/oracle) of the shipped policy on a real routing
  trace, reported against LRU on the same trace. Promote only on a positive delta with no hit
  regression.

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
binds them for routed experts) -> `internal/model/expert_spill_fit.go` (the sizing math still
waiting for a caller - R1).
