---
title: "Ultra-long-context sessions: the levels, the levers, and the naming"
description: "A framing note for first-class >100k-token session support: the four levels of the regime, which kernel levers each needs (shipped vs missing), the integration points for compaction and KV compression, and a naming verdict on 'skip turn' / 'session query'."
---

# Ultra-long-context sessions — levels, levers, naming

Date: 2026-06-22

Scope: the thinking behind the ultra-long-context proof (the exact work floor in
[`ULTRA-LONG-CONTEXT-RESULTS.md`](../benchmarks/ULTRA-LONG-CONTEXT-RESULTS.md) +
`internal/turnbench/longcontext.go` + `cmd/longctxbench`). This note is the *why* and the
*what-next*: the levels of the regime, the lever each level leans on, where compaction and
compression plug in, and a verdict on the names. Measured/shipped is separated from frontier.

Companion: [scaling-laws thesis](SCALING-LAWS-OF-AGENTS-2026-06-19.md) (this note is the
ultra-long-context specialization of its regimes 2 and 3).

---

## 1. The four levels

The regime is not one thing. Each level has a *different* dominant cost and leans on a
*different* lever. The floor numbers below are from the canonical ladder (Qwen2.5-7B geometry).

| Level | Shape | Dominant cost | The lever | Floor (vs naive / vs tuned) |
|---|---|---|---|---|
| **L0 — short session** | small P, few turns, 1 agent | raw model latency | none — prefix caching already wins | ~1× (out of regime) |
| **L1 — single >100k session** | P≈100k or deep T, 1 agent | re-reading the whole context every turn | **reread elision** (persistent KV; the turn-tax) | **~10× / 1.0×** |
| **L2 — multi-agent fleet, each >100k** | shared 100k prefix, C agents | C× duplicate prefix prefill | **cross-agent prefix share** (prefix once, cloned) | **~40×+ / ~4×** |
| **L3 — agent city** | shared 100k prefix, tens–thousands of agents | residency + invalidation + scheduling | prefix share **+ KV residency tiers + scoped invalidation** | ~150× / ~15× (and rising toward C) |

The key honest fact the floor makes explicit: **B/C (vs a warm cache) rises monotonically with
the shared-prefix fraction, from ~1 toward the agent count C.** L1 has no peer to share with, so
its win is *entirely* vs-naive (turn-tax); the vs-tuned win only appears at L2+ and grows with C.

What *changes* as you climb the levels:

- **L1 → L2**: the win stops being "don't re-read your own context" (turn-tax, a single-tenant
  property) and becomes "don't let C agents each re-read the *shared* setup" (cross-agent, the
  property a shared-slot serving engine cannot offer while preserving per-agent KV ownership).
- **L2 → L3**: the bottleneck moves *off* FLOPs entirely. At thousands of agents the hot KV does
  not fit in HBM/DRAM, so the binding constraint is **residency** (paging, tiers, recompute
  policy) and **legal reuse** (one write must not blast-invalidate every sharer). This is the
  scaling-laws "agent city" wall; the FLOP floor stops being the right meter there.

---

## 2. The lever inventory (shipped vs missing)

What the floor *assumes* exists, and whether it does. Most of L1/L2 is shipped; L3 and the
context-shrinking levers are the gap.

| Lever | Status | Where | Level it serves |
|---|---|---|---|
| Persistent per-agent KV (reread elision) | **shipped** | `internal/model` Session/KVCache; sessionbench arm B | L1 |
| Cross-agent prefix share (prefix once + clone) | **shipped** | `model.NewBatchFromPrefix`, `KVCache.Clone`; sessionbench arm C | L2 |
| Bit-exact middle-span KV eviction | **shipped** | `KVCache.Evict` (stores pre-RoPE `Kraw`, re-rotates survivors) | L1–L3 |
| Sliding-window attention (O(window) not O(L)) | **shipped** | `internal/model` `windowForLayer`/`windowLoContig` | L1–L3 |
| RoPE scaling / longrope (reach 100k positions) | **shipped (Part A)** | `internal/model/longrope.go`, `rope_scaling.go` | L1–L3 |
| RadixAttention prefix-tree sharing + policy evict | **shipped** | `internal/radixkv` | L2–L3 |
| KV residency tiers + witnessed exact-span eviction | **shipped** | `internal/cachemeta`, `internal/enginecache`, `engine/cacheevents` | L3 |
| Result-admit gate + page-out + poison re-screen | **shipped** | `internal/ctxmmu`, `internal/recall` | all |
| **Idle-lane skip / ragged batched decode** | **MISSING** | `model.StepBatch` decodes *all* C lanes every step | L2–L3 |
| **First-class context-residency query** | **MISSING** | ctxmmu query API is observability-only (`Held`/`HeldLen`/`Evicted`) | L1–L3 |
| **Compaction (drop/summarize old turns)** | **MISSING** | `TrimToWindow` designed in SLIDING-WINDOW doc, no API | L1–L3 |
| **KV compression wired into eviction** | **MISSING** | `cachemeta.MatCompressedKV` designed, not on the evict path | L3 |
| **Live wall-clock anchor at >100k** | **MISSING** | needs a model resident on a bench node | L1–L2 |

The five MISSING rows are the issue backlog this proof seeds (see §5).

---

## 3. The naming verdict — "skip turn", "session query"

The goal asks whether "skip turn" is a good name. A name-collision scan found **"skip turn",
"session query", and "reread" are unused anywhere in the tree; "elision" is *taken*** (it means
"a turn saved by omitting an optional tool call", `TurnKinds.Elision`). So the field is open —
but "skip turn" is the *wrong* name, for a precise reason: **it conflates two distinct levers and
overclaims.** You never skip a *turn* (the turn still happens and still decodes); you skip
*work within* the turn. What work, exactly, splits into two unrelated mechanisms:

1. **Reread elision** — *not re-processing context the kernel already holds.* This is the
   prefill-skip / turn-tax lever, the L1 win, and it is **already shipped** (arm B→A). It lives in
   the scaling-laws vocabulary ("reread rate"). Recommended name: **reread elision** (umbrella) /
   **prefill-skip** (the specific mechanism). NOT "skip turn".

2. **Idle-lane skip** — in fused multi-agent *decode*, *not decoding a lane that has nothing to
   produce this step* (blocked on a tool, emitted EOS, or idle). Today `StepBatch` decodes all `C`
   lanes every step, so a heterogeneous fleet wastes decode on idle agents — capping the L2 win
   below its potential. This is **not shipped** and is a genuine new lever. Recommended name:
   **idle-lane skip** / **ragged batched decode** (the geometry: a ragged, not rectangular, batch).
   This is the closest thing to what "skip turn" *gestures* at, made precise.

So: retire "skip turn"; use **reread elision** (shipped) and **ragged decode / idle-lane skip**
(new). Keep "elision" reserved for the optional-call meaning it already has — do not overload it.

**"Session query"** is the right *intent* but the wrong *granularity*: what you want to query at
ultra-long context is *which spans are resident, which are evictable, which are quarantined, and
what a candidate eviction would cost*. ctxmmu exposes only counters today. Recommended name:
**context-residency query** — a first-class, witnessable read over the span ledger (resident /
evictable / held / cleared, with the eviction blast-radius), the read side of the same coherence
state the write side (`Admit`/`Evict`) already maintains.

---

## 4. Compaction and compression — the integration points

The goal asks to "peek into" compaction and compression. Neither shrinks the context *today*;
both have a clear seam.

- **Compaction (summarize/drop old turns).** The kernel has the *mechanism* to make dropping safe
  — `KVCache.Evict` is bit-exact and sliding-window makes aged-out positions provably
  un-attended — but no *policy* that decides what to drop, and no `Session.TrimToWindow(slack)`
  API to call it on the live loop. The integration point is `internal/agent/loop.go` before the
  per-turn decode. The interesting version is **kernel-mediated external compaction**: a harness
  (e.g. Claude Code's auto-compaction) emits a summary span; the kernel `Evict`s the summarized
  range and ingests the summary *under the same result-admit gate* (so a poisoned summary cannot
  launder back in) — compaction becomes a coherence-checked span swap, not an opaque rewrite.

- **KV compression libraries.** `cachemeta.MatCompressedKV` already models a quality-gated
  compressed-KV view (`QualityEvidence`, `MaxQualityDelta`) but is not on the eviction path. The
  seam: when a span would be *evicted* under residency pressure (L3), compress-and-demote it to a
  lower tier instead of dropping it, admitting the compressed span only when the measured quality
  delta is under bound. This composes with the residency tiers already shipped; it is the L3
  "keep more context resident per byte" lever, distinct from compaction (which *removes*
  information; compression *preserves* it lossily).

The honest framing for both: they are **context-shrinking** levers that change the *L* the floor
is evaluated at, orthogonal to the reread-elimination the floor measures. A compacted/compressed
context makes *every* arm cheaper; the kernel's contribution is doing it **safely** (gated,
witnessed, non-laundering), which is the coherence-layer thesis, not a raw-speed claim.

---

## 5. What to build next (the issue seeds)

1. **Ragged batched decode / idle-lane skip** — stop decoding idle lanes in `StepBatch`; lift the
   L2 win for heterogeneous fleets. The genuine "skip turn".
2. **Context-residency query** — promote ctxmmu's observability counters to a first-class,
   witnessable read over the resident/evictable/held span ledger (+ eviction blast-radius).
3. **`Session.TrimToWindow` + kernel-mediated external compaction** — the gate-checked span-swap
   compaction seam on the live agent loop.
4. **KV compression on the eviction path** — wire `MatCompressedKV` quality-gated compression as
   a compress-and-demote step under residency pressure.
5. **Live wall-clock anchor at >100k** — validate the floor's ratios with a real `sessionbench`
   run on a bench node (model resident), closing the floor↔wall-clock loop the way `sessionbench`
   already does at small scale with `-validate`.

Each is additive and witness-bearing; none touches the frozen ABI.
