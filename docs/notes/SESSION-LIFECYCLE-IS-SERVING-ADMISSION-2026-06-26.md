---
title: "One machine, two altitudes: the session lifecycle IS serving admission (turn = the quantum)"
description: "Starting/resuming an agent session and admitting/preempting a sequence into a continuous serving batch are the same state machine seen at two altitudes; the agent turn boundary is its single admission/preemption/resume quantum. Today the repo spells that machine twice and only advisory-couples the halves. This note maps the correspondence, separates the buildable seams from the analogies, and proposes a unification epic."
date: 2026-06-26
---

# One machine, two altitudes

*Session start/resume and dynamic fused serving of random-length, flexible-turn
sessions are not two systems that need a bridge. They are one state machine viewed
from two altitudes. fak's code already implies that — it just spells the machine
twice and wires the halves together with advisory hints instead of one queue.*

## The claim in one sentence

The `internal/session` run-state machine (start / throttle / pause / drain / stop,
plus cold `Restore` re-attach and `Recontinue` re-arm) and the continuous-batch
serving operation (admit / preempt / evict / resume a sequence) are the **same
machine**, and the **agent turn boundary** is its single admission / preemption /
resume quantum. A served session is a random-length, flexible-turn continuous
batch: *start a turn* is *admit to the batch*; *throttle/yield* is *preempt*;
*drain/stop at the next boundary* is *evict + free KV*; *resume* (live or cold) is
*re-admit the sequence*.

## The evidence it is already one skeleton

The same shape — **an admit-with-a-named-reason gate, taken at a boundary, never
mid-flight** — is implemented at least three times, in three packages, with three
unrelated vocabularies:

- `internal/session/decide.go` `Decide` — the per-turn boundary gate: debit the
  turn, return proceed/why for an agent session.
- `internal/loopmgr/loopmgr.go` — the loop-run lifecycle (`LoopState`:
  `armed/running/paused/draining/stopped/disabled`), folded out of a hash-chained
  event ledger by `Summarize`, with `admit`/`refused` events.
- `internal/compute/discard_admit.go` `DecideDiscardAdmission` — the serving-side
  admission decision (`AdmitPreempt`, discard-aware), a pure verdict that moves no
  KV.

Three admission gates. None of them shares a type, a state vocabulary, or a queue
with the others. The skeleton is the same; the spelling is not.

## The correspondence

Each row pairs a session verb with its serving counterpart, marks how fused the two
are **today**, and cites the grounding symbol. Status is one of: **fused** (one
mechanism already does both), **duplicated** (both exist, no shared type/queue),
**advisory-only** (a hint flows but gates nothing live), **analogy** (a real
parallel with no buildable seam yet).

| Session verb | Serving verb | Status | Grounding |
|---|---|---|---|
| START / admit a turn (`DefaultState` on first `Decide`; `beginServedSessionTurn`) | ADMIT + PREFILL a sequence (`NativeScheduler.Admit`; `BatchPolicy.ComposeBatches`) | duplicated | `gateway/session_admit.go`, `session/decide.go` vs `modelengine/nativesched.go`, `gateway/batchsched.go` — two admit-with-reason gates, no shared queue |
| PAUSE (`Transition→Paused`, non-terminal hold) | PREEMPT / SWAP-OUT (free a running seq's KV) | advisory-only | `session/session.go` `RunState(Paused)` vs `compute/discard_admit.go` `AdmitPreempt` — Paused has no live consumer loop; `AdmitPreempt` moves no KV |
| DRAIN/STOP at boundary (`SlotEvent` `CauseDraining/CauseStopped`) | EVICT + FREE KV (`KVCache.Evict` / `kvmmu.EvictColdest`) | advisory-only | `session/scheduler.go` `SlotEvent` vs `kvmmu/attention.go` `EvictColdest`, `model/kvcache.go` `Evict` — `SlotEvent` is consumed only in-memory; the served path drives no real `Evict` |
| RECONTINUE (terminal parent + fresh-budget child trace on exhaustion) | MAX-LEN SPILL / RECOMPACT + RE-PREFILL | advisory-only | `session/table.go` `Recontinue` + `gateway/reset_score.go` `ResetScore` (shadow / recommend-only) — separate surfaces, no controller binds them |
| SNAPSHOT (`Table.Snapshot`, sorted Priority/Rev) | ACTIVE-BATCH TABLE (the lane set a step iterates) | duplicated | `session/table.go` `Snapshot` + `scheduler.go` `Pick` vs `modelengine/nativesched.go` `schedLane` — two batch tables; `Pick` names a winner no batch acts on |
| ContinuationID + Generation lineage | KV-EPOCH / ProvisionalSink epoch lineage | duplicated | `session/usage.go` `continuationID` + `session.go` `Generation` vs `abi/types.go` `SpeculationContext{Epoch,ParentEpoch}` + `Outcome{OutcomeCommitted,OutcomeSquashed,OutcomeRolledBack}` + `ProvisionalSink` — two epoch lineages, no converter |
| BUDGET (`TokensLeft`/`ContextTokensLeft` debited per turn; `Pool` ceiling) | max_tokens + KV-BLOCK BUDGET (residency positions) | duplicated | `session/session.go` `Budget` + `pool.go` `Draw` vs `kvmmu/attention.go` `EvictUnderBudget(budgetPositions)` — two budgets; the session token budget never bounds KV residency |
| PRIORITY (`State.Priority`, read by `Pick`) | SCHEDULER PRIORITY (which lanes decode this step) | advisory-only | `session/scheduler.go` `Pick` (`StrictPriority`/`WeightedFair`) — fused inside the session layer, but the serving `NativeScheduler` has no priority/fairness, so session priority never reaches a real batch |
| RunState enum (uint8, in-memory, 5 states incl. Throttled) | `loopmgr.LoopState` enum (string, ledger-folded, 6 states incl. Disabled) | duplicated | `session/session.go` `RunState` vs `loopmgr/loopmgr.go` `LoopState` — share `paused/draining/stopped/running` tokens, no shared type/String/Parse/converter |
| THROTTLE (`SetPace`; `Pace.MaxTokensPerTurn`) | PER-STEP DECODE RATE-LIMIT across co-resident lanes | analogy | real shipped artifact is a per-**turn** `max_tokens` cap (`session_admit.go` `maxTokensFor`); there is no co-resident-lane served batch to rate-limit until #401 |
| LIVE RESUME (`Transition Paused→Running`) | SWAP-IN (page preempted KV back) | analogy | `KVCache.Clone` exists but nothing splices it on `Paused→Running`; `warmPrefix` stamps `live_kv_reuse:deferred`. No KV mover is wired — the live-resume loop is **untracked** |
| COLD RESTORE / REHYDRATE (re-attach drive verbatim, Rev-preserving) | KV-RESTORE from a colder tier | analogy | `session/restore.go` `Restore` + `sessionimage` `Portability.KVIncluded=false` **by design** — resume re-attaches logical state but always re-prefills cold; there is no KV residency tier to restore from |

## Where it is duplicated today

Four things are built twice, with no shared type and no converter:

1. **The lifecycle vocabulary.** `session.RunState` (a `uint8`, in-memory) and
   `loopmgr.LoopState` (a `string`, folded from a ledger) both carry
   `paused/draining/stopped/running`. Neither imports the other; there is no shared
   definition, `String`/`Parse`, or converter. `loopmgr/registry.go` `validJobState`
   even re-uses the `loopmgr` constants for a third surface.
2. **The epoch / generation lineage.** `continuationID = sha256(trace+rev)` plus
   `State.Generation` is one lineage; `abi.SpeculationContext{Epoch,ParentEpoch}`
   plus `Outcome` is another. Same idea — a parent that spawns provisional children
   that commit or get discarded — maintained in two unrelated type families.
3. **The batch table.** `session.Table.Snapshot` (sorted into scheduler-consumption
   order) and `modelengine`'s `schedLane` set are two active-sequence tables that
   never share a queue.
4. **The budget ceiling.** The session token/context budget and the KV-block
   residency budget are independent; N sessions can oversubscribe GPU KV in a way
   the session budget never sees.

## Where it is advisory-only today

The halves are wired, but the wire gates nothing on a live served batch:

- **`TurnIntent`** (the #805/#807 conduit) is consumed — by `compute/prewarm_admit.go`,
  `compute/discard_admit.go`, and `radixkv/prewarm.go` — but **not by anything that
  gates a live served-batch decision**, because `NativeScheduler` is a shape-proof
  stub and the in-kernel served device path serializes whole forwards under one
  mutex (see `L4-INKERNEL-SERVE-AND-CONCURRENCY-FIX-2026-06-25`).
- **`SlotEvent`** (drain/stop "a slot freed") is consumed only by the in-memory
  session scheduler. Nothing routes it to `kvmmu`/`KVCache` eviction.
- **`State.Priority`** drives `Pick`, but `Pick`'s winner is the agent-loop turn
  gate, not a GPU batch admission.
- **`ResetScore`** (`gateway/reset_score.go`) computes the cut-vs-reset verdict but
  ships shadow / recommend-only; nothing acts on it.

## The fusion seams worth building

Buildable places where collapsing the two halves pays off, each grounded in shipped
primitives:

1. **Shared lifecycle vocabulary.** One state package the supervisor (`LoopState`)
   and the served session (`RunState`) both speak, with an explicit converter. This
   is the literal dedupe that grounds the whole "one machine" claim.
2. **One epoch lineage.** Make `continuationID`/`Generation` and the `abi` epoch
   family one id space with a converter, so a `Recontinue` mints a KV epoch and a
   speculative branch is a generation under the same lineage. Additive mapping; no
   change to the frozen ABI.
3. **`SlotEvent` drives real KV reclaim.** Route a boundary-taken drain/stop to an
   actual `kvmmu.EvictColdest` / `KVCache.Evict` on the served path, so "a session
   slot freed" *is* "a KV block freed" for a waiting sequence. Gated behind the
   in-kernel KVMMU flag; advisory-degrading when off.
4. **Warm-swap KV on `Paused→Running`.** Build the (currently untracked) live-resume
   loop — block on Paused, re-`Decide`, splice warm KV (`KVCache.Clone` /
   `cachemeta.MoveTo(KVRestore)`) — so a resumed session reuses warm KV instead of
   cold re-prefilling. This is the buildable half of the swap-in parallel.

Two further convergence points are real but **owned elsewhere**, and this note files
them as dependencies, not new work:

- *One admission table* (`Table.Snapshot` becomes the table the continuous batcher
  reads) converges with the engine in **#401** and the deep half of **#805**; the
  scheduler that already consumes `Snapshot` shipped in the closed **#627**. The
  unclaimed nuance is only that today's consumer is the agent-loop turn gate, never
  the GPU continuous-batch admission table.
- *Auto-firing cut-vs-reset* (promote `ResetScore` from shadow to acting, splice a
  warm prefix) is **#774**'s Phase 2, building on the closed **#739** reset
  mechanism.

## Honest fences

- **This is a naming / unification claim, not a new mechanism.** Every seam reuses
  shipped primitives. If it cannot dedupe a real twice-built thing or wire a real
  advisory edge, it ships nothing.
- **The transition sets are not isomorphic, and we do not force them equal.** Serving
  has preempt/evict the session machine lacks; the session machine has
  pause/resume/throttle a sequence-to-EOS lacks. The claim is a shared *skeleton*
  plus a shared *quantum* (the turn), not identical verb sets.
- **Cross-host live migration is an analogy only.** `sessionimage` rehydrate records
  a migration, but `KVIncluded=false` by design — "offload and bring back" is
  freeze-to-disk + cold re-prefill, never an in-flight KV swap. Out of scope.
- **Advisory stays advisory.** `TurnIntent`/`Goal`/reservations never gate
  correctness; every fused edge degrades to the GPU-visible decision (the #805 fence
  is inherited, not broken).
- **Boundary discipline is preserved.** Stop/evict is taken at the next turn
  boundary, never mid-decode.

## Position vs prior art

The nearest open epics each preserve the assumption this note denies — that there
are **two** machines:

- **#805 (the intent conduit)** is the explicit inverse: it fences itself as "the
  information edge *between*" two distinct decision-makers and pipes advisory
  `TurnIntent` into a separate GPU scheduler that still owns admission. It keeps two
  machines and wires a conduit. This note says there is one machine at two altitudes.
- **#748 (agent-OS)** frames the session as a PCB and builds a scheduler that *reads*
  it (the closed #627), but that scheduler picks among live agent sessions and is a
  wholly separate object from the GPU continuous-batch scheduler; #748 never
  identifies the two.
- **#637 (shared spine)** names a *serving* request lifecycle seam
  (admit→step→stream→reclaim) but never equates it with the *session* lifecycle, nor
  makes the turn the admission quantum.
- **#809 (speculative loop)** is about the *next* turn arriving hot, not about
  declaring turn-start and serving-admit the same operation.
- **#844 (reachability GC)** is about what stays *live* in the context heap, not
  whether session-resume and serving-resume are one admission event.

**The unclaimed gap** is the isomorphism itself: the explicit assertion that the
session lifecycle and the serving admit/preempt/evict/resume operation are facets of
one state machine whose quantum is the turn — turning the currently-separate
`session.scheduler` and the (unbuilt) GPU batch scheduler into one governed
continuous-batch admission machine over flexible-turn sessions. The frozen ABI even
reserves the substrate (`SpeculationContext`/`Outcome`/`ProvisionalSink`, `TxnID`
with KV checkpoint-rollback), yet nothing declares the unifying machine.

## The proposed epic

Filed as **#912** (`epic(session): one machine — fuse the agent-session lifecycle and
serving admission into one turn-quantum state machine`). Carries no new compute. Four
children, each a real dedupe or a real advisory-edge wiring:

1. **#913** — a shared lifecycle-state vocabulary spanning `RunState` and `loopmgr.LoopState`.
2. **#914** — unify `ContinuationID`/`Generation` with the `abi` epoch lineage (converter only).
3. **#915** — drive real KV reclaim from a drain/stop `SlotEvent` (gated behind in-kernel KVMMU).
4. **#916** — warm-swap KV on `Paused→Running` instead of cold re-prefill (the untracked
   live-resume loop).

It explicitly does **not** re-file #805's conduit, #748's scheduler, #401's batcher,
or #739's reset, and it names #401/#805-deep (admission table) and #774 (auto-fire
reset) as the two convergence points owned elsewhere.

## Non-goals / what stays where

- No new GPU batcher — that is #401/#637.
- No live cross-host migration — `KVIncluded=false` stands.
- Advisory hints stay advisory; correctness never gates on a hint.
- The GPU continuous-batch scheduler remains #401's deliverable; this epic only makes
  the unified vocabulary it would read.

## See also

- `SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md` — the drive-state table.
- `BUDGET-TRIGGERED-SESSION-RESET-2026-06-25.md` — `Recontinue` and the reset seam.
- `PORTABLE-SESSION-IMAGE-AND-SNAPSHOT-2026-06-24.md` — freeze/rehydrate (`KVIncluded`).
- `THROUGHPUT-TRUST-SHARED-SPINE-2026-06-24.md` — the serving request lifecycle spine (#637).
- `FUSION-REFCOUNT-GC-FOR-THE-AGENT-LOOP-2026-06-25.md` — the context-heap GC sibling (#844).
