---
title: "Plan mode is a control/data-plane split — extend it 100×"
description: "The third axis of fak's managed context. Autoregression fuses the plan and the actions into one token stream; that fusion is why a long session's context grows without bound. Plan mode is the first crack — it separates the durable, cheap, reviewable PLAN from the ephemeral, expensive EXECUTION. ctxplan already carries both halves as mechanism: objective.go is the control plane (a pinned span reconciled byte-for-byte, drift is a refusal), plan.go is the data plane (a knapsack of freely-elided candidates with page-back-in handles). The seam 'a pin is not a candidate' is plan mode generalized. Extending it 100× means the split is permanent (not a mode), many executions share one control plane (not one plan per run), and the control plane is itself reconciled (not merely held). This axis is what makes the temporal and magnitude axes safe: it says which spans must survive the baton and which are free to re-derive."
date: 2026-07-11
---

# Plan mode is a control/data-plane split — extend it 100×

Status: concept note. The **third** axis of a trilogy on fak's managed
context, each extending one property autoregression gives up:

| Axis | Question | Note | ctxplan seam |
|---|---|---|---|
| **Temporal** — how *long* | the session outlives any single window | [CONCEPT-PERPETUAL-SESSIONS](CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md) (+ [RELAY-BATON-SCHEMA](RELAY-BATON-SCHEMA-2026-07-01.md)) | throw the transcript away, keep the re-derivable store |
| **Magnitude** — how *wide* | one store, many views at marginal cost | [CONTEXT-VIEWS-AT-MARGINAL-COST](CONTEXT-VIEWS-AT-MARGINAL-COST-2026-07-04.md) | re-slice the same lossless image without rebuilding it |
| **Control/data** — what *kind* | the plan is a different substance from the work | **this note** | `objective.go` pin vs `plan.go` knapsack |

The first two answer *when* and *how much*. This one answers *what is a span
FOR* — is it **intent** (control) or **work-product** (data)? — and it is the
axis that makes the other two honest, because it tells you which spans must
survive the baton and which are free to be re-derived on demand.

Nothing ships from this note. Every mechanism it names is a file at HEAD; every
gap is called out as unbuilt.

## The seed: what autoregression fuses, and what plan mode cracks

Autoregression is single-threaded in a way we rarely name: there is **one**
token stream, and *deciding what to do* and *doing it* are the same sequence of
tokens sweeping the same attention. The plan ("I will edit three files, then run
the tests") and the execution (the three diffs, the 4 KB test log, the stack
trace) live in one undifferentiated ledger. That fusion is the root cause of
unbounded context growth: every turn re-encodes the *control* decision as data
tokens, and every *data* token — a tool result, a file dump, a retry — sits in
the same stream as the decisions, so the two grow together and neither can be
evicted without touching the other.

**Plan mode is the first crack in the monolith.** It separates authoring the
plan (the todolist — small, durable, the user's intent) from executing it (the
actions — large, ephemeral, the mechanism). In plan mode you write the plan
*without* running it; the plan is a distinct artifact reviewed at a fraction of
the tokens the execution will cost. That is a **control-plane / data-plane
split** by another name — the same split a database draws between the query
*plan* (EXPLAIN: tiny, legible, the intent) and the query *execution* (the
buffer-pool churn: huge, ephemeral, the mechanism). `ctxplan/doc.go` already
leans on exactly this Postgres correspondence for the magnitude axis; this note
observes that plan mode is the *control* half of the same lens.

## The split already exists in the tree, as two mechanisms and one seam

fak did not have to invent the two planes — `internal/ctxplan` already ships
both halves, built for different issues, and the seam between them is razor-sharp.

**Data plane — `plan.go`.** The execution transcript is a knapsack. A
`Candidate` carries a `Cost` and a `Benefit`; `Optimize` keeps the densest set
that fits a `Budget` and moves the rest to cold storage as an `Elision` with a
`Digest` page-back-in handle. A forecast miss is a *page fault, not a lost fact*
(`faithful.go` proves Selected+Elided partition every candidate). This is the
plane you want cheap, verbose, and disposable: work-product that can be
re-derived exactly whenever it is referenced again.

**Control plane — `objective.go`.** The active objective is an `ObjectivePin`:
a stable `PinID` assigned once and a `Digest` over its content.
`ReconcileObjective` compares the pin before and after every replan/reset and
returns exactly one closed-vocabulary outcome — and here is the load-bearing
difference from the data plane: **drift is a refusal.** If the identity survives
but the content changed (`ObjectiveDrifted`), or the pin vanished
(`ObjectiveDropped`), the caller must surface it, never proceed. A data-plane
span *wants* to be elided and re-derived; a control-plane span must survive
*byte-for-byte* or raise. Same store, opposite retention contract.

**The seam, in one sentence from the source** (`objective.go`):

> A pin is deliberately NOT a Span: it is not a candidate for the resident-view
> knapsack (Optimize never scores it, Candidates never lists it) — it is
> out-of-band structural state the planner and the reset both consult.

*"A pin is not a candidate"* is plan mode's *"the plan is not the actions,"*
already compiled into the type system. The control plane sits **outside** the
knapsack the data plane is optimized by. That is the whole idea, and it is real
today.

## What "extend it 100×" means — three multiplications

Plan mode as shipped is one plan, one execution, entered and left as a UI mode.
The 100× is not a bigger todolist; it is removing three ceilings:

1. **Permanent, not modal.** The split stops being a mode you toggle and becomes
   how *every* turn is planned: control-plane spans are pinned and reconciled;
   data-plane spans are knapsacked and freely elided. You never "leave plan
   mode" because the plan is never fused back into the transcript. `objective.go`
   already reconciles on *every* replan — the mechanism is continuous, only the
   product surface is still modal.

2. **Many executions, one control plane.** Autoregression gives one execution
   per plan. Un-fused, the control plane is a *shared* object: many executions —
   successive turns, a fan-out of sub-agents, a relay across a reset — read from
   and write back to the same pinned objective **without carrying each other's
   data plane.** A fleet can share an objective without sharing a transcript.
   This is the mechanistic root of the fleet/sidecar asks in
   [CONCEPT-CONTROL-SIDECAR-AUTOCTX](CONCEPT-CONTROL-SIDECAR-AUTOCTX-2026-07-01.md):
   that note draws the *product* boundary (which knobs the user owns); this one
   draws the *architectural* boundary the product boundary rests on. "Control
   exercised once applies everywhere; an item is the same item everywhere" is
   only possible because the control plane is a small object separable from the
   execution that produced it.

3. **The control plane is itself reconciled, not merely held.** The failure that
   sends long autonomous runs off the rails is not a lost tool result — that
   pages back in — it is a silently *rewritten objective*: the agent paraphrases
   or truncates what it is doing under a preserved identity and no longer knows
   it drifted. `objective.go`'s `ObjectiveDrifted`/`ObjectiveLog.Replay` make
   that drift a checkable, refusable event rather than a narrative one. Extended
   100×, *every* control-plane item — not just the single objective span —
   carries this contract: a structured, multi-item plan whose every step is
   digest-pinned and reconciled across replans, so "the plan survived" is an
   equality check, never a hope.

## Why this axis makes the other two safe

The temporal and magnitude axes are both *lossy-looking* operations made honest
only by knowing what may be dropped:

- **Temporal (perpetual sessions).** The baton throws the transcript away and
  keeps pointers. That is only safe if the objective pin is *the baton's most
  important field* — the one thing carried byte-for-byte while everything else
  becomes a re-derivable pointer. The control/data split is what lets the baton
  say "this survives, that re-derives" instead of guessing.
- **Magnitude (context views).** Re-slicing one store into many marginal-cost
  views is only safe if slicing can never strip a control-plane span. The knob
  census / housekeeping doctrine
  ([CONCEPT-AUTOMATIC-CONTEXT](CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md)) is
  precisely the rule "housekeeping may evict data-plane spans automatically;
  intent-plane spans it may not touch." The litmus test in that program — *a
  knob belongs to the user iff it encodes intent the system cannot infer* — is
  the control/data boundary wearing a product hat.

So the trilogy composes: the **control plane** is what a **temporal** baton
carries and what a **magnitude** view must never strip; the **data plane** is
what both axes are free to re-derive. Name the two substances and the other two
axes stop being dangerous.

## The unbuilt rung (honest gap)

Today the control plane is exactly *one* reconciled span — the single objective
pin (`objective.go`, #1583). The 100× version is a **structured plan** as a
first-class control-plane object: an ordered set of steps/todos, each with its
own `PinID`+`Digest`, each reconciled across every replan, each with a typed
disposition (preserved / advanced / dropped / drifted / query-user) reusing the
exact closed-vocabulary + `Replay` discipline `ReconcileObjective` already
proves out for the singular case. That object is what plan mode's todolist would
*be* if it were promoted from a UI artifact to a durable, addressable,
fleet-shared spine. It does not exist yet; `objective.go` is the single-item
proof that its contract is buildable without inventing new machinery.

## Honesty fences

- Nothing here ships by being written. `plan.go` (data plane) and `objective.go`
  (control plane) are real at HEAD; the multi-item structured-plan control
  object of the "unbuilt rung" is not — it is the single-objective mechanism
  generalized, and would land behind its own witness.
- The two planes share the *same trust boundary*: a sealed/tombstoned span is
  never resident in the data plane (`ElideSealed`/`ElideTombstoned`) and a pin
  that fails its own `Verify` is never trusted as preserved (`ObjectiveQueryUser`).
  The split adds a retention *contract* per plane; it does not add a trust
  bypass to either.
- This is the *mechanism* framing of the control/data boundary;
  [CONCEPT-CONTROL-SIDECAR-AUTOCTX](CONCEPT-CONTROL-SIDECAR-AUTOCTX-2026-07-01.md)
  is the *product* framing (INTENT / LEGIBILITY / HOUSEKEEPING planes). They
  describe the same seam from opposite ends; neither supersedes the other.
- "100×" is a direction, not a measured factor. The claim is architectural
  (three ceilings removed), not a benchmark; no resident-token curve is asserted
  here that `scaling.go` does not already measure for the magnitude axis.
