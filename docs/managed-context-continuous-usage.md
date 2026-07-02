---
title: "Managed-context continuous usage semantics"
description: "The product contract for long fak sessions that cross hidden context resets: what continuity means, what evidence is surfaced, and when fak asks instead of guessing."
---

# Managed-context continuous usage semantics

*Program: [managed-context](https://github.com/anthony-chaudhary/fak/issues/1570). Issue:
[#1578](https://github.com/anthony-chaudhary/fak/issues/1578).*

Continuous usage means a user can keep working through a long agent run without becoming
the context-window manager. fak may re-plan the resident view, page cold spans back in,
reset a model trace, or route through a warmer cache path; the user-facing promise is that
the objective, budget, assumptions, and evidence survive those moves in a form the user can
inspect.

This is not a promise that every byte stays resident forever. It is a promise that a hidden
context change is either behaviorally neutral or visible as a typed decision with evidence.

## The Contract

1. **The objective stays stable.** The pinned objective is carried across replans, resets,
   and session migrations by identity and digest, not by a model-authored summary. If the
   carried objective no longer matches what the user stated, fak surfaces an objective
   reconciliation outcome instead of treating the rewrite as continuity.
2. **The full history remains recoverable.** The resident view is a bounded working set over
   a lossless store. A span that leaves the current view is cold, not forgotten. If a later
   turn needs it, the page-in path produces a typed page-fault outcome rather than silently
   pretending the span was never needed.
3. **A reset carries witnessed evidence.** A reset transaction records the old trace, new
   trace, carryover seed digest, budget rearm, contributors, and omitted-span digests with
   reasons. A user or operator can ask what survived, what did not, and why.
4. **Budgets do not reset accidentally.** Token, turn, context, clarification-query, and
   wall-clock budgets move forward across a hidden reset. Resetting the context trace does
   not mint a fresh runway or erase a spent one.
5. **Assumptions are scored before use.** A context plan may rely on an assumption only after
   the assumption's source and confidence clear policy. Low-confidence, stale, inferred, or
   unknown assumptions turn into a query or refresh action before they can support an
   effectful step.
6. **Memory promotion is separate from context survival.** A fact can stay useful in current
   context without becoming durable memory. Durable promotion requires its own evidence; an
   unclassified observation defaults to expiring, not persisting.
7. **Cache state is never authority.** Provider prompt-cache warmth can lower cost and
   latency, but it never decides whether a fact may be omitted. A cold cache is a pricing
   event, not a continuity failure.
8. **Relay mode is pointer-or-nothing.** A relay leg may carry forward durable pointers,
   not a transcript summary. A load-bearing fact is either externalized and queryable by
   the next leg, or it does not survive the rotation.

## What Fak Surfaces

The continuous-usage surface is a small evidence bundle, not a transcript dump:

| Surface | What it proves | Code anchor |
|---|---|---|
| Pinned objective | The active goal survived unchanged, or drift was detected | [`ctxplan.ObjectivePin`](https://github.com/anthony-chaudhary/fak/blob/main/internal/ctxplan/objective.go) |
| Resident view plan | Which spans are resident, which are elided, and why | [`ctxplan.PlanView`](https://github.com/anthony-chaudhary/fak/blob/main/internal/ctxplan/query.go) |
| Page-fault outcome | Whether a cold span was served back in or refused | [`ctxplan.PageFaultOutcome`](https://github.com/anthony-chaudhary/fak/blob/main/internal/ctxplan/pagefault.go) |
| Reset transaction | What the reset carried, omitted, and rearmed | [`session.ResetTransaction`](https://github.com/anthony-chaudhary/fak/blob/main/internal/session/reset_transaction.go) |
| Budget envelope | Which budget axis is near exhaustion or exhausted | [`session.Budget`](https://github.com/anthony-chaudhary/fak/blob/main/internal/session/session.go) |
| Time budget | How wall-clock time carries across resets | [`session.TimeBudget`](https://github.com/anthony-chaudhary/fak/blob/main/internal/session/timebudget.go) |
| Assumption report | Which assumptions can be used, queried, or refreshed | [`ctxplan.AssessAssumptions`](https://github.com/anthony-chaudhary/fak/blob/main/internal/ctxplan/assumption.go) |
| Promotion record | Why a fact did or did not cross into durable memory | [`memq.PromotionRecord`](https://github.com/anthony-chaudhary/fak/blob/main/internal/memq/promotion.go) |

The intended product shape is simple: a long run can show the current objective, remaining
budget envelope, active resident view, unresolved assumptions, last reset transaction, and
recent memory-promotion decisions without asking the user to read the whole prompt.

## Relay Mode: No Compaction

Relay mode is the stricter continuous-usage contract planned by the perpetual-sessions
spine. It is not enforcement yet; it is the product promise future relay rungs must satisfy.

- **No-compaction promise:** relay mode never asks a model to summarize the old transcript
  and treat that summary as state. A closing leg writes a baton of durable pointers; the
  transcript is disposable once those pointers pass the externalize gate.
- **Pointer-or-nothing:** every load-bearing fact must be in a durable store first — a
  commit, ledger row, memory slug, issue, file path, or other witnessed reference. If a
  fact lives only in the transcript, rotation is refused rather than laundered into prose.
- **Flat-context invariant:** peak resident context is bounded by the per-leg envelope, not
  by total goal duration, number of rotations, or total work done. A relay that runs for a
  week should peak no higher than one leg with the same budget.
- **Re-derive by query:** a successor leg uses the baton to re-query the durable store and
  re-verify progress. The baton is an index and lineage record, not a trusted progress
  report.

This is stricter than ordinary hidden reset continuity. A reset may carry a deterministic
seed built from approved contributors; a relay rotation carries only the baton and the
ability to query durable evidence. Model-written recaps are not a third state.

## When Fak Asks Instead Of Guessing

fak should ask the user, or surface an explicit host decision, when the expensive error is a
silent wrong assumption:

- **Objective drift:** the carried objective's digest or identity no longer reconciles with
  the user's stated objective.
- **Low-confidence assumption:** an inferred, stale, unknown, or below-threshold assumption
  would affect an effectful action.
- **Ambiguous memory promotion:** a live observation looks useful but has not earned durable
  status or corroborating evidence.
- **Budget exhaustion:** the run has spent a configured turn, token, context, clarification,
  or wall-clock envelope and needs a pause, reset, or stop decision.
- **Unserved page fault:** a needed cold span cannot be safely or cheaply restored into the
  resident view.
- **Destructive memory operation:** a delete, reset, namespace wipe, or standing-profile
  rewrite requires a capability beyond ordinary recall.

The opposite cases stay automatic: replanning the resident view under budget, serving a
recoverable cold span, carrying the pinned objective intact, rearming the same remaining
budget after a reset, and preserving cache affinity when it is only a cost optimization.

## What The User No Longer Manages

The user should not have to say "summarize our context," "keep that in the prompt," or
"remember that the old reset had my real goal." Those are managed-context responsibilities:

- choose the bounded resident view for the next turn;
- keep dropped spans recoverable by digest;
- carry the pinned objective and remaining budget through hidden resets;
- show the reset transaction when continuity depends on it;
- query or refresh assumptions before using weak evidence;
- keep context-only facts out of durable memory unless promotion is justified.

What remains user-controlled is the actual objective, explicit facts, budget limits, pins,
layout preferences, and any approval to promote or delete durable memory.

## Operator Readout

After a hidden reset or long-run handoff, the minimum honest readout is:

```text
objective: preserved | established | drifted
budget: turns/tokens/context/queries/time remaining
resident_view: selected=<n> elided=<n> page_faults=<served/refused>
assumptions: use=<n> query=<n> refresh=<n>
reset: old_trace=<id> new_trace=<id> seed_digest=<sha256> omitted=<n>
memory: promoted=<n> refused_or_expired=<n> ambiguous=<n>
cache: warm | cold | unknown (cost only)
```

That readout is the user-facing contract in one screen. If it cannot be produced from
witnessed records, the system should say `not yet` rather than claim continuity.

## See Also

- [`managed-context-glossary.md`](managed-context-glossary.md) - the vocabulary this page
  relies on: assumption, resident view, pinned objective, budget envelope, reset
  transaction, context query, memory promotion, cache state, and relay vocabulary.
- [`notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md`](notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md) -
  the relay concept spine for flat-context, no-compaction perpetual sessions.
- [`CONTEXT-IS-NOT-MEMORY.md`](CONTEXT-IS-NOT-MEMORY.md) - why context survival and durable
  memory promotion are separate decisions.
- [`explainers/o1-context-window-economics.md`](explainers/o1-context-window-economics.md) -
  why a bounded resident view is cheaper than carrying a full transcript forever.
