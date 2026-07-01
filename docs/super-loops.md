---
title: "fak super loops: the operator-intent meta-loop that walks a set of loops"
description: "A super loop is keyed on an operator intent (improve quality), not a task: it walks its member loops/scorecards/gardens to read their status first, selects the worst-first member to enter, and exits on the aggregate clearing — the layer above a normal loop. Five properties separate them, and fak superloop makes the distinction executable."
---

# Super loops (`fak superloop`)

> An operator says **"improve quality"**. That is not one task — it is a standing
> intent that spans a dozen loops: the code-quality scorecard, the slop scorecard,
> the disambiguation scorecard, the gardening bundle, and more. A **super loop** is
> the thing that takes that intent, **walks those loops first to read their status**,
> and tells you **worst-first what to enter** — before it does any work.

The fleet already runs many loops. The [issue-dispatch loop](dispatch-loop.md)
resolves one issue per tick; the [RSI loop](rsi-loop.md) keeps-or-reverts one
candidate; the garden tick reaps one class of stale work; a scorecard run reports one
debt number; `fak loop drive` settles one `GOAL.md` witness. Each is a **normal
loop**: keyed on a *task* and a cadence, its tick *does* one concrete thing, and it
is a **leaf** in the work graph — it acts on the codebase or the world directly.
When a super loop spans generation-labeled work, use the
[generation super-loop budget contract](generation-super-loop-budgets.md) to keep
time, token, worker, and review capacity explicit without turning generation into
priority, a branch, or a runtime gate.

A **super loop** sits one altitude up. It is keyed on an operator **intent**, and its
tick is a **traversal over other loops**, not a task.

## The differentiation — five properties

`fak superloop explain <name>` prints this table for a registered intent, side by side
with a normal loop. The classification is executable (`internal/superloop.Classify`):
a loop is a super loop **iff all five hold**.

| Property | Super loop | Normal loop | What it means |
|---|:--:|:--:|---|
| `has_members` | yes | no | it walks ≥1 member loop; a normal loop has none |
| `walks_first` | yes | no | its tick **reads** each member's status before acting (orient-over-loops) |
| `selects_worst_first` | yes | no | it **selects** which member to enter, worst-first; a normal loop just runs its body |
| `exits_on_aggregate` | yes | no | it exits when the **fold** clears (aggregate debt ≤ floor), not on a single task's witness |
| `interior_node` | yes | no | it **mutates nothing at its own altitude** — only its members mutate the world |

The load-bearing line is the last one. A normal loop is a **leaf**: its unit of work
is a task, and it changes the repo. A super loop is an **interior node**: its unit of
work is *another loop*, and its only effects are *reading* members and *driving* them.
That is why a super loop is safe to run as a read-first orientation pass — by
construction it has nothing to commit.

## What the walk does — the four moves

```
fak superloop walk improve-quality
```

1. **WALK** — read each member's status. Cheaply and honestly: a scorecard member's
   debt comes from the pinned control-pane baseline (`tools/scorecard_baseline.json`,
   the last *measured, committed* value); a loop member's live/stale/dark state comes
   from the cross-ledger loop-health fold (`internal/loopfleet`). A member whose
   status cannot be read is surfaced as **unmeasured**, never silently treated as
   clean.
2. **SELECT** — fold the members worst-first. Dark or unmeasured leaves rank first
   (a gone-dark loop or an unknown status is the most urgent thing to enter), then by
   debt descending. The result is a **worklist**, not a pass/fail.
3. **DESCEND** — a member that is itself a container (the **garden** bundle,
   another **super loop**, or a domain-specific command surface) is surfaced as a
   *descend pointer*: its status is only knowable by walking it in turn. This is the
   recursion — *loops that themselves have many loops* — and it is the named
   follow-on the walk hands you.
4. **FOLD** — the intent is **satisfied** only when the aggregate debt is at-or-below
   its floor **and** every member was measured **and** none is dark. An unread or dark
   member can never let the intent read as done.

A real walk on this repo:

```
superloop walk: improve-quality — ACTION (superloop_debt)
  aggregate debt 657 (floor 0)  members 8  walked 7  unmeasured 0  dark 0

  worst-first — enter these in order:
  #  MEMBER                    DEBT  ACTION
  1  scorecard slop            600   enter the slop scorecard's reduce loop (its skill)
  2  scorecard disambiguation  30    enter the disambiguation scorecard's reduce loop
  3  scorecard code            24    enter the code scorecard's reduce loop
  4  scorecard ui_quality      3     enter the ui_quality scorecard's reduce loop
  5  garden garden             →     run `fak garden` then `fak garden tick`

  → worst-first: scorecard "slop" — enter the slop scorecard's reduce loop (its skill)
```

## How a super loop relates to what already exists

A super loop **generalizes the garden bundle** (`internal/gardenbundle`). The garden
is a *fixed* bundle of members folded into one OK/RED **gate**. A super loop is an
**intent-named**, **worst-first-selecting**, **recursively-nestable** bundle whose
members are themselves loops/gardens/scorecards, and whose output is a **worklist**
(what to enter next) rather than only a pass/fail. The status reads it folds are the
same ones the fleet already computes — `loopfleet` for loop health, `scorecardpane`
for scorecard debt — so a super loop adds a *view and a selection*, never a new
oracle.

## The registry

Super loops are **data** (`internal/superloop`). Each binds an operator intent to an
ordered member set; every scorecard member references a real control-pane card key
(a no-drift test enforces it). The seed set:

- **`improve-quality`** — the quality-bearing scorecards (slop, code, disambiguation,
  conflation, intent-literal, ui-quality, claim-repro) + the gardening bundle.
- **`improve-loops`** — the loop-index scorecard + the dogfood scorecard + the live
  loop ledgers (dispatch, cadence, dojo) + the gardening bundle.
- **`manage-benchmarks`** — the benchmark-DX scorecard + the `nightrun` collection
  loop + a descend pointer into `fak bench-loop status`, the benchmark-specific
  control surface.

For benchmark work, use the generic intent to orient across loop health and debt:

```
fak superloop walk manage-benchmarks
```

Then descend into the benchmark surface when the worklist points there:

```
fak bench-loop status              # registry + run catalog + ledger + local next + authority gap
fak bench-loop next                # the single next benchmark-loop action
fak bench-loop walk                # map the benchmark surfaces to enter
fak bench-loop run --apply --loop  # delegate to the local nightrun collection loop
```

```
fak superloop list                  the named super loops + their members
fak superloop explain <name>        the five-property differentiation, super vs normal
fak superloop walk <name> [--json]  walk the members' status, fold the worst-first plan
```

## Honest scope

The `walk` reads status and folds the plan; it **mutates nothing**. Actually
*entering* (driving) the worst-first member loop — and *descending* into a container
member's own walk — is the named next rung, built on the existing `fak loop drive`
and the member surfaces. The walk is the orientation half the operator's intent needs
first: *understand the status of everything under this intent, and what to do next.*
