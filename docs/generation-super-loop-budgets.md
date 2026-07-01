---
title: "Generation Super-Loop Budgets"
description: "Budget contract for recurring super loops that must keep now work moving without starving next, second-next, or future generation streams."
---

# Generation Super-Loop Budgets

**Issue:** #1653.
**Stream:** `gen/second-next`.
**Status:** planning contract for recurring loop governance.

This page is the continuation packet for agents budgeting generation-aware super
loops. It defines the minimum budget rows a future command or report should
carry, without requiring a reread of the generation epic.

## Core Rule

Generation budgets reserve attention; they do not assign priority.

Priority still answers which admitted item matters most. Shared trunk still
requires every commit to land on `main` by explicit path with the normal witness.
Runtime feature gates still decide whether code is reachable by users. Generation
budgets decide how much recurring loop capacity is reserved for each horizon so
current work is not diluted and future-facing work is not silently starved.

No generation budget authorizes a feature branch, side worktree, broad staging
sweep, force-push, or runtime exposure without a gate.

## Budget Dimensions

Every recurring super-loop plan should make four budgets visible:

| Budget | What it caps | Required row |
|---|---|---|
| Time | Wall-clock window spent walking and descending into member loops. | `max_minutes`, `cadence`, and expiry. |
| Tokens | Model/context spend for planning, worker prompts, reviews, and synthesis. | `token_ceiling`, `prompt_budget`, and review allowance. |
| Workers | Concurrent agent seats or issue workers admitted by the loop. | `max_workers`, lane leases, and collision policy. |
| Review | Human or stronger-rung review slots consumed before promotion. | reviewer, review deadline, and escalation rule. |

A loop with no explicit row for one of these dimensions is unbudgeted for that
dimension. Treat that as a hold for later-horizon work and as an operator warning
for now-work.

## Default Envelope

The default recurring envelope is conservative:

| Stream | Default recurring share | Worker posture | Review posture |
|---|---|---|---|
| `gen/now` | Majority share until current red gates, regressions, or release blockers clear. | Launchable when scoped and witnessed. | Normal focused review or automated witness. |
| `gen/next` | Reserved minority share so near-term foundation does not starve. | Launchable when gated, dogfoodable, and path-scoped. | Review checks the gate, dogfood route, and backout path. |
| `gen/second-next` | Bounded design/simulation share. | Held by default except explicit architecture, compatibility, or scheduler waves. | Review checks promotion criteria and collision cost. |
| `gen/future` | Small research/option share with an expiry. | Planning or read-only by default. | Review checks the consumer decision and recheck date. |

The exact percentages belong in a future measured report, not this contract.
Until measurement exists, a super loop must at least report which streams received
capacity, which were held, and why.
The hardware-style sizing model for worker seats, GPU/model seats, tokens,
attention/context, review bandwidth, and lane leases lives in
[`docs/generation-agent-capacity-model.md`](generation-agent-capacity-model.md).

## Admission Rules

Before a recurring super loop spends budget on a generation item:

1. Confirm the issue has exactly one generation label and a matching milestone.
2. Confirm the item is a leaf, planning artifact, simulation, or explicit
   operator override. Do not spend worker budget on an unsplit epic.
3. Confirm the expected paths and run the same lane/tree arbitration used by
   dispatch.
4. Confirm runtime exposure is gated or inert for non-now code.
5. Record which budget dimension is being consumed and which witness will retire
   the spend.

If two streams compete for the same scarce budget, prefer the item whose witness
retires the strongest blocker. When that is unclear, emit a review escalation row
instead of letting the loop guess.

## Promotion And Retirement

Promotion evidence for a budget rule is a measured or witnessed reduction in
loop ambiguity:

- A super-loop walk reports per-generation time, token, worker, and review
  consumption without hiding held work.
- A `gen/next` reservation produces a scoped, witnessed artifact that reduces a
  repeated dispatch collision, missing gate, or dogfood gap.
- A `gen/second-next` design budget produces a compatibility or simulation row a
  `gen/next` issue can consume.
- A `gen/future` research budget produces a decision memo, standards signal, or
  assumption update before its expiry.

Demotion or retirement evidence is equally concrete:

- A stream repeatedly consumes budget without producing a witness.
- The reserved share displaces current red gates or release blockers.
- The runtime gate leaks behavior into current users.
- The review queue becomes the bottleneck and no stronger-rung review capacity
  exists.
- The invalidating assumptions below fail.

Do not promote, demote, or retire a budget by changing a label alone. Name the
budget row, witness, and assumption that changed.

## Invalidating Assumptions

This contract depends on these assumptions:

- Super-loop walks can read enough issue metadata to classify streams cheaply.
- Token and worker consumption can be attributed to a loop member or issue.
- Review capacity is visible enough to budget, not just discovered after a pileup.
- `gen/next` reservations reduce starvation without becoming priority laundering.
- Operators can tolerate a small future-facing research share with explicit expiry.

If an assumption fails, demote the budget rule or replace it with a measured
report. Do not create generation branches or hidden worker pools to escape the
budget.

## Future Implementation Hooks

A future implementation should add one narrow, witnessed surface:

- `fak superloop walk --json` can expose `generation_budget` rows per member.
- `fak dispatch wave --json` can report generation worker share and held counts.
- A scorecard can compare planned share against witnessed consumption for the last
  window.

The first useful fixture should prove that now/next/second-next/future rows are
rendered, later-horizon rows require expiry, and the report keeps priority,
shared trunk, and runtime gates separate from generation.
