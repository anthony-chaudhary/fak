---
title: "Generation Loop Scheduling"
description: "Scheduler contract for running now, next, second-next, and future generation loops concurrently on one shared trunk."
---

# Generation Loop Scheduling

**Issue:** #1654.
**Stream:** `gen/second-next`.
**Status:** scheduler design contract for concurrent generation loops.

This page is the continuation packet for agents working on generation-aware
dispatch. It narrows the scheduler behavior without requiring a reread of the
whole generation epic.

## Core Rule

Generation is a scheduling hint, not a queue silo.

The dispatch loop may use generation labels to choose which readiness checks and
operator gates apply, but it must still rank concrete work by lane pressure,
priority, path safety, and witness quality. A `gen/future` item can be urgent to
study, and a `gen/now` item can wait if it has no safe path or witness.

Generation remains orthogonal to the three other controls:

- Priority answers which admitted item matters most.
- Shared trunk answers where every commit lands: `main`, explicit path, signed,
  and witnessed.
- Runtime feature gates answer whether shipped code is reachable by users.

No generation stream authorizes a feature branch, side worktree, broad staging
sweep, force-push, or un-gated runtime exposure.

## Scheduler Buckets

Each open generation issue maps to exactly one scheduler bucket after routing:

| Bucket | Default launch policy | Required evidence before launch |
|---|---|---|
| `gen/now` | Eligible in the normal dispatch loop. | Concrete issue scope, expected paths, and the ordinary focused witness. |
| `gen/next` | Eligible when its write paths are scoped and runtime exposure is inert or gated. | A gate, schema, dogfood route, dry-run mode, or no-op scaffold that keeps current users safe. |
| `gen/second-next` | Held for design, compatibility, simulation, or operator-approved waves unless explicitly requested. | A simulation, compatibility policy, dependency edge, or scheduling contract that can be promoted into `gen/next`. |
| `gen/future` | Read-only or planning-only by default; launch write workers only with an override. | A research memo, standards analogue, assumption ledger, or decision model with a named consumer. |

The buckets can run in the same dispatch window only when their path leases are
disjoint. Different generation labels do not make overlapping trees safe.

## Scheduling Algorithm

A generation-aware scheduler should apply this order:

1. Read the issue labels, milestone, body, expected paths, and routed lane.
2. Refuse or hold an issue with no concrete witness, no expected path scope, or
   mismatched generation label and milestone.
3. Classify the generation bucket and required exposure gate.
4. Run the lane/path arbitration check before launch.
5. Build a wave from disjoint leases, then sort within the admitted set by lane
   pressure, priority, stale-risk, and witness strength.
6. If two admitted items collide, launch the item whose witness retires the
   stronger current blocker and mark the other `collision_deferred`.
7. Record a status row that names issue, generation bucket, lane, expected paths,
   launch decision, blocker reason, override reason if any, and witness.

The scheduler should not run four independent global queues. That hides shared
path contention and creates stale work in the later streams. One global
admission pass with generation-aware gates keeps the trunk collision model
visible.

## Contention Handling

| Condition | Scheduler action |
|---|---|
| Two generation streams want the same path tree. | Serialize by lease. Prefer the issue with the stronger witness or operator priority, and mark the other `collision_deferred`. |
| A later-horizon item needs a current runtime path. | Require a default-off feature gate, dry-run command, fixture-only path, or explicit operator override before launch. |
| A `gen/second-next` or `gen/future` item has no consumer. | Hold it as planning work; do not launch a write worker until it names a downstream decision or interface. |
| A label/milestone mismatch appears. | Hold for intake repair; do not guess the stream from the title. |
| The issue is an epic rather than a leaf. | Require a child issue or a planning artifact. Do not hand the whole epic to a worker as a write task. |

Contention is a scheduling result, not a failure. A clean `collision_deferred`
row is better evidence than two workers racing the same tree.

## Operator Overrides

An override may admit a held `gen/second-next` or `gen/future` worker, but the
override must be explicit and expiring. Record:

- The issue and stream.
- The narrow write paths or read-only scope.
- The reason the work must run in this window.
- The runtime exposure gate, or why the artifact is inert.
- The witness that will promote, demote, retire, or re-hold the item.
- The expiry condition: one wave, one commit, one report, or one named deadline.

An override changes launch eligibility only. It does not change priority,
shared-trunk rules, runtime exposure, or the required witness.

## Promotion And Retirement

Promotion evidence for scheduler behavior is concrete:

- A dry-run or live dispatch readout shows a later-horizon item launched safely
  because its path lease, exposure gate, and witness were all present.
- A compatibility simulation or dependency edge lets a `gen/second-next` item
  become `gen/next` foundation work.
- A planning artifact is consumed by a command, issue view, route policy, or
  operator report.

Demotion or retirement evidence is equally concrete:

- The expected path repeatedly collides with current product work.
- The runtime gate leaks behavior into current users.
- The witness is stale and no cheap re-witness path exists.
- A simpler shipped design supersedes the option.
- The invalidating assumptions below fail.

Do not promote, demote, or retire by label movement alone. The issue comment or
commit sidecar must name the witness and the assumption it changed.

## Invalidating Assumptions

This scheduling contract depends on these assumptions:

- Generation labels and milestones stay cheap for agents to read.
- `dos arbitrate` regions are granular enough to separate real concurrent work.
- Operators can identify runtime exposure gates before launch.
- Issue bodies usually carry enough path and witness detail to classify a leaf.
- A single admitted-wave scheduler stays clearer than independent per-generation
  queue workers.

If any assumption fails, update this contract or demote the mechanism. Do not
work around the failure by creating generation branches.

## Future Implementation Hooks

The next code slice should be small and witnessed. Reasonable options:

- Add a generation decision column to `fak dispatch route --json`.
- Add a dry-run `fak dispatch tick` readout that explains held
  `gen/second-next` and `gen/future` issues.
- Add a fixture test proving a wave admits disjoint `gen/now` plus `gen/next`
  work and defers an overlapping `gen/future` item with `collision_deferred`.

Use [`docs/generation-concurrent-trunk-playbook.md`](generation-concurrent-trunk-playbook.md)
for the branchless team rules and [`docs/dispatch-loop.md`](dispatch-loop.md)
for the live issue-dispatch pipeline.
