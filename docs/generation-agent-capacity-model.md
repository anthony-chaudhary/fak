---
title: "Generation Agent Capacity Model"
description: "Hardware-style capacity model for sizing agent workers, token spend, GPU/model seats, attention, and review queues by generation stream."
---

# Generation Agent Capacity Model

**Issue:** #1662.
**Stream:** `gen/second-next`.
**Status:** planning model for generation-aware fleet capacity.

This page gives a future agent the capacity model needed to size generation-aware
dispatch waves and review queues without rereading the generation epic.

## Core Rule

Capacity is a constraint model, not a priority model.

Priority still answers which admitted item matters most. Shared trunk still
requires one `main` branch, explicit path commits, and witnessed closes. Runtime
feature gates still decide whether shipped code is visible. Generation capacity
answers which scarce resource each horizon consumes and what evidence would
increase, reduce, or retire that allocation.

Do not solve capacity pressure with generation branches, hidden worktrees, or
ungated runtime exposure.

## Capacity Resources

Agent-fleet capacity has hardware-style supply constraints:

| Resource | Hardware analogue | Agent constraint | Sizing signal |
|---|---|---|---|
| Worker seats | Build slots / validation benches | Concurrent headless workers and account seats. | live worker count, account availability, `max_workers`, cooldowns |
| GPU/model seats | Scarce accelerator time | GPU-backed serve, model routing, or high-cost judge/review runs. | queue depth, model availability, per-run cost |
| Token budget | Power / wafer starts | Prompt, context, synthesis, and review tokens consumed by a wave. | token ceiling, prompt size, retry count |
| Attention/context | SRAM/cache residency | Human and model focus available for one stream before context thrash. | issue count, stale witnesses, handoff length |
| Review bandwidth | QA and signoff benches | Human review, stronger-rung witness, or release readiness slots. | open review queue, reviewer expiry, gate latency |
| Lane leases | Floorplan/routing congestion | File-tree regions that cannot be safely edited concurrently. | `dos arbitrate`, collision count, repartition advice |

The model is useful only when each row has a measured or witnessed source. A
capacity row without a source is a planning assumption, not a scheduling fact.

## Generation Fit

Use the resources differently by horizon:

| Stream | Capacity posture | Expansion evidence | Reduction evidence |
|---|---|---|---|
| `gen/now` | Receives first claim on worker/review capacity for red gates, regressions, release blockers, and trunk hygiene. | Current blocker cleared faster with no collision or review backlog increase. | Now work has no witness, no path scope, or is actually future-gated. |
| `gen/next` | Receives reserved worker and token capacity for gated, dogfoodable foundation work. | A reservation produces a shipped gate, schema, dogfood route, or dispatch simplification. | Reserved work repeatedly fails to produce artifacts or leaks runtime exposure. |
| `gen/second-next` | Receives bounded simulation, compatibility, and scheduling capacity. | Simulation or dependency evidence creates a concrete `gen/next` child. | The option collides repeatedly or a simpler shipped design supersedes it. |
| `gen/future` | Receives small, expiring research/decision capacity. | Research changes a decision, assumption, or standards/market watch row. | No consumer appears before expiry or the signal is superseded. |

A high-priority `gen/future` issue can receive capacity when the decision window
is real. A weak `gen/now` issue can be held when it lacks a witness or safe path.
That is the orthogonality requirement in scheduling form.

## Sizing Formula

For a dispatch or super-loop window, compute capacity in this order:

1. Hard cap: `min(worker seats, account seats, host health, lane lease safety)`.
2. Review cap: `min(hard cap, available review/witness slots)`.
3. Token cap: hold candidates whose prompt/retry/review spend would exceed the
   window's token ceiling.
4. Generation envelope: reserve visible slices for `gen/now` and `gen/next`;
   admit `gen/second-next` and `gen/future` only through bounded explicit rows.
5. Collision price: reduce the launch set to disjoint lane/path regions.

The resulting wave should report:

```json
{
  "generation_capacity": {
    "window": "2026-07-01T00:00:00Z/PT2H",
    "worker_cap": 4,
    "review_cap": 2,
    "token_ceiling": 120000,
    "generation_counts": {"gen/now": 2, "gen/next": 1, "gen/second-next": 0, "gen/future": 0},
    "held": [{"issue": 1662, "generation": "gen/second-next", "reason": "explicit capacity row required"}]
  }
}
```

These numbers are examples of shape, not a claim about this host's live capacity.

## Promotion And Retirement

Promotion evidence for this model is a before/after capacity readout:

- A wave report names worker, token, review, and lane-lease caps and uses them to
  size a safe launch set.
- A `gen/next` reservation reduces stale issues, repeated collisions, or missing
  dogfood gates without increasing current red work.
- A later-horizon capacity row produces a child issue, simulation, or decision a
  nearer stream consumes.

Demotion or retirement evidence is equally concrete:

- Capacity rows cannot be measured cheaply.
- Review or token ceilings are routinely exceeded.
- Generation reservations hide higher-priority current work.
- The runtime feature gate leaks behavior into current users.
- The invalidating assumptions below fail.

Do not promote, demote, or retire the model by changing labels alone. Name the
measured capacity row and the witness that changed it.

## Invalidating Assumptions

This model depends on these assumptions:

- Worker, account, token, review, and lease capacity can be read cheaply enough
  before launch.
- Token spend can be attributed to issues or loop members well enough to budget.
- Review capacity is a real limiter, not just an after-the-fact complaint.
- Lane leases approximate actual edit conflicts closely enough for wave sizing.
- Operators can distinguish a high-value future decision from vague future work.

If those assumptions fail, replace the model with the stronger measured surface.
Do not keep stale capacity math as an operator-facing fact.

## Future Implementation Hooks

Reasonable next slices:

- Add `generation_capacity` to `fak dispatch wave --json`.
- Add a capacity table to `fak superloop walk --json`.
- Add a fixture that sizes a four-issue wave and proves `gen/second-next` is held
  without an explicit capacity row.

Use [`docs/generation-super-loop-budgets.md`](generation-super-loop-budgets.md)
for the budget envelope and [`docs/generation-loop-scheduling.md`](generation-loop-scheduling.md)
for admission and conflict arbitration.
