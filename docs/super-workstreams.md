---
title: "fak super workstreams: queue progression, dynamic lane leases, and context safety"
description: "A super workstream organizes multi-task execution queues across dynamic lane leases while preserving context safety over long turns. It bridges high-level intent, just-in-time lease acquisition, and O(1) carryover handoffs."
---

# Super Workstreams (`fak superstream`)

> An operator or loop says **"progress this sequence of tasks across the codebase"**.
> In a shared trunk and an LLM-driven environment, running multiple tasks in one session
> without coordination causes two severe failures: **context bloat** (linear accumulation
> of turns dilutes attention and exhausts token limits) and **coarse trunk lockout** (holding
> one broad lane lease across the entire session starves peers, or skipping leases causes collisions).
> A **Super Workstream** coordinates an ordered queue of tasks, acquires and releases lane leases
> dynamically per task, and enforces **O(1) context safety** across long sessions.

The repository already provides specialized execution primitives:
- [Issue-dispatch](dispatch-loop.md) (`fak dispatch tick`): runs exactly one issue under one lease, exiting cleanly.
- [Super loops](super-loops.md) (`fak superloop`): reads member loops and scorecards worst-first to select what to enter, but mutates nothing at its own altitude and does not sequence multi-task queues.
- [Fleet waves](explainers/ultracode-multi-agent-dogfood.md) (`/fleet-wave` / `/super-loop`): launches N detached concurrent workers across separate accounts, prioritizing parallel fan-out over sequential queue continuity.

A **Super Workstream** sits at the functional coordinator level for sequential, long-horizon work.
It balances three interacting requirements:

| Dimension | Uncontrolled Session | Super Workstream |
|---|---|---|
| **Task Queue** | Ad-hoc, unstructured execution | Prioritized, explicit queue with per-item turn budgets and witness criteria |
| **Lane Leases** | Coarse lock held for entire session (or no lease at all) | Just-in-time per-item lease acquisition, collision avoidance via disjoint queue skipping, immediate release |
| **Context Safety** | Unbounded linear transcript (40–100+ turns), prompt degradation | Strict per-item turn caps, O(1) `StreamCarryoverSeed` handoffs, and `ctxplan` layout integration |

---

## Architectural Pillars

```
                      ┌────────────────────────────────────────┐
                      │        Super Workstream Intent         │
                      │      (Queue of Ordered WorkItems)      │
                      └───────────────────┬────────────────────┘
                                          │
                        For each Item in Queue (Sequential)
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
┌───────────────────────┐                                   ┌───────────────────────┐
│ 1. Dynamic Lane Lease │                                   │ 2. Context Safety     │
│   • Request admission │                                   │   • Turn/token budget │
│   • If COLLISION_RISK,│                                   │   • Monitor pressure  │
│     skip to disjoint  │                                   │   • At boundary/limit:│
│   • Acquire & release │                                   │     emit O(1) seed    │
└───────────┬───────────┘                                   └───────────┬───────────┘
            │                                                           │
            └─────────────────────────────┬─────────────────────────────┘
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ 3. Execution & Witness Gate            │
                      │   • Bounded implementation             │
                      │   • Verify real witness command        │
                      │   • Commit strictly by explicit path   │
                      └───────────────────┬────────────────────┘
                                          │
                                   [Advance Queue]
```

### 1. Dynamic Per-Item Lane Leasing

Holding a lease on `internal/**` for two hours while an agent works on three consecutive
tasks creates severe contention on the maintainers' shared trunk.

Under a Super Workstream:
- **Just-In-Time Acquisition**: The stream evaluates `laneadmit.Decide` against active leases
  (`refs/fak/locks/*`) only when the item is ready to execute.
- **Contention Reordering (`FindNextDisjointItem`)**: If Item 1 requires `internal/gateway`
  and is blocked by a peer's live lease (`COLLISION_RISK`), the stream inspects downstream
  pending items in the queue. If Item 2 requires `docs/` and is completely disjoint, the
  stream skips to Item 2, maintaining forward velocity rather than idling.
- **Immediate Release**: As soon as an item is committed or reaches a terminal state,
  its lane lease is released back to the trunk pool before the stream touches the next item.

### 2. Context Safety and O(1) Carryover

Long-running agent loops naturally decay as turn count rises:
1. Tool outputs from Task 1 (compiler logs, test stacks) clutter attention during Task 3.
2. Token budgets get consumed by stale conversational history.
3. System and task pins lose saliency in a 100k-token prompt.

A Super Workstream maintains context safety through two mechanisms:
- **Context Pressure Monitoring**: `EvaluateContextSafety` tracks current item turns against
  `MaxTurns` (default 8). Crossing 70% emits `PRESSURE_WARN`. Reaching 100% requires a context
  boundary reset (`RESET_REQUIRED`).
- **`StreamCarryoverSeed`**: At item boundaries or upon hitting turn limits, the stream
  discards the bulky transcript and distills state into an O(1) handoff seed:
  - Completed items as compact summaries (`ID`, `Title`, `CommitSHA`, `WitnessResult`).
  - Active item and next pending item.
  - Preserved stream invariants (`BasePins`).
  - Recommended `ctxplan.Layout`:
    - `Base`: Stream invariants (exact).
    - `Recent`: Previous task's commit receipt (planned).
    - `Current`: Active work item descriptor (exact).
    - `Deep`: History pointers (pointer-only).

This seed can be passed across conversational turns or subagent invocations, ensuring that
Turn 1 of Task 4 has the same pristine context quality as Turn 1 of Task 1.

---

## The Stream State Machine (`superstream.DecideStep`)

The stream progression is driven by a pure, deterministic state machine:

| Current Item Status | Condition | Next Action |
|---|---|---|
| `ItemPending` | Lane is free | `ACQUIRE_LEASE` |
| `ItemPending` | Lane is contended, disjoint item exists | `YIELD_CONTENDED` (skip to disjoint item) |
| `ItemPending` | Lane is contended, no disjoint item | `YIELD_CONTENDED` (pause/yield) |
| `ItemLeaseAcquired`| Lease registered | `EXECUTE_ITEM` |
| `ItemExecuting` | Turn limit hit | `RESET_CONTEXT` (emit seed, re-arm) |
| `ItemExecuting` | Work ready | `WITNESS_AND_COMMIT` |
| `ItemCommitted` | Lease still held | `RELEASE_LEASE` |
| `ItemCompleted` | Queue has remaining items | `ADVANCE_QUEUE` |
| All Terminal | Leases cleared | `STREAM_COMPLETE` |

---

## CLI Reference

### 1. Inspect Workstream Plan
```bash
fak superstream plan [--spec <path>] [--json]
```
Displays queue ordering, target lanes, per-item turn allocations, and acceptance witnesses.

### 2. Step the Stream
```bash
fak superstream step [--spec <path>] [--holder <id>] [--json]
```
Evaluates the next deterministic transition (e.g. acquire lease, execute, commit, release).

### 3. Generate Carryover Seed
```bash
fak superstream carryover [--spec <path>] [--json]
```
Outputs the compact O(1) `StreamCarryoverSeed` for boundary resets.

### 4. Query Status
```bash
fak superstream status [--spec <path>] [--json]
```
Reports total progress, held leases, turn/token spend, and context safety status.

---

## Integration with Natural Context Controls

Super Workstreams coordinate natively with existing fak context control facilities:
- **`internal/ctxplan`**: Directly provisions the 4-area `Layout` (`Base`, `Current`, `Recent`, `Deep`).
- **`internal/gateway/session_budget`**: Serves as the seed payload for `--reset-on-budget` and `continuation_id`.
- **`internal/ctxmmu` / `headroom`**: Cleans per-turn tool results of ANSI/progress-bar noise during item execution.
- **Coordinator Cleanliness**: Keeps the high-level coordinator free of raw tool executions, dispatching bounded sub-tasks.
