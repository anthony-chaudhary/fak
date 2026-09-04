---
name: super-workstream
description: "Organize and progress multi-task queues under dynamic lane leases with long-turn context safety. Use when an operator or loop needs to execute a sequence of work units across lanes while preventing context degradation and trunk lock contention."
---

# /super-workstream — queue progression, dynamic lane leases, and context safety

> **The queue-ordered workstream coordinator.** Bridges high-level operator intents,
> just-in-time lane leases, and rigorous long-turn context safety. It allows one skill,
> loop, or autonomous agent to progress multiple tasks sequentially without suffering
> from context bloat or monopolizing shared trunk lanes.

The fleet runs many work engines: `fak dispatch tick` handles one issue under one lease;
`fak superloop` walks scorecards and member loops to find the single worst-first entry;
`/super-loop` fans out N detached processes across accounts. But when an agent needs to
advance a **coherent sequence of multiple work items** over a long session, it hits two
opposing failure modes:

1. **Context Bloat & Attention Dilution**: Running 5+ tasks in one continuous raw context
   window accumulates 40–100 turns. Old command outputs pollute attention, prompt caching
   degrades, and the model confuses requirements across tasks.
2. **Coarse Trunk Lockout**: Taking one broad lane lease across the whole multi-task session
   blocks peer agents and human contributors on the shared trunk. Conversely, omitting lane
   leases leads to collisions (`COLLISION_RISK`).

A **Super Workstream** resolves this by operating as a functional coordinator that executes
an ordered queue through three synchronized disciplines:

```
[ Queue of WorkItems ]
         │
         ▼
┌─────────────────────────┐
│ 1. Dynamic Lane Lease   │  Acquire just-in-time for active item; release immediately on exit.
│    (laneadmit / yield)  │  If contended by a peer, skip to a downstream disjoint item.
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 2. Bounded Execution    │  Execute item within a strict turn budget (default 8 turns).
│    & Witness Gate       │  Prove real execution (CLAIM_TEST_GREEN); commit by explicit path.
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 3. Context Safety       │  Evaluate context pressure. At item boundaries or turn ceilings,
│    & O(1) Carryover     │  emit StreamCarryoverSeed, reset transcript, and re-arm clean context.
└─────────────────────────┘
```

---

## Core Principles

### 1. Dynamic Per-Item Lane Leasing
Never hold a coarse lane lease across multiple queue items.
- Acquire the exact lane/tree lease (`laneadmit.Request`) immediately before executing the active item.
- If the head item's lane is held by a peer session (`COLLISION_RISK`), check for a downstream
  disjoint item in the queue (`FindNextDisjointItem`). If one exists, progress that item immediately;
  otherwise, yield until the peer lease clears.
- As soon as the item is committed, failed, or yielded, release the lease before advancing the queue.

### 2. Bounded Turn Budget per Work Unit
Every work item carries a defined turn budget (`MaxTurns`, default 8).
- Keep the coordinator context clean: do not execute long multi-page compiler outputs or exploratory
  grep loops in the coordinator context.
- When an item approaches 70% of its turn budget (`PRESSURE_WARN`), focus on completing the minimal
  reproduction and verification.
- If an item exhausts its turn budget without completion (`RESET_REQUIRED`), checkpoint its partial
  progress, release the lane lease, and re-arm with a fresh turn.

### 3. Context Safety via `StreamCarryoverSeed`
To sustain long runs across dozens of tasks, the stream discards raw historical transcripts at item
boundaries and carries only a compact O(1) **Carryover Seed**:
- **Stream ID & Intent**: High-level objective.
- **Queue Progress**: Completed count, remaining items.
- **Closed Receipts**: For finished items, preserve ONLY `{ID, Title, CommitSHA, WitnessStatus}`.
- **Stream Pins**: Invariants (e.g. "keep main buildable", "explicit path commits only").
- **ctxplan Layout**: Reconstructs the context frame using `ctxplan.Layout`:
  - `Base`: Stream invariants and pins (exact precision).
  - `Recent`: Previous item's summary receipt (planned precision).
  - `Current`: Active work item descriptor and acceptance criteria (exact precision).
  - `Deep`: Historical git/CAS pointers (pointer precision).

---

## Step-by-Step Execution Workflow

### Step 0: Define or Inspect the Workstream Plan

Inspect the queued items, target lanes, and context budgets:

```bash
fak superstream plan [--spec path/to/stream.json] [--json]
```

### Step 1: Query Next Safe Step (`fak superstream step`)

Evaluate the single next transition from the pure state machine:

```bash
fak superstream step [--spec path/to/stream.json] [--holder <id>] [--json]
```

The decision returns one of:
- `ACQUIRE_LEASE`: Head item is ready and lane is free.
- `YIELD_CONTENDED`: Head item lane is held; tells you whether a disjoint item is ready to skip to.
- `EXECUTE_ITEM`: Lease is held; proceed with implementation.
- `WITNESS_AND_COMMIT`: Execution finished; execute verification and commit by explicit path.
- `RELEASE_LEASE`: Changes committed/settled; release the lane lease.
- `RESET_CONTEXT`: Turn limit or item boundary reached; re-arm coordinator context.
- `ADVANCE_QUEUE`: Move to the next pending item.
- `STREAM_COMPLETE`: All queue items terminal.

### Step 2: Acquire Lane Lease & Execute Item

Under `ACQUIRE_LEASE`, request admission via `laneadmit.Decide` and register the lease:

```bash
# Verify admission against live leases
fak lease check --lane <LANE> --tree <PATHS>
```

Execute the work item under bounded turns:
- Make minimal edits within the leased tree.
- Comply with AGENTS.md: do not modify files outside the leased tree.

### Step 3: Witness Verification & Explicit-Path Commit

Before claiming done, execute the real witness command:

```bash
# Example witness
fak validate --mine <PATHS>
```

Commit strictly by explicit paths:

```bash
fak commit --preview --path <PATHS> -m "feat(scope): <description> (fak <leaf>)"
fak commit --path <PATHS> -m "feat(scope): <description> (fak <leaf>)"
```

### Step 4: Release Lane Lease

Immediately upon committing:
- Release the held lease so trunk peers can proceed without delay.

### Step 5: Check Context Safety & Emit Carryover Seed

Inspect context health:

```bash
fak superstream carryover [--spec path/to/stream.json]
```

If context reset is required:
- Flush previous conversational turn history.
- Feed the output of `fak superstream carryover` as the initial user/system prompt for the next turn.
- Advance the queue to the next work item.

---

## Integration with fak Natural Context Controls

| Context Control Layer | How Super Workstream Integrates |
|---|---|
| **`ctxplan`** | Carryover seed maps directly into `ctxplan.Layout` (Base pins, Current item, Recent summary, Deep pointer). |
| **`session.Budget`** | When running under `fak serve`/`fak guard`, the carryover seed serves as the carryover seed for `--reset-on-budget` / `continuation_id`. |
| **`ctxmmu` / `headroom`** | Strips ANSI escape sequences and carriage-return redraws from tool results during item execution. |
| **Clean Coordinator** | Coordinator retains O(1) state; delegates execution sub-tasks to isolated child processes. |

---

## Worked Example

An operator wants to execute three tasks in one workstream:
1. `task-gateway`: Update rate limiter in `internal/gateway`
2. `task-docs`: Refresh CLI reference in `docs/`
3. `task-engine`: Optimize KV allocation in `internal/engine`

```bash
# 1. View initial plan
$ fak superstream plan
Super Workstream: stream-sample (Intent: sample-workstream-progression)
Budget: 40 max total turns, 200000 max tokens, 8 turns/item
Context Safety: SAFE (item "task-gateway" has 8 turns remaining)

#  TASK ID              LANE      TURNS  STATUS   WITNESS
1  task-gateway-parity  gateway   8      pending  go test ./internal/gateway/...
2  task-docs-refresh    docs      6      pending  fak validate --mine docs/integrations/**
3  task-engine-audit    engine    10     pending  go test ./internal/engine/...

# 2. Get step decision
$ fak superstream step
Super Workstream Step Decision: ACQUIRE_LEASE
Reason: lane "gateway" is free; acquire lease stream:stream-sample:task-gateway-parity:superstream-agent
Next Safe Step: acquire lane lease stream:stream-sample:task-gateway-parity:superstream-agent over [internal/gateway/**]

# 3. Item 1 executes and commits -> release lease -> emit carryover
$ fak superstream carryover
Stream Carryover Seed (O(1) Context Handshake)
Stream ID: stream-sample (Intent: sample-workstream-progression)
Completed Items: 1 / 3
Active Task: task-docs-refresh (refresh integration documentation, Lane: docs)
Remaining Budget: 34 turns, 185000 tokens
```
