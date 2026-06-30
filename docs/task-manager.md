---
title: "fak task manager: in-process runtime state"
description: "How fak's internal/taskmgr folds live process, task, step, and concept state into a Manager.Snapshot, with the fak task sample CLI."
---

# fak task manager concept

`internal/taskmgr` is the process-local runtime state surface for "what is this
process doing right now?" It is separate from:

- `internal/sharedtask`: collaborative task records and patch/event semantics.
- `internal/session`: live session control state and budgets.
- `internal/trajectory`: per-turn historical analysis rows.

The task manager is the live, in-process fold: a long-running `fak` front door can
create a `Manager`, mark tasks and steps as work advances, and expose
`Manager.Snapshot()` to an operator or health endpoint.

## Snapshot model

Each snapshot carries:

- process identity: pid, Go OS/arch/version, start time, snapshot time, uptime.
- process resource sample: wall seconds, runtime CPU seconds when the Go runtime
  exposes it, heap/sys memory, and goroutine count.
- task rows: id, title, state, runtime seconds, progress, ETA when progress has a
  positive `done` and a known `total`, current step, liveness class, beat counters,
  and resource delta since task start.
- step rows: id, title, concept bucket, state, runtime seconds, progress, ETA, and
  liveness/resource delta since step start.
- concept usage: per-concept aggregation over steps, so an operator can see time
  spent in buckets such as `observe`, `adjudicate`, `tool`, `model`, or `verify`.

ETA is deliberately absent when it would be guesswork: no known total, no positive
progress, elapsed time of zero, or a terminal task/step.

Liveness is an in-loop progress heartbeat, not a process scan. `Beat` marks that
the task or step body is still advancing; `SetProgress` also counts as a beat.
Running records with no beats are `idle`; records with a recent beat are `live`;
records whose last beat is older than the manager's liveness timeout are
`stalled`. Terminal records return to `idle` while preserving beat metadata for
diagnostics.

## CLI proof surface

`fak task sample` emits the same snapshot shape for the current command process:

```powershell
go run ./cmd/fak task sample --json --task build --step tests --concept verify --done 2 --total 10 --unit phase
```

Human output is available without `--json`:

```powershell
go run ./cmd/fak task sample --task build --concept observe
```

This command is a sample/export surface, not a scheduler. The load-bearing API is
the `internal/taskmgr.Manager` type.

`fak task handoff` is the completion-to-next-work gate. It reads a
`fak.task-handoff.v1` JSON file and refuses unless the record carries:

- a `task` with `state: "done"` and a `witness.verified_state:
  "verified_done"`;
- a `current_state` summary describing where the item stands now;
- either one or two `next_steps` or a `no_next_step_reason`.

Dry-run mode prints the stable GitHub issue create/update plan. `--live` is the
only mode that calls `gh`, and each next step is deduped by an HTML marker in the
issue body.

The planned hardening for generated follow-up issues is tracked in
[`docs/notes/NEXT-STEP-SCOPING-GUARDS-2026-06-30.md`](notes/NEXT-STEP-SCOPING-GUARDS-2026-06-30.md):
default issue creation should require explicit current state, in-scope and
out-of-scope boundaries, done condition, witness, route, and acceptance gate
before a machine-created issue becomes dispatchable.

Minimal handoff input:

```json
{
  "schema": "fak.task-handoff.v1",
  "current_state": "The implementation is committed; the remaining proof is a live issue sync smoke.",
  "task": {
    "task_id": "task_push_next",
    "title": "Push next work",
    "state": "done",
    "witness": {
      "verified_state": "verified_done",
      "source": "commit-audit",
      "sha": "deadbeef"
    }
  },
  "next_steps": [
    {
      "key": "task_push_next/live-smoke",
      "title": "Run live task handoff issue sync smoke",
      "body": "Exercise `fak task handoff --live` against a disposable follow-up issue.",
      "reason": "Dry-run planning is covered; live gh behavior still needs an operator-owned witness.",
      "labels": ["agent-handoff"]
    }
  ]
}
```

## Embedding shape

```go
m := taskmgr.NewManager()
task, _ := m.StartTask(taskmgr.TaskSpec{
    TaskID: "release",
    Title:  "Build release",
    Total:  10,
    Unit:   "phase",
})
step, _ := task.StartStep(taskmgr.StepSpec{
    StepID:  "tests",
    Concept: "verify",
    Total:   4,
    Unit:    "suite",
})

_ = task.SetProgress(2, 10, "phase")
_ = step.SetProgress(1, 4, "suite")
_ = step.Beat()
snapshot := m.Snapshot()
```

The package is stdlib-only and has injectable clock/resource sampling, so tests can
prove elapsed-time, resource-delta, and ETA math without sleeping.

## Non-goals in this rung

This is not yet a durable task service, a distributed scheduler, a process monitor
for other PIDs, or a fleet-level progress oracle. Those can be built on top by
publishing snapshots through the existing gateway, shared-task, or a2a surfaces.
