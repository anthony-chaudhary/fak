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
