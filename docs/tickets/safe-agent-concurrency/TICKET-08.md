<!-- fak-cmd-key: safe-agent-concurrency-guarded-wrapper-lifecycle-v1 -->
<!-- fak-ticket-mirror: safe-agent-concurrency-08 -->
# feat(cmd): complete durable attempt lifecycle in the guarded dispatch wrapper

```routing
lane: cmd
paths: ["cmd/fak/dispatch_tick.go", "cmd/fak/guard_child.go", "cmd/fak/guard_child_supervision.go"]
expected_steps: 7
```

## Working spine
The existing `fak dispatch tick` command is the input trigger; guarded-wrapper engine execution consumes TICKET-02's durable handoff, registers the exact child process, and drives lifecycle transitions; a durable running/terminal receipt exposes its outcome across crash windows.

## Current state
`cmd/fak/dispatch_tick.go:896` reaches the production spawn path, `cmd/fak/dispatch_tick_broker.go:60-95` admits it, and `cmd/fak/guard_child_supervision.go` owns child start, wait, and containment. `internal/agentqueue/store_lifecycle.go:73-212` can register a wrapper, mark it running, and complete it, but the live `cmd/fak` path does not consume an attempt ID, nonce, or durable state path from `agentqueue`.

## Why this is next
TICKET-02 can safely begin and hand off a launch, but only the wrapper observes the exact child PID/start instant and exit result needed to finish durable ownership.

## Parent context
Prerequisite: TICKET-02 durable agentqueue launch handoff. Related public scheduler epic [fak#11175](https://github.com/anthony-chaudhary/fak/issues/11175) and queue controls [fak#8891](https://github.com/anthony-chaudhary/fak/issues/8891).

## Core through-line
Typed attempt handoff -> guarded wrapper starts child -> exact PID/start registration -> `MarkRunning` -> `CompleteAttempt` -> restart/crash-window witness. TICKET-03 later joins this lifecycle to host-grant admission.

## Gold-plating boundary
Only `cmd/fak/**` changes. Do not redesign agentqueue persistence, add another supervisor, bypass `SpawnBroker`, weaken DOS/containment, or implement fleet-scale qualification.

## Done condition / witness
The guarded wrapper validates TICKET-02's handoff, registers the exact child PID/start identity, marks running after successful process enrollment, and records one fenced terminal result. If an upstream caller already supplied grant metadata, preserve it without making host admission a prerequisite for this lifecycle leaf.

Witness: `go test ./cmd/fak -run "Dispatch.*AttemptLifecycle|GuardedWrapper.*Registration|AgentQueue.*CrashWindow" -count=1`

## Witness
`go test ./cmd/fak -run "Dispatch.*AttemptLifecycle|GuardedWrapper.*Registration|AgentQueue.*CrashWindow" -count=1`

## Definition of done
- [ ] Refuse missing, malformed, expired, or mismatched attempt handoff data before child start.
- [ ] Register the exact child PID/start instant and preserve any supplied grant metadata without acquiring a new grant.
- [ ] Call `MarkRunning` only after successful process enrollment.
- [ ] Call `CompleteAttempt` once for clean exit, failure, and cancellation.
- [ ] Fence duplicate wrappers and stale terminal writers.
- [ ] Reconcile start-before-register, register-before-running, and running-before-completion crash windows.
- [ ] Preserve existing guarded spawn, containment, and lane behavior.

## Acceptance gate
TICKET-02 is landed, the witness exits 0, and production tests prove exact process ownership through every lifecycle transition. Host-grant ownership is the separate TICKET-03/TICKET-04 acceptance gate.

## Closure binding
The resolving public commit cites this issue and its TICKET-02 prerequisite, carrying `(fak cmd)`.

## Lane
cmd

## Likely files
- `cmd/fak/dispatch_tick.go`
- `cmd/fak/guard_child.go`
- `cmd/fak/guard_child_supervision.go`
- focused guarded-wrapper lifecycle tests under `cmd/fak/`
- optional focused wrapper lifecycle files under `cmd/fak/`

## Expected steps
7

## Value

- Centrality: Core
- P1 Context: advanced - joins durable queue identity to the real child process.
- P2 Net value: advanced - eliminates unknown ownership across wrapper crash windows.
- P3 Adaptation: preserved - consumes existing lifecycle and broker contracts.
- P4 Operations: advanced - makes running, cancellation, completion, and recovery auditable.

## Work estimate
Estimate: 7 points

## Overall completion contribution
Contribution: 7/39 points

## Completion standard
development

github_issue: anthony-chaudhary/fak#13491
