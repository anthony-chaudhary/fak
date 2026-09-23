<!-- fak-agentqueue-key: safe-agent-concurrency-durable-launch-handoff-v1 -->
<!-- fak-ticket-mirror: safe-agent-concurrency-02 -->
# feat(agentqueue): hand durable launch identity to the guarded dispatch wrapper

```routing
lane: agentqueue
paths: ["internal/agentqueue/controller.go", "internal/agentqueue/actuator.go", "internal/agentqueue/store_lifecycle.go"]
expected_steps: 6
```

## Working spine
A production `Controller.Tick` invocation is the input trigger; the `agentqueue` lifecycle engine advances its reserved attempt to `launching` and passes the nonce and state reference to the guarded dispatch command; a durable handoff receipt exposes the exact attempt for TICKET-08 to consume.

## Current state
`internal/agentqueue/controller.go:83-117` reserves starts before `Actuate`; `internal/agentqueue/actuator.go:40-75` invokes `fak dispatch tick` as one opaque command. `internal/agentqueue/store_lifecycle.go:20-212` already provides fenced `BeginLaunching`, `RegisterWrapper`, `MarkRunning`, and `CompleteAttempt`, but the actuator does not begin or hand off that lifecycle.

## Why this is next
The queue must persist the launch nonce before process creation so a later guarded-wrapper change can register the exact child and finish the lifecycle without guessing after a crash.

## Parent context
Public scheduler epic [fak#11175](https://github.com/anthony-chaudhary/fak/issues/11175) and queue controls [fak#8891](https://github.com/anthony-chaudhary/fak/issues/8891). TICKET-08 is the required `cmd/fak` wrapper half; this ticket deliberately stops at the cross-process handoff.

## Core through-line
Reserved attempt -> fenced `BeginLaunching` -> nonce/state handoff to existing `dispatch tick` -> deterministic wrapper enrollment contract -> focused actuator witness.

## Gold-plating boundary
Only `internal/agentqueue/**` changes. Do not edit `cmd/fak`, implement wrapper registration, mark running/completed from the parent process, add a supervisor, or replace `fak dispatch tick`.

## Done condition / witness
Every actuated reservation is first fenced as `launching`, and the exact nonce, attempt ID, and absolute durable state path are passed to `dispatch tick` for TICKET-08 to consume.

Witness: `go test ./internal/agentqueue -run "Controller.*Launch|Actuate.*Lifecycle|BeginLaunching" -count=1`

## Witness
`go test ./internal/agentqueue -run "Controller.*Launch|Actuate.*Lifecycle|BeginLaunching" -count=1`

## Definition of done
- [ ] Call `BeginLaunching` before invoking the command runner with a bounded deadline.
- [ ] Pass attempt ID, launch nonce, and absolute state path through explicit dispatch arguments or a typed handoff file.
- [ ] Keep retries idempotent for the winning nonce and fence competing nonces.
- [ ] Record failure only when non-start is proved; leave a possibly-started wrapper `held/unknown` for fenced reconciliation.
- [ ] Cover reserve-before-start, duplicate controller, stale nonce, and restart readback.
- [ ] Document TICKET-08 as the required dependent wrapper stage.

## Acceptance gate
The witness exits 0 and tests prove handoff identity without modifying or mocking a new wrapper lifecycle in `cmd/fak`.

## Closure binding
The resolving public commit cites this issue and carries `(fak agentqueue)`; closure does not close TICKET-08.

## Lane
agentqueue

## Likely files
- `internal/agentqueue/controller.go`
- `internal/agentqueue/actuator.go`
- `internal/agentqueue/store_lifecycle.go`
- focused tests beside those files

## Expected steps
6

## Value

- Centrality: Core
- P1 Context: advanced - binds a durable attempt to the process-launch boundary.
- P2 Net value: advanced - removes the reserve-to-start ambiguity without widening scope.
- P3 Adaptation: preserved - reuses the existing lifecycle store and dispatch command.
- P4 Operations: advanced - creates a crash-recoverable handoff for the wrapper stage.

## Work estimate
Estimate: 5 points

## Overall completion contribution
Contribution: 5/39 points

## Completion standard
development

github_issue: anthony-chaudhary/fak#13490
