<!-- fak-agentqueue-key: safe-agent-concurrency-per-goal-production-admission-v2 -->
<!-- fak-ticket-mirror: safe-agent-concurrency-03 -->
# feat(agentqueue): enroll production controller in bounded per-goal host admission

```routing
lane: agentqueue
paths: ["internal/agentqueue/controller.go", "internal/agentqueue/agentqueue.go", "internal/agentqueue/controller_test.go"]
expected_steps: 7
```

## Working spine
A production `Controller.Tick` invocation is the input trigger; the scheduling and durable `hostgrant` engines execute per-goal admission before the actuator starts work; a durable queued, granted, or typed refusal receipt makes the decision observable across controller processes.

## Current state
`internal/agentsched/governor.go:138-205,290-458` and `priority_queue.go:10-200` provide bounded admission and priority ordering, while `internal/microagent/fairsched.go:10-159` and `budgetqueue.go:10-282` provide tenant fairness and token backpressure. `internal/hostgrant` already has a durable cross-process resource store. Production `internal/agentqueue/controller.go:83-117` does not consume these contracts; `agentsched` callers are limited to harness benchmarks. Per-process scheduler instances alone cannot enforce a node-wide cap.

## Why this is next
Once TICKET-01 provides stable goal identity, the production controller is the narrowest real seam where queued work can be bounded per goal before process launch.

## Parent context
Public scheduler epic [fak#11175](https://github.com/anthony-chaudhary/fak/issues/11175). Closed fak#11177 and fak#11183 supplied scheduler and herd-test foundations; reuse them rather than reopening or duplicating those features. Reuse semantics from closed [fak#11840](https://github.com/anthony-chaudhary/fak/issues/11840).

## Core through-line
Durable queued intent -> fair goal selection -> one cross-process host grant -> existing actuator -> exact release and production reachability witness.

## Gold-plating boundary
Only `internal/agentqueue/**` changes. Do not edit `internal/agentsched` or `internal/microagent`, add another scheduler, build cluster placement, or optimize benchmark throughput.

## Done condition / witness
The production controller uses adapters over existing scheduling primitives and the durable `hostgrant` store, parks work when capacity is full, releases only exact owned grants after terminal proof, and demonstrates non-starvation under skew.

Witness: `go test ./internal/agentqueue -run "PerGoal|FairAdmission|Controller.*Admission" -count=1`

## Witness
`go test ./internal/agentqueue -run "PerGoal|FairAdmission|Controller.*Admission" -count=1`

## Definition of done
- [ ] Add agentqueue-local adapters that consume existing `agentsched` and `microagent` contracts.
- [ ] Enroll `Controller` before `Actuate` using immutable `GoalID` and the shared `hostgrant` store.
- [ ] Enforce bounded global and per-goal queued/running/token limits.
- [ ] Preserve deterministic ordering and weighted progress for eligible goals.
- [ ] Release an exact grant on proved completion, failure, or cancellation; hold uncertain launch/descendant state for reconciliation.
- [ ] Return typed retry/backpressure receipts at capacity.
- [ ] Prove controller-path reachability with skew/cancel/restart and two controller processes competing for one process seat.

## Acceptance gate
The witness exits 0, all implementation changes remain under `internal/agentqueue/**`, and no parallel scheduling algorithm is introduced.

## Closure binding
The resolving public commit cites this issue and carries `(fak agentqueue)`.

## Lane
agentqueue

## Likely files
- `internal/agentqueue/controller.go`
- `internal/agentqueue/agentqueue.go`
- `internal/agentqueue/controller_test.go`
- optional new adapter and focused test files under `internal/agentqueue/`

## Expected steps
7

## Value

- Centrality: Core
- P1 Context: advanced - connects stable goal identity to real production admission.
- P2 Net value: advanced - bounds overload while preserving progress across goals.
- P3 Adaptation: advanced - adapts existing scheduler and microagent primitives without modifying them.
- P4 Operations: advanced - supplies typed pressure and exact release behavior.

## Work estimate
Estimate: 7 points

## Overall completion contribution
Contribution: 7/39 points

## Completion standard
development

github_issue: anthony-chaudhary/fak#13492
