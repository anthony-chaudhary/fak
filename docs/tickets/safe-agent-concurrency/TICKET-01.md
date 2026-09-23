<!-- fak-agentqueue-key: safe-agent-concurrency-goal-task-lineage-v1 -->
<!-- fak-ticket-mirror: safe-agent-concurrency-01 -->
# feat(agentqueue): bind primary goal identity separately from child task and session identity

```routing
lane: agentqueue
paths: ["internal/agentqueue/prompt_task.go", "internal/agentqueue/agentqueue.go", "internal/agentqueue/store.go"]
expected_steps: 6
```

## Working spine
An existing prompt-task submission request is the input trigger; the `agentqueue` persistence engine validates and stores one immutable primary `GoalID` alongside distinct root-task, parent-task, task, session, intent, and attempt identities; the returned durable task handle is the observable receipt and survives restart readback.

## Current state
`internal/goalregistry/goalregistry.go:52-56,82-93` stores durable goal identity, while `internal/goalregistry/goalregistry_test.go:60-75` proves structural relations are not execution parentage. Closed fak#8119 already threads explicit `FAK_GOAL_ID` through guard, account, dispatch-worker, and microagent launch lineage. The remaining gap is narrower: `internal/agentqueue/prompt_task.go:36-56` stores `TaskID` and `ParentSessionID`, and `internal/agentqueue/agentqueue.go:42-78` stores intents and attempts, but these durable queue records do not retain goal/task lineage for queue accounting.

## Why this is next
Per-goal fairness, cancellation, accounting, and restart recovery require a stable goal key before production admission can aggregate child work safely.

## Parent context
Public scheduler epic [fak#11175](https://github.com/anthony-chaudhary/fak/issues/11175). Reuse closed fak#8119's explicit launch `FAK_GOAL_ID`, closed runtime adapter [fak#11840](https://github.com/anthony-chaudhary/fak/issues/11840), and queue controls [fak#8891](https://github.com/anthony-chaudhary/fak/issues/8891); coordinate with open task ingress [fak#11414](https://github.com/anthony-chaudhary/fak/issues/11414) without claiming it complete.

## Core through-line
Explicit lineage in durable queue records -> unambiguous goal ownership across sessions and retries -> queryable goal/task accounting -> round-trip and restart witness.

## Gold-plating boundary
No graph database, distributed tracing system, new goal registry, replacement task API, or changes to guard/account/dispatch-worker/microagent launch lineage already owned by fak#8119. Never infer a goal from prompt text, issue number, repository path, or process name.

## Done condition / witness
The durable queue round-trips and restarts without conflating goal, task, and session identity.

Witness: `go test ./internal/agentqueue -run "GoalLineage|TaskLineage|PromptTask" -count=1`

## Witness
`go test ./internal/agentqueue -run "GoalLineage|TaskLineage|PromptTask" -count=1`

## Definition of done
- [ ] Add a validated execution-lineage value with `GoalID`, `RootTaskID`, `ParentTaskID`, `TaskID`, and `ParentSessionID`.
- [ ] Persist lineage in prompt handles, intents, attempts, and launch receipts with explicit old-snapshot compatibility.
- [ ] Keep `GoalID` immutable across retry and restart reconciliation.
- [ ] Reject mismatched goal/root/parent combinations and scoped idempotency collisions.
- [ ] Cover multiple sessions under one goal and multiple goals associated with one session.
- [ ] Use synthetic identifiers only in fixtures.

## Acceptance gate
The witness exits 0, compatibility behavior is explicit, and the source diff remains inside the `agentqueue` lane.

## Closure binding
The resolving public commit cites this issue and carries `(fak agentqueue)`.

## Lane
agentqueue

## Likely files
- `internal/agentqueue/prompt_task.go`
- `internal/agentqueue/agentqueue.go`
- `internal/agentqueue/store.go`
- focused tests beside those files

## Expected steps
6

## Value

- Centrality: Core
- P1 Context: advanced - establishes the identity key required by every per-goal control.
- P2 Net value: advanced - prevents cross-goal accounting and cancellation ambiguity.
- P3 Adaptation: preserved - extends existing durable records and task handles.
- P4 Operations: advanced - makes restart and audit ownership deterministic.

## Work estimate
Estimate: 5 points

## Overall completion contribution
Contribution: 5/39 points

## Completion standard
development

github_issue: anthony-chaudhary/fak#13489
