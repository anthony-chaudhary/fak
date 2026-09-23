<!-- fak-toolprocgate-key: safe-agent-concurrency-shared-tool-hostgrant-v2 -->
<!-- fak-ticket-mirror: safe-agent-concurrency-04 -->
# feat(toolprocgate): charge heavy child starts to the shared durable host grant

```routing
lane: toolprocgate
paths: ["internal/toolprocgate/spawnbroker.go", "internal/toolprocgate/spawnbroker_test.go"]
expected_steps: 7
```

## Working spine
An existing `SpawnBroker.Admit` request from a public launcher is the input trigger; the shared `hostgrant` engine executes capacity acquisition for a classified heavy child; the broker returns a transferable grant or typed capacity-refusal receipt to its caller before process start.

## Current state
`internal/hostgrant/hostgrant.go:57-154` provides durable cross-process acquire, transfer, and release. `internal/toolprocgate/spawnbroker.go:65-151,288-405` admits starts and sanitizes metadata, but does not acquire shared host capacity. Existing `cmd/fak` dispatch and guarded-child brokers call `SpawnBroker`; this ticket changes their consumed contract without editing those callers.

## Why this is next
Process-local broker decisions cannot bound aggregate heavy children from concurrent public entrypoints; the broker is the shared pre-start seam that can consume the durable host authority once.

## Parent context
Public scheduler epic [fak#11175](https://github.com/anthony-chaudhary/fak/issues/11175), queue controls [fak#8891](https://github.com/anthony-chaudhary/fak/issues/8891), and open task ingress [fak#11414](https://github.com/anthony-chaudhary/fak/issues/11414). TICKET-07 owns 100/1,000 cross-entrypoint qualification; this leaf provides the tool-process grant contract only.

## Core through-line
Heavy spawn attempt -> existing `hostgrant` acquisition -> sanitized transferable grant metadata -> existing caller launch -> broker-level admission and cleanup witness.

## Gold-plating boundary
Only `internal/toolprocgate/**` changes. Do not edit `internal/hostgrant`, `cmd/fak`, task tools, or queue packages; do not add scale qualification, distributed admission, or hardware tuning.

## Done condition / witness
`SpawnBroker` acquires the existing durable grant for classified heavy starts, emits sanitized grant identity/generation metadata, and returns typed capacity refusal without starting work.

Witness: `go test ./internal/toolprocgate -run "SharedHostGrant|SpawnBroker.*Grant|HeavyChildAdmission" -count=1`

## Witness
`go test ./internal/toolprocgate -run "SharedHostGrant|SpawnBroker.*Grant|HeavyChildAdmission" -count=1`

## Definition of done
- [ ] Define a conservative heavy-child resource classification inside `toolprocgate`.
- [ ] Acquire existing `hostgrant` capacity using stable request/goal/task/attempt identity.
- [ ] Return sanitized grant ID, generation, and transfer metadata to existing broker callers.
- [ ] Avoid double acquisition on idempotent replay and fence mismatched ownership.
- [ ] Return typed full/backpressure receipts without invoking the launcher.
- [ ] Cover concurrent acquisition, replay, stale ownership, cancellation-before-launch, and redaction.
- [ ] Keep fixtures synthetic and aggregate only.

## Acceptance gate
The witness exits 0, the diff remains under `internal/toolprocgate/**`, and callers can consume the extended grant without a new broker path.

## Closure binding
The resolving public commit cites this issue and carries `(fak toolprocgate)`; TICKET-07 remains the scale-qualification owner.

## Lane
toolprocgate

## Likely files
- `internal/toolprocgate/spawnbroker.go`
- `internal/toolprocgate/spawnbroker_test.go`
- optional focused adapter/test files under `internal/toolprocgate/`

## Expected steps
7

## Value

- Centrality: Core
- P1 Context: advanced - unifies heavy-child admission at the shared broker seam.
- P2 Net value: advanced - prevents cross-process capacity oversubscription.
- P3 Adaptation: advanced - consumes existing durable grants and caller contracts.
- P4 Operations: advanced - exposes typed pressure and transferable ownership metadata.

## Work estimate
Estimate: 5 points

## Overall completion contribution
Contribution: 5/39 points

## Completion standard
development

github_issue: anthony-chaudhary/fak#13493
