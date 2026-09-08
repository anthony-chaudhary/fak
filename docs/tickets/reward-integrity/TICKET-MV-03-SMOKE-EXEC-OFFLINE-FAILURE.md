<!-- fak-testquality-key: smoke-exec-offline-failure-propagation -->
# fix(smoke-exec): propagate offline-agent failure instead of printing success

```routing
lane: testquality
paths: ["Makefile", "cmd/fak/validate_smoke_test.go"]
expected_steps: 4
priority: P1
class: infra
```

## Parent context

Child of the open QA-process quality program #3831; complements closed #3838's test-run witness by repairing a concrete green-but-unexecuted smoke path.

## Current state

The `smoke-exec` Make target describes itself as proof of offline agent execution, but its offline command ends with `|| true` and discards all output. The next command unconditionally prints `smoke-exec OK`. A nonzero exit from `fak agent --offline` therefore leaves the supposedly blocking smoke target green. The isolated `fak validate --smoke` path already records this class of failure; the Make target does not preserve that contract.

No open or closed issue found by targeted search owns this exact Makefile failure-propagation seam.

## Why this is next

`smoke-exec` is invoked by `make ci`, `make test-fast`, and the documented verification workflow. Its unconditional success directly weakens several higher-level done claims.

## Classification

- Centrality: Enabling verification integrity.
- P1 context: advanced — the smoke result corresponds to actual offline-agent completion.
- P2 net value: advanced — prevents CI and local verification from converting a real failure into a success claim.
- P3 adaptation: preserved — retain the existing fast smoke sequence and offline mode.
- P4 operations: advanced — failures keep their exit status and concise diagnostic instead of being silently discarded.

## Core through-line

Freshly built binary -> run offline agent with a controlled report path -> require exit zero and validate the durable report/output -> print success only after the witness is present.

## Working spine

Make target -> freshly built `fak` -> offline agent child process -> exit status plus report validation -> success marker or propagated failure.

## Gold-plating boundary

- No live-provider call, model download, agent planner redesign, or broad smoke framework.
- No new shell script; repository policy forbids loose glue scripts.
- Do not weaken the policy ALLOW/DENY checks already in the target.
- A structural grep alone is insufficient unless paired with a forced failing execution witness.

## Done condition

- [ ] Any nonzero offline-agent exit makes `make smoke-exec` nonzero.
- [ ] The target validates its offline report or semantic completion marker before printing success.
- [ ] A deterministic failure-injection test proves the previous masked-failure case.
- [ ] Normal `make smoke-exec` remains hermetic and passes without credentials or hardware.

## Definition of done

The Make target cannot print success after an offline-agent failure, and a deterministic regression proves both failing and passing execution branches.

## Verifiable Witness

```text
make smoke-exec
go test ./cmd/fak -run 'Test.*Smoke.*Offline.*Failure' -count=1
```

The regression must execute a controlled offline-agent failure and assert that the smoke wrapper returns nonzero and never emits `smoke-exec OK`. The normal hermetic target must still pass.

## Witness

The normal smoke target succeeds, while an injected offline-agent failure returns nonzero and produces no success marker.

## Acceptance gate

The focused Go regression and normal `make smoke-exec` both pass; the injected failure returns nonzero and emits no success marker.

## Closure binding

The resolving commit cites this issue and carries `(fak testquality)`.

## Likely files

- `Makefile`
- `cmd/fak/validate_smoke_test.go`

## Lane

`testquality`

## Expected steps

4
