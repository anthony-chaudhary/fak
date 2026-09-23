<!-- fak-cmd-key: safe-agent-concurrency-guarded-run-facade-v1 -->
<!-- fak-ticket-mirror: safe-agent-concurrency-11 -->
# feat(cmd): expose one guarded host-grant run facade for external callers

```routing
lane: cmd
paths: ["cmd/fak/main.go", "cmd/fak/host_grant.go", "cmd/fak/guard_hostgrant.go", "cmd/fak/host_grant_test.go"]
expected_steps: 7
```

## Working spine
Expose a bounded `fak host-grant run` command, or equivalent stable CLI, that joins shared tool-process admission to a contained child and transfers one host grant to the exact child process. Public queue callers may attach their durable handoff; external callers provide explicit stable request lineage.

## Current state
`cmd/fak/guard_hostgrant.go:35-117` already provides opt-in hostgrant acquire, managed start, parent-to-child transfer, and release for the internal guard child path. `cmd/fak/guard_child.go:919-1038` separately admits through `toolprocgate.SpawnBroker`, prepares durable registration, and builds the child. Those package-main helpers are not an executable contract private callers can use without importing public `internal` packages, and composing both paths can charge the same managed start twice.

## Why this is next
TICKET-03 and TICKET-04 establish queue and tool admission, while TICKET-08 owns durable wrapper lifecycle. Their first usable public join must be an executable seam so private orchestration can consume the policy without an illegal `fak/internal` import.

## Parent context
Child of public scheduler epic [fak#11175](https://github.com/anthony-chaudhary/fak/issues/11175). Native prerequisites: `safe-agent-concurrency-03`, `safe-agent-concurrency-04`, and `safe-agent-concurrency-08`.

## Core through-line
CLI command input with explicit stable request identity and optional public queue handoff -> hostgrant and toolprocgate engine execution -> one acquire or inherited transfer -> managed child start -> durable typed start/terminal receipt -> exact release after proved tree exit.

## Gold-plating boundary
No daemon, RPC service, new scheduler, replacement for `hostgrant`, replacement for `toolprocgate`, private package import, or broad migration of callers. Keep the existing guard opt-in compatible and define explicit behavior when a valid grant is already carried.

## Done condition / witness
An external process can invoke the CLI with synthetic stable identity, observe one admitted managed child and typed receipt, and prove a carried grant is transferred rather than acquired again. A full grant store returns promptly with typed retry-after so callers can park intents without a CLI process per waiter.

Witness 1: `go test ./cmd/fak -list '^TestHostGrantRunContract$'` must print exactly `TestHostGrantRunContract`; absence is failure.

Witness 2: `go test ./cmd/fak -run '^TestHostGrantRunContract$' -count=1 -v` must execute acquire, inherited-transfer, start-refusal, cancellation, and release subtests with a fake child executable.

## Witness
Run both named witness commands above; a zero-test run cannot satisfy the first command's exact-name requirement.

## Definition of done
- [ ] Add one documented `cmd/fak` run facade with explicit argv, working directory, environment, cancellation, and JSON receipt behavior.
- [ ] Validate a supplied TICKET-03/TICKET-08 durable handoff; otherwise require explicit stable request lineage from an external caller without fabricating a public queue record.
- [ ] Admit heavy starts through the TICKET-04 tool-process contract and acquire one host process seat when no grant is carried.
- [ ] Validate and transfer a carried grant to the exact child PID/start identity without a second acquire.
- [ ] Release exactly once after proved child-tree exit or proved non-start; hold uncertain starts and receipt-write failures for fenced reconciliation.
- [ ] Return typed refusal with retry-after and terminal receipts without leaking argv secrets or environment values; queued callers do not create CLI waiters.
- [ ] Preserve the existing `fak guard` hostgrant opt-in path and cover compatibility in focused tests.

## Acceptance gate
Both witness commands pass, the named test executes every required subtest with positive fake acquire/start/transfer counters, and the diff stays in the `cmd` lane.

## Closure binding
The resolving public commit cites this issue and carries `(fak cmd)`.

## Lane
cmd

## Likely files
- `cmd/fak/main.go`
- `cmd/fak/host_grant.go`
- `cmd/fak/guard_hostgrant.go`
- `cmd/fak/host_grant_test.go`

## Expected steps
7

## Value

- Centrality: Core
- P1 Context: advanced - exposes the public executable join needed by private production callers.
- P2 Net value: advanced - prevents duplicate capacity charging and unowned child starts.
- P3 Adaptation: preserved - reuses the existing guard, hostgrant, toolprocgate, and lifecycle seams.
- P4 Operations: advanced - emits typed receipts and deterministic release evidence at the process boundary.

## Work estimate
Estimate: 5 points

## Overall completion contribution
Contribution: 5/39 points

## Completion standard
development

github_issue: anthony-chaudhary/fak#13494

github_issue: anthony-chaudhary/fak#13494
