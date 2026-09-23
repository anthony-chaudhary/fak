<!-- fak-cmd-key: safe-agent-concurrency-managed-launch-census-v1 -->
<!-- fak-ticket-mirror: safe-agent-concurrency-12 -->
# feat(cmd): route long-lived managed starts through guarded host admission

```routing
lane: cmd
paths: ["cmd/fak/issueorchestrator.go", "cmd/fak/orchestration_launch.go", "cmd/fak/dispatch_canary.go", "cmd/fak/managed_launch_census_test.go"]
expected_steps: 7
```

## Working spine
Migrate the bounded census of long lived `cmd/fak` agent launches to the admitted engine behind TICKET-11; existing in-process callers need not start a nested CLI. Keep light bounded helper processes explicitly classified.

## Current state
`configureDispatchSpawn` has nine production callers. Three long lived managed starts remain outside the shared admission seam: `cmd/fak/issueorchestrator.go:721-745` starts an OpenCode worker, `cmd/fak/orchestration_launch.go:731-833` starts guarded orchestration workers, and `cmd/fak/dispatch_canary.go:361-390` runs a guarded canary worker. `cmd/fak/dispatch_tick_worker.go:349-364` is already brokered. `project.go`, `dispatch_project_fields.go`, and `scoreboardpost.go` launch bounded Git or GitHub helpers rather than agents or heavy tools and should remain uncharged unless their behavior changes.

## Why this is next
The public facade from TICKET-11 has production value only when every known long lived `cmd/fak` agent start uses it and a source census fails when a new unmanaged start appears.

## Parent context
Child of public scheduler epic [fak#11175](https://github.com/anthony-chaudhary/fak/issues/11175). Native prerequisite: `safe-agent-concurrency-11`.

## Core through-line
Production caller invocation -> TICKET-11 admission engine execution for each classified managed start -> exact lifecycle identity -> durable admission receipt and a census response that rejects future unmanaged starts.

## Gold-plating boundary
No conversion of bounded Git/GitHub probes, no repository-wide `exec.Command` rewrite, no scheduler replacement, and no changes outside `cmd/fak`. Classification must follow actual child lifetime and role rather than executable name alone.

## Done condition / witness
The three named long lived production starts enter through the TICKET-11 admission engine, preserve their existing worktree/log/stdio behavior, and a source census fails on a seeded unmanaged managed-start fixture.

Witness 1: `go test ./cmd/fak -list '^TestManagedLaunchCensusContract$'` must print exactly `TestManagedLaunchCensusContract`; absence is failure.

Witness 2: `go test ./cmd/fak -run '^TestManagedLaunchCensusContract$' -count=1 -v` must execute issue-orchestrator, orchestration-worker, dispatch-canary, light-helper, and seeded-violation subtests.

## Witness
Run both named witness commands above; a zero-test run cannot satisfy the first command's exact-name requirement.

## Definition of done
- [ ] Record the nine current `configureDispatchSpawn` callers with managed-agent, heavy-tool, or light-helper classification.
- [ ] Route issue-orchestrator worker start through TICKET-11 while preserving logs, worktree environment, and lease cleanup.
- [ ] Route orchestration worker start through TICKET-11 while preserving worktree handoff, PID identity, launch probe, and failure cleanup.
- [ ] Route dispatch canary worker start through TICKET-11 while preserving stdin, transcript, exit status, and cancellation.
- [ ] Keep the already brokered dispatch-tick worker on its existing admitted path without double acquisition.
- [ ] Keep project, dispatch-project-fields, and scoreboard Git/GitHub helpers classified as light and uncharged.
- [ ] Add a fail-closed census test whose seeded unmanaged managed start is detected and whose migrated callers emit typed admission receipts.

## Acceptance gate
Both witness commands pass, all five named subtests execute, the seeded violation is rejected, and no classified managed start in `cmd/fak` bypasses the admitted facade.

## Closure binding
The resolving public commit cites this issue and carries `(fak cmd)`.

## Lane
cmd

## Likely files
- `cmd/fak/issueorchestrator.go`
- `cmd/fak/orchestration_launch.go`
- `cmd/fak/dispatch_canary.go`
- `cmd/fak/managed_launch_census_test.go`

## Expected steps
7

## Value

- Centrality: Core
- P1 Context: advanced - connects public admission to the production agent launch entrypoints.
- P2 Net value: advanced - removes known unmanaged long lived starts and prevents silent regression.
- P3 Adaptation: preserved - migrates three concrete callers and retains light helper classification.
- P4 Operations: advanced - makes admission reachability and caller coverage mechanically auditable.

## Work estimate
Estimate: 5 points

## Overall completion contribution
Contribution: 5/39 points

## Completion standard
development

github_issue: anthony-chaudhary/fak#13495

github_issue: anthony-chaudhary/fak#13495
