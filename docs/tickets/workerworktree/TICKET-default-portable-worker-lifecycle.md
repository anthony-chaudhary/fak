# feat(workerworktree): default portable isolation and governed landing for issue workers

<!-- fak-workerworktree-key: default-portable-worker-lifecycle -->
<!-- github-issue: 12379 -->

```routing
lane: workerworktree
paths:
  - cmd/fak/issueorchestrator.go
  - cmd/fak/issueorchestrator_test.go
  - cmd/fak/dispatch_tick.go
  - cmd/fak/dispatch_tick_worker.go
expected_steps: 7
priority: P1
dependencies:
  - fak#11235
```

## Current state

The portable `fak worktree worker prepare|land|reap` commands already accept an arbitrary `--root`, but worker admission does not consistently use them. `issue-orchestrator` defaults `--worktree` and `--auto-land` off, dispatch defaults `FLEET_WORKER_WORKTREE` off, and preparation failure can fall through to launching in the shared checkout. The low-level isolated land path is nominally default-on but #11235 owns making its failure terminal rather than falling back to shared-index mutation.

## Parent ref

#11235

## Why now

Managed workers are already the documented landing path, yet ordinary launchers require opt-in flags and can abandon isolation exactly when preparation or reconciliation fails. This prevents the same safe lifecycle from being used predictably across repository roots.

## Problem frame

- Centrality: Core
- P1: advanced - every issue worker starts isolated and reaches a governed landing receipt by default.
- P2: advanced - one root-parameterized lifecycle works for the current repository and companion repositories without duplicated merge engines.
- P3: preserved - preparation or landing failure retains recoverable worker state and never mutates the shared checkout.
- P4: advanced - launch, land, push, and reap states are explicit in machine-readable receipts.

## Working spine

Issue dispatch -> portable worktree prepare for the requested root -> worker execution in detached checkout -> fail-closed governed land -> authoritative read-back -> reap only after durable completion.

## Core through-line

Make the existing root-parameterized worker-worktree lifecycle the default admission and completion route for OpenCode issue workers, with explicit opt-out compatibility and no implicit shared-checkout fallback.

## Scope

Flip launcher defaults, fail closed on prepare errors, pass the selected repository root through every lifecycle phase, and expose consistent JSON/human receipts. Reuse #11235 for low-level land conflict/CAS behavior.

## Concrete repro witness

```powershell
rg -n 'spawn-opencode|worktree|auto-land|FLEET_WORKER_WORKTREE' cmd/fak/issueorchestrator.go cmd/fak/dispatch_tick.go cmd/fak/dispatch_tick_worker.go
```

Current defaults show worker isolation and automatic landing are opt-in, while failed preparation does not prevent shared-root execution.

## Exact file:line seams

- `cmd/fak/issueorchestrator.go:145-158` (default flags)
- `cmd/fak/issueorchestrator.go:481-487` (worktree preparation result)
- `cmd/fak/issueorchestrator.go:544-590` (OpenCode process launch)
- `cmd/fak/dispatch_tick_worker.go:198-203` (environment default)
- `cmd/fak/dispatch_tick.go:863-879` (dispatch preparation and fallback)

## Blast radius and affected lanes

- Primary lane: `workerworktree`
- Affected surface: issue-orchestrator and dispatch worker admission in `cmd/fak`.
- Unaffected: model execution, serving, public ABI wire types, hardware paths, and module files.

## Quarantined fallback mechanism

If prepare or governed land fails, do not launch or mutate the shared root. Retain the detached worktree and typed reconciliation receipt for retry. Legacy shared-checkout execution is available only through an explicit off/unsafe compatibility selection covered by tests.

## Scoped acceptance criteria

- [ ] 1. [SW-VERIFIED] OpenCode issue-orchestrator and dispatch workers prepare a detached managed worktree by default for the selected `--root`.
- [ ] 2. [SW-VERIFIED] Prepare failure is terminal and cannot spawn a worker with the shared repository as its working directory.
- [ ] 3. [SW-VERIFIED] Successful workers enter governed land and authoritative read-back by default; reap occurs only after durable completion.
- [ ] 4. [SW-VERIFIED] #11235 failure states remain fail-closed: conflict or exhausted CAS preserves worker state and never reaches shared-index apply/commit.
- [ ] 5. [SW-VERIFIED] An explicit opt-out retains compatibility and is visible in receipts as unsafe/shared mode.
- [ ] 6. [SW-VERIFIED] A disposable second repository root completes the same prepare -> execute -> land -> read-back lifecycle without source changes or private imports.
- [ ] 7. [SW-VERIFIED] Human and JSON output distinguish prepared, running, reconciliation-required, locally landed, remotely published, and reaped states.

## Gold-plating boundary

No distributed consensus, new VCS, GUI, background daemon, module changes, or repository-specific merge implementation. Do not duplicate the low-level CAS/remerge work owned by #11235.

## Likely files

- `cmd/fak/issueorchestrator.go`
- `cmd/fak/issueorchestrator_test.go`
- `cmd/fak/dispatch_tick.go`
- `cmd/fak/dispatch_tick_worker.go`

## Witness

`go test ./cmd/fak -run 'TestIssueOrchestrator.*Worktree|TestDispatchTick.*Worktree' -count=1` must exit 0.

### Verifiable Witness

```powershell
go test ./cmd/fak -run 'TestIssueOrchestrator.*Worktree|TestDispatchTick.*Worktree' -count=1
```

## Done condition

Default launch is isolated and root-portable, preparation and land failures are fail-closed, explicit opt-out is tested, lifecycle receipts are truthful, and the targeted command tests pass.

## Done condition / witness

The targeted command test and `go test ./cmd/fak -count=1` both exit 0 in an isolated validation checkout.

## Acceptance gate

`go test ./cmd/fak -count=1`

## Closure binding

The resolving signed-off commit cites the created GitHub issue number, carries `(fak workerworktree)`, is read back on `origin/main`, and is pushed before closure.

## Lane

workerworktree
