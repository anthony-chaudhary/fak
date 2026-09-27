# fix(ops): admit mutable scheduled runs through managed worktrees

<!-- fak-ops-key: ops-run-managed-worktree-admission -->

```routing
repo: fak
lane: cmd
paths:
  - cmd/fak/ops_run.go
  - cmd/fak/ops_run_test.go
expected_steps: 4
priority: P1
dependencies: []
```

## Current state

`runOpsRun` resolves the caller-supplied `--workspace` at
`cmd/fak/ops_run.go:575-604`. The effectful production executor then constructs
the harness child at `cmd/fak/ops_run.go:992-1001` and assigns that same path to
`cmd.Dir`. A scheduled mutable run therefore edits the caller's checkout and can
share its Git index and untracked state with peer work.

The local issue corpus contains no ticket that owns managed-worktree admission
for this call path:

- Public fak#13364, **closed**, added explicit workspace binding. Its contract
  explicitly excluded a worktree manager and labels cwd as neither a sandbox nor
  a lease.
- Public fak#12648, **open**, owns local cron and repository mutexes in
  `cmd/fak/cron_run.go` and `cmd/fak/ops_schedule.go`. Serialization prevents
  overlap but still executes the mutable child in the shared checkout.
- Public fak#12480, **closed**, added managed worktrees only to Codex
  orchestration launches. It is reusable precedent, not an owner of `ops run`.

## Value

- Centrality: Core repository mutation safety for scheduled agent work.
- P1: Advanced. Every mutable scheduled harness child receives a detached
  checkout before it can execute.
- P2: Advanced. Concurrent runs stop sharing the caller's index and untracked
  files.
- P3: Preserved. Workspace canonicalization, subprocess cancellation, receipts,
  and dry-run behavior remain intact.
- P4: Preserved. Reuse the public managed-worktree lifecycle rather than adding
  another allocator or lease format.

## Working spine

`fak ops run` scheduled invocation -> canonical source root -> bounded managed
worktree preparation and durable owner lease -> child execution in the managed
cwd with worktree environment -> existing terminal receipt and lifecycle
handling.

## Core through-line

Before any effectful `runOpsRun` child starts, prepare a run-scoped managed
worktree from the selected repository root. Bind the actual child cwd and Git
environment to that checkout, and fail closed if preparation or durable lease
creation is unavailable.

## Reuse decision

Reuse `workerworktree.PrepareOwnedBounded` and `workerworktree.WorktreeEnv`.
`cmd/fak/orchestration_launch.go:713-840` already proves this public `cmd/fak`
adapter sequence: bounded preparation, complete worktree identity, environment
binding, process start, owner handoff, and pre-owner cleanup. Preparation already
writes the durable worker lease in
`internal/workerworktree/workerworktree.go:563-581`. Do not import private
dispatch or lease packages.

## Exact implementation seams

- `cmd/fak/ops_run.go` — insert managed-worktree admission between canonical
  workspace resolution and `executeOpsRun`; bind the effective worktree identity
  into the existing launch receipt.
- `cmd/fak/ops_run_test.go` — exercise the real parser and production executor
  with helper subprocesses and an injectable preparation failure.

## Scoped acceptance criteria

- [ ] An effectful scheduled `runOpsRun` invocation calls bounded managed
  preparation before `exec.CommandContext`; the child observes the prepared
  worktree as cwd and receives `workerworktree.WorktreeEnv`.
- [ ] Two concurrently admitted mutable helper runs selecting the same source
  root observe distinct managed cwd paths and distinct durable owner leases.
- [ ] Each helper writes its sentinel only in its managed checkout. The source
  root's tracked status, untracked census, and `.git/index` digest are identical
  before and after both runs.
- [ ] Preparation or lease persistence failure returns a typed refusal before
  process start; a child-start counter remains zero and the source root is
  unchanged.
- [ ] The execution receipt records source root, effective worktree path, base
  SHA, and lease identity. It never reports the caller checkout as isolated.
- [ ] Existing missing/non-directory workspace refusals and read-only dry-run
  behavior remain green and allocate no worktree.

## Production-path acceptance witness

Add `TestOpsRunScheduledMutableManagedWorktree`, which creates one real Git root,
records its index digest and status, and concurrently invokes the real
`runOpsRun` parser twice with helper child processes. Do not replace
`opsRunExecute`: the exercised path must be
`runOpsRun -> resolveOpsRunWorkspace -> workerworktree.PrepareOwnedBounded ->
executeOpsRun -> exec.CommandContext`. Assert distinct child cwd paths, distinct
lease identities readable from both worktrees, sentinel confinement, unchanged
root evidence, and truthful receipts. A preparation-refusal subtest must prove
zero child starts.

## Gold-plating boundary

Do not add a worktree manager, lease schema, distributed lock, Git format, or
private import. Do not weaken fak#12648's local mutex contract, redesign landing,
or claim OS-level filesystem sandboxing. Retain dirty worktrees for the existing
governed land/reap lifecycle.

## Witness

```powershell
go test ./cmd/fak -run '^TestOpsRunScheduledMutableManagedWorktree$' -count=1 -v
```

The command must list and execute concurrent-success and preparation-refusal
subtests. A no-tests-to-run result is not a pass.

## Verifiable Witness

`go test ./cmd/fak -run '^TestOpsRunScheduledMutableManagedWorktree$' -count=1 -v`

## Done condition

The production `runOpsRun` path launches mutable scheduled children only after
managed-worktree and lease admission; the focused witness proves distinct cwd
and leases, no source-root mutation, truthful receipts, and fail-closed
preparation; package tests and vet pass in an isolated validation checkout.

