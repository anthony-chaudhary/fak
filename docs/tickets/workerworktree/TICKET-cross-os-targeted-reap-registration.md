# fix(workerworktree): preserve cross-OS registrations during targeted reap

<!-- fak-workerworktree-key: cross-os-targeted-reap-registration -->

```routing
repo: fak
lane: workerworktree
paths:
  - internal/workerworktree/lease.go
  - internal/workerworktree/workerworktree.go
  - internal/workerworktree/lease_test.go
expected_steps: 4
priority: P1
dependencies: []
```

## Current state

A WSL invocation of `fak worktree worker prepare --root /mnt/c/work/fak
--wt-root /tmp/fak-sandbox-workers-01a0` successfully created
`/tmp/.../fak-worker-wt-worktree-878b711de32d`. About one minute later, `fak
worktree worker land` failed at `git ls-files`. The worker's `.git` still named
`/mnt/c/work/fak/.git/worktrees/fak-worker-wt-worktree-878b711de32d`, but that
shared administration directory no longer existed. A Windows `worktree/reap-dd02`
lane was concurrent. This temporal evidence does not identify which process or
call removed the registration.

The public lifecycle contains a concrete mechanism consistent with the result.
`SweepDeadWorktrees` recognizes a `/tmp/...` or `/mnt/...` Git directory as a
foreign-platform registration and skips direct deletion at
`internal/workerworktree/lease.go:210-213`. If the same sweep reaps any other
local registration, however, it finishes with repository-wide `git worktree
prune --expire now` at `lease.go:355-358`. Windows Git cannot stat a live WSL
`/tmp` checkout and may therefore remove its shared admin directory. Other
worker cleanup paths in `internal/workerworktree/workerworktree.go` also use an
unscoped `git worktree prune` after targeted cleanup.

## Existing owners and dedupe

- Public fak#11813, **closed**, introduced foreign-OS path recognition and
  preservation during sweep inspection.
- Public fak#11814, **closed**, extended foreign-platform Git registration
  preservation and its direct no-reap test.

Those fixes do not own the mixed-cohort case where one eligible local reap
triggers a broad Git prune that independently removes a foreign registration.
No open native ticket in the local issue corpus covers that reciprocal case.

## Value

- Centrality: Core managed-worktree lifecycle integrity.
- P1: Advanced. A worker prepared in WSL retains its shared Git administration
  metadata while Windows performs unrelated cleanup.
- P2: Advanced. Targeted reap can continue without invalidating live work in a
  different OS namespace.
- P3: Preserved. Existing dead-local eligibility, owner/lease protection, and
  dirty-worktree retention remain unchanged.
- P4: Preserved. Narrow the cleanup operation; do not add cross-OS PID probing or
  a new lease format.

## Working spine

WSL prepare against a Windows-hosted common Git directory -> durable foreign
registration -> unrelated Windows local reap -> targeted administrative cleanup
only -> WSL worker Git commands remain valid -> governed land continues.

## Core through-line

Once inspection classifies a registration as foreign-platform and therefore
unverifiable locally, no cleanup consequence earned by another worktree may
remove it. Reap only the registration selected by the existing owner, lease,
dirty-state, and path guards.

## Exact implementation seams

- `internal/workerworktree/lease.go` — replace or fence the trailing broad prune
  after a mixed `SweepDeadWorktrees` pass.
- `internal/workerworktree/workerworktree.go` — keep single-worktree reap and
  partial-prepare cleanup from broad-pruning unrelated foreign registrations.
- `internal/workerworktree/lease_test.go` — add the mixed foreign-live plus
  local-dead regression and bounded cleanup call audit.

## Scoped acceptance criteria

- [ ] A Windows sweep containing a foreign `/tmp/.../.git` registration and one
  eligible dead local registration reaps the local entry and preserves the
  foreign admin directory byte-for-byte.
- [ ] The same witness proves no repository-wide `git worktree prune` can run as
  a side effect of targeted sweep/reap while a foreign registration is present.
- [ ] A WSL worker whose `.git` points through that preserved admin directory can
  run `git ls-files` after the Windows cleanup and remains eligible for the
  existing land path.
- [ ] Local dead registrations are still removed, and active-owner, live-lease,
  dirty, idle-pool, and unknown-owner protections remain green.
- [ ] Cleanup failure is reported or retained for retry; it never falls back to
  a broad prune that can delete unrelated registrations.

## Production-path acceptance witness

Add `TestSweepDeadWorktreesMixedCrossOSRegistration`. On Windows, initialize one
real repository on a Windows-visible path, seed a live WSL-style registration
whose admin `gitdir` points to `/tmp/fak-sandbox-workers-.../.git`, and seed one
eligible dead local managed registration. Invoke the real
`SweepDeadWorktrees` path used by worker prepare/reap. Assert exactly the local
entry is reported and removed, the foreign admin files are unchanged, and the
recorded Git calls contain no unscoped prune. Then, through WSL, bind a temporary
live checkout to the preserved registration and prove `git ls-files` succeeds
after the Windows cleanup. Do not infer the original remover from this test.

## Gold-plating boundary

Do not add distributed leases, translate or probe foreign PIDs, copy Git admin
directories, redesign worktree storage, or serialize all Windows and WSL work.
Do not weaken local dead-worktree cleanup. Scope the change to preventing a
targeted cleanup from broad-pruning unrelated registrations.

## Witness

```powershell
go test ./internal/workerworktree -run 'TestSweepDeadWorktreesMixedCrossOSRegistration|TestSweepDeadWorktrees_PreservesForeignPlatformRegistration|TestSweepDeadWorktrees_ReapsLocalDeadRegistration' -count=1 -v
```

All three named tests must execute; a no-tests-to-run result is not a pass.

## Verifiable Witness

`go test ./internal/workerworktree -run 'TestSweepDeadWorktreesMixedCrossOSRegistration|TestSweepDeadWorktrees_PreservesForeignPlatformRegistration|TestSweepDeadWorktrees_ReapsLocalDeadRegistration' -count=1 -v`

## Done condition

The mixed production sweep reaps its eligible local target without issuing a
repository-wide prune or changing the live foreign registration; the WSL
checkout still executes `git ls-files`; existing local eligibility protections
remain green; the resolving public commit is independently read back.

