# fix(worktree): retire one superseded flow checkout by exact proof

<!-- fak-workerworktree-key: exact-flow-retire-superseded-worktree -->

```routing
repo: fak
lane: workerworktree
paths:
  - cmd/fak-flow/recover_dead.go
  - platform/worktree/recover_dead_target.go
  - cmd/fak-flow/recover_dead_contract_test.go
expected_steps: 4
priority: P2
dependencies: []
```

## Current state

The sandbox acceptance change was independently landed on `origin/main` as
`c6224136` with its two tickets in `e79c2e2e`, then integrated at `59d8944e`.
The original flow checkout held byte-identical copies of all five paths. A
scoped `fak-flow commit --paths ...` preserved them on its ticket branch, but
its isolated-index contract left the live index at the previous HEAD. `git
status --short` therefore still showed `MM` for code and tests and `D` plus
untracked copies for the two new tickets, despite a committed branch tip.

`fak-flow recover-dead sandbox-localized-accept --repo-root ... --json` refused
that exact target because the checkout appeared dirty. `fak commit` with the
same five explicit paths refused before staging with `PEER_WIP_COLLISION`:
other sessions were attributed to the paths in the shared repository. A
read-only `fak-flow sweep --age-floor 1h --json` also aborted on an unrelated
invalid worktree registration, before it could classify the target. The exact
checkout was retained and checkpointed at
`refs/fak/wip/ticket-sandbox-localized-accept`; its lane lease was released.

The existing `fak worktree worker reap --superseded-by` proof applies only to
managed worker paths, not `.worktrees/ticket-*`. Existing `recover-dead` has no
supersession argument. No native exact-target flow cleanup path was available
for this already-landed, byte-equivalent checkout.

## Parent context

This is a follow-up to the public sandbox-localized prepared-land acceptance
ticket. It owns only the flow-checkout cleanup gap observed after that ticket
was landed and does not depend on the cross-OS targeted-reap ticket.

## Why this is next

Parallel agent work already produces isolated flow checkouts and shared-path
ownership metadata. An exact cleanup path is needed before coordinators can
retire completed lanes without touching the peer worktree fleet.

## Value

- Centrality: Core managed-worktree lifecycle integrity.
- P1: Advanced. A coordinator can retire its one completed flow checkout
  without a repository-wide sweep or mutation of peer checkouts.
- P2: Advanced. Index drift after an isolated-index commit does not strand
  already-landed work.
- P3: Preserved. Dirty unique work, active owners, and uncertain proofs remain
  protected.
- P4: Preserved. Reuse the existing exact-target recovery and archive format;
  do not add a new allocator, lease type, or broad cleanup pass.

## Working spine

Exact flow ticket and superseding commit -> owner and lease fence -> trusted
trunk ancestry -> complete changed-path and untracked-path byte proof ->
recoverable archive -> targeted checkout removal and registration cleanup.

## Core through-line

Extend `fak-flow recover-dead` with an opt-in `--superseded-by <commit>` proof
for one named `.worktrees/ticket-*` checkout. Admit a dirty target only when
every effective worktree byte, including an index-only mismatch or a path that
looks deleted in the old index but exists on disk, is equal to the named commit
on the fetched trunk. Preserve the existing fail-closed default.

## Exact implementation seams

- `cmd/fak-flow/recover_dead.go` — parse and report the optional supersession
  proof and keep the existing dry-run default.
- `platform/worktree/recover_dead_target.go` — fence one target, prove its
  complete effective file set, archive it, and remove only its registration.
- `cmd/fak-flow/recover_dead_contract_test.go` — production CLI acceptance and
  negative cases with a real Git repository.

## Likely files

- `cmd/fak-flow/recover_dead.go`
- `platform/worktree/recover_dead_target.go`
- `cmd/fak-flow/recover_dead_contract_test.go`

## Scoped acceptance criteria

- [ ] The original default still refuses a dirty target and leaves all bytes
  and refs intact.
- [ ] `--superseded-by` accepts a commit proven on fetched trunk ancestry and
  retires only the named ticket when every dirty or untracked path matches it.
- [ ] A stale live index, including `MM` and `D` plus untracked copies after
  `fak-flow commit`, does not change the byte proof's result.
- [ ] Unique untracked bytes, changed tracked bytes, symlink substitution,
  moving supersession, live owner or lease, and unknown Git state all refuse
  before removal, with a typed reason.
- [ ] A broken sibling registration neither blocks the exact-target operation
  nor gets pruned as a side effect.
- [ ] Successful cleanup retains an inspectable recovery ref and a receipt
  naming the removed checkout and superseding commit.

## Production-path acceptance witness

Add `TestRecoverDeadSupersededDirtyFlowCheckout`. Create a real repository with
two flow worktrees. Land one ticket's complete file set on trunk, advance its
branch with `fak-flow commit --paths`, and verify the live index reports
`MM`/`D` plus untracked copies. Invoke `runRecoverDead` in dry-run and apply
modes with `--superseded-by`; assert exact archive and removal of that ticket,
and unchanged bytes, ref, and registration of the sibling. Run the negative
cases through the same CLI path.

## Gold-plating boundary

Do not broaden bulk sweep eligibility, reset a shared index, force-remove
unique work, or introduce a second worktree allocator. Reuse the existing
exact-target recovery fence and archive path.

## Witness

`go test ./cmd/fak-flow ./platform/worktree -run 'TestRecoverDeadSupersededDirtyFlowCheckout|TestRecoverDead' -count=1 -v`

All named tests must execute; a no-tests-to-run result is not a pass.

## Acceptance gate

The focused real-Git tests above and `go vet ./cmd/fak-flow ./platform/worktree`
pass, and a read-only inspection proves the target's archived content equals
the superseding trunk commit while the sibling registration remains usable.

## Closure binding

The resolving public commit names this ticket key and carries the focused
witness result; the native ticket closes only after the exact-target production
path is independently read back.

## Verifiable Witness

`go test ./cmd/fak-flow ./platform/worktree -run 'TestRecoverDeadSupersededDirtyFlowCheckout|TestRecoverDead' -count=1 -v`

## Done condition

The exact-target production command retires an already-landed flow checkout
with index drift while preserving unique WIP and every sibling; the resolving
public commit is independently read back.
