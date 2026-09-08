# fix(workerworktree): preserve sibling workspace topology during candidate validation

<!-- fak-workerworktree-key: portable-candidate-validation-root -->
<!-- github-issue: 12447 -->

```routing
lane: workerworktree
paths:
  - internal/workerworktree/land.go
  - internal/workerworktree/workerworktree_live_test.go
expected_steps: 5
priority: P1
dependencies:
  - fak#11235
```

## Current state

`fak worktree worker land --root <repo> --verify go-build` checks the prospective merge commit in a system temporary directory. A repository whose checked-in `go.work` refers to a sibling module therefore resolves the sibling relative to the wrong parent and refuses an otherwise valid cross-repository land.

## Why this is next

Default-on isolation is now shipped, so verified landing must work for every selected repository root before companion consumers can rely on that default safely.

## Parent context

#11235

## Problem frame

- Centrality: Core
- P1: advanced - the canonical landing primitive verifies and lands both its own and companion repository roots.
- P2: advanced - one root-parameterized implementation preserves repository-relative workspace semantics.
- P3: preserved - verification failure remains fail-closed and the worker/recovery ref remain intact.
- P4: advanced - the existing typed refusal/success receipts remain authoritative.

## Working spine

Selected repository root -> isolated merge commit -> topology-preserving candidate checkout -> compilation witness -> compare-and-swap -> typed receipt.

## Core through-line

Materialize the isolated post-merge candidate next to the selected repository root so checked-in relative workspace paths keep their original parent topology during verification.

## Gold-plating boundary

Do not add a module resolver, copy sibling repositories, import non-SDK code, weaken verification, or duplicate the merge engine. This is only candidate-directory placement plus a live disposable-root witness.

## Concrete repro witness

```powershell
fak worktree worker land --root C:\scratch\companion --worktree <worker> --base-sha <sha> --paths app.go --verify go-build
```

The current command refuses before CAS with `go: cannot load module ../sibling listed in go.work file` because its candidate directory is under the system temporary root rather than beside the selected repository.

## Exact file:line seams

- `internal/workerworktree/land.go:744-770` (`landIsolated`, post-merge candidate construction)
- `internal/workerworktree/workerworktree_live_test.go:218-320` (isolated candidate verification witnesses)

## Blast radius and affected lanes

- Primary lane: `workerworktree`
- Affected: `fak worktree worker land --root ... --verify go-build` for repositories with relative sibling modules.
- Unaffected: merge construction, CAS/retry semantics, serving, model execution, ABI, and unrelated implementation packages.

## Quarantined fallback mechanism

Until fixed, the verifier returns a typed failure before CAS and preserves the worker and recovery ref. No shared-index or unverified fallback is permitted.

## Scoped acceptance criteria

- [ ] 1. [SW-VERIFIED] Candidate verification materializes beside the selected root and keeps `go.work` sibling paths valid.
- [ ] 2. [SW-VERIFIED] A disposable pair of sibling repositories completes verified isolated landing through an arbitrary `--root`.
- [ ] 3. [SW-VERIFIED] Failed verification still leaves trunk HEAD, index, worktree, and worker unchanged.
- [ ] 4. [SW-VERIFIED] Existing public single-module landing tests remain green.
- [ ] 5. [SW-VERIFIED] No non-SDK imports or repository-specific module names enter the implementation.

## Definition of done

- [ ] Topology-preserving candidate placement is implemented.
- [ ] Sibling-workspace and existing single-module tests pass.

## Likely files

- `internal/workerworktree/land.go`
- `internal/workerworktree/workerworktree_live_test.go`

## Witness

`go test ./internal/workerworktree -run 'TestLiveLand.*(Sibling|Verified)' -count=1` must exit 0.

### Verifiable Witness

```powershell
go test ./internal/workerworktree -run 'TestLiveLand.*(Sibling|Verified)' -count=1
```

## Done condition

The public root-parameterized lander verifies a candidate from a sibling-workspace repository without weakening fail-closed behavior, and the focused live tests pass.

## Done condition / witness

The focused live witness and `go test ./internal/workerworktree -count=1` both exit 0 in an isolated checkout.

## Acceptance gate

`go test ./internal/workerworktree -count=1`

## Closure binding

The resolving signed-off commit cites the GitHub issue number, carries `(fak workerworktree)`, and is read back on `origin/main` before closure.

## Expected steps

5

## Work estimate

Estimate: 2 points

## Overall completion contribution

Contribution: 2/10 points for the portable default-on landing campaign.


## Lane

workerworktree
