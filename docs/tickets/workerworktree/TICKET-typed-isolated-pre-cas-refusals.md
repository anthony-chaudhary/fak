# fix(workerworktree): type every isolated pre-CAS refusal as reconciliation-required

<!-- fak-workerworktree-key: typed-isolated-pre-cas-refusals -->
<!-- github-issue: 12448 -->

```routing
lane: workerworktree
paths:
  - internal/workerworktree/land.go
  - internal/workerworktree/workerworktree_test.go
expected_steps: 4
priority: P1
dependencies:
  - fak#11235
  - fak#12447
```

## Current state

The default isolated landing route is fail-closed before compare-and-swap, but several refusal branches return an untyped `Result` instead of the portable recovery contract `Code: reconciliation-required` plus `Preserved: true`.

Observed on `origin/main` `5074b5cbc8de413259fea2ff135d0ba7778b903c` in `internal/workerworktree/land.go`:

- post-apply disambiguation invariant failure;
- recovery-anchor publication failure;
- required remote recovery witness failure;
- selected-root resolution failure;
- topology-preserving candidate temporary-directory creation failure;
- candidate worktree checkout failure; and
- candidate verification/build failure.

Apply conflicts, detached HEAD or identity errors, isolated index/commit construction failures, and exhausted CAS retries already use the shared reconciliation helper and return typed preserved-state receipts. The branches above construct `Result{OK:false,...}` directly, leaving `Code` empty and `Preserved` false even though worker and recovery state are intentionally retained.

## Why this is next

#11235 established fail-closed isolated landing, and #12447 preserved sibling workspace topology for candidate verification. Normalizing the remaining refusal receipts completes the portable machine contract those default-on callers consume.

## Parent context

#11235

## Problem frame

- Centrality: Core
- P1: advanced - every isolated pre-CAS refusal becomes machine-routable without parsing human reason text.
- P2: preserved - one public root-parameterized landing primitive remains consumable by both `fak` and companion repositories.
- P3: preserved - trunk HEAD, the shared index, the selected root, and the worker remain unchanged on refusal.
- P4: advanced - typed receipts consistently distinguish reconciliation-required preserved state from success and no-op outcomes.

## Working spine

Selected repository root -> isolated candidate construction -> pre-CAS gate -> typed preserved-state refusal -> explicit reconciliation.

## Core through-line

Route every isolated pre-CAS refusal through the existing reconciliation-result helper while retaining the branch-specific reason, detail, disambiguation, recovery-ref, and remote-recovery evidence.

## Gold-plating boundary

Do not alter merge construction, retry policy, candidate verification semantics, topology placement, recovery publication policy, or explicit `FAK_LAND_ISOLATED=off` shared-mode behavior. Do not add a second merge engine or repository-specific logic.

## Concrete repro witness

```powershell
$env:FAK_FAST='0'
.\test.ps1 ./internal/workerworktree -run 'TestLandIsolated.*(Disambiguation|Recovery|Remote|Candidate|Verification).*RequiresReconciliation' -count=1 -v
```

Before the fix, the focused cases report an empty `Code` and `Preserved == false` even though landing refuses before CAS and retains the candidate state.

## Exact file:line seams

- `internal/workerworktree/land.go:709-712` (`landIsolated`, post-apply disambiguation refusal)
- `internal/workerworktree/land.go:725-738` (`landIsolated`, recovery publication refusals)
- `internal/workerworktree/land.go:748-778` (`landIsolated`, candidate materialization and verification refusals)
- `internal/workerworktree/workerworktree_test.go:800-1060` (isolated landing receipt and mutation witnesses)

## Blast radius and affected lanes

- Primary lane: `workerworktree`
- Affected: public `fak worktree worker land --root ...` callers, including companion repository landing processes and issue/OpenCode orchestration.
- Unaffected: merge composition, CAS/retry semantics, serving, model execution, ABI, and explicit legacy shared-mode landing.

## Quarantined fallback mechanism

Current trunk already refuses these paths before CAS and preserves the worker; callers must retain the worker and inspect human `Reason` text until typed normalization lands. No shared-index, unverified, or automatic-reap fallback is permitted.

## Scoped acceptance criteria

- [ ] 1. [SW-VERIFIED] Every listed isolated pre-CAS refusal returns `Code: reconciliation-required` and `Preserved: true`.
- [ ] 2. [SW-VERIFIED] Existing `Reason`, `Detail`, `RecoveryRef`, `RemoteRecovery`, and `Disambiguation` evidence remains populated where applicable.
- [ ] 3. [SW-VERIFIED] Focused table-driven tests cover each normalized refusal branch.
- [ ] 4. [SW-VERIFIED] Trunk HEAD, shared index, shared worktree, and worker changes remain unchanged on each refusal.
- [ ] 5. [SW-VERIFIED] Success receipts and explicit `FAK_LAND_ISOLATED=off` shared-mode behavior remain unchanged.

## Definition of done

- [ ] All isolated pre-CAS refusal receipts share the portable reconciliation-required contract.
- [ ] Focused and full `internal/workerworktree` tests pass from a clean checkout.

## Likely files

- `internal/workerworktree/land.go`
- `internal/workerworktree/workerworktree_test.go`

## Witness

`go test ./internal/workerworktree -run 'TestLandIsolated.*(Disambiguation|Recovery|Remote|Candidate|Verification).*RequiresReconciliation' -count=1` must exit 0 and assert the typed fields plus state preservation for every normalized branch.

### Verifiable Witness

```powershell
go test ./internal/workerworktree -run 'TestLandIsolated.*(Disambiguation|Recovery|Remote|Candidate|Verification).*RequiresReconciliation' -count=1
```

## Done condition

Every default isolated refusal that occurs before CAS emits `reconciliation-required` with preserved state, without changing branch-specific evidence or shared-mode behavior.

## Done condition / witness

The focused receipt witness and `go test ./internal/workerworktree -count=1` both exit 0 in an isolated checkout.

## Acceptance gate

`go test ./internal/workerworktree -count=1`

## Closure binding

The resolving signed-off commit cites GitHub issue #12448, carries `(fak workerworktree)`, and is read back on `origin/main` before closure.

## Expected steps

4

## Work estimate

Estimate: 2 points

## Overall completion contribution

Contribution: 1/10 point for the portable default-on landing campaign.

## Lane

workerworktree
