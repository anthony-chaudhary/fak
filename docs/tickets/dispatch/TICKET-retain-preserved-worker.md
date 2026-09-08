# fix(dispatch): retain preserved worker after land refusal

<!-- fak-dispatch-key: retain-preserved-worker-after-land-refusal -->
<!-- github-issue: 12449 -->

```routing
lane: cmd/fak/dispatch
paths:
  - cmd/fak/dispatch_tick_witness.go
  - cmd/fak/dispatch_tick_witness_test.go
  - cmd/fak/dispatch_tick_witness_land_retry_test.go
expected_steps: 4
priority: P1
dependencies:
  - fak#11235
  - fak#12379
```

## Current state

The default dispatch completion path consumes a structured `workerworktree.Result`, but reaps the managed worker and deletes its `.worktree` sidecar after every land attempt. A fail-closed `reconciliation-required` result may explicitly report `Preserved: true`, yet the caller destroys the retained recovery state.

## Why this is next

Default-on managed issue workers are only safe when a refused landing stays recoverable. The public low-level lander already preserves conflicts and terminal refusals; this consumer must honor that contract.

## Parent context

#12379 and #11235.

## Problem frame

- Centrality: Core - this is the default issue-worker completion path.
- P1: advanced - managed workers land through one public primitive.
- P2: advanced - only durable success authorizes reap and sidecar removal.
- P3: preserved - retryable readback races retain the existing bounded retry behavior.
- P4: advanced - terminal receipts expose the result code, retained path, and reconciliation action.

## Working spine

Dead managed worker -> structured land result -> transient retry or terminal classification -> success-only reap -> sidecar retained or removed -> operator-visible receipt.

## Core through-line

Propagate the structured land result through dispatch completion. Reap exactly once and remove the sidecar only after a durable successful land; keep the worktree and sidecar whenever the result is preserved or requires reconciliation.

## Gold-plating boundary

Do not change the low-level merge/CAS implementation, candidate validation topology, unsafe shared-root opt-out, worker launch admission, or add an implicit discard path.

## Concrete repro witness

```powershell
$env:FAK_FAST='0'
.\test.ps1 ./cmd/fak -run 'Test(DeterministicRefusalRetainsWorker|RefusedLandExhaustionRetainsWorker|CleanLandSingleAttemptThenReap)$' -count=1 -v
```

At the base revision, the refusal tests assert the stale `[land,reap]` sequence and therefore prove recoverable state is destroyed.

## Exact file:line seams

- `cmd/fak/dispatch_tick_witness.go:75-140` (`landAndReapWorkerWorktree`, `landAndReapWorkerWorktreeDefault`)
- `cmd/fak/dispatch_tick_witness_test.go:350-430` (sidecar success/refusal lifecycle witnesses)
- `cmd/fak/dispatch_tick_witness_land_retry_test.go:63-104` (refusal and success sequencing witnesses)

## Blast radius and affected lanes

- Primary lane: `cmd/fak/dispatch`
- Affected: automatic completion of managed issue/OpenCode workers.
- Unaffected: low-level workerworktree merge/CAS, shared-root explicit opt-out, serving, model execution, and unrelated commands.

## Quarantined fallback mechanism

Until fixed, the low-level lander still fails closed and preserves its recovery ref. Operators can invoke the public worktree reconciliation commands manually, but automatic dispatch completion must not silently discard the retained worker.

## Scoped acceptance criteria

- [x] `Code: reconciliation-required` or `Preserved: true` prevents reap and sidecar deletion.
- [x] Deterministic conflicts and exhausted transient retry budgets retain the managed worktree.
- [x] Durable successful land reaps exactly once and removes the sidecar.
- [x] Human and JSON diagnostics expose the structured refusal, retained path, and next reconciliation action.
- [x] No implicit discard path or low-level landing behavior changes.

## Definition of done

- [x] The dispatch completion seam returns and consumes structured `workerworktree.Result`.
- [x] Focused refusal, retry exhaustion, success, sidecar, and JSON receipt tests pass.
- [x] The affected package compiles and `go vet ./cmd/fak` passes.

## Likely files

- `cmd/fak/dispatch_tick_witness.go`
- `cmd/fak/dispatch_tick_witness_test.go`
- `cmd/fak/dispatch_tick_witness_land_retry_test.go`

## Witness

`go test ./cmd/fak -run 'Test(DeterministicRefusalRetainsWorker|RefusedLandExhaustionRetainsWorker|CleanLandSingleAttemptThenReap)$' -count=1` must exit 0.

## Done condition

Every terminal or preserved land refusal leaves the worker and `.worktree` sidecar available for reconciliation, while a durable successful land still performs one reap and removes the sidecar.

## Acceptance gate

The focused dispatch completion tests, a compile-only `cmd/fak` package run, and `go vet ./cmd/fak` pass.

## Validation receipts

- RED: the base implementation failed `TestDeterministicRefusalRetainsWorker` and `TestRefusedLandExhaustionRetainsWorker` because both observed an unwanted reap.
- GREEN: the focused eight-test dispatch completion slice exits zero.
- GREEN: `.\test.ps1 ./cmd/fak -run '^$' -count=1` exits zero.
- GREEN: `go vet ./cmd/fak` exits zero.
- The unfiltered `cmd/fak` suite was sampled for 90 seconds with no failure output, then bounded rather than waiting on unrelated long-running host probes; it is not claimed as a full-suite witness.

## Closure binding

The resolving signed-off commit cites #12449, carries `(fak dispatch)`, and is independently landed/read back before issue closure.

## Expected steps

4

## Work estimate

Estimate: 2 points.

## Overall completion contribution

Contribution: 1/10 points for the portable default-on landing campaign.

## Lane

cmd/fak/dispatch
