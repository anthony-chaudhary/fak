<!-- fak-issueorchestrator-key: opencode-halo-terminal-proof-admission -->
<!-- github-issue: https://github.com/anthony-chaudhary/fak/issues/12072 -->
# fix(issueorchestrator): require source-bound Halo proof before terminal clear

GitHub issue: [fak#12072](https://github.com/anthony-chaudhary/fak/issues/12072)

```routing
lane: issueorchestrator
paths: ["cmd/fak/issueorchestrator.go", "cmd/fak/issueorchestrator_test.go", "internal/issueorchestrator/harvest.go", "internal/issueorchestrator/harvest_test.go"]
expected_steps: 7
priority: P1
dependencies: ["#12034"]
```

## Lane

`issueorchestrator`

## Parent context

[fak#12034](https://github.com/anthony-chaudhary/fak/issues/12034) is the hardware-proof producer; this issue is its terminal proof-consumer child.

## Dependencies

- after: #12034 must land the canonical source-bound Halo validation receipt and fail-closed producer semantics. This leaf consumes that contract; it must not invent a second receipt schema.

## Current state

`FormatOpencodePrompt` recognizes Halo-related issues and asks workers to run an early probe, coordinate access, and return source-bound physical evidence, but that code is only a prompt producer. The terminal consumer is the `fak issue-orchestrator --harvest` path. `SpawnedChatRecord` and `OpencodeSpawnReceipt` carry spawn metadata only; `loadLeavesFromReceipt` reduces each chat to a `LeafReceipt`; and `ReconcileWave` can classify and auto-land a leaf as `VERIFIED_CLEARED` through `WitnessChecker` without receiving or validating any Halo proof. Prompt text therefore remains advisory: a worker can reach terminal clear with no source digest, built-executable digest, admission witness, device identity, or required subkernel result.

The completed audit also found that the current producer path can bind only the base `HEAD`, execute a pre-existing remote binary, accept a PASS-shaped receipt with zero subkernels or empty digests, and leave skipped subkernels out of the overall verdict. Those producer defects belong to #12034; this issue begins only after its corrected contract is available.

### Concrete repro witness

From current `origin/main` (`56a32e8e1ca9f047b5204637656d08b0a54f9b15`):

```powershell
go test ./internal/issueorchestrator -run 'TestHarvest_(FourStateTaxonomy|LoadFromReceiptPath|AutoLand)$' -count=1
rg -n "RequireSourceBinding|source_digest|binary_digest|lease_mode|subkernels" cmd/fak/issueorchestrator.go cmd/fak/issueorchestrator_test.go internal/issueorchestrator/harvest.go internal/issueorchestrator/harvest_test.go
```

Observed output:

```text
ok github.com/anthony-chaudhary/fak/internal/issueorchestrator 0.017s
# rg exits 1 with no matches
```

The passing tests prove that verified-clear and auto-land decisions currently need only the generic witness callback and commit metadata. The empty search proves the actual spawn/harvest path carries none of the canonical physical-proof fields.

## Why this is next

The prompt remediation and host-local GPU lease are now on trunk, but neither converts worker prose into enforceable terminal evidence. The next safe step is to specify the consumer contract now and dispatch it only after #12034 makes source-bound admitted receipts trustworthy.

## Working spine

OpenCode spawn receipt -> `loadLeavesFromReceipt` -> `LeafReceipt` -> canonical #12034 proof validation in `ReconcileWave` -> terminal clear/auto-land or durable `PENDING` reason.

## Value

- Centrality: Enabling (qualified serving and harness work on the physical Halo path)
- P1 Context: advanced - terminal state retains the exact source, binary, admission, device, and subkernel evidence instead of trusting prose.
- P2 Net value: advanced - invalid or stale physical runs cannot consume operator review as apparent completion.
- P3 Adaptation: preserved - this is a deterministic terminal gate, not a new adaptive controller.
- P4 Operations: advanced - hardware validation becomes a machine-enforced obligation with a specific pending reason and next action.

## Core through-line

For Halo-relevant OpenCode work, extend the existing spawn/harvest receipt path to carry the canonical proof produced by #12034 and refuse `VERIFIED_CLEARED` or auto-land unless that proof validates. A complete proof must bind the executed source revision (and dirty patch digest when applicable), the built executable, the admitted shared/exclusive hardware lease, the actual engine/device identity, command and exit status, and every required non-skipped correctness result. Missing, malformed, source-mismatched, all-skipped, or non-passing evidence leaves hardware validation `PENDING` with a stable machine-readable reason and next action. Non-Halo work remains unchanged.

This is the thin vertical slice: worker terminal result -> canonical Halo proof validation -> terminal clear or durable pending reason. The validator must call the #12034 contract rather than reimplementing its rules.

## Gold-plating boundary

- Do not change `internal/amdgpu`, `internal/gpulease`, `internal/flock`, `internal/issueorchestrator/opencode.go`, the OpenCode plugin, or any unrelated `cmd/fak` file; those are producer, admission, prompt, and presentation concerns.
- Do not add a second hardware receipt schema, a distributed lease service, or a general policy engine.
- Do not invent `hardware_validation.go` or another adapter file without a demonstrated terminal call path; use the existing spawn and harvest seams.
- Do not launch new physical benchmarks or claim throughput improvement. This leaf enforces proof consumption; #12034 supplies the bounded physical producer witness.
- Do not hard-code operator hostnames, endpoints, credentials, or lab topology.
- Do not allow unit fixtures, historical receipts, simulated execution, base-`HEAD` identity alone, or prompt compliance to satisfy physical proof.

## Verifiable Witness

```bash
go test ./internal/issueorchestrator -run 'Test(ReconcileWaveRejectsHaloWithoutSourceBoundProof|LoadLeavesFromReceiptPreservesHaloProof)' -count=1
go test ./cmd/fak -run 'TestIssueOrchestratorHarvest(RejectsHaloWithoutSourceBoundProof|AcceptsCompleteHaloProof)' -count=1
go vet ./internal/issueorchestrator/... ./cmd/fak/...
```

### Witness

The internal-package matrix must prove `loadLeavesFromReceipt` preserves the canonical proof and `ReconcileWave` rejects an absent proof; PASS with empty source, executable, device, or admission identity; source mismatch; zero required subkernels; all-skipped subkernels; a missing required subkernel; and any non-passing required result. The CLI matrix must prove the harvest branch refuses terminal clear/auto-land for an invalid Halo proof, accepts one complete canonical #12034 receipt, and preserves non-Halo behavior. The vet command checks both packages after the narrow regressions.

## Acceptance gate

All three witness commands exit 0 on the resolving commit, and the focused matrices demonstrate that advisory prompt text cannot clear the terminal gate without a complete canonical #12034 receipt.

### File:Line Seam

- `cmd/fak/issueorchestrator.go:20 (SpawnedChatRecord)` carries per-worker receipt data into the harvest surface.
- `cmd/fak/issueorchestrator.go:34 (OpencodeSpawnReceipt)` is the persisted spawn envelope.
- `cmd/fak/issueorchestrator.go:122 (runIssueOrchestrator harvest branch)` constructs `HarvestOptions` and emits the terminal result.
- `internal/issueorchestrator/harvest.go:43 (LeafReceipt)` is the per-leaf terminal evidence record.
- `internal/issueorchestrator/harvest.go:70 (HarvestOptions)` carries witness policy into reconciliation.
- `internal/issueorchestrator/harvest.go:84 (ReconcileWave)` sets `VERIFIED_CLEARED` and can invoke auto-land.
- `internal/issueorchestrator/harvest.go:220 (loadLeavesFromReceipt)` converts spawn receipts into harvest leaves.

### Blast Radius & Affected Lanes

- Primary lane: `internal/issueorchestrator` plus the existing `cmd/fak` issue-orchestrator entry point.
- Hard dependency lane: `internal/amdgpu` via #12034; consume its public contract without editing that lane here.
- Unaffected lanes: GPU admission, compute kernels, serving, installer, dispatch, and unrelated OpenCode work.

### Quarantined Fallback Mechanism

Until #12034 lands, keep this leaf blocked and leave Halo hardware validation `PENDING`; do not add a permissive shim. After implementation, missing or invalid proof fails closed only for Halo-relevant terminal decisions. Unit fixtures remain `[SW-VERIFIED]` and can never substitute for the physical producer witness.

## Scoped Acceptance Criteria

- [ ] 1. [SW-VERIFIED] The terminal decision for Halo-relevant work requires the canonical source-bound receipt exported by #12034.
- [ ] 2. [SW-VERIFIED] Absent, malformed, source-mismatched, admission-missing, zero-subkernel, all-skipped, missing-required, and non-passing proofs each refuse terminal clear with a stable `PENDING` reason.
- [ ] 3. [SW-VERIFIED] A complete canonical proof clears the gate, and non-Halo work preserves its existing terminal behavior.
- [ ] 4. [SW-VERIFIED] Tests prove prompt wording or a three-line worker summary alone cannot satisfy the gate.
- [ ] 5. [SW-VERIFIED] The implementation touches only the four routed spawn/harvest paths and directly reuses the #12034 validator contract.
- [ ] 6. [SW-VERIFIED] Focused tests and package vet pass on the landed commit.
- [ ] 7. [SW-VERIFIED] No fixture, mock, historical receipt, or simulated device result is represented as a physical hardware witness.

## Done condition

After #12034 lands, this leaf is done when the `fak issue-orchestrator --harvest` path consumes that exact contract, the fail-closed matrix passes, and a Halo task without complete source-bound admitted proof remains explicitly pending instead of `VERIFIED_CLEARED` or auto-landable.

## Closure binding

The resolving commit cites #12072, carries the `(fak issueorchestrator)` trailer, touches only the four routed paths, and is pushed to `origin/main` after the acceptance gate passes.

## Likely files

- `cmd/fak/issueorchestrator.go`
- `cmd/fak/issueorchestrator_test.go`
- `internal/issueorchestrator/harvest.go`
- `internal/issueorchestrator/harvest_test.go`

## Duplicate search

Reviewed [#11955](https://github.com/anthony-chaudhary/fak/issues/11955), [#11997](https://github.com/anthony-chaudhary/fak/issues/11997), [#12034](https://github.com/anthony-chaudhary/fak/issues/12034), and a full snapshot of all 2,783 open `anthony-chaudhary/fak` issues. #11955 removes proof-plugin stdout contamination; #11997 synchronizes the embedded plugin asset; #12034 produces source-bound admitted hardware evidence. No open issue owns the terminal proof-consumer gate or the dedupe key `opencode-halo-terminal-proof-admission`.
