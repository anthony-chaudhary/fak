<!-- fak-issueorchestrator-key: opencode-halo-terminal-proof-admission -->
<!-- github-issue: https://github.com/anthony-chaudhary/fak/issues/12072 -->
# fix(issueorchestrator): require source-bound Halo proof before terminal clear

GitHub issue: [fak#12072](https://github.com/anthony-chaudhary/fak/issues/12072)

```routing
lane: issueorchestrator
paths: ["internal/issueorchestrator/opencode.go", "internal/issueorchestrator/opencode_test.go", "internal/issueorchestrator/hardware_validation.go", "internal/issueorchestrator/hardware_validation_test.go"]
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

`FormatOpencodePrompt` recognizes Halo-related issues and asks workers to run an early probe, coordinate access, and return source-bound physical evidence. The terminal path does not consume or validate that evidence. Prompt text therefore remains advisory: a worker can report a terminal PASS or three-line receipt with no source digest, built-executable digest, admission witness, device identity, or required subkernel result, and the issue-orchestrator layer has no fail-closed reason to keep the hardware obligation pending.

The completed audit also found that the current producer path can bind only the base `HEAD`, execute a pre-existing remote binary, accept a PASS-shaped receipt with zero subkernels or empty digests, and leave skipped subkernels out of the overall verdict. Those producer defects belong to #12034; this issue begins only after its corrected contract is available.

### Concrete repro witness

From current `origin/main` (`56a32e8e1ca9f047b5204637656d08b0a54f9b15`):

```powershell
go test ./internal/issueorchestrator -run 'TestFormatOpencodePrompt_HaloHardwareValidation' -count=1
rg -n "RequireSourceBinding|source_digest|binary_digest|lease_mode|subkernels" internal/issueorchestrator/opencode.go internal/issueorchestrator/opencode_test.go
```

Observed output:

```text
ok github.com/anthony-chaudhary/fak/internal/issueorchestrator 0.015s
# rg exits 1 with no matches
```

The passing test proves prompt wording only. The empty search proves there is no terminal proof consumer at either existing code seam.

## Why this is next

The prompt remediation and host-local GPU lease are now on trunk, but neither converts worker prose into enforceable terminal evidence. The next safe step is to specify the consumer contract now and dispatch it only after #12034 makes source-bound admitted receipts trustworthy.

## Working spine

Halo task terminal result -> canonical #12034 receipt parser/validator -> fail-closed admission decision -> terminal clear or durable `PENDING` reason.

## Value

- Centrality: Enabling (qualified serving and harness work on the physical Halo path)
- P1 Context: advanced - terminal state retains the exact source, binary, admission, device, and subkernel evidence instead of trusting prose.
- P2 Net value: advanced - invalid or stale physical runs cannot consume operator review as apparent completion.
- P3 Adaptation: preserved - this is a deterministic terminal gate, not a new adaptive controller.
- P4 Operations: advanced - hardware validation becomes a machine-enforced obligation with a specific pending reason and next action.

## Core through-line

For Halo-relevant OpenCode work, carry the canonical proof produced by #12034 into the issue-orchestrator terminal decision and refuse terminal clear unless it validates. A complete proof must bind the executed source revision (and dirty patch digest when applicable), the built executable, the admitted shared/exclusive hardware lease, the actual engine/device identity, command and exit status, and every required non-skipped correctness result. Missing, malformed, source-mismatched, all-skipped, or non-passing evidence leaves hardware validation `PENDING` with a stable machine-readable reason and next action. Non-Halo work remains unchanged.

This is the thin vertical slice: worker terminal result -> canonical Halo proof validation -> terminal clear or durable pending reason. The validator must call the #12034 contract rather than reimplementing its rules.

## Gold-plating boundary

- Do not change `internal/amdgpu`, `internal/gpulease`, `internal/flock`, `cmd/fak`, or the OpenCode plugin; those are producer, admission, CLI, and presentation concerns.
- Do not add a second hardware receipt schema, a distributed lease service, or a general policy engine.
- Do not launch new physical benchmarks or claim throughput improvement. This leaf enforces proof consumption; #12034 supplies the bounded physical producer witness.
- Do not hard-code operator hostnames, endpoints, credentials, or lab topology.
- Do not allow unit fixtures, historical receipts, simulated execution, base-`HEAD` identity alone, or prompt compliance to satisfy physical proof.

## Verifiable Witness

```bash
go test ./internal/issueorchestrator -run 'Test(HaloTerminalProofAdmission|FormatOpencodePrompt_HaloHardwareValidation)' -count=1
go vet ./internal/issueorchestrator/...
```

### Witness

The focused matrix must reject an absent proof; PASS with empty source, executable, device, or admission identity; source mismatch; zero required subkernels; all-skipped subkernels; a missing required subkernel; and any non-passing required result. It must accept one complete canonical #12034 receipt, preserve terminal behavior for non-Halo issues, and return a stable `PENDING` reason plus exact next action for each rejection. The vet command checks the complete package after the narrow regression witness.

## Acceptance gate

Both witness commands exit 0 on the resolving commit, and the focused matrix demonstrates that advisory prompt text cannot clear the terminal gate without a complete canonical #12034 receipt.

### File:Line Seam

- `internal/issueorchestrator/opencode.go:11 (isHaloHardwareRelevant)` selects work requiring physical Halo proof.
- `internal/issueorchestrator/opencode.go:37 (FormatOpencodePrompt)` currently emits advisory instructions without a terminal gate.
- `internal/issueorchestrator/opencode_test.go:86 (TestFormatOpencodePrompt_HaloHardwareValidation)` currently asserts wording only.
- `internal/issueorchestrator/hardware_validation.go:1` will contain the narrow canonical-proof adapter after #12034 lands.
- `internal/issueorchestrator/hardware_validation_test.go:1` will contain the fail-closed terminal matrix.

### Blast Radius & Affected Lanes

- Primary lane: `internal/issueorchestrator`.
- Hard dependency lane: `internal/amdgpu` via #12034; consume its public contract without editing that lane here.
- Unaffected lanes: GPU admission, compute kernels, serving, installer, dispatch, and unrelated OpenCode work.

### Quarantined Fallback Mechanism

Until #12034 lands, keep this leaf blocked and leave Halo hardware validation `PENDING`; do not add a permissive shim. After implementation, missing or invalid proof fails closed only for Halo-relevant terminal decisions. Unit fixtures remain `[SW-VERIFIED]` and can never substitute for the physical producer witness.

## Scoped Acceptance Criteria

- [ ] 1. [SW-VERIFIED] The terminal decision for Halo-relevant work requires the canonical source-bound receipt exported by #12034.
- [ ] 2. [SW-VERIFIED] Absent, malformed, source-mismatched, admission-missing, zero-subkernel, all-skipped, missing-required, and non-passing proofs each refuse terminal clear with a stable `PENDING` reason.
- [ ] 3. [SW-VERIFIED] A complete canonical proof clears the gate, and non-Halo work preserves its existing terminal behavior.
- [ ] 4. [SW-VERIFIED] Tests prove prompt wording or a three-line worker summary alone cannot satisfy the gate.
- [ ] 5. [SW-VERIFIED] The implementation touches only the four routed `internal/issueorchestrator` paths and directly reuses the #12034 validator contract.
- [ ] 6. [SW-VERIFIED] Focused tests and package vet pass on the landed commit.
- [ ] 7. [SW-VERIFIED] No fixture, mock, historical receipt, or simulated device result is represented as a physical hardware witness.

## Done condition

After #12034 lands, this leaf is done when the issue-orchestrator terminal path consumes that exact contract, the fail-closed matrix passes, and a Halo task without complete source-bound admitted proof remains explicitly pending instead of terminally clear.

## Closure binding

The resolving commit cites #12072, carries the `(fak issueorchestrator)` trailer, touches only the four routed paths, and is pushed to `origin/main` after the acceptance gate passes.

## Likely files

- `internal/issueorchestrator/opencode.go`
- `internal/issueorchestrator/opencode_test.go`
- `internal/issueorchestrator/hardware_validation.go`
- `internal/issueorchestrator/hardware_validation_test.go`

## Duplicate search

Reviewed [#11955](https://github.com/anthony-chaudhary/fak/issues/11955), [#11997](https://github.com/anthony-chaudhary/fak/issues/11997), [#12034](https://github.com/anthony-chaudhary/fak/issues/12034), and a full snapshot of all 2,783 open `anthony-chaudhary/fak` issues. #11955 removes proof-plugin stdout contamination; #11997 synchronizes the embedded plugin asset; #12034 produces source-bound admitted hardware evidence. No open issue owns the terminal proof-consumer gate or the dedupe key `opencode-halo-terminal-proof-admission`.
