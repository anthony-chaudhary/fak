<!-- fak-amdgpu-key: strix-receipt-physical-authority -->
# security(amdgpu): make Strix physical credit unforgeable from caller-authored receipts

GitHub issue: https://github.com/anthony-chaudhary/fak/issues/12376

## Current state

`StrixValidationReceipt` is described as verified hardware evidence, but `ComputeDigest` in `internal/amdgpu/strix_receipt.go:699-713` is an ordinary caller-recomputable hash. `Validate`, `validExecutionEvidence`, and `validateExecutionCredit` (`:789-906,967-1042`) check only caller-authored field shapes and internal consistency. There is no verifier-held authority, signature verification, or independent artifact read-back.

The package's own `validStrixReceipt` fixture (`internal/amdgpu/strix_validation_test.go:20-47`) fills every source, binary, shader, lease, command, and raw-output digest with the same repeated synthetic value, sets device/lease/exit booleans, recomputes the outer digest, and earns validation and candidate evaluation. Negative tests mutate a sealed receipt without recomputing its digest, so they do not exercise the attacker model.

## Parent context

Parent: #11572. This is a source-code prerequisite for #12034.

## Why now

The open physical campaign treats the v2 receipt validator as its credit boundary, but the current synthetic fixture proves that boundary accepts entirely caller-authored evidence. Hardware capture must not proceed on a forgeable admission primitive.

## Problem frame

- Priority: P0 trust-floor defect.
- Centrality: Enabling (truthful Strix native-performance admission).
- P1: advanced - caller-authored JSON can no longer mint apparent physical parity or performance credit.
- P2: advanced - invented timings cannot feed promotion and acceptance consumers.
- P3: preserved - harden the existing receipt boundary without a second receipt framework.
- P4: advanced - tests distinguish payload integrity from execution authority and require independent read-back.

## Core through-line

Untrusted serialized receipt -> verifier-controlled source/artifact/execution read-back or opaque authority -> physical-credit decision. A digest proves only that bytes are unchanged, never that hardware executed them.

## Working spine

`StrixValidationReceipt.Validate` and `CreditEligible` are the single admission spine before any consumer may grant physical credit.

## Blast radius and affected lanes

- Primary lane: `amdgpu`.
- Consumers: Strix receipt validation and legacy candidate evaluation.
- Related but separate: qwen38 scoreboard cell authority and live hardware capture campaigns.

## Quarantined fallback mechanism

Integrity-readable historical receipts may remain parseable, but absent trusted execution authority they must be explicitly non-credit. No simulated or self-authored fallback may upgrade to physical.

## Scoped acceptance criteria

- [ ] Recomputing a digest over fabricated evidence does not earn physical credit.
- [ ] Serialized fields cannot construct the trusted authority needed for physical credit.
- [ ] Verification binds source, binary, shader, command, device, and raw output to evidence independently observed by the verifier.
- [ ] Existing historical receipts remain readable but non-credit unless they satisfy the new authority contract.
- [ ] Adversarial tests recompute every public digest after fabrication and still receive refusal.

## Gold-plating boundary

No physical benchmark run, kernel tuning, new scoreboard, cryptographic PKI, service mutation, or historical-result promotion. Reuse the opaque-authority direction from #12295/#12270 where applicable.

## Done condition

No plain JSON document, even if internally consistent and freshly rehashed, can cause `Validate`, `CreditEligible`, or `EvaluateReceipt` to award Strix physical execution credit without verifier-controlled authority.

## Witness

The deterministic test witness is the adversarial receipt-authority regression below.

### Non-Forgeable Witness

```text
go test ./internal/amdgpu -run '^TestStrixValidationReceiptRejectsSelfAuthoredPhysicalCredit$' -count=1
```

The adversarial fixture must fabricate all public fields, recompute all public hashes, and prove the result remains non-credit.

## Acceptance gate

`go test ./internal/amdgpu -run '^TestStrixValidationReceiptRejectsSelfAuthoredPhysicalCredit$' -count=1`

## Closure binding

The resolving commit cites this issue in its subject and carries a `(fak amdgpu)` trailer.

## Likely files

- `internal/amdgpu/strix_receipt.go`
- `internal/amdgpu/strix_validation_test.go`
- `internal/amdgpu/strix_candidate_test.go`

## Lane

`amdgpu`

## Dependencies and dedupe

Make this a prerequisite/blocker for open #12034, whose present scope captures one physical JSON artifact and assumes the code-local receipt hardening is sufficient. Open #12295 authenticates controller semantics, while #12270 protects a different qwen38 scoreboard package. Closed #12065 fixed empty/tampered digests but not attacker-recomputed digests.

## Expected steps

6

```routing
lane: amdgpu
paths: internal/amdgpu/strix_receipt.go, internal/amdgpu/strix_validation_test.go, internal/amdgpu/strix_candidate_test.go
expected_steps: 6
```
