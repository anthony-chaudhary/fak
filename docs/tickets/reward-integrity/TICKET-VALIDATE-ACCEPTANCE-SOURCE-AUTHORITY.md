<!-- fak-validate-key: acceptance-source-bound-hardware-authority -->
# security(validate): bind hardware acceptance to source, binary, and independent execution authority

GitHub issue: https://github.com/anthony-chaudhary/fak/issues/12381

## Current state

`cmd/fak/validate_acceptance.go:511-611` strengthens fanout acceptance with `Verified`, provenance strings, capture-command text, a raw artifact path, and a recomputed SHA-256. Those checks establish payload integrity and reject known synthetic markers, but all positive inputs can still be authored together by the receipt producer. The raw artifact is not bound to the source revision, mapped validator binary, shader/model identities, or a verifier-controlled execution authority.

The empirical roofline branch delegates to receipt verification, whose physical-witness/signature recurrence is tracked by reopened #11724. Even after that point fix, the generic acceptance boundary needs an explicit rule: a producer-authored receipt plus producer-authored raw file is not independent hardware proof.

## Parent context

Parent: #11572. Related completed hardening: #11802 and #11965. Related recurrence: #11724.

## Why now

The acceptance command is a release-facing consumer of the same evidence under active Strix campaigns. Producer-authored strings and a matching producer-authored raw file must not become the final authority after lower-level receipt repairs land.

## Problem frame

- Priority: P0 acceptance trust-floor defect.
- Centrality: Enabling (truthful release-facing hardware acceptance).
- P1: advanced - arbitrary hardware-success fields and a matching raw file cannot satisfy acceptance.
- P2: advanced - source-bound authority replaces optimization against known fixture digests and strings.
- P3: preserved - extend the current entry point with one reusable authority contract.
- P4: advanced - the verifier independently binds executable and source identity and rejects unavailable bindings.

## Core through-line

Acceptance input -> independently resolve executing source/binary/model/shader identity -> verify raw capture binding and trusted execution authority -> compute acceptance. Missing authority yields non-credit, never a string-based pass.

## Working spine

The existing `runAcceptanceValidation` entry point resolves the receipt and raw capture, obtains verifier-controlled identity authority, and only then computes a success report.

## Blast radius and affected lanes

- Primary lane: `validate-acceptance`.
- Affected consumers: generic acceptance report and Strix fanout/roofline acceptance adapters.
- Related producer-specific authority remains owned by amdgpu/qwen38 leaves.

## Quarantined fallback mechanism

Historical receipts remain integrity-readable but non-credit. Unsupported platforms or missing executable/artifact observation return a typed unavailable/refused result; they do not fall back to claimed provenance strings.

## Scoped acceptance criteria

- [ ] A self-authored receipt and matching self-authored raw file cannot pass hardware acceptance.
- [ ] Acceptance binds a full source revision and mapped binary digest to the captured execution.
- [ ] Relevant model/shader/artifact identities are checked against verifier-observed bytes.
- [ ] Producer strings such as `hardware_execution`, command text, and `Verified=true` are never sufficient authority.
- [ ] Missing, stale, mismatched, or unsupported authority fails before any success report is emitted.
- [ ] Both fanout and empirical-roofline adapters use the same trust distinction between integrity and execution proof.

## Gold-plating boundary

No hardware run, PKI deployment, benchmark threshold change, new acceptance campaign, kernel work, or service lifecycle mutation. Do not reintroduce known-digest denylists as the primary defense.

## Done condition

The generic acceptance entry point can award hardware credit only when source, executing binary, required artifacts, and raw capture are bound by verifier-controlled evidence; fully rehashed producer-authored fixtures remain refused.

## Witness

The deterministic test witness is the acceptance-authority adversarial regression below.

### Non-Forgeable Witness

```text
go test ./cmd/fak -run '^TestAcceptanceRejectsSelfAuthoredSourceUnboundHardwareProof$' -count=1
```

Preserve the existing synthetic, missing-artifact, digest-mismatch, and simulated-roofline refusal matrix.

## Acceptance gate

`go test ./cmd/fak -run '^TestAcceptanceRejectsSelfAuthoredSourceUnboundHardwareProof$' -count=1`

## Closure binding

The resolving commit cites this issue in its subject and carries a `(fak validate-acceptance)` trailer.

## Likely files

- `cmd/fak/validate_acceptance.go`
- `cmd/fak/validate_acceptance_test.go`

## Lane

`validate-acceptance`

## Dependencies and dedupe

Closed #11802 removed automatic synthetic witness generation. Closed #11965 added raw-artifact integrity and explicitly noted that integrity is not independent hardware attestation. Reopened #11724 owns roofline receipt verification. This issue owns the remaining generic source/binary/execution authority at the acceptance consumer.

## Expected steps

6

```routing
lane: validate-acceptance
paths: cmd/fak/validate_acceptance.go, cmd/fak/validate_acceptance_test.go
expected_steps: 6
```
