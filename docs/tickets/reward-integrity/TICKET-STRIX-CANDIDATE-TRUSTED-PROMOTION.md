<!-- fak-amdgpu-key: strix-candidate-trusted-promotion -->
# fix(amdgpu): require trusted receipts and pinned baselines for Strix candidate promotion

GitHub issue: https://github.com/anthony-chaudhary/fak/issues/12377

## Current state

`NewStrixCandidateRegistry` seeds historical constants from a JSON artifact and immediately constructs seven `VERIFIED_LIFT` comparisons (`internal/amdgpu/strix_candidate.go:135-397`). They enter the scoreboard as `PROMOTED` without a current receipt, even though `EvaluateReceipt` separately rejects historical v1 receipts for current performance credit (`:500-520`).

The raw `EvaluateCandidate` path accepts receipt-free `StrixAblationResult` values. `evaluateCandidateInternal` prefers the caller's `BaselineArm` over the pinned registry baseline (`:650-684`), then trusts caller parity/noise inputs. An inflated baseline and small candidate latency can therefore mint a promotion.

## Parent context

Parent: #11572. Depends on the Strix receipt physical-authority repair.

## Why now

The registry is already a credit consumer and currently starts green with seven historical promotions. Leaving it unchanged would bypass stronger receipt work through a separate raw evaluation path.

## Problem frame

- Priority: P1 performance-credit integrity defect.
- Centrality: Enabling (truthful Strix candidate selection).
- P1: advanced - a newly constructed registry reports no wins before current work occurs.
- P2: advanced - callers cannot select the denominator that determines their speedup reward.
- P3: preserved - retain reference metadata and safe registry APIs while changing credit-bearing transitions only.
- P4: advanced - promotion becomes authority-backed and recomputed against the pinned baseline.

## Core through-line

Pinned reference metadata + trusted current candidate observation -> registry recomputes comparison -> promotion only after provenance, parity, and threshold gates. Historical constants remain references, not live wins.

## Working spine

Trusted current observation enters `EvaluateReceipt`, is compared with the registry-owned baseline, and alone may update the scoreboard to `PROMOTED`.

## Blast radius and affected lanes

- Primary lane: `amdgpu`.
- Affected surface: legacy Strix candidate registry and scoreboard.
- Unaffected: qwen38 scoreboard-v2 and physical runner implementations.

## Quarantined fallback mechanism

On missing authority, preserve the candidate row as `UNVERIFIED`/neutral reference metadata; never fall back to `PROMOTED`. Invalid caller baselines are rejected or ignored in favor of pinned data.

## Scoped acceptance criteria

- [ ] A new registry contains no receipt-free `PROMOTED` rows.
- [ ] Historical benchmark constants remain labeled reference-only.
- [ ] Caller-supplied baseline latency cannot replace the pinned baseline for credit.
- [ ] Raw receipt-free evaluation cannot award promotion.
- [ ] Trusted evaluation recomputes speedup, lift, parity, and verdict from admitted evidence.

## Gold-plating boundary

No benchmark rerun, new candidate dimensions, threshold tuning, hardware contact, or migration to qwen38 scoreboard-v2. Do not delete useful historical reference metadata.

## Done condition

Every `PROMOTED` Strix candidate is derived from trusted current evidence and the registry-owned pinned denominator; constructing a registry or supplying raw caller metrics cannot create a win.

## Witness

The deterministic test witness is the focused registry promotion suite below.

### Non-Forgeable Witness

```text
go test ./internal/amdgpu -run '^TestStrixCandidateRegistry_(StartsUncredited|RejectsCallerBaselinePromotion|PromotesTrustedReceipt)$' -count=1
```

## Acceptance gate

`go test ./internal/amdgpu -run '^TestStrixCandidateRegistry_' -count=1`

## Closure binding

The resolving commit cites this issue in its subject and carries a `(fak amdgpu)` trailer.

## Likely files

- `internal/amdgpu/strix_candidate.go`
- `internal/amdgpu/strix_candidate_test.go`

## Lane

`amdgpu`

## Dependencies and dedupe

Depends on the trusted receipt/authority boundary. Exact issue searches for `StrixCandidateRegistry` and `VERIFIED_LIFT scoreboard` returned no owner. Open #12270 is analogous but applies to `internal/qwen38quantrun`, not this registry.

## Expected steps

5

```routing
lane: amdgpu
paths: internal/amdgpu/strix_candidate.go, internal/amdgpu/strix_candidate_test.go
expected_steps: 5
```
