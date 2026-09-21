<!-- fak-public-issue: anthony-chaudhary/fak#13456 -->
<!-- fak-model-key: model-top15-20260921-rmsnorm-leaf -->
# refactor(model): isolate the serial RMSNorm reference

```routing
repo: fak
lane: model
paths: ["internal/model/norm/scalar.go","internal/model/norm/scalar_test.go","internal/model/forward.go","internal/model/rmsnorm_leaf_parity_test.go"]
expected_steps: 4
```

## Working spine
Make native-model improvements local to a small numerical contract or adapter, with exact software feedback.

## Current state
The serial float32 RMSNorm reference is coupled to the parent model package even though its contract is small and self-contained.

Source baseline: `7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b`. The shared FFN pilot landed at `360d204b49ca19fa68305847751f19583538adf7` (#13435). This issue is proposed work; new tests are not yet implemented or passing.

### File:Line seam
- internal/model/forward.go:873 rmsnorm (at the pinned baseline).
- internal/model/forward.go:891 rmsnormInto; distinct reduction policy (at the pinned baseline).
- internal/model/arch.go:580 rmsnormCfg (at the pinned baseline).
Read-only reproduction: `git show 7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b:internal/model/forward.go`.
Observed result: The serial float32 RMSNorm reference is coupled to the parent model package even though its contract is small and self-contained.

## Why this is next
Rank 11/15. Makes a high-centrality numerical reference independently testable without mixing incompatible norm policies.

## Parent context
Follows [#13435](https://github.com/anthony-chaudhary/fak/issues/13435), contributing to composable model blocks in [#11085](https://github.com/anthony-chaudhary/fak/issues/11085).
Dependencies: None; ready after the landed #13435 baseline.
Reuse `internal/model/ffn/README.md` and the existing adapters; no new registry.

## Core through-line
Extract ONLY the allocating plain-weight rmsnorm arithmetic to a stdlib-only leaf and retain the model wrapper. Preserve serial float32 sum-of-squares, epsilon placement, sqrt conversion, multiplication order and one output allocation.

Consumer: plain rmsnormCfg and final/decoder normalization through rmsnorm. The witness must exercise this real adapter or existing fixture, beyond merely compiling a new helper.

## Gold-plating boundary
Do not touch rmsnormInto/fdot, gain1p, LayerNorm, KDA float64 accumulation, fusion or device normalization. Preserve current invalid-geometry behavior.
Own only the listed files. Use deterministic software fixtures; hardware qualification is separate. A newly discovered requirement that expands this unit needs a separate scope decision.

## Done condition / witness
The specified behavior is preserved and the existing consumer uses the shared code, or the requested test/benchmark directly exercises that consumer.
Witness: `go test ./internal/model/norm ./internal/model -run '^(TestSerialRMSNormBitExact|TestRMSNormLeafAdapterParity)$' -count=1`

## Verifiable Witness
Run from the public checkout with `GOWORK=off` and its documented host toolchain:
```powershell
go test ./internal/model/norm ./internal/model -run '^(TestSerialRMSNormBitExact|TestRMSNormLeafAdapterParity)$' -count=1
go test ./internal/model/norm ./internal/model -count=1
```
NEW tests to author (not present or passing yet): TestSerialRMSNormBitExact, TestRMSNormLeafAdapterParity.
The owning-package suite supplies broader regression coverage.
Before accepting a filtered green result, run `go test ./internal/model/norm ./internal/model -list '^(TestSerialRMSNormBitExact|TestRMSNormLeafAdapterParity)$'` and verify that every named test appears. Zero matches or skipped required cases fail acceptance. Record leaf feedback separately from the full owning-package result; both are required.


## Definition of done
- [ ] The specified change is complete within the owned files and exercised through its named consumer.
- [ ] Independent original-arithmetic fixtures prove the specified numerical, ownership, error-order and allocation invariants.
- [ ] Every named check executes, the owning-package suite passes after the last edit, and the evidence is attached.

## Acceptance gate
All three criteria pass. Compare finite and signed-zero results with `math.Float32bits`; compare NaN/Inf behavior to the original operation and retain exposed payload bits where applicable. No tolerance substitution or arithmetic reassociation. Test-only units leave production code unchanged.
Blast radius: lane `model`, packages ./internal/model/norm ./internal/model, and their existing callers. Keep the current implementation until the atomic migration passes; no partial migration lands.

## Closure binding
Land a verified, signed-off commit with `Fixes #<this issue>` and subject suffix `(fak model)`. Use independent implementation, test and judge contexts when both code and tests change. Close after the commit and witness land on main; ticket publication does not complete implementation.

## Lane
model

## Likely files
- internal/model/norm/scalar.go
- internal/model/norm/scalar_test.go
- internal/model/forward.go
- internal/model/rmsnorm_leaf_parity_test.go
Absent test/leaf paths are intended additions; no unrelated file ownership is implied.

## Expected steps
4
1. Pin old scalar reduction and ownership in independent tiny tests.
2. Extract only the plain serial implementation.
3. Verify wrapper parity and unchanged allocation behavior.
4. Run owning package checks after the final edit.

## Value
- Centrality: Core
- P1 Context: advanced - pinned source seams and file ownership avoid broad repeated reconnaissance.
- P2 Net value: advanced - Makes a high-centrality numerical reference independently testable without mixing incompatible norm policies.
- P3 Adaptation: N/A - model/runtime policy stays unchanged.
- P4 Operations: preserved - existing dispatch remains with its owner and exact software evidence gates landing.

## Work estimate
Estimate: 3 points. Agent estimate, not an operator-supplied duration. Risk: medium numerical seam.

## Overall completion contribution
Contribution: 3/34 points for this fifteen-ticket planning cohort only. Scope estimates are not measured throughput or proof of the aspirational 10x/3x outcomes.

## Completion standard
development

github_issue: anthony-chaudhary/fak#13456
