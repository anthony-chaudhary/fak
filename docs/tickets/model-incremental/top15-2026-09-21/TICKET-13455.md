<!-- fak-public-issue: anthony-chaudhary/fak#13455 -->
<!-- fak-model-key: model-top15-20260921-activation-leaf -->
# refactor(model): isolate exact scalar activation primitives

```routing
repo: fak
lane: model
paths: ["internal/model/activation/scalar.go","internal/model/activation/scalar_test.go","internal/model/arch.go","internal/model/forward.go"]
expected_steps: 4
```

## Working spine
Make native-model improvements local to a small numerical contract or adapter, with exact software feedback.

## Current state
Three tiny scalar formulas can currently be tested only inside the large model package.

Source baseline: `7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b`. The shared FFN pilot landed at `360d204b49ca19fa68305847751f19583538adf7` (#13435). This issue is proposed work; new tests are not yet implemented or passing.

### File:Line seam
- internal/model/arch.go:551 act; 562 geluTanh; 570 geluErf (at the pinned baseline).
- internal/model/forward.go:938 silu (at the pinned baseline).
Read-only reproduction: `git show 7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b:internal/model/arch.go`.
Observed result: Three tiny scalar formulas can currently be tested only inside the large model package.

## Why this is next
Rank 10/15. A second fast numerical feedback boundary; no change to runtime architecture or model configuration.

## Parent context
Follows [#13435](https://github.com/anthony-chaudhary/fak/issues/13435), contributing to composable model blocks in [#11085](https://github.com/anthony-chaudhary/fak/issues/11085).
Dependencies: [#13454](https://github.com/anthony-chaudhary/fak/issues/13454) must land first.
Reuse `internal/model/ffn/README.md` and the existing adapters; no new registry.

## Core through-line
Move the unchanged SiLU, GELU-tanh and GELU-erf formulas into a dependency-light internal/model/activation package. Keep existing model function names as delegating wrappers and leave Config selection/precedence in act. Preserve every float32/float64 conversion and operation order.

Consumer: act and existing direct silu/gelu callers through their unchanged wrappers. The witness must exercise this real adapter or existing fixture, beyond merely compiling a new helper.

## Gold-plating boundary
No Config/backend imports in the leaf, approximations, new activation kinds, policy selector or GPU changes.
Own only the listed files. Use deterministic software fixtures; hardware qualification is separate. A newly discovered requirement that expands this unit needs a separate scope decision.

## Done condition / witness
The specified behavior is preserved and the existing consumer uses the shared code, or the requested test/benchmark directly exercises that consumer.
Witness: `go test ./internal/model/activation ./internal/model -run '^(TestScalarActivationsBitExact|TestFFNAdapterActivationMatrix)$' -count=1`

## Verifiable Witness
Run from the public checkout with `GOWORK=off` and its documented host toolchain:
```powershell
go test ./internal/model/activation ./internal/model -run '^(TestScalarActivationsBitExact|TestFFNAdapterActivationMatrix)$' -count=1
go test ./internal/model/activation ./internal/model -count=1
```
NEW tests to author (not present or passing yet): TestScalarActivationsBitExact.
Existing regression checks once dependency rank 9 lands: TestFFNAdapterActivationMatrix.
Before accepting a filtered green result, run `go test ./internal/model/activation ./internal/model -list '^(TestScalarActivationsBitExact|TestFFNAdapterActivationMatrix)$'` and verify that every named test appears. Zero matches or skipped required cases fail acceptance. Record leaf feedback separately from the full owning-package result; both are required.


## Definition of done
- [ ] The specified change is complete within the owned files and exercised through its named consumer.
- [ ] Independent original-arithmetic fixtures prove the specified numerical, ownership, error-order and allocation invariants.
- [ ] Every named check executes, the owning-package suite passes after the last edit, and the evidence is attached.

## Acceptance gate
All three criteria pass. Compare finite and signed-zero results with `math.Float32bits`; compare NaN/Inf behavior to the original operation and retain exposed payload bits where applicable. No tolerance substitution or arithmetic reassociation. Test-only units leave production code unchanged.
Blast radius: lane `model`, packages ./internal/model/activation ./internal/model, and their existing callers. Keep the current implementation until the atomic migration passes; no partial migration lands.

## Closure binding
Land a verified, signed-off commit with `Fixes #<this issue>` and subject suffix `(fak model)`. Use independent implementation, test and judge contexts when both code and tests change. Close after the commit and witness land on main; ticket publication does not complete implementation.

## Lane
model

## Likely files
- internal/model/activation/scalar.go
- internal/model/activation/scalar_test.go
- internal/model/arch.go
- internal/model/forward.go
Absent test/leaf paths are intended additions; no unrelated file ownership is implied.

## Expected steps
4
1. Author independent literal-formula tests for finite extrema, signed zero, infinities and NaNs.
2. Move formulas without algebraic rewriting.
3. Retain model wrappers and policy precedence.
4. Run leaf tests, adapter matrix and matched activation benchmarks.

## Value
- Centrality: Core
- P1 Context: advanced - pinned source seams and file ownership avoid broad repeated reconnaissance.
- P2 Net value: advanced - A second fast numerical feedback boundary; no change to runtime architecture or model configuration.
- P3 Adaptation: N/A - model/runtime policy stays unchanged.
- P4 Operations: preserved - existing dispatch remains with its owner and exact software evidence gates landing.

## Work estimate
Estimate: 2 points. Agent estimate, not an operator-supplied duration. Risk: low-medium.

## Overall completion contribution
Contribution: 2/34 points for this fifteen-ticket planning cohort only. Scope estimates are not measured throughput or proof of the aspirational 10x/3x outcomes.

## Completion standard
development

github_issue: anthony-chaudhary/fak#13455
