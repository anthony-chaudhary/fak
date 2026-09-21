<!-- fak-public-issue: anthony-chaudhary/fak#13458 -->
<!-- fak-model-key: model-top15-20260921-scaled-accumulate -->
# refactor(model): share ordered expert accumulation

```routing
repo: fak
lane: model
paths: ["internal/model/ffn/weighted.go","internal/model/ffn/weighted_test.go","internal/model/moe.go","internal/model/v41_forward.go","internal/model/weighted_expert_parity_test.go"]
expected_steps: 4
```

## Working spine
Make native-model improvements local to a small numerical contract or adapter, with exact software feedback.

## Current state
Standard and V4.1 routed experts repeat elementwise weighted accumulation, a reduction order that future changes must preserve.

Source baseline: `7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b`. The shared FFN pilot landed at `360d204b49ca19fa68305847751f19583538adf7` (#13435). This issue is proposed work; new tests are not yet implemented or passing.

### File:Line seam
- internal/model/moe.go:595 routed weighted add (at the pinned baseline).
- internal/model/v41_forward.go:2542 device gate/up handled branch; 2564 generic branch (at the pinned baseline).
Read-only reproduction: `git show 7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b:internal/model/moe.go`.
Observed result: Standard and V4.1 routed experts repeat elementwise weighted accumulation, a reduction order that future changes must preserve.

## Why this is next
Rank 13/15. Creates a reusable accumulation contract for two live consumers and the shared-expert follow-on.

## Parent context
Follows [#13435](https://github.com/anthony-chaudhary/fak/issues/13435), contributing to composable model blocks in [#11085](https://github.com/anthony-chaudhary/fak/issues/11085).
Dependencies: None; ready after the landed #13435 baseline.
Reuse `internal/model/ffn/README.md` and the existing adapters; no new registry.

## Core through-line
Add ffn.AddScaled(dst, src []float32, weight float32) error. Validate equal lengths before mutation; allow equal empty slices as a no-op. Iterate increasing element indices using the existing dst[i] += weight*src[i] expression, zero allocations, no finite-value policy. Adapt the standard MoE loop and BOTH V4.1 full-forward loops (v41GateUpHandled and generic branches); preserve the outer expert-slot order.

Consumer: moeFFN.apply and V4.1 full-forward expert accumulation. The witness must exercise this real adapter or existing fixture, beyond merely compiling a new helper.

## Gold-plating boundary
No sorting/grouping/parallel reduction, changed weight precision, FMA rewrite, route selection or hardware work. Document caller alias ownership.
Own only the listed files. Use deterministic software fixtures; hardware qualification is separate. A newly discovered requirement that expands this unit needs a separate scope decision.

## Done condition / witness
The specified behavior is preserved and the existing consumer uses the shared code, or the requested test/benchmark directly exercises that consumer.
Witness: `go test ./internal/model/ffn ./internal/model -run '^(TestAddScaledContract|TestWeightedExpertAccumulationParity)$' -count=1`

## Verifiable Witness
Run from the public checkout with `GOWORK=off` and its documented host toolchain:
```powershell
go test ./internal/model/ffn ./internal/model -run '^(TestAddScaledContract|TestWeightedExpertAccumulationParity)$' -count=1
go test ./internal/model/ffn ./internal/model -count=1
```
NEW tests to author (not present or passing yet): TestAddScaledContract, TestWeightedExpertAccumulationParity.
The owning-package suite supplies broader regression coverage.
Before accepting a filtered green result, run `go test ./internal/model/ffn ./internal/model -list '^(TestAddScaledContract|TestWeightedExpertAccumulationParity)$'` and verify that every named test appears. Zero matches or skipped required cases fail acceptance. Record leaf feedback separately from the full owning-package result; both are required.


## Definition of done
- [ ] The specified change is complete within the owned files and exercised through its named consumer.
- [ ] Independent original-arithmetic fixtures prove the specified numerical, ownership, error-order and allocation invariants.
- [ ] Every named check executes, the owning-package suite passes after the last edit, and the evidence is attached.

## Acceptance gate
All three criteria pass. Compare finite and signed-zero results with `math.Float32bits`; compare NaN/Inf behavior to the original operation and retain exposed payload bits where applicable. No tolerance substitution or arithmetic reassociation. Test-only units leave production code unchanged.
Blast radius: lane `model`, packages ./internal/model/ffn ./internal/model, and their existing callers. Keep the current implementation until the atomic migration passes; no partial migration lands.

## Closure binding
Land a verified, signed-off commit with `Fixes #<this issue>` and subject suffix `(fak model)`. Use independent implementation, test and judge contexts when both code and tests change. Close after the commit and witness land on main; ticket publication does not complete implementation.

## Lane
model

## Likely files
- internal/model/ffn/weighted.go
- internal/model/ffn/weighted_test.go
- internal/model/moe.go
- internal/model/v41_forward.go
- internal/model/weighted_expert_parity_test.go
Absent test/leaf paths are intended additions; no unrelated file ownership is implied.

## Expected steps
4
1. Test prevalidation, ordered accumulation and zero allocations independently.
2. Add the small leaf helper.
3. Migrate the standard loop and both V4.1 branches, leaving outer order unchanged; force each V4.1 branch in parity fixtures.
4. Compare multiple-expert results bit-for-bit against their old loops.

## Value
- Centrality: Core
- P1 Context: advanced - pinned source seams and file ownership avoid broad repeated reconnaissance.
- P2 Net value: advanced - Creates a reusable accumulation contract for two live consumers and the shared-expert follow-on.
- P3 Adaptation: N/A - model/runtime policy stays unchanged.
- P4 Operations: preserved - existing dispatch remains with its owner and exact software evidence gates landing.

## Work estimate
Estimate: 3 points. Agent estimate, not an operator-supplied duration. Risk: medium numerical order.

## Overall completion contribution
Contribution: 3/34 points for this fifteen-ticket planning cohort only. Scope estimates are not measured throughput or proof of the aspirational 10x/3x outcomes.

## Completion standard
development

github_issue: anthony-chaudhary/fak#13458
