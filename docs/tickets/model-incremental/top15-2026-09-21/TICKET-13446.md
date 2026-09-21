<!-- fak-public-issue: anthony-chaudhary/fak#13446 -->
<!-- fak-model-key: model-top15-20260921-gated-row -->
# refactor(model): share gated row activation with batched MLP

```routing
repo: fak
lane: model
paths: ["internal/model/ffn/gated.go","internal/model/ffn/gated_test.go","internal/model/arch.go","internal/model/batched_ffn_composition_test.go"]
expected_steps: 4
```

## Working spine
Make native-model improvements local to a small numerical contract or adapter, with exact software feedback.

## Current state
Gated owns a useful scalar row contract, but batchedGatedMLP repeats the same activation-times-up loop in fuseGatedMLPPanels.

Source baseline: `7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b`. The shared FFN pilot landed at `360d204b49ca19fa68305847751f19583538adf7` (#13435). This issue is proposed work; new tests are not yet implemented or passing.

### File:Line seam
- internal/model/ffn/gated.go:17 Gated (at the pinned baseline).
- internal/model/arch.go:230 fuseGatedMLPPanels (at the pinned baseline).
Read-only reproduction: `git show 7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b:internal/model/ffn/gated.go`.
Observed result: Gated owns a useful scalar row contract, but batchedGatedMLP repeats the same activation-times-up loop in fuseGatedMLPPanels.

## Why this is next
Rank 1/15. Unlocks two disjoint downstream adopters while making scalar and batched gating one change point.

## Parent context
Follows [#13435](https://github.com/anthony-chaudhary/fak/issues/13435), contributing to composable model blocks in [#11085](https://github.com/anthony-chaudhary/fak/issues/11085).
Dependencies: None; ready after the landed #13435 baseline.
Reuse `internal/model/ffn/README.md` and the existing adapters; no new registry.

## Core through-line
Add ffn.ApplyInPlace(gate, up []float32, activate Activation) error. Validate nonempty equal lengths and nonnil activation before mutation, retain exclusive gate ownership and read-only up. Have Gated delegate its loop, then project once; Gated must still validate a nil down callback BEFORE any mutation. Adopt ApplyInPlace after the existing panel bias loop. Preserve the adapter's existing zero-row no-op without weakening the leaf contract.

Consumer: Model.batchedGatedMLP -> fuseGatedMLPPanels, and all existing Gated adapters. The witness must exercise this real adapter or existing fixture, beyond merely compiling a new helper.

## Gold-plating boundary
Keep projection, bias, Config, device selection and topology outside ffn. Do not add a graph/registry or change the Gated signature.
Own only the listed files. Use deterministic software fixtures; hardware qualification is separate. A newly discovered requirement that expands this unit needs a separate scope decision.

## Done condition / witness
The specified behavior is preserved and the existing consumer uses the shared code, or the requested test/benchmark directly exercises that consumer.
Witness: `go test ./internal/model/ffn ./internal/model -run '^(TestApplyInPlaceContract|TestBatchedGatedMLPApplyInPlaceParity|TestGatedValidatesBeforeMutation|TestBlockTopologyComposition|TestWeightFreeFamilyConformance)$' -count=1`

## Verifiable Witness
Run from the public checkout with `GOWORK=off` and its documented host toolchain:
```powershell
go test ./internal/model/ffn ./internal/model -run '^(TestApplyInPlaceContract|TestBatchedGatedMLPApplyInPlaceParity|TestGatedValidatesBeforeMutation|TestBlockTopologyComposition|TestWeightFreeFamilyConformance)$' -count=1
go test ./internal/model/ffn ./internal/model -count=1
```
NEW tests to author (not present or passing yet): TestApplyInPlaceContract, TestBatchedGatedMLPApplyInPlaceParity.
Existing regression checks: TestGatedValidatesBeforeMutation, TestBlockTopologyComposition, TestWeightFreeFamilyConformance.
Before accepting a filtered green result, run `go test ./internal/model/ffn ./internal/model -list '^(TestApplyInPlaceContract|TestBatchedGatedMLPApplyInPlaceParity|TestGatedValidatesBeforeMutation|TestBlockTopologyComposition|TestWeightFreeFamilyConformance)$'` and verify that every named test appears. Zero matches or skipped required cases fail acceptance. Record leaf feedback separately from the full owning-package result; both are required.


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
- internal/model/ffn/gated.go
- internal/model/ffn/gated_test.go
- internal/model/arch.go
- internal/model/batched_ffn_composition_test.go
Absent test/leaf paths are intended additions; no unrelated file ownership is implied.

## Expected steps
4
1. Author independent prevalidation/order/alias-ownership tests and a legacy panel oracle.
2. Implement ApplyInPlace and preserve Gated's full prevalidation.
3. Migrate only fuseGatedMLPPanels and retain empty-panel behavior.
4. Run exact parity, allocation checks, and owning-package tests.

## Value
- Centrality: Core
- P1 Context: advanced - pinned source seams and file ownership avoid broad repeated reconnaissance.
- P2 Net value: advanced - Unlocks two disjoint downstream adopters while making scalar and batched gating one change point.
- P3 Adaptation: N/A - model/runtime policy stays unchanged.
- P4 Operations: preserved - existing dispatch remains with its owner and exact software evidence gates landing.

## Work estimate
Estimate: 3 points. Agent estimate, not an operator-supplied duration. Risk: low-medium.

## Overall completion contribution
Contribution: 3/34 points for this fifteen-ticket planning cohort only. Scope estimates are not measured throughput or proof of the aspirational 10x/3x outcomes.

## Completion standard
development

github_issue: anthony-chaudhary/fak#13446
