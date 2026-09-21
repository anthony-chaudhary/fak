<!-- fak-public-issue: anthony-chaudhary/fak#13457 -->
<!-- fak-model-key: model-top15-20260921-softmax-leaf -->
# refactor(model): share stable scalar softmax outside model

```routing
repo: fak
lane: model
paths: ["internal/model/softmax/scalar.go","internal/model/softmax/scalar_test.go","internal/model/forward.go","internal/model/moe.go","internal/model/softmax_leaf_parity_test.go"]
expected_steps: 4
```

## Working spine
Make native-model improvements local to a small numerical contract or adapter, with exact software feedback.

## Current state
Attention and router helpers duplicate the same max-subtracted scalar softmax with different ownership contracts.

Source baseline: `7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b`. The shared FFN pilot landed at `360d204b49ca19fa68305847751f19583538adf7` (#13435). This issue is proposed work; new tests are not yet implemented or passing.

### File:Line seam
- internal/model/forward.go:920 softmaxInPlace (at the pinned baseline).
- internal/model/moe.go:508 softmaxOf (at the pinned baseline).
Read-only reproduction: `git show 7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b:internal/model/forward.go`.
Observed result: Attention and router helpers duplicate the same max-subtracted scalar softmax with different ownership contracts.

## Why this is next
Rank 12/15. One small contract replaces two implementations and gives attention/router changes a local test boundary.

## Parent context
Follows [#13435](https://github.com/anthony-chaudhary/fak/issues/13435), contributing to composable model blocks in [#11085](https://github.com/anthony-chaudhary/fak/issues/11085).
Dependencies: None; ready after the landed #13435 baseline.
Reuse `internal/model/ffn/README.md` and the existing adapters; no new registry.

## Core through-line
Provide InPlace and Copy in a dependency-light softmax leaf. Copy allocates once and delegates the exact in-place arithmetic. Keep model wrappers and preserve max comparison, float32 exponent conversion/sum/division order, mutation and empty-input panic semantics.

Consumer: attention callers of softmaxInPlace and MoE router callers of softmaxOf. The witness must exercise this real adapter or existing fixture, beyond merely compiling a new helper.

## Gold-plating boundary
No online softmax, SIMD, new non-finite policy, tolerance substitution, stable-sort or attention algorithm changes.
Own only the listed files. Use deterministic software fixtures; hardware qualification is separate. A newly discovered requirement that expands this unit needs a separate scope decision.

## Done condition / witness
The specified behavior is preserved and the existing consumer uses the shared code, or the requested test/benchmark directly exercises that consumer.
Witness: `go test ./internal/model/softmax ./internal/model -run '^(TestScalarSoftmaxBitExact|TestSoftmaxLeafAdapterParity)$' -count=1`

## Verifiable Witness
Run from the public checkout with `GOWORK=off` and its documented host toolchain:
```powershell
go test ./internal/model/softmax ./internal/model -run '^(TestScalarSoftmaxBitExact|TestSoftmaxLeafAdapterParity)$' -count=1
go test ./internal/model/softmax ./internal/model -count=1
```
NEW tests to author (not present or passing yet): TestScalarSoftmaxBitExact, TestSoftmaxLeafAdapterParity.
The owning-package suite supplies broader regression coverage.
Before accepting a filtered green result, run `go test ./internal/model/softmax ./internal/model -list '^(TestScalarSoftmaxBitExact|TestSoftmaxLeafAdapterParity)$'` and verify that every named test appears. Zero matches or skipped required cases fail acceptance. Record leaf feedback separately from the full owning-package result; both are required.


## Definition of done
- [ ] The specified change is complete within the owned files and exercised through its named consumer.
- [ ] Independent original-arithmetic fixtures prove the specified numerical, ownership, error-order and allocation invariants.
- [ ] Every named check executes, the owning-package suite passes after the last edit, and the evidence is attached.

## Acceptance gate
All three criteria pass. Compare finite and signed-zero results with `math.Float32bits`; compare NaN/Inf behavior to the original operation and retain exposed payload bits where applicable. No tolerance substitution or arithmetic reassociation. Test-only units leave production code unchanged.
Blast radius: lane `model`, packages ./internal/model/softmax ./internal/model, and their existing callers. Keep the current implementation until the atomic migration passes; no partial migration lands.

## Closure binding
Land a verified, signed-off commit with `Fixes #<this issue>` and subject suffix `(fak model)`. Use independent implementation, test and judge contexts when both code and tests change. Close after the commit and witness land on main; ticket publication does not complete implementation.

## Lane
model

## Likely files
- internal/model/softmax/scalar.go
- internal/model/softmax/scalar_test.go
- internal/model/forward.go
- internal/model/moe.go
- internal/model/softmax_leaf_parity_test.go
Absent test/leaf paths are intended additions; no unrelated file ownership is implied.

## Expected steps
4
1. Pin both old bodies including signed zero, NaN/Inf and empty panic behavior.
2. Add InPlace/Copy with exact arithmetic and ownership.
3. Delegate both existing wrappers.
4. Run bit, allocation and representative-size benchmarks.

## Value
- Centrality: Core
- P1 Context: advanced - pinned source seams and file ownership avoid broad repeated reconnaissance.
- P2 Net value: advanced - One small contract replaces two implementations and gives attention/router changes a local test boundary.
- P3 Adaptation: N/A - model/runtime policy stays unchanged.
- P4 Operations: preserved - existing dispatch remains with its owner and exact software evidence gates landing.

## Work estimate
Estimate: 3 points. Agent estimate, not an operator-supplied duration. Risk: medium numerical seam.

## Overall completion contribution
Contribution: 3/34 points for this fifteen-ticket planning cohort only. Scope estimates are not measured throughput or proof of the aspirational 10x/3x outcomes.

## Completion standard
development

github_issue: anthony-chaudhary/fak#13457
