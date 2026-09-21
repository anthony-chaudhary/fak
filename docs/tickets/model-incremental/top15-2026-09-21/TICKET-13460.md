<!-- fak-public-issue: anthony-chaudhary/fak#13460 -->
<!-- fak-model-key: model-top15-20260921-fusion-policy -->
# refactor(model): centralize gated FFN fusion eligibility

```routing
repo: fak
lane: model
paths: ["internal/model/moe.go","internal/model/moe_fusion_policy_test.go"]
expected_steps: 3
```

## Working spine
Make native-model improvements local to a small numerical contract or adapter, with exact software feedback.

## Current state
Several adapters independently express the same SiLU-only, bias-free requirement for a fused gated MLP.

Source baseline: `7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b`. The shared FFN pilot landed at `360d204b49ca19fa68305847751f19583538adf7` (#13435). This issue is proposed work; new tests are not yet implemented or passing.

### File:Line seam
- internal/model/moe.go:127 dense eligibility (at the pinned baseline).
- internal/model/moe.go:232 and 285 expert eligibility (at the pinned baseline).
- internal/model/moe.go:639 Qwen shared eligibility (at the pinned baseline).
Read-only reproduction: `git show 7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b:internal/model/moe.go`.
Observed result: Several adapters independently express the same SiLU-only, bias-free requirement for a fused gated MLP.

## Why this is next
Rank 15/15. Keeps a shared policy decision consistent while arithmetic components remain independent of device/session state.

## Parent context
Follows [#13435](https://github.com/anthony-chaudhary/fak/issues/13435), contributing to composable model blocks in [#11085](https://github.com/anthony-chaudhary/fak/issues/11085).
Dependencies: None; ready after the landed #13435 baseline.
Reuse `internal/model/ffn/README.md` and the existing adapters; no new registry.

## Core through-line
Extract only the common model-local eligibility decision and use it at the four existing sites. Keep backend/session type assertions, resident weight checks, kernel calls, fallback order and counters at the caller. Preserve short-circuit behavior and avoid introducing allocations for the predicate. Preserve activation-flag checks before the first bias lookup and short-circuit bias lookups in their original order; do not eagerly evaluate bias arguments that were previously skipped.

Consumer: denseSwiGLU, expertSwiGLU HAL/Metal, and qwen35SharedExpert fused branches. The witness must exercise this real adapter or existing fixture, beyond merely compiling a new helper.

## Gold-plating boundary
No fusion policy inside the mathematical leaf; no new backend capability, route activation or numerical change. Do not broaden any current eligibility condition.
Own only the listed files. Use deterministic software fixtures; hardware qualification is separate. A newly discovered requirement that expands this unit needs a separate scope decision.

## Done condition / witness
The specified behavior is preserved and the existing consumer uses the shared code, or the requested test/benchmark directly exercises that consumer.
Witness: `go test ./internal/model -run '^(TestGatedMLPFusionEligibility|TestExpertSwiGLUUsesResidentDeviceQ4KGateUp|TestQwen35MoESharedExpertAdded)$' -count=1`

## Verifiable Witness
Run from the public checkout with `GOWORK=off` and its documented host toolchain:
```powershell
go test ./internal/model -run '^(TestGatedMLPFusionEligibility|TestExpertSwiGLUUsesResidentDeviceQ4KGateUp|TestQwen35MoESharedExpertAdded)$' -count=1
go test ./internal/model -count=1
```
NEW tests to author (not present or passing yet): TestGatedMLPFusionEligibility.
Existing regression checks: TestExpertSwiGLUUsesResidentDeviceQ4KGateUp, TestQwen35MoESharedExpertAdded.
Before accepting a filtered green result, run `go test ./internal/model -list '^(TestGatedMLPFusionEligibility|TestExpertSwiGLUUsesResidentDeviceQ4KGateUp|TestQwen35MoESharedExpertAdded)$'` and verify that every named test appears. Zero matches or skipped required cases fail acceptance. Record leaf feedback separately from the full owning-package result; both are required.


## Definition of done
- [ ] The specified change is complete within the owned files and exercised through its named consumer.
- [ ] Independent original-arithmetic fixtures prove the specified numerical, ownership, error-order and allocation invariants.
- [ ] Every named check executes, the owning-package suite passes after the last edit, and the evidence is attached.

## Acceptance gate
All three criteria pass. Compare finite and signed-zero results with `math.Float32bits`; compare NaN/Inf behavior to the original operation and retain exposed payload bits where applicable. No tolerance substitution or arithmetic reassociation. Test-only units leave production code unchanged.
Blast radius: lane `model`, packages ./internal/model, and their existing callers. Keep the current implementation until the atomic migration passes; no partial migration lands.

## Closure binding
Land a verified, signed-off commit with `Fixes #<this issue>` and subject suffix `(fak model)`. Use independent implementation, test and judge contexts when both code and tests change. Close after the commit and witness land on main; ticket publication does not complete implementation.

## Lane
model

## Likely files
- internal/model/moe.go
- internal/model/moe_fusion_policy_test.go
Absent test/leaf paths are intended additions; no unrelated file ownership is implied.

## Expected steps
3
1. Table-test default SiLU, each/both GELU flags and each bias-presence position.
2. Extract only the common decision and preserve caller-specific guards.
3. Verify positive and negative software dispatch fixtures and no predicate allocations.

## Value
- Centrality: Core
- P1 Context: advanced - pinned source seams and file ownership avoid broad repeated reconnaissance.
- P2 Net value: advanced - Keeps a shared policy decision consistent while arithmetic components remain independent of device/session state.
- P3 Adaptation: N/A - model/runtime policy stays unchanged.
- P4 Operations: preserved - existing dispatch remains with its owner and exact software evidence gates landing.

## Work estimate
Estimate: 2 points. Agent estimate, not an operator-supplied duration. Risk: medium admission policy.

## Overall completion contribution
Contribution: 2/34 points for this fifteen-ticket planning cohort only. Scope estimates are not measured throughput or proof of the aspirational 10x/3x outcomes.

## Completion standard
development

github_issue: anthony-chaudhary/fak#13460
