<!-- fak-public-issue: anthony-chaudhary/fak#13454 -->
<!-- fak-model-key: model-top15-20260921-activation-matrix -->
# test(model): cover the FFN adapter activation matrix

```routing
repo: fak
lane: model
paths: ["internal/model/ffn_composition_test.go"]
expected_steps: 3
```

## Working spine
Make native-model improvements local to a small numerical contract or adapter, with exact software feedback.

## Current state
Current parity cases use different activations per adapter rather than checking each adapter against all supported activation choices.

Source baseline: `7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b`. The shared FFN pilot landed at `360d204b49ca19fa68305847751f19583538adf7` (#13435). This issue is proposed work; new tests are not yet implemented or passing.

### File:Line seam
- internal/model/ffn_composition_test.go:8 TestFFNGatedAdaptersBitExact (at the pinned baseline).
- internal/model/arch.go:551 act (at the pinned baseline).
Read-only reproduction: `git show 7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b:internal/model/ffn_composition_test.go`.
Observed result: Current parity cases use different activations per adapter rather than checking each adapter against all supported activation choices.

## Why this is next
Rank 9/15. A one-file guard that lowers risk for the activation extraction and later adapter work.

## Parent context
Follows [#13435](https://github.com/anthony-chaudhary/fak/issues/13435), contributing to composable model blocks in [#11085](https://github.com/anthony-chaudhary/fak/issues/11085).
Dependencies: None; ready after the landed #13435 baseline.
Reuse `internal/model/ffn/README.md` and the existing adapters; no new registry.

## Core through-line
Extend independent old-body parity across dense, routed V4.1 and shared V4.1 adapters for default SiLU, GELU-tanh, GELU-erf and both flags (tanh wins). Keep dense bias coverage and confirm up/input buffers remain unchanged where owned by the caller.

Consumer: the three production adapters from #13435. The witness must exercise this real adapter or existing fixture, beyond merely compiling a new helper.

## Gold-plating boundary
Test-only; reuse existing fixtures and independent legacy bodies. Do not weaken bit equality to a tolerance or make hardware claims.
Own only the listed files. Use deterministic software fixtures; hardware qualification is separate. A newly discovered requirement that expands this unit needs a separate scope decision.

## Done condition / witness
The specified behavior is preserved and the existing consumer uses the shared code, or the requested test/benchmark directly exercises that consumer.
Witness: `go test ./internal/model -run '^(TestFFNAdapterActivationMatrix|TestFFNGatedAdaptersBitExact)$' -count=1`

## Verifiable Witness
Run from the public checkout with `GOWORK=off` and its documented host toolchain:
```powershell
go test ./internal/model -run '^(TestFFNAdapterActivationMatrix|TestFFNGatedAdaptersBitExact)$' -count=1
go test ./internal/model -count=1
```
NEW tests to author (not present or passing yet): TestFFNAdapterActivationMatrix.
Existing regression checks: TestFFNGatedAdaptersBitExact.
Before accepting a filtered green result, run `go test ./internal/model -list '^(TestFFNAdapterActivationMatrix|TestFFNGatedAdaptersBitExact)$'` and verify that every named test appears. Zero matches or skipped required cases fail acceptance. Record leaf feedback separately from the full owning-package result; both are required.


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
- internal/model/ffn_composition_test.go
Absent test/leaf paths are intended additions; no unrelated file ownership is implied.

## Expected steps
3
1. Make a table covering all three adapters and four policies.
2. Compare float32 bits against original arithmetic, including sign-sensitive finite inputs.
3. Retain allocation assertions as no-regression checks.

## Value
- Centrality: Core
- P1 Context: advanced - pinned source seams and file ownership avoid broad repeated reconnaissance.
- P2 Net value: advanced - A one-file guard that lowers risk for the activation extraction and later adapter work.
- P3 Adaptation: N/A - model/runtime policy stays unchanged.
- P4 Operations: preserved - existing dispatch remains with its owner and exact software evidence gates landing.

## Work estimate
Estimate: 2 points. Agent estimate, not an operator-supplied duration. Risk: low.

## Overall completion contribution
Contribution: 2/34 points for this fifteen-ticket planning cohort only. Scope estimates are not measured throughput or proof of the aspirational 10x/3x outcomes.

## Completion standard
development

github_issue: anthony-chaudhary/fak#13454
