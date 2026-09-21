<!-- fak-public-issue: anthony-chaudhary/fak#13453 -->
<!-- fak-model-key: model-top15-20260921-shared-bench -->
# test(model): benchmark the real V4.1 shared FFN adapter

```routing
repo: fak
lane: model
paths: ["internal/model/v41_forward_test.go","internal/model/v41_weights_test.go","internal/model/v41_config_test.go","internal/model/ffn_composition_test.go"]
expected_steps: 4
```

## Working spine
Make native-model improvements local to a small numerical contract or adapter, with exact software feedback.

## Current state
The shipped pilot times routed and dense adapters but records only allocation counts for the shared expert; the reduced-fixture chain accepts *testing.T.

Source baseline: `7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b`. The shared FFN pilot landed at `360d204b49ca19fa68305847751f19583538adf7` (#13435). This issue is proposed work; new tests are not yet implemented or passing.

### File:Line seam
- internal/model/v41_forward_test.go:47 v41ReducedModelLayers (at the pinned baseline).
- internal/model/v41_weights_test.go:16 v41TestReducedConfig (at the pinned baseline).
- internal/model/v41_config_test.go:17 readDeepSeekV41Config (at the pinned baseline).
- internal/model/ffn_composition_test.go:168 BenchmarkFFNGatedAdapters (at the pinned baseline).
Read-only reproduction: `git show 7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b:internal/model/v41_forward_test.go`.
Observed result: The shipped pilot times routed and dense adapters but records only allocation counts for the shared expert; the reduced-fixture chain accepts *testing.T.

## Why this is next
Rank 8/15. Fills the one unmeasured production adapter and makes an existing tiny fixture reusable by later benchmarks.

## Parent context
Follows [#13435](https://github.com/anthony-chaudhary/fak/issues/13435), contributing to composable model blocks in [#11085](https://github.com/anthony-chaudhary/fak/issues/11085).
Dependencies: None; ready after the landed #13435 baseline.
Reuse `internal/model/ffn/README.md` and the existing adapters; no new registry.

## Core through-line
Generalize only readDeepSeekV41Config, v41TestReducedConfig and v41ReducedModelLayers to testing.TB, preserving all data and Helper calls. Add BenchmarkFFNGatedAdapters/V41Shared with exact original-body and adapter cases for SiLU and GELU; exclude setup from timing.

Consumer: existing reduced model tests and real v41SharedExpertSwiGLU benchmark. The witness must exercise this real adapter or existing fixture, beyond merely compiling a new helper.

## Gold-plating boundary
Test/benchmark files only. No network/downloads, real weights, production fixture mutation, or noisy timing pass threshold.
Own only the listed files. Use deterministic software fixtures; hardware qualification is separate. A newly discovered requirement that expands this unit needs a separate scope decision.

## Done condition / witness
The specified behavior is preserved and the existing consumer uses the shared code, or the requested test/benchmark directly exercises that consumer.
Witness: `go test ./internal/model -run '^(TestFFNGatedAdaptersBitExact)$' -count=1`

## Verifiable Witness
Run from the public checkout with `GOWORK=off` and its documented host toolchain:
```powershell
go test ./internal/model -run '^(TestFFNGatedAdaptersBitExact)$' -count=1
go test ./internal/model -count=1
```
NEW benchmark: BenchmarkFFNGatedAdapters/V41Shared, with original/adapter SiLU and GELU cases.
Existing regression checks: TestFFNGatedAdaptersBitExact.
Before accepting a filtered green result, run `go test ./internal/model -list '^(TestFFNGatedAdaptersBitExact)$'` and verify that every named test appears. Zero matches or skipped required cases fail acceptance. Record leaf feedback separately from the full owning-package result; both are required.
Benchmark witness: go test ./internal/model -run '^TestFFNGatedAdaptersBitExact$' -bench '^BenchmarkFFNGatedAdapters/V41Shared' -benchmem -cpu=1 -benchtime=500ms -count=6. Require all four shared original/adapter and activation cases to emit samples; report workload, toolchain, measured ranges and allocations, without a timing threshold or hardware speedup claim.

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
- internal/model/v41_forward_test.go
- internal/model/v41_weights_test.go
- internal/model/v41_config_test.go
- internal/model/ffn_composition_test.go
Absent test/leaf paths are intended additions; no unrelated file ownership is implied.

## Expected steps
4
1. Widen the three fixture parameter types only.
2. Add exact legacy and adapter shared-expert cases.
3. Run correctness and six benchmark samples with allocations.
4. Record workload, toolchain, ranges and limitations.

## Value
- Centrality: Core
- P1 Context: advanced - pinned source seams and file ownership avoid broad repeated reconnaissance.
- P2 Net value: advanced - Fills the one unmeasured production adapter and makes an existing tiny fixture reusable by later benchmarks.
- P3 Adaptation: N/A - model/runtime policy stays unchanged.
- P4 Operations: preserved - existing dispatch remains with its owner and exact software evidence gates landing.

## Work estimate
Estimate: 2 points. Agent estimate, not an operator-supplied duration. Risk: low.

## Overall completion contribution
Contribution: 2/34 points for this fifteen-ticket planning cohort only. Scope estimates are not measured throughput or proof of the aspirational 10x/3x outcomes.

## Completion standard
development

github_issue: anthony-chaudhary/fak#13453
