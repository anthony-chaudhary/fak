<!-- fak-public-issue: anthony-chaudhary/fak#13450 -->
<!-- fak-model-key: model-top15-20260921-host-batch -->
# refactor(model): share gated rows in host-batched experts

```routing
repo: fak
lane: model
paths: ["internal/model/moe_host_batch.go","internal/model/moe_host_batch_ffn_test.go"]
expected_steps: 3
```

## Working spine
Make native-model improvements local to a small numerical contract or adapter, with exact software feedback.

## Current state
Host-batched expert rows have another activation-times-up loop.

Source baseline: `7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b`. The shared FFN pilot landed at `360d204b49ca19fa68305847751f19583538adf7` (#13435). This issue is proposed work; new tests are not yet implemented or passing.

### File:Line seam
- internal/model/moe_host_batch.go:163 batchedExpertDelta (at the pinned baseline).
- internal/model/moe_host_batch.go:181 repeated row loop (at the pinned baseline).
Read-only reproduction: `git show 7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b:internal/model/moe_host_batch.go`.
Observed result: Host-batched expert rows have another activation-times-up loop.

## Why this is next
Rank 5/15. Independent two-file adopter once the row contract lands.

## Parent context
Follows [#13435](https://github.com/anthony-chaudhary/fak/issues/13435), contributing to composable model blocks in [#11085](https://github.com/anthony-chaudhary/fak/issues/11085).
Dependencies: [#13446](https://github.com/anthony-chaudhary/fak/issues/13446) must land first.
Reuse `internal/model/ffn/README.md` and the existing adapters; no new registry.

## Core through-line
Apply the row primitive from ticket rank 1 to the existing gate/up row slices. Preserve row boundaries, expert routing, grouped projections, down projection and accumulation order.

Consumer: split-host MoE batching -> batchedExpertDelta. The witness must exercise this real adapter or existing fixture, beyond merely compiling a new helper.

## Gold-plating boundary
No routing/grouping changes, cross-row aliasing, new batching framework or device work.
Own only the listed files. Use deterministic software fixtures; hardware qualification is separate. A newly discovered requirement that expands this unit needs a separate scope decision.

## Done condition / witness
The specified behavior is preserved and the existing consumer uses the shared code, or the requested test/benchmark directly exercises that consumer.
Witness: `go test ./internal/model -run '^(TestHostBatchedFFNApplyInPlaceParity|TestGLMMoeFFNSplitHostBatchedMatchesLoop|TestMoEFFNSplitHostBatchedMatchesLoop)$' -count=1`

## Verifiable Witness
Run from the public checkout with `GOWORK=off` and its documented host toolchain:
```powershell
go test ./internal/model -run '^(TestHostBatchedFFNApplyInPlaceParity|TestGLMMoeFFNSplitHostBatchedMatchesLoop|TestMoEFFNSplitHostBatchedMatchesLoop)$' -count=1
go test ./internal/model -count=1
```
NEW tests to author (not present or passing yet): TestHostBatchedFFNApplyInPlaceParity.
Existing regression checks: TestGLMMoeFFNSplitHostBatchedMatchesLoop, TestMoEFFNSplitHostBatchedMatchesLoop.
Before accepting a filtered green result, run `go test ./internal/model -list '^(TestHostBatchedFFNApplyInPlaceParity|TestGLMMoeFFNSplitHostBatchedMatchesLoop|TestMoEFFNSplitHostBatchedMatchesLoop)$'` and verify that every named test appears. Zero matches or skipped required cases fail acceptance. Record leaf feedback separately from the full owning-package result; both are required.


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
- internal/model/moe_host_batch.go
- internal/model/moe_host_batch_ffn_test.go
Absent test/leaf paths are intended additions; no unrelated file ownership is implied.

## Expected steps
3
1. Add multirow old-loop bit parity and ownership fixtures.
2. Replace only the row loop with ApplyInPlace.
3. Run host-batch-versus-loop tests and allocation checks.

## Value
- Centrality: Core
- P1 Context: advanced - pinned source seams and file ownership avoid broad repeated reconnaissance.
- P2 Net value: advanced - Independent two-file adopter once the row contract lands.
- P3 Adaptation: N/A - model/runtime policy stays unchanged.
- P4 Operations: preserved - existing dispatch remains with its owner and exact software evidence gates landing.

## Work estimate
Estimate: 2 points. Agent estimate, not an operator-supplied duration. Risk: low-medium.

## Overall completion contribution
Contribution: 2/34 points for this fifteen-ticket planning cohort only. Scope estimates are not measured throughput or proof of the aspirational 10x/3x outcomes.

## Completion standard
development

github_issue: anthony-chaudhary/fak#13450
