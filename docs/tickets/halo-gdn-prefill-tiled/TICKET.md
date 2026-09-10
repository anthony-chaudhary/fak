# perf(vulkan): tile GDN recurrent prefill state for Qwen3.8

<!-- fak-compute-key: halo-gdn-prefill-state-tile-12533 -->

```routing
lane: internal/compute
paths: ["internal/compute/shaders/qwen35_gdn_prefill_tiled.comp", "internal/compute/shaders/qwen35_gdn_prefill_norm.comp", "internal/compute/vulkan_shim.cpp", "internal/compute/vulkan_gdn_prefill_tiled.go", "internal/compute/vulkan_gdn_prefill_tiled_test.go", "internal/compute/build_vulkan.ps1", "internal/computebuild/vulkan.go", "internal/computebuild/computebuild_test.go"]
expected_steps: 7
github_issue: 12533
```

Tracking: [fak #12533](https://github.com/anthony-chaudhary/fak/issues/12533).
Status: opt-in implementation, scoped package checks, and physical component qualification complete. The broader suite timed out in an unchanged scaffold test; whole-model qualification is blocked by the original-arm parity failure. Default remains off; no closure claim.

## Parent context

Existing #12533 owns the recurrent-state tile; #12531 and #12532 own adjacent normalization work. #12098 remains open and retains the matched physical whole-model comparison gate required for default promotion.

## Current state

At public base `3858867f25b163f3b8f57c12f88f79cf02d277b3`, the native Qwen3.8 sequence path already batches prompt projections. Its GDN recurrent kernel assigns one 64-thread workgroup to a value head, walks prompt tokens serially, and repeatedly reads and updates the global recurrent-state matrix. The supported dense geometry has 16 key heads, 48 value heads, and 128-dimensional keys and values. This exposes limited workgroup parallelism and repeated state accesses; no current full-model stage profile establishes the percentage of prefill time attributable to this kernel.

The existing issue's original in-place plan shared its recurrent shader with #12531 and #12532. This additive variant uses new shader files and its own parallel output RMS pass, so implementing the executable slice does not require those edits to finish. Coordinate dispatch integration with their owners; keep the existing shader as the scalar control.

## Problem frame

- Centrality: Core; cold prompt processing through the native local model session.
- P1 Context: advanced; expose the recurrent-state matrix through a two-dimensional tile instead of serial key walks per value column.
- P2 Net value: advanced - require a measured component gain before asserting a user-visible improvement.
- P3 Adaptation: preserved; retain state layout, grouped-head mapping, recurrence semantics, and scalar fallback.
- P4 Operations: advanced; explicit opt-in and observed dispatch counters identify the executed arm.

For native Qwen3.8 on Vulkan / Problem: repeated recurrent-state work during prompt ingestion / Today: one workgroup per value head / Better because: tile state and retain it across tokens / Witness: independent output/state parity plus matched physical timings.

## Why this is next

Native prompt projections already batch. This leaf addresses the exposed recurrent-state work while independent matrix-kernel qualification proceeds, with no claimed percentage attribution before measurement.

## Core through-line

`Session.Prefill` -> `Session.tryQwen35SequencePrefill` -> `vulkanBackend.Qwen35SequencePrefill` -> `fvk_qwen35_gdn_preprojected_f32` -> opt-in tiled recurrence -> parallel gated RMS output -> existing GDN output projection and continuation state.

Add one fak-native variant for 128x128 recurrent heads, grouped 16 key / 48 value heads, and panels of at least eight tokens. Preserve the ordinary session call path. Normal application use requires explicit `FAK_VULKAN_GDN_PREFILL_TILED=1`; the default remains off. Unsupported shapes, short panels, and the forced scalar arm retain the existing kernel. Read, evolve, and commit the same persistent recurrent state and preserve convolution carry across calls.

## Gold-plating boundary

No Q4 cooperative GEMM work, model-loader changes, new model architectures, generic graph scheduler, alternative inference runtime, public API change, or default promotion. The variant does not edit the original recurrent shader. Its output RMS pass is part of this candidate, so timing establishes the combined variant's effect. Whole-model improvement remains pending a same-build comparison; primitive speed alone cannot establish it.

## Likely files

Only `internal/compute` and `internal/computebuild` are implementation packages. The routing block names the proposed new shaders, native dispatch, selector and counter wiring, independent tests, and shader registry. This ticket is their local specification for existing #12533.

The original three-file estimate expands to nine paths: eight implementation, test, and build paths plus this ticket. The additive variant avoids conflicting edits to the original shader, registers both shader build entry points, and includes independently authored behavioral tests. Scope remains seven steps across two implementation packages.

## Lane

Public `internal/compute`, with the existing `internal/computebuild` shader registry. Other quantization, attention, and serving work proceeds independently.

## Witness

Source witness at the stated base:

```text
git grep -n -E 'for \(int t|state\[si\]|int d = lane' 3858867f25b163f3b8f57c12f88f79cf02d277b3 -- internal/compute/shaders/qwen35_gdn_recurrent.comp
```

Observed structure: the token loop encloses serial key-dimension loops that decay/read and update/read `state[si]`; value columns are distributed by lane. This command demonstrates the source mechanism, not measured hardware traffic or timing.

Source seams: `internal/compute/vulkan_qwen35_sequence.go:563` invokes GDN preprojected execution; `internal/compute/vulkan_shim.cpp:2329` defines that native dispatcher; `internal/model/qwen35_hal.go:1041` invokes the backend sequence entry. Source line numbers describe the pinned base.

## Verifiable Witness

Independent witness commands (PowerShell; replace the device substring with the observed intended device):

```powershell
$env:FAK_VULKAN_REQUIRE_DEVICE = '1'
$env:FAK_VULKAN_EXPECT_DEVICE = '<observed intended device substring>'
$env:FAK_VULKAN_GDN_PREFILL_TILED_PROFILE = '1'
go test ./internal/computebuild -count=1
go test -tags vulkan ./internal/compute -run TestVulkanQwen35GDNTiledPrefillParityAndSplitContinuation -count=1 -v
go test -tags vulkan ./internal/compute -run '^$' -bench BenchmarkVulkanQwen35GDNTiledPrefillComponentAB -benchtime=6x -count=1
```

The profile environment variable is required: the component benchmark skips without it. The device variables make the witness require the intended physical device. The independent test and benchmark force the scalar and candidate arms themselves; the normal application opt-in remains separate. Use isolated build caches and retain the exact command, environment, compiler, and device identities with each run.

Require the intended test to execute without skips on the named physical device. Compare scalar and candidate from identical fixed-seed preprojected inputs and nonzero recurrent/convolution carry. Check finite core vectors, final state, grouped heads, small-norm inputs, values exposing reduction error, multiple tokens, split-prefill continuation, short-panel fallback, unsupported geometry, and forced scalar selection. Freeze explicit output/state tolerances before observing candidate results; preserve the original oracle. Positive candidate counters are required for supported panels and zero counters for scalar controls and single-token continuation.

## Qualification snapshot

On 2026-09-10, the runtime frozen at source checkpoint `b9e391e91226c17f405f7fa09251995dbeaea168`, based on `3858867f25b163f3b8f57c12f88f79cf02d277b3`, passed shader compilation and SPIR-V validation, native C++ compilation, and a targeted nine-path public boundary check. Its added-line leak audit also passed. The landing source changes only this ticket, the independent GDN test, and the build verifier's module count from 43 to 45. The other six source paths are byte-identical to that component freeze; both shaders, the native shim, and the Go adapter retain their measured bytes.

Independent physical Radeon RX 7600 and Strix Halo runs passed all five output/state/continuation and fallback cases, including eight-token unsupported geometry. A fresh isolated Strix Halo run passed the six-pair component acceptance gate. The component result applies to the declared preprojected GDN call, including its output RMS pass; it establishes no full-model stage attribution or end-to-end throughput claim.

After correcting the stale module count, the complete `internal/computebuild` suite passed on Linux. The normal `internal/compute` suite passed with only `TestScaffoldGeneratedTaxonomyBuildsAndTestsGreen` excluded. The earlier broader run timed out after 15 minutes inside that unchanged test's nested Go build; it is not a passing full-suite result. Reproduce the scoped compute check with `go test ./internal/compute -count=1 -skip '^TestScaffoldGeneratedTaxonomyBuildsAndTestsGreen$'`.

Operator evidence is retained in the private companion receipt set `docs/benchmarks/receipts/halo-gdn-prefill-12533-20260910/`: `final-component-correctness.raw.log`, `component-ab-n6-isolated.raw.log`, and `component-ab-n6.json`, with `component-isolated-pre-snapshot.txt`, `component-isolated-post-snapshot.txt`, and `component-lifecycle-restoration.txt`. The private receipt set binds the runtime source, binary and shader identities, raw paired samples, device and system state, and independent validation. This public ticket records the gate verdict without reproducing private measurements.

Same-build whole-model and continuation qualification remains blocked under #12098. A coexistence run was aborted because of memory pressure and supplies no correctness or performance credit. A later isolated Q2_K_XL original-arm control failed the existing strict logit-parity gate despite matching generated token IDs; the GDN candidate was disabled and its arm was skipped. `fullmodel-isolated-env0-control.raw.log` retains that failure. It supplies no candidate or whole-model performance credit. Default promotion remains blocked until the separate whole-model gate is witnessed.

## Done condition

- [x] [SW-VERIFIED] New registered shaders and capability/shape-gated dispatch provide the complete opt-in path through the ordinary native sequence entry.
- [x] [SW-VERIFIED] Independent tests preserve finite outputs, recurrent state, convolution carry, grouped-head mapping, and split-prefill/continuation behavior under fixed predeclared tolerances.
- [x] [SW-VERIFIED] Short panels, unsupported geometry, and forced scalar controls execute the original path; candidate dispatch is observed only for admitted opt-in cases.
- [x] [SW-VERIFIED] The complete build-verifier suite, compute suite excluding the exact unchanged scaffold test above, shader compilation/validation, and targeted public boundary checks pass on the landing source. The broader-suite timeout remains disclosed.
- [x] [HW-WITNESSED] At least six interleaved scalar/candidate component pairs after excluded warmup preserve correctness and show at least 5% lower candidate median latency for the declared geometry; retain P50/P90, uncertainty, raw samples, source/binary/shader identities, driver, clocks/governor, thermals, memory state, and actual dispatch counts.
- [ ] [HW-WITNESSED] Same-build full-model native sequence and continuation correctness and matched timing are recorded separately before any default or end-to-end performance claim.

## Definition of done

The opt-in variant is integrated and independently verified with preserved output and state semantics and the issue's physical component gate. Same-build whole-model evidence remains a separate requirement for default promotion or an end-to-end performance claim. A failed physical gate is retained as a rejected candidate, not reported as completion of #12533.

## Acceptance gate

Independent software output/state parity and the six-pair physical component gate above must pass for the same candidate. Default promotion additionally requires same-build whole-model correctness and timing.

## Closure binding

The resolving commit references #12533 and its exact source-bound witness. Leave the issue open while its component correctness or physical performance criteria remain unfulfilled.

## Working spine

1. Freeze the supported geometry, scalar oracle, selector, and acceptance tolerances.
2. Implement the additive tiled recurrence and output RMS shaders.
3. Register the shaders and gate native dispatch with observed counters.
4. Run independent output/state/fallback/continuation tests and build checks.
5. Bind an exact candidate and collect the six-pair physical component comparison.
6. Collect same-build full-model correctness and timing when an exclusive hardware window is available.
7. Review evidence, retain any failed arm honestly, and land only the behavior its evidence supports.
