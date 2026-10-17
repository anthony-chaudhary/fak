# perf(vulkan): tile GDN recurrent prefill state for Qwen3.8

<!-- fak-compute-key: halo-gdn-prefill-state-tile-12533 -->

```routing
lane: internal/compute
paths: ["internal/compute/shaders/qwen35_gdn_prefill_tiled.comp", "internal/compute/shaders/qwen35_gdn_prefill_norm.comp", "internal/compute/shaders/qwen35_gdn_verify_tiled.comp", "internal/compute/vulkan_shim.cpp", "internal/compute/vulkan_gdn_prefill_tiled.go", "internal/compute/vulkan_gdn_prefill_tiled_test.go", "internal/compute/vulkan_qwen35_gdn.go", "internal/compute/build_vulkan.ps1", "internal/computebuild/vulkan.go", "internal/computebuild/computebuild_test.go", "THIRD_PARTY_NOTICES.md"]
expected_steps: 12
github_issue: 12533
```

Tracking: [fak #12533](https://github.com/anthony-chaudhary/fak/issues/12533).
Status: opt-in register-resident routes implemented for prefill (tokens >= 8, `qwen35_gdn_prefill_tiled`) and decode/verify (tokens 1..7, `qwen35_gdn_verify_tiled`); fak#12533 is closed on that opt-in component scope. This ticket now owns default promotion of both routes ([Default promotion (2026-10-17)](#default-promotion-2026-10-17)), gated by fak#12098. Open: tokens=1 runtime route witness, kernel-only timing (unmeasured), and fak#12735 logit parity (original-arm control still failing on Q2_K_XL). Both default-flip constants remain `false`; no default or end-to-end performance claim.

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

Add one fak-native variant for 128x128 recurrent heads, grouped 16 key / 48 value heads, and panels of at least eight tokens. Preserve the ordinary session call path. Normal application use requires explicit `FAK_VULKAN_GDN_PREFILL_TILED=1`; the default remains off. Panels of 1-7 tokens take the separate verify-width kernel `qwen35_gdn_verify_tiled` under the same opt-in and its own self-check verdict. Unsupported shapes and the forced scalar arm retain the existing kernel. Read, evolve, and commit the same persistent recurrent state and preserve convolution carry across calls.

## Gold-plating boundary

No Q4 cooperative GEMM work, model-loader changes, new model architectures, generic graph scheduler, alternative inference runtime, or public API change. Default promotion is in scope only through the gated steps in [Default promotion (2026-10-17)](#default-promotion-2026-10-17). The variant does not edit the original recurrent shader. Its output RMS pass is part of this candidate, so timing establishes the combined variant's effect. Whole-model improvement remains pending a same-build comparison; primitive speed alone cannot establish it.

## Likely files

Only `internal/compute` and `internal/computebuild` are implementation packages. The routing block names the proposed new shaders, native dispatch, selector and counter wiring, independent tests, and shader registry. This ticket is their local specification for existing #12533.

The original three-file estimate expands to nine paths: eight implementation, test, and build paths plus this ticket. The additive variant avoids conflicting edits to the original shader, registers both shader build entry points, and includes independently authored behavioral tests. Scope remains seven steps across two implementation packages. Default promotion adds steps 8-12 and three routing paths: the verify-width shader, the Go GDN entry (`vulkan_qwen35_gdn.go`), and `THIRD_PARTY_NOTICES.md`, which changes only if step 4 adapts upstream code.

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

## Default promotion (2026-10-17)

This ticket owns the register-resident GDN default flip for both width classes, gated by fak#12098 and coordinated with fak#12537:

| Class | Tokens per call | Kernels | Flip constant (origin/main `10989c198`) |
| --- | --- | --- | --- |
| Prefill | >= 8 | `qwen35_gdn_prefill_tiled` + `qwen35_gdn_prefill_norm` | `kGdnTiledDefaultFlipEarned`, `internal/compute/vulkan_shim.cpp:8108` |
| Decode / verify | 1..7 | `qwen35_gdn_verify_tiled` + `qwen35_gdn_prefill_norm` | `kGdnVerifyDefaultFlipEarned`, `internal/compute/vulkan_shim.cpp:8118` |

### Process cause

The original scope excluded default promotion. The verify-width route arrived after the component gate, so neither its timing nor a whole-model A/B was ever owned. fak#12098 is cited by the flip-constant comments but owns no kernel. The component benchmark timed a synchronized readback, so the kernel time that would justify a flip was never measured.

### Problem

- Stock `qwen35_gdn_recurrent` runs one 64-lane workgroup per value head (48 workgroups). For every token it makes two global-memory read-modify-write passes over the 128x128 fp32 state.
- A device-timestamp profile on a Strix Halo node at fak `344f353f09` puts the GDN recurrent stage at about 6.7 ms per decode token. That is about 140 us per GDN layer across 48 GDN layers (derived).
- Tiled kernel-only time is unknown. The component benchmark's declared `timing_scope` is the resident GDN call plus a synchronized output read, issued as three one-shot submits. At P512, the 2026-09-10 receipt's tiled arm is dominated by a ~12.6 MB readback (512 x 48 x 128 fp32) from uncached HOST_VISIBLE memory. The component gate passed but places no bound on kernel time.
- Opt-out gap: the prefill-class selector (`tiledRequested`, `vulkan_shim.cpp:8109-8110`) never tests `optOut`. Once `kGdnTiledDefaultFlipEarned` is `true`, `FAK_VULKAN_GDN_PREFILL_TILED=0` would no longer force stock for tokens >= 8; only the verify-class selector (`:8119`) honors `=0`. No counter records opt-out selections.
- State traffic sets the floor. Each GDN layer holds 3.1 MB of fp32 state, 151 MB across 48 layers, and that exceeds the on-die cache. At tokens=1 a register-resident kernel still moves at least 6.3 MB per layer (one read, one write), about 25-33 us at 190-256 GB/s (derived). At tokens >= 8 that traffic is amortized over the panel. Estimated decode saving: about 4-5 ms/token if the verify kernel reaches about 150-200 GB/s of effective state bandwidth.
- Per-token GDN cost does not depend on context length, so the decode saving carries over unchanged to long-context Q4_K_M operation (fak#11572) while attention's share grows.

### Change

1. **Route check (tokens=1).** Source trace at origin/main: `Session.qwen35LinearHAL` (`internal/model/qwen35_hal.go:766`) -> `Qwen35GDNDecode` -> `Qwen35GDNPreprojected(tokens=1)` (`internal/compute/vulkan_qwen35_gdn.go:54`) -> `fvk_qwen35_gdn_preprojected_f32` (`:188`). The exception is `FAK_DISABLE_VECTOR_GDN`, `FAK_VECTORIZED_DELTANET=0`, or the backend option, which select the host scalar recurrence with full readback. Partial-commit prefix replay (`vulkan_qwen35_partial_commit.go:152`) calls the same native entry. `qwen35_gdn_decode` in `internal/amdgpu` is a parity-receipt selector label, not a dispatch route. Witness the route at runtime: each decoded token makes 48 verify-class selections (verify plus stock counts from `fvk_debug_gdn_verify_tiled_snapshot`). Under opt-in, ts-stage rows show `kernel=qwen35_gdn_verify_tiled groups=16x48`; on the stock route they show `kernel=qwen35_gdn_recurrent groups=48x1`. If any production decode or replay path reaches GDN through another entry, route it through this selector behind the same `g_gdn_verify_verdict`. Never add a second selector.
2. **Kernel-only timing.** Add a device-timestamp arm to `BenchmarkVulkanQwen35GDNTiledPrefillComponentAB`. Record conv, recurrent, and norm for each arm inside one batch under `FAK_VULKAN_TIMESTAMP_PROFILE=1`, and keep readback and validation outside the timed region. Emit per-stage `kernel_ns` with `timing_scope: "device timestamps; no readback"`. Sweep P in {1, 4, 7, 8, 16, 32, 128, 512} and report us/token/layer per kernel. Cross-check in situ against whole-model ts-stage rows for decode and a 512-token prefill chunk. Timestamps are BOTTOM_OF_PIPE deltas between barrier-separated dispatches, and the profile disables batch cycling, so use them for attribution, not throughput.
3. **Parity.** Use the fak#12735 logit oracle unchanged: cosine >= 0.99999 and relative L2 <= 1e-4 over complete logit vectors. Compare the token reference against the whole and split sequence arms with `FAK_VULKAN_GDN_PREFILL_TILED` unset, `=1`, and `=0`. Add greedy equivalence with a predeclared tie tolerance: a token mismatch is allowed only where the reference top-2 logit gap is below a frozen epsilon. Greedy text identity across fresh processes is not an oracle. Judge the candidate only after the fak#12735 original-arm control passes on the same artifact and build. Add a compute-level state-continuity test: one state chain across calls at tokens 512 -> 8 -> 7 -> 1, the reverse order, and a partial-commit replay prefix, all against the CPU oracle with tolerances frozen before the run.
4. **Adapt if slow.** Two triggers. For the prefill class: kernel time above ~15 us/token/layer at P512 (~0.72 ms/token over 48 layers). For tokens=1: the verify kernel moves its 6.3 MB/layer below ~150 GB/s effective (~42 us/layer, ~2.0 ms/token). 15 us/layer sits below the tokens=1 bandwidth floor, so that bound applies to the prefill class only. On either trigger, ADAPT the layout of [ggml-org/llama.cpp@f74e2fa6a31cc33ab4ee4ae516641cfe62df3924 `ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp`](https://github.com/ggml-org/llama.cpp/blob/f74e2fa6a31cc33ab4ee4ae516641cfe62df3924/ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp): a register-resident state shard per lane, single-subgroup workgroups, `subgroupClusteredAdd` column reductions, and lanes-per-column as a specialization constant. The upstream file has no per-file header (repository license MIT, ggml authors). Add an MIT attribution header with URL and sha to the adapted shader and an entry in `THIRD_PARTY_NOTICES.md`. Sweep columns per workgroup (lanes per column 4/8/16/32 at the device's reported subgroup size, 32 or 64) on the target GPU, and record the winner per device identity in the receipt. Keep the variant behind the same selector, self-check, and counters, and register it in `internal/computebuild`.
5. **Whole-model A/B and flip.** Run the fak#12098 protocol: three uncounted warmups per arm, at least five measured pairs in AB/BA order, CV <= 5%. Swap, throttle, fallback, or a missing counter makes the result `NOT_COMPARABLE`. Arms: decode (tg128) and 512-token prefill, `FAK_VULKAN_GDN_PREFILL_TILED` unset vs `=1`, same build and node, thermally unflagged, plus a same-node llama.cpp pair. Before either flip, make `=0` force stock in both selectors and add an opt-out counter exposed through the debug snapshot and dispatch profile. Flip each constant only on its own width-class receipt. The self-check verdict stays a hard gate after the flip, so a declining device stays stock. A zero-code A/B (`=1` vs unset at tip, same-node llama.cpp pair, thermally unflagged arms) is in flight. Its result is pending and is not cited here.

### Acceptance witness

PowerShell; substitute the observed device substring and checkpoint path:

```powershell
$env:FAK_VULKAN_REQUIRE_DEVICE = '1'
$env:FAK_VULKAN_EXPECT_DEVICE = '<observed intended device substring>'
go test -tags vulkan ./internal/compute -run 'TestVulkanQwen35GDN(TiledPrefillParityAndSplitContinuation|VerifyWidthParityAndRoute|SelfCheckRunsOnceBeforeForward|SelfCheckForcedDisagreementFallsBack)$' -count=1 -v
$env:FAK_QWEN35_SEQUENCE_FULL_MODEL_WITNESS = '1'
$env:FAK_QWEN38_GGUF = '<checkpoint path>'
$env:FAK_VULKAN_TIMESTAMP_PROFILE = '1'
$env:FAK_VULKAN_TIMESTAMP_PROFILE_EVERY = '1'
foreach ($arm in 'unset', '1', '0') {
  if ($arm -eq 'unset') { Remove-Item Env:FAK_VULKAN_GDN_PREFILL_TILED -ErrorAction SilentlyContinue } else { $env:FAK_VULKAN_GDN_PREFILL_TILED = $arm }
  go test -tags vulkan ./internal/model -run '^TestVulkanFullModelSequencePrefillStandalone$' -count=1 -v *> "gdn-arm-$arm.log"
}
Select-String -Path gdn-arm-*.log -Pattern 'ts-stage kernel=qwen35_gdn_', '"passed"'
```

- [ ] [SW-VERIFIED] With the default path forced on (a debug seam standing in for both flip constants), `FAK_VULKAN_GDN_PREFILL_TILED=0` selects stock in both selectors and increments the opt-out counter. Route assertions compare counters or typed sentinels, never log prose.
- [ ] [SW-VERIFIED] The cross-class state-continuity test (512 -> 8 -> 7 -> 1, reverse, and partial-commit replay) passes against the CPU oracle with frozen tolerances.
- [ ] [HW-WITNESSED] Route: under `=1`, decode shows 48 verify-class selections per token, ts-stage `qwen35_gdn_verify_tiled groups=16x48`, and no tokens=1 `qwen35_gdn_recurrent` dispatch. Under `=0` the reverse holds.
- [ ] [HW-WITNESSED] A kernel-only device-timestamp receipt per width class, P 1..512, with no readback in the timed region, bound to source, binary, shader, and driver identity.
- [ ] [HW-WITNESSED] The fak#12735 oracle passes for unset, `=1`, and `=0` on one artifact and build, and greedy equivalence with tie tolerance holds.
- [ ] [HW-WITNESSED] The fak#12098-protocol A/B shows tg128 and 512-token prefill not regressed and the GDN stage reduced before the class's constant flips. Estimated targets: decode GDN <= 2.0 ms/token (from ~6.7); prefill GDN <= ~0.72 ms/token at P512.
- [ ] Step 4 is executed, or recorded as not triggered with the measured kernel times cited.

### Scope fence

Excluded: chunked/WY GDN (a later stage after the flip; in-house reference `internal/compute/metal/wyf_gdn.metal`, chunk size 32), f16 recurrent state, a multi-sequence grid dimension (B > 1), MTP state-snapshot slots (partial commit already replays prefixes), conv+recurrent fusion (fak#12537), and any change to the self-check deviation bound or the fak#12735 thresholds. No constant flips without its own width-class receipt.

### Coordination

fak#12537 fuses GDN convolution and recurrence on the same `qwen35_gdn_*` shaders, the same `fvk_qwen35_gdn_preprojected_f32` ABI, and the same `vulkan_shim.cpp` seam. Allow one lane at a time on those paths; the second lander rebases onto the first lander's selector. A fused kernel inherits the width-class selector, self-check verdict, opt-out, and counters, and repeats steps 2, 3, and 5 before it can become default. Every A/B names the recurrent kernel that ran, from counters and ts-stage rows.

### Related

fak#12533 (closed; opt-in component), fak#12098 (promotion gate), fak#12735 (logit oracle; control failing), fak#12537 (conv+recurrent fusion), fak#12687 (K<=4 draft verify in one resident panel; consumes the 1..7 route), fak#11572 (long-context campaign). Upstream: [ggml-org/llama.cpp@f74e2fa6a31c](https://github.com/ggml-org/llama.cpp/tree/f74e2fa6a31cc33ab4ee4ae516641cfe62df3924) (MIT).

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
8. Witness the tokens=1 route at runtime (Default promotion step 1).
9. Measure kernel-only time per width class with device timestamps (step 2).
10. Pass the fak#12735 oracle and cross-class state continuity (step 3).
11. Adapt the upstream layout only if a step-4 trigger fires (step 4).
12. Close the opt-out gap, run the fak#12098 A/B, and flip each class on its own receipt (step 5).
