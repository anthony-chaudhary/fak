<!-- fak-compute-key: vulkan-q4k-wave32-decode -->
# perf(compute): qualify Wave32 cooperative Q4_K decode on gfx1151

```routing
lane: compute
paths: ["internal/compute/vulkan_shim.cpp", "internal/compute/vulkan_q4k_wave32.go", "internal/compute/vulkan_q4k_wave32_test.go", "internal/computebuild/vulkan.go", "internal/computebuild/computebuild_test.go"]
expected_steps: 8
```

GitHub: [#12175](https://github.com/anthony-chaudhary/fak/issues/12175)

## Parent issue

Parent campaign: [#11572](https://github.com/anthony-chaudhary/fak/issues/11572). Evidence dependencies: [#11919](https://github.com/anthony-chaudhary/fak/issues/11919), [#12001](https://github.com/anthony-chaudhary/fak/issues/12001), and [#12054](https://github.com/anthony-chaudhary/fak/issues/12054).

## Value

- **For:** fak-native Q4_K decode on the AMD gfx1151 Vulkan serving path.
- **Problem:** the current correct kernel gives each invocation a complete output-row dot product; it does not test whether one Wave32 cooperatively dividing a 256-value Q4_K block can improve a matched physical decode workload.
- **Today:** `q4k_matmul.comp` uses a 64-thread local size, flattened row/token indexing, and a serial eight-group inner loop per invocation. Q4_K correctness and a real-shape physical control exist, but there is no Q4-specific subgroup capability gate or cooperative pipeline.
- **Better because:** a separate 32-lane, two-output-row candidate would make one falsifiable change while preserving the current shader as the runtime control and rollback path.
- **Witness:** source-bound software parity plus exclusive, paired A/B/B/A physical measurements against the scalar control from the same revision.
- Centrality: Core
- P1: advanced - tests a bounded optimization on the fak-native decode critical path.
- P2: preserved - fak retains ownership of kernel, dispatch, admission, and fallback.
- P3: advanced - reuses the existing Q4_K format, Vulkan HAL, shader build, and physical profile to make one isolated candidate.
- P4: preserved - acceptance requires independently checked quality and matched physical evidence; analytical or simulated results cannot claim speedup.

## Why now

#11919 supplies a source-bound Q4_K control, #12054 supplies a cooperative Vulkan precedent, and #12001 supplies negative controls that make a one-variable Q4_K experiment actionable without speculative framework work.

## Current state

Classification: **PARTIAL**.

PRESENT:

- Correct Q4_K Vulkan dequantization/GEMV, device-resident weights, and multi-token correctness.
- A source-bound `17408x5120`, `P=1` physical profile from #11919.
- Generic required-subgroup-size pipeline plumbing in `vulkan_shim.cpp`.
- A physically qualified cooperative Q8 design precedent in #12054.

ABSENT:

- Query and admission of compute subgroup arithmetic for the Q4 path.
- An independently implemented Wave32 cooperative Q4_K shader and pipeline.
- Candidate-versus-scalar runtime selection and a source-bound physical KEEP/REJECT receipt.

This ticket records a hypothesis, not a performance claim. #12001 is direct negative evidence against assuming cooperation is faster: its one-Wave32-per-row Q2_K candidate passed parity but was 2.01x slower at `P=1` and 2.97x slower at `P=4`; an eight-lane partition was 1.926x and 2.442x slower; and four live token accumulators made `P=4` 10.8x slower. Its later two-dimensional scalar dispatch candidate improved its Q2_K primitive, but has not established any Q4_K or full-model gain.

## Core through-line

Resident Q4_K row blocks plus resident f32 activation -> Q4-specific subgroup-capability admission -> separate 32-lane/two-row cooperative decode pipeline -> subgroup FP32 reduction -> resident output -> strong CPU-reference parity -> paired scalar/candidate physical receipt -> KEEP or REJECT without removing the scalar fallback.

The candidate maps one 32-lane subgroup to two output rows. For each 256-value Q4_K block, each lane owns eight K values, accumulates two FP32 row partials, and participates in one subgroup reduction per row; lane zero writes. Select the candidate only for `P=1` and a reported/effectively required subgroup size of 32. Unsupported devices and all `P>1` panels retain the scalar pipeline.

## Working spine

Q4_K input and f32 activation -> admitted Vulkan Wave32 candidate -> output -> parity result -> durable paired physical KEEP/REJECT receipt.

## Scope

One separate cooperative Q4_K decode pipeline, its exact capability selector, parity coverage, and physical A/B receipt contract.

## Gold-plating boundary

- No other quantization format, full-model graph, command-graph reuse, scheduler, or model-loader change.
- No multi-token weight sharing, four-accumulator panel, Wave64 specialization, integer-dot path, or generalized subgroup framework.
- No replacement of the scalar shader or silent CPU/external-runtime fallback.
- No full-model or tokens/s claim from a primitive microbenchmark.
- No hardware access or physical result is required merely to land a software candidate; physical acceptance remains unchecked until witnessed.

## Prior art and clean-room/legal boundary

Authoritative inspiration was inspected at `ggml-org/llama.cpp@5d806aa2575e01e126651fd69ab1ab6cefff861d`, accessed 2026-09-08:

- [`mul_mat_vec.comp`](https://github.com/ggml-org/llama.cpp/blob/5d806aa2575e01e126651fd69ab1ab6cefff861d/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp)
- [`mul_mat_vec_base.glsl`](https://github.com/ggml-org/llama.cpp/blob/5d806aa2575e01e126651fd69ab1ab6cefff861d/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl)
- [`dequant_funcs.glsl`](https://github.com/ggml-org/llama.cpp/blob/5d806aa2575e01e126651fd69ab1ab6cefff861d/ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl)
- [`ggml-vulkan.cpp`](https://github.com/ggml-org/llama.cpp/blob/5d806aa2575e01e126651fd69ab1ab6cefff861d/ggml/src/ggml-vulkan/ggml-vulkan.cpp)
- [MIT license](https://github.com/ggml-org/llama.cpp/blob/5d806aa2575e01e126651fd69ab1ab6cefff861d/LICENSE)

The upstream source is MIT-licensed and its design is legally usable as inspiration in this Apache-2.0 project. Implement from the public Q4_K mathematical format and the independently stated mapping above; do not copy upstream source, comments, identifiers, or test vectors. If upstream code is ever incorporated directly, retain its MIT copyright/license notice. Refresh the study if the pinned shader/host files, target driver, or shader compiler changes.

Required Vulkan contracts are documented by Khronos: [`VkPhysicalDeviceSubgroupProperties`](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceSubgroupProperties.html), [`VkSubgroupFeatureFlagBits`](https://docs.vulkan.org/refpages/latest/refpages/source/VkSubgroupFeatureFlagBits.html), the [subgroup guide](https://docs.vulkan.org/guide/latest/subgroups.html), and [`VkPipelineShaderStageCreateInfo`](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageCreateInfo.html).

## Done condition

- [ ] [SW-VERIFIED] The new shader compiles for Vulkan 1.2, its SPIR-V validates, and the tagged shim/test binary links.
- [ ] [SW-VERIFIED] Devices without compute subgroup arithmetic or effective/required subgroup size 32 select the unchanged scalar pipeline without attempting an invalid candidate pipeline.
- [ ] [SW-VERIFIED] Crafted Q4_K fixtures cover all eight packed scale/min groups, both nibbles, odd row tails, and `in={256,512,768}` with finite output.
- [ ] [SW-VERIFIED] Candidate `P=1` parity holds at exact argmax, relative L2 `<=1e-4`, and cosine `>=0.99999`; `P>1` is witnessed on the scalar fallback.
- [ ] [HW-WITNESSED] The target gfx1151 device reports subgroup arithmetic and an effective subgroup size of 32, with no validation error or fallback in the candidate arm.
- [ ] [HW-WITNESSED] Same-revision scalar/candidate A/B/B/A receipts contain at least ten post-warm samples per arm for `17408x5120,P=1` and `5120x17408,P=1`, including source, shader, binary, compiler, driver, device, clock/headroom, allocation, and raw-sample identity.
- [ ] [HW-WITNESSED] KEEP requires at least 10% median improvement on both real shapes, while scalar `P=3/P=4` fallback regresses no more than 5%; otherwise record REJECT and retain scalar default.
- [ ] A primitive KEEP only unlocks a separately scoped native full-model receipt under #11572; it does not establish tokens/s itself.

## Definition of done

All software checks above pass, and a source-bound physical receipt records either KEEP or REJECT without weakening or removing the scalar fallback.

## Verifiable Witness

Software:

```powershell
pwsh internal/compute/build_vulkan.ps1 shaders
go test ./internal/computebuild -run TestVulkanShadersCompleteness -count=1
go test ./internal/compute -run 'TestVulkanQ4KMatMulMatchesCPUReference|TestVulkanQ4KWave32' -count=1
go vet ./internal/compute ./internal/computebuild
```

Physical, after software review and under the repository's exclusive GPU lease/idle gate:

```sh
FAK_VULKAN_SPIRV="$PWD/build/spirv" \
FAK_VULKAN_REQUIRE_DEVICE=1 \
FAK_VULKAN_EXPECT_DEVICE=8060S \
FAK_VULKAN_Q4K_PROFILE=1 \
FAK_VULKAN_DISPATCH_PROFILE=1 \
./build/compute.test -test.run '^TestVulkanQ4KWave32PhysicalAB$' -test.v
```

The physical test must time MatMul plus synchronized read, excluding upload, validation, and free, matching #11919. The historical #11919 median is context only; acceptance compares freshly built scalar and candidate arms from the same source revision.

## Acceptance gate

Software admission requires every `[SW-VERIFIED]` criterion. Runtime default admission additionally requires every `[HW-WITNESSED]` criterion and the stated 10% two-shape KEEP threshold.

## Closure binding

The resolving implementation commit cites this issue number and carries the `(fak compute)` leaf trailer. A documentation-only ticket commit does not close the implementation issue.

## Work estimate

Estimate: 5 points (medium). Uncertainty is dominated by driver capability behavior and physical variance; revise through the parent denominator if implementation evidence changes.

## Overall completion contribution

Contribution: 5/100 points. The 100-point parent baseline is a provisional campaign denominator, not a percentage-complete claim.

## Completion standard

experiment

## File:Line seams

- `internal/compute/shaders/q4k_matmul.comp:3` — retained scalar control and current local size.
- `internal/compute/shaders/q4k_matmul.comp:27` — current flattened row/token ownership and serial row dot.
- `internal/compute/vulkan_shim.cpp:623` — existing optional required-subgroup-size pipeline seam.
- `internal/compute/vulkan_shim.cpp:798` — device property/feature query seam.
- `internal/compute/vulkan_shim.cpp:1556` — current flattened Q4 dispatch seam.
- `internal/compute/vulkan_q4k_profile_test.go:14` — existing physical real-shape oracle.

## Blast radius and fallback

Primary blast radius is `internal/compute`; `internal/model`, other quant formats, CUDA, Metal, and private appliance lanes remain unaffected. The quarantined fallback is the unchanged scalar Q4_K shader/pipeline. A missing capability, compile failure, parity failure, or missed physical KEEP gate leaves or restores scalar selection.

## Likely files

- `internal/compute/shaders/q4k_matmul_wave32.comp`
- `internal/compute/vulkan_shim.cpp`
- `internal/compute/build_vulkan.ps1`
- `internal/compute/vulkan_q4k_test.go`
- `internal/compute/vulkan_q4k_profile_test.go`
- `internal/compute/vulkan_q4k_wave32.go`
- `internal/compute/vulkan_q4k_wave32_test.go`
- `internal/computebuild/vulkan.go`
- `internal/computebuild/computebuild_test.go`

## Lane

`compute`; one public native-serving leaf under #11572.
