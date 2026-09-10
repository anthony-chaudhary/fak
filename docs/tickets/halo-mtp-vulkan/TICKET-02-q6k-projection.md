<!-- fak-compute-key: vulkan-native-packed-q6k-matmul-v1 -->
<!-- fak-public-issue: anthony-chaudhary/fak#12403 -->

# feat(vulkan): execute one resident Q6_K projection without F32 materialization

```routing
lane: strix-vulkan-q6k
paths: ["internal/compute/shaders/q6k_matmul.comp", "internal/compute/vulkan.go", "internal/compute/vulkan_shim.cpp", "internal/compute/vulkan_backend.h", "internal/computebuild/vulkan.go", "internal/computebuild/computebuild_test.go", "internal/compute/build_vulkan.ps1", "internal/compute/vulkan_q6k_test.go"]
expected_steps: 8
```

## Parent context

Parent campaign: #11572. Coordinates with #9352, #9820, #11917, and
#12387. Physical comparison promotion additionally requires #12098.
The retained Qwen3.8 MTP draft leaf #12538 needs this primitive because its
value/down tensors and shared output head remain exact Q6_K.

## Current state

At public source `19d160b9fdac5203c518a713f09ceda93f5a5b68`, the CPU and
CUDA backends execute packed Q6_K directly, while Vulkan rejects Q6_K upload
and matmul. The canonical CPU oracle is `dequantQ6K` and `rawKRowDot` in
`internal/compute/quant_q4k.go`; CUDA's `k_q6k_gemm` independently implements
the same 210-byte superblock layout.

The loaded Qwen3.8-27B Q4_K_M inventory binds one next-token predictor and
retains Q6_K projections. Expanding them to F32 or requantizing them changes
the checkpoint representation and materially increases residency. The issue's
original three-file limit omitted the C ABI, optional kernel registration, and
both canonical shader build lists, so it could not produce a reachable path.

Centrality: Core. Portfolio tier: 2 (serving). Priority: P0.

- P1 Context: advanced - supplies the missing retained-draft projection path.
- P2 Net value: advanced - keeps one exact packed representation resident.
- P3 Adaptation: preserved - CPU is the oracle, never a hidden runtime fallback.
- P4 Operations: advanced - explicit capability admission distinguishes old bundles.

## Why this is next

The target verifier and retained MTP loader now expose this exact missing
operation. One portable packed kernel plus its existing Vulkan registry seam is
the smallest complete slice that lets #12538 remain device resident.

## Working spine

Exact GGUF Q6_K bytes -> padded device-local upload -> optional Vulkan kernel
capability -> packed dequantized dot product -> F32 result -> observed receipt.

## Core through-line

Add a portable `q6k_matmul` shader for `Y[P,out] = X[P,in] * W[out,in]^T`.
Each 256-weight superblock remains the exact 210-byte GGML layout: 128 low-bit
bytes, 64 high-bit bytes, 16 signed scales, and one little-endian F16 block
scale. Upload those bytes once without F32 expansion or requantization, pad only
the descriptor allocation to a four-byte boundary, and route Vulkan MatMul and
BatchedMatMul for Q6_K through the shader for P=1..4.

Load the shader as an optional pipeline. A runtime using an older otherwise
valid SPIR-V bundle must still initialize, report `SupportsQ6KMatMul() == false`,
and let the MTP constructor return its typed pre-execution refusal. A new bundle
that contains the valid shader reports true. No request may silently execute a
CPU Q6_K projection after Vulkan admission.

Host validation rejects malformed rank, a reduction dimension not divisible by
256, mismatched raw length, dimensions outside signed shader integers, or packed
byte addressing beyond the Vulkan storage-buffer/uint32 envelope before upload
or dispatch. Real required shapes include 5,120 and 17,408 reduction dimensions
and the 248,320 by 5,120 shared head.

## Gold-plating boundary

No Q5_K path, cooperative-matrix rewrite, fused SwiGLU/down shader, fused
RMSNorm/head shader, command graph, allocator redesign, model-loader change,
acceptance-policy change, external engine, or default-on serving promotion.
Do not add an F32 materialization, Q8 requantization, CPU fallback, or permanent
stub. Performance tuning follows only after this direct scalar tracer is exact.

## Done condition

- [ ] [SW-VERIFIED] Vulkan uploads exact packed Q6_K bytes once into device-local
  weight storage; allocation-only tail padding cannot change the logical bytes.
- [ ] [SW-VERIFIED] MatMul and BatchedMatMul for P=1,2,4 match the canonical CPU
  Q6_K oracle within the established floating-point tolerance and exact argmax.
- [ ] [SW-VERIFIED] Tests cover mixed signed scales, all ql/qh bit groups,
  two-byte-aligned odd superblock starts, output tails, repeated calls, and the
  5,120/17,408 reduction dimensions without allocating a full vocabulary head.
- [ ] [SW-VERIFIED] The 248,320 by 5,120 head passes checked size/admission math
  without overflow and remains inside the supported uint32 descriptor envelope.
- [ ] [SW-VERIFIED] Missing or invalid optional `q6k_matmul.spv` does not break
  an older complete Vulkan bundle; capability is false and the MTP route refuses
  before allocations. A present valid pipeline reports capability true.
- [ ] [SW-VERIFIED] The C ABI, Go shader registry, registry test, and existing
  PowerShell builder name the same selector and buffer/push-constant contract.
- [ ] [HW-WITNESSED] A source-bound required-device oracle executes the real
  Vulkan Q6_K path with zero CPU fallback and exact argmax.
- [ ] [HW-WITNESSED] A same-artifact N>=5 interleaved A/B reports raw samples,
  P50/P90 and confidence intervals, unchanged accepted output and bounded
  residency. A performance claim requires at least 5% c1 lift; otherwise retain
  the rejection receipt and track the next lever.

## Witness

Deterministic packed oracle and source-contract witness:

```text
CGO_ENABLED=1 FAK_VULKAN_DISPATCH_PROFILE=1 FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/compute -run 'TestVulkanQ6K' -count=1
```

Compile and validate the standalone shader locally:

```text
glslc -O --target-env=vulkan1.2 -fshader-stage=comp internal/compute/shaders/q6k_matmul.comp -o <scratch>/q6k_matmul.spv
spirv-val --target-env vulkan1.2 <scratch>/q6k_matmul.spv
```

The device test must execute rather than skip when a required Vulkan device is
declared. Hardware performance remains unmeasured until the physical gate runs.

## Done condition / witness

Done condition: exact packed Q6_K upload and P=1..4 Vulkan projection execute
without materialization, requantization, or CPU fallback while older shader
bundles receive a typed pre-allocation refusal.

Witness: the focused oracle, shader compilation/validation, source registry
contract, required-device execution, and matched physical A/B above.

## Verifiable Witness

Compare output against `Default().MatMul` / `Default().BatchedMatMul` over the
same raw 210-byte blocks. Observe the concrete Vulkan route and transfer counters;
a capability field alone is insufficient. Bind physical results to the clean
source, exact model hash, native library and shader digests, device, driver,
compiler, clocks, power, thermals, and every sample.

## Blast radius and affected lanes

The implementation spans the existing Vulkan runtime and its canonical shader
builders in `internal/compute` and `internal/computebuild`. CPU, CUDA, Metal,
model loading, target verification, gateway policy, and installer code remain
unchanged.

## Quarantined fallback mechanism

The Q6_K pipeline is optional at Vulkan initialization so already-installed
shader bundles stay usable for their existing dtypes. Capability false is an
explicit typed admission refusal for resident MTP; it never authorizes a CPU
projection. Malformed Q6_K data fails before device mutation.

## Likely files

- `internal/compute/shaders/q6k_matmul.comp`
- `internal/compute/vulkan.go`
- `internal/compute/vulkan_shim.cpp`
- `internal/compute/vulkan_backend.h`
- `internal/computebuild/vulkan.go`
- `internal/computebuild/computebuild_test.go`
- `internal/compute/build_vulkan.ps1`
- `internal/compute/vulkan_q6k_test.go`

## Lane

strix-vulkan-q6k

## Expected steps

8: implement shader; bind optional pipeline; export capability and C ABI; retain
raw upload; route matmul; align builders; run independent oracle; qualify device.

## Acceptance gate

Software landing requires the packed oracle, source-contract tests, shader
compile/validation, affected-package checks, and zero hidden fallback. Hardware
evidence gates performance credit and promotion, not the bounded software slice.

## Definition of done

The reachable optional Vulkan path retains exact Q6_K bytes, passes the packed
CPU oracle and capability/refusal tests, and ships in both canonical shader
builders. Physical promotion remains separately bound to the declared A/B gate.

## Closure binding

Resolving commits cite #12403 and #12538 and include independent test provenance.
Keep #12403 open until the physical gate passes or a measured rejection is
preserved with the next implementation step tracked.

## Work estimate

Estimate: 5 points. Contribution: the packed Q6_K projection prerequisite for
the resident MTP draft leaf.

## Completion standard

development plus physical qualification.
