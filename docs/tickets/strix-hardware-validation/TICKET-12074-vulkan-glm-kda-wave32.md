<!-- fak-compute-key: vulkan-glm-kda-wave32 -->
# feat(compute): physically qualify a gfx1151 Wave32 GLM KDA recurrent step

```routing
lane: compute
paths: ["internal/compute/shaders/*glm_kda*", "internal/compute/vulkan_*kda*", "internal/model/vulkan_glm_kda_physical_test.go"]
expected_steps: 6
```

GitHub: [#12074](https://github.com/anthony-chaudhary/fak/issues/12074)

## Current state

The predecessor supplied only an analytical Go register contract. The public
runtime had no compiled GLM KDA shader, no Vulkan dispatch seam, no enforced
Wave32 pipeline, and no physical comparison against the production recurrence.

This leaf adds two fixed-D physical kernels. Both map one head to one 128-thread
workgroup and one invocation to one state column. The matched parent writes the
decayed column to device memory and rereads it; the candidate retains the column
privately until its sole final write. Pipeline creation explicitly requests a
32-lane subgroup. Unsupported devices refuse the operation rather than falling
back.

## Core through-line

Resident `[64,128,128]` state plus resident Q/K/V/alpha/beta -> strict geometry,
ownership, and no-alias preflight -> explicit Wave32 Vulkan pipeline -> one
rank-one recurrent update per head -> resident state/output -> durable physical
receipt against `model.StepGLM5NextKDAHead`.

## Gold-plating boundary

- No full GLM checkpoint or end-to-end model wiring in this leaf.
- No second analytical occupancy model.
- No Qwen GDN changes or unrelated shader bundle.
- No inference from local size to wave size; RADV must report subgroup size 32.
- No simulated result may stand in for physical latency or compiler resources.

## Done condition

- Both shader variants compile to SPIR-V and the tagged model test binary links.
- Fixed production geometry matches the production CPU recurrence.
- Timed dispatches perform no H2D/D2H transfer and cannot silently fall back.
- RADV reports Wave32 plus real register, spill, code, scratch, and occupancy data.
- The candidate is retained only when repeated matched A/B exceeds the 1% keep gate.

## Verifiable Witness

```powershell
go test ./internal/compute -run 'TestGLMKDA' -count=1
go test ./internal/computebuild -run 'TestVulkanShadersCompleteness' -count=1
go test ./internal/model -run 'TestGLM5NextKDA' -count=1
go vet ./internal/compute ./internal/model ./internal/computebuild
```

Physical source-bound witness:

```sh
exec 9</tmp/fak-gpu.lease
flock -n 9
sleep 4
RADV_DEBUG=shaders \
FAK_VULKAN_SPIRV="$PWD/build/spirv" \
FAK_VULKAN_REQUIRE_DEVICE=1 \
FAK_VULKAN_EXPECT_DEVICE=8060S \
FAK_VULKAN_GLM_KDA_PHYSICAL=1 \
FAK_VULKAN_DISPATCH_PROFILE=1 \
./build/model.test -test.run '^TestVulkanGLMKDAWave32PhysicalAB$' -test.v
```

## Likely files

- `internal/compute/glm_kda.go`
- `internal/compute/vulkan_glm_kda.go`
- `internal/compute/vulkan_shim.cpp`
- `internal/compute/shaders/glm_kda_recurrent_reread.comp`
- `internal/compute/shaders/glm_kda_recurrent_wave32.comp`
- `internal/model/vulkan_glm_kda_physical_test.go`

## Lane

`compute`, with a production-math witness in `internal/model`

## Scoped acceptance criteria

- [x] [SW-VERIFIED] Exhaustive production-geometry ownership proves every state/output element has exactly one writer.
- [x] [SW-VERIFIED] Strict F32 row-major geometry, backend ownership, live allocation, variant, and pairwise no-alias checks fail before dispatch.
- [x] [SW-VERIFIED] Both shaders compile to SPIR-V; the candidate alone contains function-private `[128]float` retention.
- [x] [HW-WITNESSED] RADV reports required subgroup size 32 for both 128-thread pipelines.
- [x] [HW-WITNESSED] 62 attributed dispatches execute with zero timed H2D/D2H and no fallback.
- [x] [HW-WITNESSED] Parent and candidate state/output match `model.StepGLM5NextKDAHead` at cosine above 0.999999 and max error below 2e-4.
- [x] [HW-WITNESSED] Candidate median is 2.30x and 2.40x faster in two qualifying runs, clearing the 1% keep gate.
- [x] [HW-WITNESSED] RADV compiler stats report 256 VGPRs, zero VGPR spills, zero scratch, five subgroups/SIMD, and 258 VMEM instructions for the retained candidate.

Physical receipt:
[`strix-vulkan-glm-kda-wave32-20260908.json`](../../benchmarks/receipts/strix-vulkan-glm-kda-wave32-20260908.json).

## Non-blocking unrelated suite observations

The broad untagged package sweep also reported the existing simulated doorbell
sub-100us timing gate at 173.555us and three one-ULP Qwen chunked/scalar equality
failures. Those files and execution paths are outside this GLM KDA leaf; focused
KDA tests, vet, the five-gate boundary audit, and the physical Vulkan witness are
green.
