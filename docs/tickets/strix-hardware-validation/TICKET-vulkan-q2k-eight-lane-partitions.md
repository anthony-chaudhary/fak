# perf(compute): test Q2_K row and token partitioning on gfx1151

<!-- fak-compute-key: vulkan-q2k-eight-lane-row-partitions -->

```routing
lane: internal/compute
paths: ["internal/compute/shaders/q2k_matmul.comp", "internal/compute/vulkan_shim.cpp", "internal/compute/vulkan_q2k_test.go"]
expected_steps: 4
```

## Current state

The retained Vulkan Q2_K shader assigns one lane to a complete output row. A
one-Wave32-per-row candidate was numerically correct but physically measured
2.01 times slower for decode and 2.97 times slower for four-token prefill, then
reverted. Issue #12001 requires a different occupancy hypothesis and matched
physical evidence.

## Core through-line

The first candidate assigned eight adjacent lanes to each row and reused decoded
weights across four tokens. It passed physical parity but regressed decode by
1.926 times and P4 prefill by 2.442 times against the exact retained parent, so
it was rejected. The next isolated hypothesis retains the parent's measured
one-invocation-per-token-row arithmetic but maps rows and tokens through a native
two-dimensional dispatch. This removes per-invocation integer divide/modulo. A
separate fused-expression candidate proved neutral for decode and 1.8 percent
slower for P4, isolating dispatch indexing as the next hypothesis.

## Gold-plating boundary

Do not repeat the rejected full-Wave32 row mapping, alter Q2_K storage, change
accuracy floors, or claim a speedup from simulation. KEEP only after matched
physical A/B evidence beats both retained parent bars and a raw model receipt
improves without fallback.

## Done condition / witness

- [x] [SW-VERIFIED] The shader and Vulkan shim compile without an overlay.
- [x] [HW-WITNESSED] Token panels 1, 3, 4, and 7 meet cosine and per-token argmax parity.
- [x] [HW-WITNESSED] Matched physical A/B beats 223733 ns/op decode and 299272 ns/op P4 prefill.
- [ ] A source-bound full-model raw receipt improves with zero backend fallback.

## Verifiable Witness

```bash
FAK_VULKAN_SPIRV=<spirv-dir> ./compute.test \
  -test.run '^TestVulkanQ2KMatMulMatchesCPUReference$' -test.v
FAK_VULKAN_SPIRV=<spirv-dir> ./compute.test \
  -test.run '^$' -test.bench '^BenchmarkVulkanQ2KMatMul$' -test.benchmem -test.count 4
```

## Rejected candidate receipts

The physical A/B/B/A receipt for the eight-lane candidate is
`docs/benchmarks/receipts/strix-q2k-gpu-20260908-eightlane-abba.json`. Its ten
samples per arm measured 79,028.3 vs 152,190.4 ns/op for decode and 125,118.0
vs 305,511.5 ns/op for P4 prefill (parent vs candidate). The exclusive lease,
idle admission state, exact source blobs, compiled SPIR-V hashes, binary hashes,
and parity panels are bound in that receipt. The failed primitive gate correctly
prevented a full-model run.

The four-token panel-only candidate is preserved in
`docs/benchmarks/receipts/strix-q2k-gpu-20260908-panel-abba.json`. It improved
decode from 79,225.1 to 66,604.1 ns/op but regressed P4 prefill from 127,390.3
to 1,378,949.1 ns/op. This isolates shorter-lived fused arithmetic as the next
hypothesis while rejecting multiple live token accumulators.

The fused-expression/register-lifetime candidate is preserved in
`docs/benchmarks/receipts/strix-q2k-gpu-20260908-register-abba.json`. Decode was
neutral within noise (80,125.1 vs 79,599.4 ns/op) and P4 regressed 1.8 percent
(126,397.7 vs 128,699.9 ns/op), parent vs candidate. It was rejected before a
full-model run.

The qualifying primitive candidate is bound in
`docs/benchmarks/receipts/strix-q2k-gpu-20260908-2d-abba.json`. Native 2-D
dispatch improved decode by 15.114 percent (79,690.5 to 67,646.1 ns/op) and P4
prefill by 10.240 percent (127,356.4 to 114,315.2 ns/op), with two allocations
per operation in both arms. The candidate remains unlanded until the independent
full-model gate can run without replacing the active service.

## File:Line seams

- `internal/compute/shaders/q2k_matmul.comp` (`main`)
- `internal/compute/vulkan_shim.cpp` (`fvk_q2k_matmul_f32`)
- `internal/compute/vulkan_q2k_test.go` (`TestVulkanQ2KMatMulMatchesCPUReference`)

## Blast radius and affected lanes

- Affected: Vulkan Q2_K GEMV/GEMM dispatch on gfx1151.
- Unaffected: CPU, CUDA, Q4_K, Q8, and non-Vulkan builds.
- Fallback: candidate is reverted if either primitive bar or model receipt fails.

## Likely files

- `internal/compute/shaders/q2k_matmul.comp`
- `internal/compute/vulkan_shim.cpp`
- `internal/compute/vulkan_q2k_test.go`

## Expected steps

4

## Lane

internal/compute

## Work estimate

Estimate: 3 points.

## Overall completion contribution

Contribution: 3/3 points.

## Completion standard

production

## Target operating envelope

acceptance pass rate: = 100 percent

## Witnessed operating envelope

acceptance pass rate: = 0 percent pending physical witness
