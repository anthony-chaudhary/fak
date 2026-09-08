<!-- fak-compute-key: vulkan-radv-kv-contiguization -->
# reject(compute): whole-cache RADV KV contiguization before attention

```routing
lane: compute
paths: ["internal/compute/vulkan_ops.go", "internal/compute/vulkan_shim.cpp", "internal/compute/shaders/attention.comp", "internal/compute/shaders/radv_contiguize_f32.comp"]
expected_steps: 6
```

GitHub: [#12053](https://github.com/anthony-chaudhary/fak/issues/12053)

## Current state

The modeled FP16 pass was never connected to Vulkan, whose real KV ABI is FP32.
A source-bound tracer implemented the actual FP32 path entirely on-device and
proved bit-exact attention output. Robust physical A/B then found net latency
regressions at 32K, 64K, and 128K, so the candidate is rejected and must not be
merged into the production path.

## Core through-line

Resident FP32 K/V -> vectorized in-device transpose -> unchanged attention over
head-contiguous indices -> exact output -> matched physical A/B -> reject when
the complete operation costs more than direct strided attention.

## Gold-plating boundary

- Do not merge a negative-performance mechanism for structural completeness.
- Do not convert the public KVStore ABI or keep a second per-layer cache without a memory receipt.
- Do not restart, stop, or replace the active Strix service.
- Do not promote host channel simulation into physical bandwidth evidence.

## Done condition

- The proposed shader is compiled and dispatched on the physical target.
- Baseline and candidate outputs are compared bit-for-bit.
- Interleaved steady-state samples cover 32K, 64K, and 128K.
- A negative result prevents default integration and corrects the stale benchmark claim.

## Verifiable Witness

```powershell
FAK_VULKAN_RADV_TEST_NPOS=32768 go test -v ./internal/compute -run TestVulkanAttention_RADVContiguization -count=1
FAK_VULKAN_RADV_TEST_NPOS=65536 go test -v ./internal/compute -run TestVulkanAttention_RADVContiguization -count=1
FAK_VULKAN_RADV_TEST_NPOS=131072 go test -v ./internal/compute -run TestVulkanAttention_RADVContiguization -count=1
```

## Likely files

- `internal/compute/shaders/radv_contiguize_f32.comp`
- `internal/compute/shaders/attention.comp`
- `internal/compute/vulkan_shim.cpp`
- `internal/compute/vulkan_ops.go`

## Lane

`compute` (public runtime)

## Scoped acceptance criteria

- [x] [SW-VERIFIED] The transpose index map is a total bijection and preserves GQA attention by element substitution.
- [x] [SW-VERIFIED] The live Vulkan call site compiled with checked shader-index and resource bounds.
- [x] [HW-WITNESSED] The 8060S executed 11 attributed dispatches per context depth with bit-exact output.
- [x] [HW-WITNESSED] Interleaved medians regressed by 4.48% at 32K, 2.01% at 64K, and 5.08% at 128K.
- [x] [DECISION] Reject whole-cache pre-attention contiguization; retain no production code from the candidate.

Physical receipt:
[`strix-vulkan-radv-contiguization-reject-20260908.json`](../../benchmarks/receipts/strix-vulkan-radv-contiguization-reject-20260908.json).
