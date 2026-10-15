# TICKET-02: V4.1 device kernels have no hardware run after waived landing

## Current state

These V4.1 device-dispatch kernels landed with no Vulkan-tagged build, opt-in witness run, or silicon qualification: 0d0dcda7e63 query latent RMSNorm; fa265cd4174 KV latent RMSNorm; e8340033f8d FFN input RMSNorm; 0bae85dd15b final RMSNorm dispatch; 6a54495ecd4 transposed mHC projection; d6e79ea3d3d shared expert activation; 31de0e44223, 4cd54773544 tail RoPE; c0a18c1c522, fb6724d7efb shared latent sink attention; 10237ca5dd6 indexer score; 0b7df47254a opt-in attention witnesses.

## Working spine

1. Build with `-tags vulkan` for the Strix Halo target.
2. Run each opt-in witness (`internal/compute/vulkan_v41_*_test.go`, `internal/model/*_halo_test.go`) on silicon.
3. Record per-kernel parity against the CPU reference.
4. File one narrow issue for each kernel that fails parity.

## Dependency

The CPU reds in TICKET-01 should be green first. If they are not, report results against the exact commit.

## Witness

An opt-in witness log per kernel from a real Strix Halo node. Label a result HW-witnessed only when it comes from a silicon run.

## Per-kernel silicon parity results (2026-10-15, strix3)

[HW-WITNESSED] One Strix Halo appliance, `strix3`, device
`integrated:AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, driver
`vendor=0x1002 device=0x1586`, runtime `vulkan-1.4.354`. Prebuilt
`-tags vulkan` witness archives were reused (no hot-path compile): compute
witness source `dff9be67b` (`/home/fak/v41-witness/out-dff9be67b/compute.test`,
`compute_idx.test`), model witness source `9ffd0f361e`
(`/home/fak/v41-witness/bf16-9ffd-model.test`). Every covered commit below is an
ancestor of those trees.

| Covered commit | Kernel | Witness | Result |
| --- | --- | --- | --- |
| c0a18c1c522, fb6724d7efb | shared latent sink attention | `TestVulkanV41SharedAttentionPhysical` (compute) | PASS — 25 ops, parity max_abs=0 |
| c0a18c1c522, fb6724d7efb | shared attention model callbacks | `TestV41SharedAttentionVulkan` (model) | PASS — 16 nonempty dispatches, contraction max_abs ≤ 3.3e-7 |
| 10237ca5dd6 | indexer score | `TestVulkanV41IndexerScorePhysical` (compute) | PASS — 15 dispatches, bit-exact vs oracle |
| 31de0e44223, 4cd54773544 | tail RoPE | `TestVulkanV41TailRoPEQK` (compute) | PASS |
| 0b7df47254a | opt-in attention witnesses | source of the two witnesses above | satisfied |
| 0d0dcda7e63, fa265cd4174, e8340033f8d, 0bae85dd15b | latent/FFN/final RMSNorm | TICKET-02c (`TestV41NormRoPECallbacksHalo`, fak#13768 CLOSED) | PASS 5/5 on Halo |
| 6a54495ecd4 | transposed mHC projection | TICKET-02a/02d; F32 F32-parity drift filed as fak#13808 (OPEN) | filed issue |
| d6e79ea3d3d | shared expert activation | TICKET-02b (`TestV41ClampedDeviceSwiGLUHalo`, fak#13769 CLOSED) | PASS |

Reproduction (compute side; run from `internal/compute` of the witness tree):

```bash
FAK_V41_SHARED_ATTENTION_WITNESS=1 FAK_VULKAN_REQUIRE_DEVICE=1 \
  FAK_VULKAN_DISPATCH_PROFILE=1 FAK_VULKAN_TIMESTAMP_PROFILE=1 \
  FAK_VULKAN_TIMESTAMP_PROFILE_EVERY=1 \
  FAK_VULKAN_EXPECT_DEVICE="AMD Radeon 8060S Graphics (RADV STRIX_HALO)" \
  FAK_VULKAN_SPIRV=<spirv dir> \
  ./compute.test -test.run '^TestVulkanV41SharedAttentionPhysical$' -test.count=1 -test.v
```

A stale pre-`03c9c111e` model binary fails `TestV41SharedAttentionVulkan/plain-window`
("a later operation changed previously owned output"); `03c9c111e` fixed that test-only
hand-back, and the witness passes on the fixed trunk tree. The compute-side passes are
unaffected.

## Done condition

- [x] Every covered kernel has a silicon parity result: a pass, or a filed issue.
