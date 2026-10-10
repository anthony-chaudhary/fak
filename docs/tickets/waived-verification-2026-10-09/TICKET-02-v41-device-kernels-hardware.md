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

## Done condition

- [ ] Every covered kernel has a silicon parity result: a pass, or a filed issue.
