// alpha_moe_fused.cuh - structure-only contract for W8A8 MoE up+down+accumulate fusion.
//
// ADAPT of Aleph-Alpha/Alpha-MoE (Apache-2.0),
// csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_up_down_acc.cu and extra/pseudocode.txt
// @0fbed62bdda3702fd5d4b8c497e68746f0304c3a: one kernel computes both FFN projections (up and
// down) with fused activation and local accumulate so intermediate activations stay in
// registers/shared memory without a global-memory round-trip. No upstream code is copied.
//
// SKELETON ONLY: constants. No tuning on unleased hardware, no speed claim; included only under
// FAK_CUDA_ALPHA_MOE_FUSED. The Go oracle in alpha_moe_fused_test.go reads these #defines.
#ifndef FAK_ALPHA_MOE_FUSED_CUH
#define FAK_ALPHA_MOE_FUSED_CUH

#define FAK_ALPHA_MOE_STAGE_UP 1
#define FAK_ALPHA_MOE_STAGE_ACT 2
#define FAK_ALPHA_MOE_STAGE_DOWN 3
#define FAK_ALPHA_MOE_STAGE_ACC 4
#define FAK_ALPHA_MOE_INTERMEDIATE_IN_SMEM 1

#endif  // FAK_ALPHA_MOE_FUSED_CUH
