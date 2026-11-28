// megakernel_sm120_decode.cuh - structure-only description of a persistent sm_120a decode pass.
//
// producer warp, flag-in-data handoff in place of grid barriers, and int8 weight rows with fused
// dequant. No upstream code is copied. Measured upstream on sm_120a; nothing here is tuned or
// measured on that card class.
//
// SKELETON ONLY: constants. No kernel, no loader change, no speed claim; included only under
// FAK_CUDA_MEGAKERNEL_SM120_DECODE. The Go oracle in megakernel_sm120_decode_test.go reads these.
#ifndef FAK_MEGAKERNEL_SM120_DECODE_CUH
#define FAK_MEGAKERNEL_SM120_DECODE_CUH

#define FAK_MK120_CTAS_PER_SM 1
#define FAK_MK120_RING_SLOTS 4
#define FAK_MK120_ROW_WIDTH 8  // int8 elements per weight row in the oracle fixture

#endif  // FAK_MEGAKERNEL_SM120_DECODE_CUH
