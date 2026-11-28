// megakernel_nvfp4_decode.cuh - structure-only contract for Blackwell NVFP4 persistent decode.
//
// ADAPT of Luce-Org/lucebox-hub (Apache-2.0), optimizations/megakernel/kernel_gb10_nvfp4.cu
// @f3cc3ec9c56810c724981e8ae5114539e9500b8a: fused NVFP4 dequant GEMVs inside a single-launch
// decode pass targeting sm_120/sm_121a. No upstream code is copied.
//
// SKELETON ONLY: constants. No device run claim, no model loader change, no speed claim; included
// only under FAK_CUDA_MEGAKERNEL_NVFP4_DECODE. The Go oracle in megakernel_nvfp4_decode_test.go
// reads these #defines.
#ifndef FAK_MEGAKERNEL_NVFP4_DECODE_CUH
#define FAK_MEGAKERNEL_NVFP4_DECODE_CUH

#define FAK_NVFP4_DEC_GROUP_SIZE 16
#define FAK_NVFP4_DEC_PACKED_PER_BYTE 2
#define FAK_NVFP4_DEC_SM_MIN 120
#define FAK_NVFP4_DEC_SINGLE_LAUNCH 1

#endif  // FAK_MEGAKERNEL_NVFP4_DECODE_CUH
