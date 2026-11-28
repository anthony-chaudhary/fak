// fp8_gemm_allreduce.cuh - structure-only contract for SM90 fused FP8 rowwise GEMM+all-reduce.
//
// 5-stage SMEM pipeline, m64n128k32 e4m3 MMA shape, and P2P partial publish via atomic flag words.
// No upstream code is copied.
//
// SKELETON ONLY: constants. No P2P/NVLink hardware use, no speed claim; included only under
// FAK_CUDA_FP8_GEMM_ALLREDUCE. The Go oracle in fp8_gemm_allreduce_test.go reads these #defines.
#ifndef FAK_FP8_GEMM_ALLREDUCE_CUH
#define FAK_FP8_GEMM_ALLREDUCE_CUH

#define FAK_FI_FP8_AR_CTA_THREADS 512
#define FAK_FI_FP8_AR_PIPELINE_STAGES 5
#define FAK_FI_FP8_AR_MMA_M 64
#define FAK_FI_FP8_AR_MMA_N 128
#define FAK_FI_FP8_AR_MMA_K 32
#define FAK_FI_FP8_AR_FLAG_EMPTY 0
#define FAK_FI_FP8_AR_FLAG_READY 1

#endif  // FAK_FP8_GEMM_ALLREDUCE_CUH
