// mkernel_collective.cuh - structure-only phase contract for fused collective-compute operators.
//
// operators that fuse GEMM with all-reduce, all-gather and reduce-scatter, driven by a session
// that advances through declared phases. No upstream code is copied.
//
// SKELETON ONLY: constants. No multi-GPU run, no NVLink/RDMA use, no speed claim; included only
// under FAK_CUDA_MKERNEL_COLLECTIVE. The Go oracle in mkernel_collective_test.go reads these.
#ifndef FAK_MKERNEL_COLLECTIVE_CUH
#define FAK_MKERNEL_COLLECTIVE_CUH

#define FAK_MKC_OP_GEMM_AR 1
#define FAK_MKC_OP_AG_GEMM 2
#define FAK_MKC_OP_GEMM_RS 3

#define FAK_MKC_PHASE_GATHER 1
#define FAK_MKC_PHASE_COMPUTE 2
#define FAK_MKC_PHASE_REDUCE 3
#define FAK_MKC_PHASE_FINISH 4

#endif  // FAK_MKERNEL_COLLECTIVE_CUH
