// pit_sparse_prefill.cuh - structure-only contract for PIT mixed-sparse prefill attention.
//
// columns, fp16/bf16, 64x64 blocks, head_dim in {16,32,64,128} (else pad). No upstream code
// is copied.
//
// SKELETON ONLY: constants. No hardware run, no speed claim. The Go oracle in
// pit_sparse_prefill_test.go reads these #defines.
#ifndef FAK_PIT_SPARSE_PREFILL_CUH
#define FAK_PIT_SPARSE_PREFILL_CUH

#define FAK_PIT_BLOCK_M 64
#define FAK_PIT_BLOCK_N 64
#define FAK_PIT_HEAD_DIM_MIN 16
#define FAK_PIT_HEAD_DIM_MAX 128

#endif  // FAK_PIT_SPARSE_PREFILL_CUH
