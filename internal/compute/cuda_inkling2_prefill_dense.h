#ifndef FAK_CUDA_INKLING2_PREFILL_DENSE_H
#define FAK_CUDA_INKLING2_PREFILL_DENSE_H
#ifdef __cplusplus
extern "C" {
#endif

/* inkling2 prefill dense projections (cuda/inkling2_prefill_dense.cuh): Y[T][out] = X[T][in] * W^T.
 * fcuda_ink2p_dense_mmq runs the int8 tensor-core GEMM; kind 8 = Q8_0 (W int8 codes, S per-32 scales),
 * 4/5/6 = Q4_K/Q5_K/Q6_K raw super-blocks (S unused). cudaErrorInvalidValue (1) before any enqueue
 * means the shape or alignment is outside the kernel and the caller should use its legacy launcher. */
int fcuda_ink2p_dense_mmq(const void *W, const float *S, const float *X, float *Y, int out, int in, int T, int kind);
int fcuda_ink2p_gemm_f32_v2(const float *W, const float *X, float *Y, int rows, int cols, int T);

#ifdef __cplusplus
}
#endif
#endif
