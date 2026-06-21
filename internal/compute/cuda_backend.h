/* cuda_backend.h — the flat C ABI between the Go cgo wrapper (cuda.go) and the CUDA
 * kernels (cuda_kernels.cu). It is the typed hardware seam DIRECTION.md sanctions: the
 * Go side owns the Backend interface and re-checks every result; this header carries
 * only data (device pointers + shapes), never trust. All matmul/elementwise ops are
 * f32 — the first device backend is an Approx peer of the cpuref Reference, held to the
 * argmax-exact + logit-cosine gate, NOT to bit-identity. Quantized weight dtypes are
 * not yet implemented on device (the Go MatMul panics with a clear message; see the
 * deferred-work issues). */
#ifndef FAK_CUDA_BACKEND_H
#define FAK_CUDA_BACKEND_H
#include <stddef.h>  /* size_t */
#include <stdint.h>  /* uint8_t — needed by the AWQ kernels below; do NOT rely on a
                      * transitive include (WSL's nvcc pulls it in via <stddef.h>, but a
                      * datacenter CUDA toolchain — GCP DLVM, DGX — does not, and the
                      * header then fails to compile with "identifier uint8_t is
                      * undefined", which cascades into bogus "C linkage" overload errors
                      * on the matching definitions in cuda_kernels.cu). */
#ifdef __cplusplus
extern "C" {
#endif

/* fcuda_init selects device 0, creates the cuBLAS handle (idempotent), and reports the
 * device name (into name, up to namelen-1 chars), SM version (major*10+minor), and total
 * VRAM bytes. Returns 0 on success, non-zero if no CUDA device is reachable. */
int fcuda_init(char *name, int namelen, int *sm, size_t *total_mem);

/* device memory + transfers (the residency seam). */
void *fcuda_malloc(size_t bytes);
void fcuda_free(void *d);
void fcuda_h2d(void *d, const void *h, size_t bytes);
void fcuda_d2h(void *h, const void *d, size_t bytes);
void fcuda_d2d(void *dst, const void *src, size_t bytes);
void fcuda_sync(void);

/* y[P,out] = x[P,in] @ W[out,in]^T   (all row-major f32) via cuBLAS SGEMM. */
void fcuda_matmul_f32(const float *dW, const float *dX, float *dY, int out, int in, int P);

/* per-row RMSNorm: y[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2) + eps) * w[:]  (rows x n). */
void fcuda_rmsnorm_f32(const float *dX, const float *dW, float *dY, int rows, int n, float eps);

/* RoPE (HF non-interleaved rotate_half) on x[nHeads*headDim] at absolute position pos. */
void fcuda_rope_f32(float *dX, int pos, int nHeads, int headDim, double theta);

/* SwiGLU: y = silu(g) * u, elementwise, length n. */
void fcuda_swiglu_f32(const float *dG, const float *dU, float *dY, int n);

/* dst += src (length n); and dst[r,:] += bias[:] (rows x width). */
void fcuda_add_f32(float *dDst, const float *dSrc, int n);
void fcuda_add_bias_f32(float *dDst, const float *dBias, int rows, int width);

/* append n floats from src into dstBase at a SCALAR float offset (the KV-append, kernel
 * form so cudaGraphExecUpdate can patch the offset across a growing cache). */
void fcuda_kv_write(float *dstBase, const float *src, int offset, int n);

/* Decode attention: q[nH*hd] (one position), K/V [nPos, nKV*hd] row-major; causal by
 * construction (the cache holds exactly the attendable keys). grp = nH/nKV. out[nH*hd].
 * Allocates a scratch scores buffer internally (freed before return). */
void fcuda_attention_f32(const float *dQ, const float *dK, const float *dV, float *dOut,
                         int nPos, int maxPos, int nH, int nKV, int hd, float scale);

/* argmax over logits[n]: returns the SMALLEST index attaining the maximum value (the
 * cpuref first-max tie-break), copied back to the host as the single scalar fence. */
int fcuda_argmax_f32(const float *dLogits, int n);

/* AWQ (Activation-aware Weight Quantization) 4-bit kernels.
 * AWQ format: 4-bit weights packed 2 per byte (nibble-packed), per-channel scales.
 * Dequantization: weight = scale[o] * (code - 8), where 8 is the zero-point. */

/* fcuda_awq_gemv: y[out] = AWQ[out,in] @ x[in] where AWQ is 4-bit packed [out, in/2].
 * dScales[out] holds per-channel scales. */
void fcuda_awq_gemv(const uint8_t *dW, const float *dScales, const float *dX, float *dY, int out, int in);

/* fcuda_awq_gemm: Y[P,out] = X[P,in] @ AWQ[out,in]^T (batched AWQ matmul). */
void fcuda_awq_gemm(const uint8_t *dW, const float *dScales, const float *dX, float *dY, int out, int in, int P);

/* CUDA-graph decode: capture one token's whole op stream on g_stream, then replay it as a
 * single launch — collapsing ~600 WSL CUDA calls/token into one (the only way past the
 * proven ~12 tok/s op-per-call floor). begin starts capture; end_launch ends it,
 * instantiates, launches, fences, and frees the graph. Both return 0 on success. The
 * caller guarantees: no cudaMalloc happens during capture (the pool is pre-warmed and the
 * KV is fixed-capacity), and the host stays pinned to one OS thread across the token. */
int fcuda_graph_begin(void);
int fcuda_graph_end_launch(void);
/* fcuda_graph_reset drops the kept exec graph so the next session captures fresh (the exec
 * is tied to one session's buffer addresses). Called at session start. */
void fcuda_graph_reset(void);

#ifdef __cplusplus
}
#endif
#endif
