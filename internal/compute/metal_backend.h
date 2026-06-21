/* metal_backend.h — the flat C ABI between the Go cgo wrapper (metal.go) and the Metal /
 * MetalPerformanceShaders shim (metal_shim.m). It mirrors cuda_backend.h function-for-
 * function so metal.go is a near-mechanical port of cuda.go: the Go side owns the Backend
 * interface and re-checks every result; this header carries only data (device buffer
 * handles + shapes), never trust.
 *
 * The device "pointer" here is an opaque handle — a void* the shim maps to an
 * id<MTLBuffer> (allocated with MTLResourceStorageModeShared on Apple Silicon's unified
 * memory). It is NOT a host-dereferenceable address from Go: all host<->device transfers
 * go through fmetal_h2d / fmetal_d2h, and device<->device copies (the KV append/grow) go
 * through fmetal_copy_at. Same Caps{DeviceMemory:true} contract as the CUDA/Vulkan
 * backends — a device tensor is never reinterpreted as a host slice.
 *
 * Every op is f32: this is an *Approx* peer of the cpuref *Reference* — held to the
 * argmax-exact + logit-cosine gate, NOT to bit-identity. MPSMatrixMultiplication (a
 * different reduction order than the model's fdot tree) is what makes that distinction
 * real and honest. Quantized weight dtypes are not implemented on device yet; the Go
 * MatMul refuses them with a clear message (see the deferred-work issues / #300 scope). */
#ifndef FAK_METAL_BACKEND_H
#define FAK_METAL_BACKEND_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* fmetal_init creates the system default Metal device + a command queue, compiles the
 * embedded MSL compute library into pipeline states (one per elementwise/reduction op),
 * and reports the device name (into name, up to namelen-1 chars, e.g. "Apple M3 Pro").
 * Returns 0 on success, non-zero if no Metal device is reachable or a pipeline fails to
 * build (leaving cpu-ref as the only registered backend). */
int fmetal_init(char *name, int namelen);

/* device memory (the residency seam). fmetal_malloc returns a shared-storage MTLBuffer
 * handle of at least `bytes` (rounded up off 0); fmetal_free releases it. */
void *fmetal_malloc(size_t bytes);
void fmetal_free(void *buf);

/* host<->device + device<->device transfers over unified shared memory (plain memcpy over
 * [buffer contents]; every op commits+waits synchronously, so contents are coherent). */
void fmetal_h2d(void *dstBuf, const void *host, size_t bytes);
void fmetal_d2h(void *host, void *srcBuf, size_t bytes);
/* copy `bytes` from srcBuf[srcOff..] into dstBuf[dstOff..] (byte offsets) — the KV
 * append/grow primitive (the buffer-handle analogue of cudaMemcpyDeviceToDevice). */
void fmetal_copy_at(void *dstBuf, size_t dstOff, void *srcBuf, size_t srcOff, size_t bytes);

/* y[P,out] = x[P,in] @ W[out,in]^T  (all row-major f32) via MPSMatrixMultiplication
 * (transposeRight=YES, interiorColumns=in). */
void fmetal_matmul_f32(void *dW, void *dX, void *dY, int out, int in, int P);

/* per-row RMSNorm: y[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2) + eps) * w[:]  (rows x n). */
void fmetal_rmsnorm_f32(void *dX, void *dW, void *dY, int rows, int n, float eps);

/* RoPE (HF non-interleaved rotate_half) on x[nHeads*headDim] at absolute position pos. */
void fmetal_rope_f32(void *dX, int pos, int nHeads, int headDim, float theta);

/* SwiGLU: y = silu(g) * u, elementwise, length n. */
void fmetal_swiglu_f32(void *dG, void *dU, void *dY, int n);

/* dst += src (length n); and dst[r,:] += bias[:] (rows x width). */
void fmetal_add_f32(void *dDst, void *dSrc, int n);
void fmetal_add_bias_f32(void *dDst, void *dBias, int rows, int width);

/* Decode attention: q[nH*hd] (one position), K/V [nPos, nKV*hd] row-major; causal by
 * construction (the cache holds exactly the attendable keys). grp = nH/nKV. out[nH*hd].
 * One thread per query head, flash-style online softmax (no scores scratch buffer). */
void fmetal_attention_f32(void *dQ, void *dK, void *dV, void *dOut,
                          int nPos, int nH, int nKV, int hd, float scale);

/* argmax over logits[n]: returns the SMALLEST index attaining the maximum value (the
 * cpuref first-max tie-break), copied back to the host as the single scalar fence. */
int fmetal_argmax_f32(void *dLogits, int n);

#ifdef __cplusplus
}
#endif
#endif
