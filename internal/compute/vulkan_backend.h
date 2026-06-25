/* vulkan_backend.h — the flat C ABI between the Go cgo wrapper (vulkan.go) and the
 * Vulkan compute shim (vulkan_shim.cpp). It mirrors cuda_backend.h function-for-function
 * so vulkan.go is a near-mechanical port of cuda.go: the Go side owns the Backend
 * interface and re-checks every result; this header carries only data (device buffer
 * handles + shapes), never trust.
 *
 * The device "pointer" here is an opaque handle (a void* the shim maps to a VkBuffer +
 * its bound VkDeviceMemory), NOT a host-dereferenceable address — same contract as the
 * CUDA device pointer. All ops are f32: this is an *Approx* peer of the cpuref
 * *Reference*, held to the argmax-exact + logit-cosine gate, NOT to bit-identity (Vulkan
 * GLSL fma/reduction order differs from the model's fdot tree, which is what makes the
 * Approx classification real and honest). Quantized weight dtypes are not implemented on
 * device yet; the Go MatMul refuses them with a clear message.
 *
 * Compiled ONLY under `-tags vulkan`; the default `go build ./cmd/fak` excludes it, so the
 * shipped artifact stays one pure-Go binary. */
#ifndef FAK_VULKAN_BACKEND_H
#define FAK_VULKAN_BACKEND_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* fvk_init creates the Vulkan instance, picks a physical device (prefers a discrete GPU,
 * falls back to the first available), creates the logical device + a compute queue, builds
 * one compute pipeline per kernel from the SPIR-V modules in spirv_dir, and reports the
 * device name (into name, up to namelen-1 chars). *is_discrete is set to 1 for a discrete
 * GPU, 0 otherwise (the Go side surfaces it in Tier so a CPU/software device — e.g. Mesa
 * lavapipe — can be told apart from the real GPU). Returns 0 on success, non-zero if no
 * Vulkan compute device is reachable or pipeline creation fails. */
int fvk_init(char *name, int namelen, int *is_discrete, const char *spirv_dir);

/* device memory + transfers (the residency seam). The returned handle is an opaque
 * VkBuffer wrapper; it is NOT a host pointer. */
void *fvk_malloc(size_t bytes);
/* fvk_malloc_hostvis allocates a storage buffer in HOST-VISIBLE memory directly, skipping the
 * device-local attempt fvk_malloc makes. The residency-budget path uses it to place cold
 * weights host-side DELIBERATELY (in a chosen order) instead of letting hot weights lose the
 * device-local allocation race and spill unpredictably. Same opaque handle + STORAGE_USAGE as
 * fvk_malloc; the compute path reads it the same way, just slower (over PCIe). */
void *fvk_malloc_hostvis(size_t bytes);
void fvk_free(void *d);
void fvk_h2d(void *d, const void *h, size_t bytes);
void fvk_d2h(void *h, const void *d, size_t bytes);
void fvk_d2d(void *dst, const void *src, size_t bytes);
/* fvk_d2d_off copies `bytes` from src[0..] into dst at byte offset dst_off — the
 * device-resident KV append (write a new K/V row at the tail of the layer buffer). */
void fvk_d2d_off(void *dst, size_t dst_off, const void *src, size_t bytes);
/* General device->device range copy. Both handles are opaque Buffer* values; offsets are
 * byte offsets into those buffers, never pointer arithmetic on the handles themselves. */
void fvk_d2d_range(void *dst, size_t dst_off, const void *src, size_t src_off, size_t bytes);
void fvk_sync(void);
/* Drop all currently recycled device buffers. Safe after a session has closed and flushed. */
void fvk_trim_pool(void);
/* Drop recycled buffers when the pool grows past max_buffers; intended for token boundaries. */
void fvk_trim_pool_if_over(size_t max_buffers);
/* Debug-only introspection for backend tests. Returns the VkMemoryPropertyFlags that the
 * opaque buffer was actually allocated with, or 0 for nil. */
uint32_t fvk_debug_buffer_props(const void *d);
int fvk_debug_buffer_is_host_visible(const void *d);
int fvk_debug_buffer_is_device_local(const void *d);

/* y[P,out] = x[P,in] @ W[out,in]^T   (all row-major f32). */
void fvk_matmul_f32(const void *dW, const void *dX, void *dY, int out, int in, int P);
/* first argmax of x[1,in] @ W[out,in]^T without materializing the logits vector. */
int fvk_matmul_argmax_f32(const void *dW, const void *dX, int out, int in);

/* fvk_have_q8 reports 1 if the device + driver support the 8-bit-storage / int8-arithmetic
 * features the Q8 GEMM shader needs (queried at init); 0 otherwise. The Go side falls back to
 * f32 weights when this is 0, so Q8 is an optional fast path, never a correctness dependency. */
int fvk_have_q8(void);
/* Per-resource storage-buffer cap discovered at init. fvk_max_buffer_bytes is the effective
 * single-buffer ceiling fak must respect: min(maxStorageBufferRange, maxMemoryAllocationSize)
 * when both are known, otherwise the known cap, or 0 when unknown. */
uint64_t fvk_max_buffer_bytes(void);
uint64_t fvk_max_storage_buffer_range(void);
uint64_t fvk_max_memory_allocation_size(void);

/* Q8_0 quantized GEMM: y[P,out] = dequant(Wq[out,in], scales[out,in/32]) applied to x[P,in].
 * Wq is int8 weight codes (out*in bytes), Wscale is per-block f32 scales (out*(in/32) floats),
 * both pre-quantized and uploaded ONCE. The activation x is f32 and quantized per-block on the
 * device. This is the 4× weight-memory cut over fvk_matmul_f32 and the Q8-vs-Q8 parity path. */
void fvk_q8_matmul_f32(const void *dWcodes, const void *dWscale, const void *dX, void *dY,
                       int out, int in, int P);
/* Two Q8_0 projections over the same f32 X in one dispatch. */
void fvk_q8_matmul2_f32(const void *dW0codes, const void *dW0scale,
                        const void *dW1codes, const void *dW1scale,
                        const void *dX, void *dY0, void *dY1,
                        int out0, int out1, int in, int P);
/* Three Q8_0 projections over the same f32 X in one dispatch. */
void fvk_q8_matmul3_f32(const void *dW0codes, const void *dW0scale,
                        const void *dW1codes, const void *dW1scale,
                        const void *dW2codes, const void *dW2scale,
                        const void *dX, void *dY0, void *dY1, void *dY2,
                        int out0, int out1, int out2, int in, int P);
/* RMSNorm(X) followed by two Q8_0 projections over the normalized row (FFN gate/up, Q8). */
void fvk_rmsnorm_q8_matmul2_f32(const void *dW0codes, const void *dW0scale,
                                const void *dW1codes, const void *dW1scale,
                                const void *dX, const void *dNorm, void *dY0, void *dY1,
                                int out0, int out1, int in, int P, float eps);
/* RMSNorm(X) followed by Q/K/V Q8_0 projections over the normalized row (attention in, Q8). */
void fvk_rmsnorm_q8_matmul3_f32(const void *dWqcodes, const void *dWqscale,
                                const void *dWkcodes, const void *dWkscale,
                                const void *dWvcodes, const void *dWvscale,
                                const void *dX, const void *dNorm, void *dQ, void *dK, void *dV,
                                int qOut, int kOut, int vOut, int in, int P, float eps);
/* dD[P,out] += (silu(g) * u) @ Wcodes[out,in]^T with a Q8_0 down projection (FFN tail, Q8). */
void fvk_swiglu_q8_matmul_add_f32(const void *dWcodes, const void *dWscale,
                                  const void *dG, const void *dU, void *dD,
                                  int out, int in, int P);
/* first argmax of RMSNorm(x) @ W[out,in]^T without materializing normalized x or logits. */
int fvk_rmsnorm_matmul_argmax_f32(const void *dW, const void *dX, const void *dNorm,
                                  int out, int in, float eps);
/* dY[P,out] += x[P,in] @ W[out,in]^T; used to fuse projection + residual add. */
void fvk_matmul_add_f32(const void *dW, const void *dX, void *dY, int out, int in, int P);
/* Two projections over the same X in one dispatch. */
void fvk_matmul2_f32(const void *dW0, const void *dW1, const void *dX,
                     void *dY0, void *dY1, int out0, int out1, int in, int P);
/* Q/K/V projections over the same X in one dispatch. */
void fvk_matmul3_f32(const void *dWq, const void *dWk, const void *dWv, const void *dX,
                     void *dQ, void *dK, void *dV, int qOut, int kOut, int vOut, int in, int P);
/* per-row RMSNorm: y[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2) + eps) * w[:]  (rows x n). */
void fvk_rmsnorm_f32(const void *dX, const void *dW, void *dY, int rows, int n, float eps);
/* RMSNorm(X) followed by one projection over the normalized row. */
void fvk_rmsnorm_matmul_f32(const void *dW, const void *dX, const void *dNorm,
                            void *dY, int out, int in, int P, float eps);
/* RMSNorm(X) followed by two projections over the normalized row. */
void fvk_rmsnorm_matmul2_f32(const void *dW0, const void *dW1, const void *dX, const void *dNorm,
                             void *dY0, void *dY1, int out0, int out1, int in, int P, float eps);
/* RMSNorm(X) followed by Q/K/V projections over the normalized row. */
void fvk_rmsnorm_matmul3_f32(const void *dWq, const void *dWk, const void *dWv,
                             const void *dX, const void *dNorm, void *dQ, void *dK, void *dV,
                             int qOut, int kOut, int vOut, int in, int P, float eps);

/* RoPE (HF non-interleaved rotate_half) on x[nHeads*headDim] at absolute position pos. */
void fvk_rope_f32(void *dX, int pos, int nHeads, int headDim, double theta);

/* SwiGLU: y = silu(g) * u, elementwise, length n. */
void fvk_swiglu_f32(const void *dG, const void *dU, void *dY, int n);
/* dY[P,out] += (silu(g[P,in]) * u[P,in]) @ W[out,in]^T. */
void fvk_swiglu_matmul_add_f32(const void *dW, const void *dG, const void *dU,
                               void *dY, int out, int in, int P);

/* dst += src (length n); and dst[r,:] += bias[:] (rows x width). */
void fvk_add_f32(void *dDst, const void *dSrc, int n);
void fvk_add_bias_f32(void *dDst, const void *dBias, int rows, int width);

/* Decode attention: q[nH*hd] (one position), K/V [nPos, nKV*hd] row-major; causal by
 * construction (the cache holds exactly the attendable keys). grp = nH/nKV. out[nH*hd]. */
void fvk_attention_f32(const void *dQ, const void *dK, const void *dV, void *dOut,
                       int nPos, int nH, int nKV, int hd, float scale);

/* argmax over logits[n]: returns the SMALLEST index attaining the maximum value (the
 * cpuref first-max tie-break), copied back to the host as the single scalar fence. */
int fvk_argmax_f32(const void *dLogits, int n);

/* ---- batched submission (the throughput lever) --------------------------------
 * Without batching, every primitive op submits its own command buffer and waits on a
 * fence — ~300 GPU round-trips per token, which dominates wall-clock. Between
 * fvk_batch_begin() and the next host fence, the compute ops (matmul/rmsnorm/rope/
 * swiglu/add/add_bias/attention) instead RECORD into one shared command buffer with a
 * compute->compute barrier between them, and are submitted ONCE when the batch flushes.
 * A host-fence op (fvk_d2h / fvk_argmax_f32) auto-flushes the pending batch first, so a
 * caller just brackets a token's forward pass with begin()/…/Read and pays one submit.
 * fvk_batch_flush() forces a submit+wait (idempotent if nothing is pending). Calling
 * begin() when already batching is a no-op; the model loop brackets each token. */
void fvk_batch_begin(void);
void fvk_batch_flush(void);

#ifdef __cplusplus
}
#endif
#endif
