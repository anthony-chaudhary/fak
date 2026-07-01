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
/* Current CUDA device memory from cudaMemGetInfo. Returns 0 on success and fills free/total;
 * returns non-zero when the CUDA runtime cannot provide a fresh snapshot. */
int fcuda_mem_info(size_t *free_mem, size_t *total_mem);

/* device memory + transfers (the residency seam). */
void *fcuda_malloc(size_t bytes);
void *fcuda_malloc_managed(size_t bytes);
void fcuda_free(void *d);
void fcuda_trim_pool_large(size_t max_keep_bytes);
void fcuda_h2d(void *d, const void *h, size_t bytes);
void fcuda_d2h(void *h, const void *d, size_t bytes);
void fcuda_d2d(void *dst, const void *src, size_t bytes);
void fcuda_sync(void);

/* Device-specific allocation/copy helpers for multi-GPU collectives. The ordinary
 * fcuda_malloc/fcuda_h2d/fcuda_d2h path is device-0 + pooled. These helpers deliberately use
 * direct cudaMalloc/cudaFree on the requested device so rank-r collective buffers cannot be
 * recycled into the device-0 pool. */
int fcuda_set_device(int device);
void *fcuda_malloc_on(int device, size_t bytes);
void fcuda_free_on(int device, void *d);
void fcuda_h2d_on(int device, void *d, const void *h, size_t bytes);
void fcuda_d2h_on(int device, void *h, const void *d, size_t bytes);
void fcuda_d2d_on(int device, void *dst, const void *src, size_t bytes);

/* async host-transfer witness (#482): cumulative bytes copied device->host since the last
 * reset. The two host fences are the only d2h transfers and both add to it — fcuda_d2h (a
 * full Read) adds the vector bytes, fcuda_argmax_f32 adds only sizeof(int) — so an Argmax-only
 * decode step reads sizeof(int) here while a full-logits Read reads vocab*4. That is the
 * witness that greedy decode pulls only the token id across the bus, never the logits vector. */
size_t fcuda_hostxfer_bytes(void);
void fcuda_hostxfer_reset(void);

/* y[P,out] = x[P,in] @ W[out,in]^T   (all row-major f32) via cuBLAS SGEMM. */
void fcuda_matmul_f32(const float *dW, const float *dX, float *dY, int out, int in, int P);

/* fcuda_set_tf32 toggles the cuBLAS math mode for the f32 SGEMM above (Lever 4 of the
 * H100-KERNEL-5X-ROADMAP). on!=0 routes the f32 prefill GEMMs through Hopper/Ampere TENSOR
 * CORES at TF32 input precision with F32 accumulation (a large compute-bound prefill win at a
 * small, disclosed mantissa-only precision cost); on==0 (default) keeps the pedantic FP32-core
 * path the recorded cosine floors were witnessed against. The Go side reads FAK_CUDA_TF32 at
 * init and applies it; a host can flip it post-init via compute.EnableCUDATF32(). Idempotent
 * and safe before fcuda_init (no handle => no-op). The F16 HGEMM path is unaffected. */
void fcuda_set_tf32(int on);

/* fp16 compute path (#484, Caps.UploadDtype + tensor-core HGEMM). __half pointers are
 * passed as void* so THIS header stays free of <cuda_fp16.h> — the cgo type-check (`go vet
 * -tags cuda`) parses it with a plain host compiler and no CUDA toolkit (the #479/#482/#483
 * bar); the .cu casts back to __half. */

/* fcuda_f32_to_f16 narrows a staged f32 device buffer to F16 element-for-element (row-major:
 * dDstHalf[i] = (half)dSrc[i]). The H2D copy lands f32; this is the device-side dtype narrow
 * that makes the resident weight F16 — the Upload(t, F16) narrowing under Caps.UploadDtype. */
void fcuda_f32_to_f16(void *dDstHalf, const float *dSrc, int n);

/* fcuda_f32_to_f16_T narrows AND transpose-repacks a row-major f32 weight [out,in] into a
 * COLUMN-MAJOR F16 weight [out,in] (element (o,i) at o + i*out): dDstHalf[o + i*out] =
 * (half)dSrc[o*in + i]. This is the `Layout` repack at H2D — a ColMajor weight is laid out
 * once at upload so the HGEMM consumes it with op_N (no per-call transpose). */
void fcuda_f32_to_f16_T(void *dDstHalf, const float *dSrc, int out, int in);

/* fcuda_matmul_f16: Y[P,out] = X[P,in] @ W[out,in]^T via cuBLAS tensor-core HGEMM
 * (cublasGemmEx, CUDA_R_16F inputs, CUBLAS_COMPUTE_32F accumulate, CUBLAS_GEMM_DEFAULT_TENSOR_OP).
 * W is resident as __half (uploaded under Caps.UploadDtype); X is f32 and is converted to
 * __half in an internal scratch; Y is f32 (the F32 accumulate keeps the rest of the op chain —
 * RMSNorm/RoPE/SwiGLU/Attention — f32 and unchanged). colMajor==0 => W is row-major [out,in]
 * (op_T, lda=in, the SGEMM recipe); colMajor!=0 => W was transpose-repacked to col-major
 * [out,in] at H2D by fcuda_f32_to_f16_T (op_N, lda=out). Both compute the same Y. */
void fcuda_matmul_f16(const void *dWhalf, const float *dX, float *dY, int out, int in, int P, int colMajor);

/* native quantized device GEMM (#485): the weight stays narrow in VRAM (int8 codes / Q4_K
 * super-block bytes) and the GEMM consumes it directly — no dequant-to-f32 round trip, so the
 * VRAM/bandwidth win the quantized format buys is kept. Both are Approx peers of the cpuref
 * Reference (per-dtype recorded cosine floors in cuda.go: cudaQ8CosineMin tighter than
 * cudaQ4KCosineMin), NOT bit-identity. The activation arrives f32-resident; the kernels quantize
 * (Q8_0) or dequant-fuse (Q4_K) on device, accumulate in F32, and write f32 Y so RMSNorm/RoPE/
 * SwiGLU/Attention stay f32 and unchanged. */

/* fcuda_q8_matmul_f32: Y[P,out] = X[P,in] @ W[out,in]^T where W is resident Q8_0 — int8 codes
 * dCodes[out*in] plus per-block(=block) f32 scales dScales[out*(in/block)] (the side-channel a
 * real Q8 weight carries). X (f32) is quantized to int8 ON DEVICE per block (d=amax/127, the
 * cpuref q8round), then each block's integer dot is scaled by (weight block scale * activation
 * block scale) — the same per-block scheme as cpuref qdot8scalar, so the dynamic range of every
 * group is carried in full f32 and only the in-block code rounds. in must be divisible by block. */
void fcuda_q8_matmul_f32(const int8_t *dCodes, const float *dScales, const float *dX, float *dY,
                         int out, int in, int P, int block);

/* fcuda_q4k_matmul_f32: Y[P,out] = X[P,in] @ W[out,in]^T where W is resident Q4_K — the raw
 * llama.cpp k-quant super-block bytes dQ4K, 256 elements per 144-byte super-block (f16 d, f16
 * dmin, 12 bytes of 6-bit-packed sub-block scales+mins, 128 bytes of 4-bit codes). The kernel
 * DEQUANTS each weight fused into the GEMM tile — w = d*scale*code - dmin*min, with (scale,min)
 * unpacked per 32-elem sub-block exactly as the GGUF loader's getScaleMinK4 — and dots it with the
 * f32 activation, F32 accumulate. in must be divisible by 256. There is no activation quant on this
 * path (the weight, not the activation, is the narrow operand). */
void fcuda_q4k_matmul_f32(const uint8_t *dQ4K, const float *dX, float *dY, int out, int in, int P);

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
 *
 * fcuda_attention_f32 is the NAIVE baseline (#486): it materializes a full scores[nH*nPos]
 * row in a persistent global scratch (g_attn_scratch, grown once to nH*maxPos) and makes
 * four passes over it. Retained only as the fused-vs-naive microbench baseline.
 *
 * fcuda_flash_attention_f32 is the FUSED replacement on the live Attention path (#486): a
 * FlashAttention online-softmax kernel that streams the KV window with a running (max, sum,
 * acc) so NO scores[nPos] buffer is materialized — its only scratch is per-block shared
 * memory (query row + reduction row), so there is no per-call global allocation. maxPos is
 * accepted for a signature parallel to the naive baseline but is unused. Same f32 result as
 * the naive kernel up to reduction order (the Approx cudaFlashAttnCosineMin floor). */
void fcuda_attention_f32(const float *dQ, const float *dK, const float *dV, float *dOut,
                         int nPos, int maxPos, int nH, int nKV, int hd, float scale);
void fcuda_flash_attention_f32(const float *dQ, const float *dK, const float *dV, float *dOut,
                               int nPos, int maxPos, int nH, int nKV, int hd, float scale);

/* GLM-MoE-DSA sparse attention (model.glmDsaAttendCached's inner loop) for ONE query position
 * over nSel host-SELECTED, gathered, causal keys: per query head h, scores[i]=scale·dot(q_h,
 * selK_i_h), softmax over i, out_h=Σ softmax_i·selV_i_h. selK is [nSel, nH*kd], selV [nSel, nH*vd]
 * (the host gather laid all nH heads contiguous per selected position: head h at i*nH*kd+h*kd /
 * i*nH*vd+h*vd). kd (qkNope+qkRope) and vd DIFFER under MLA, so both are carried. Online-softmax
 * (running max/sum/acc) — no scores[nSel] row. The selection itself (index scores + top-k) is
 * computed host-side, so this attends the SAME keys as the host loop; Approx vs cpuref only in f32
 * reduction order (cudaDsaSparseAttnCosineMin). out[nH*vd]. */
void fcuda_dsa_sparse_attend_f32(const float *dQ, const float *dSelK, const float *dSelV, float *dOut,
                                 int nSel, int nH, int kd, int vd, float scale);

/* argmax over logits[n]: returns the SMALLEST index attaining the maximum value (the
 * cpuref first-max tie-break), copied back to the host as the single scalar fence. */
int fcuda_argmax_f32(const float *dLogits, int n);

/* GLM-MoE-DSA learned-indexer SCORE + top-k SELECTION for ONE query position, on-device.
 * For each cached key k (0..nKeys-1) with position k<=queryPos: score(k) = Σ_h weights[h] *
 * relu(scale * Σ_d indexQ[h*indexDim+d] * indexK[k*indexDim+d]); the per-key/per-head dot is
 * accumulated in DOUBLE precision so the device scores match the host f64 scores bit-closely
 * (selection-stable, not merely cosine-close — the indexer drives a discrete top-k, so it must be
 * reduction-faithful). dIndexQ is [nH*indexDim], dIndexK [nKeys*indexDim], dWeights [nH], all
 * device-resident. The kernel writes the per-key scores to dScores[nKeys] (device scratch), then a
 * selection kernel picks the top-k positions (score descending, ties by lower position — the
 * dsaTopKIndices order). The f64 score scratch is allocated internally. The selected positions are
 * copied back into host outIdx[0..ret-1]; ret = min(topK, #valid keys) is returned. outIdx must have
 * room for topK ints. */
int fcuda_dsa_index_select_f32(const float *dIndexQ, const float *dIndexK, const float *dWeights,
                               int nKeys, int nH, int indexDim, int queryPos,
                               int topK, float scale, int *outIdx);

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
/* fcuda_graph_abort ends an open capture and discards it WITHOUT launching — the recovery
 * half of fcuda_graph_begin for a Go-side panic mid-capture. Clears the stream's capture
 * state (and any sticky error) so the next op/request runs normally instead of cascading. */
void fcuda_graph_abort(void);
/* fcuda_graph_prewarm deepens every pooled scratch size class by `extra` spare buffers,
 * called OUTSIDE capture right before fcuda_graph_begin so a captured decode forward that
 * holds several same-size transients live at once (per-layer RMSNorm outputs, etc.) is served
 * entirely from the free list and never hits an illegal mid-capture cudaMalloc (#969). */
void fcuda_graph_prewarm(int extra);

/* NK/CKR: shared NCCL/CUDA error-handling macros for the NCCL device-collective .cu files
 * (cuda_nccl.cu, cuda_nccl_pg.cu) — one copy instead of each file duplicating the same
 * boilerplate. The guard lets a .cu file define its own before including this header if it
 * ever needs different behavior. The macro bodies name ncclResult_t/cudaError_t/fprintf,
 * which must be visible at the point of USE (not at this definition) — every .cu file in this
 * repo includes cuda_backend.h first and only expands these macros later, after
 * <cuda_runtime.h>/<nccl.h>/<stdio.h> follow, so this stays a standalone-parseable plain-C
 * header (no expansion happens here). */
#ifndef NK
#define NK(call) do { ncclResult_t _r = (call); if (_r != ncclSuccess) { \
  fprintf(stderr, "fak-nccl: %s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(_r)); \
  return (int)_r; } } while (0)
#endif

#ifndef CKR
#define CKR(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
  fprintf(stderr, "fak-nccl: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
  return (int)_e + 1000; } } while (0)
#endif

/* ---- NCCL device collectives (#971, multi-GPU expert/tensor parallelism) ----------------
 *
 * The cross-DEVICE reduction seam the CollectiveBackend interface needs to be REAL: a true
 * NCCL all-reduce/all-gather/reduce-scatter over distinct GPUs, not a host-staged copy. This
 * first cut is SINGLE-PROCESS, MULTI-GPU (ncclCommInitAll over the visible devices) — exactly
 * the shape of the acceptance witness (a device tensor reduced across 2 GPUs matching cpu-ref),
 * and the in-process form the one-process fak serve drives. (Multi-PROCESS NCCL over a rank
 * file is the follow-on; this proves the device communicator works first.)
 *
 * Build adds NCCL to the link only under `-tags cuda,nccl` (build_cuda.sh:
 * FAK_CUDA_NCCL=1). Plain `-tags cuda` stays single-device and does not require NCCL headers or
 * libnccl; the default pure-Go binary never sees any of this.
 *
 * fcuda_nccl_init(n): create one communicator per device 0..n-1 in a single process (the
 * canonical ncclCommInitAll single-process-multi-GPU setup). Returns 0 on success, an NCCL/
 * CUDA error code otherwise (and registers nothing — Caps().Collective stays false on the Go
 * side until this returns 0). Idempotent: a second call with the same n is a no-op success. */
int fcuda_nccl_init(int n);

/* fcuda_nccl_world(): the number of ranks (devices) the communicator was initialized over,
 * or 0 if fcuda_nccl_init has not succeeded. The Go side reads it to advertise world_size and
 * to validate that a collective's part count matches the communicator. */
int fcuda_nccl_world(void);

/* fcuda_nccl_allreduce_f32(dbufs, n, count): in-place NCCL all-reduce-SUM of `count` f32
 * elements, where dbufs[r] is the device pointer of rank r's buffer RESIDENT ON DEVICE r. All
 * n buffers must be the same length (count). On return every dbufs[r] holds the element-wise
 * sum across all ranks (the row-parallel partial-sum collective). Wrapped in ncclGroupStart/
 * End and fenced (cudaDeviceSynchronize per device) so the result is materialized on return.
 * Returns 0 on success, the NCCL/CUDA error code otherwise. */
int fcuda_nccl_allreduce_f32(void **dbufs, int n, int count);

/* fcuda_nccl_allgather_f32(dsend, drecv, n, sendcount): NCCL all-gather. dsend[r] (sendcount
 * f32 on device r) is gathered into drecv[r] (n*sendcount f32 on device r), rank-ordered
 * (drecv = send[0]||send[1]||...). Equal shard sizes only (the even-band column-parallel
 * gather). Returns 0 on success. */
int fcuda_nccl_allgather_f32(void **dsend, void **drecv, int n, int sendcount);

/* fcuda_nccl_reducescatter_f32(dsend, drecv, n, recvcount): NCCL reduce-scatter-SUM. dsend[r]
 * (n*recvcount f32 on device r) is reduced element-wise across ranks, then scattered so
 * drecv[r] (recvcount f32 on device r) holds rank r's 1/n band of the reduced vector — the
 * dual of all-gather (AllReduceSum == AllGather(ReduceScatter)). Returns 0 on success. */
int fcuda_nccl_reducescatter_f32(void **dsend, void **drecv, int n, int recvcount);

/* ---- Multi-process NCCL process group (ncclGetUniqueId/ncclCommInitRank) ---------------
 *
 * The multi-PROCESS follow-on the block above documents as deferred: one OS process holds
 * ONE GPU and joins a group of `world` such processes via an out-of-band-distributed unique
 * ID, instead of one process owning every device (ncclCommInitAll above). This is the shape
 * a sharded EP serve needs — N separate `fak serve --expert-parallel N` processes, one per
 * GPU — to swap its host DistComm reduce (internal/model/dist_collective.go) for a real
 * device-tensor reduce. Distinct globals from the ncclCommInitAll path above: the two
 * collectives are independent and must not share communicator state.
 *
 * Build/link gating is identical to the block above (`-tags cuda,nccl`, FAK_CUDA_NCCL=1). */

/* fcuda_nccl_pg_get_unique_id(out_id): fills out_id (caller-owned buffer, exactly
 * NCCL_UNIQUE_ID_BYTES==128 bytes) via ncclGetUniqueId. Called ONCE by the rank that will
 * bootstrap the group (rank 0); the resulting bytes are distributed to every other rank by
 * the caller (over DistComm.BroadcastFromRoot, an existing host TCP primitive — no new
 * transport needed for a 128-byte out-of-band payload). Returns 0 on success. */
int fcuda_nccl_pg_get_unique_id(void *out_id);

/* fcuda_nccl_pg_init(id, world, rank, device): cudaSetDevice(device), then
 * ncclCommInitRank(&g_pg_comm, world, id, rank) — every process in the group calls this with
 * the SAME id (from fcuda_nccl_pg_get_unique_id, distributed by the caller) and ITS OWN
 * rank/device. This call blocks until every rank in the group has called it (NCCL's
 * rendezvous). Returns 0 on success, an NCCL/CUDA error code otherwise; on failure the
 * process group is left uninitialized (a retry needs a fresh unique ID). */
int fcuda_nccl_pg_init(const void *id, int world, int rank, int device);

/* fcuda_nccl_pg_allreduce_f32(dbuf, count): in-place NCCL all-reduce-SUM of `count` f32
 * elements at dbuf, resident on THIS process's device, through g_pg_comm. Unlike
 * fcuda_nccl_allreduce_f32 above (one process, N device pointers, ncclGroupStart/End over
 * all of them), this process holds exactly one buffer on exactly one device — no group
 * bracket needed, mirroring the natural one-rank-per-process NCCL call shape. Returns 0 on
 * success, the NCCL/CUDA error code otherwise. Fails closed (returns non-zero, does not
 * silently no-op) if the process group has not been initialized. */
int fcuda_nccl_pg_allreduce_f32(void *dbuf, int count);

/* fcuda_nccl_pg_destroy(): ncclCommDestroy(g_pg_comm) and resets the process-group globals so
 * a later fcuda_nccl_pg_init can form a fresh group. Safe to call when no group is active
 * (no-op). Called on serve shutdown. */
void fcuda_nccl_pg_destroy(void);

#ifdef __cplusplus
}
#endif
#endif
