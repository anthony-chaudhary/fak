// On-device MoE token align-and-sort per expert + weighted-sum finalize.
// Included by cuda_kernels.cu (one-line #include, build wiring owned by the
// follow-on grouped-GEMM ticket) so the kernels share g_stream, the pooled
// allocator and the transfer witnesses of the single libfakcuda object.
//
// ADAPT of vllm-project/vllm csrc/moe/moe_align_sum_kernels.cu at pin
// ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0): moe_align_block_size_kernel
// + count_and_sort_expert_tokens_kernel + moe_sum_vec_kernel, re-expressed for fak's
// flat C ABI. Divergences from upstream, all deliberate:
//   - No CUB: the per-expert exclusive prefix sum is a hand-rolled two-level
//     warp-shuffle scan (32 warps x 32 lanes), so num_experts <= 1024 fits one
//     1024-thread block, like upstream's main kernel.
//   - The finalize carries an explicit float32 gate-weight operand
//     (out[t,j] = sum_k w[t,k] * in[t,k,j]); upstream folds the gating weights
//     into the fused-MoE epilogue. All-ones weights reproduce upstream's sum.
//   - The sum kernel is the scalar f32 path only (upstream's 16B-vectorized and
//     bf16/fp16 variants are dropped): every multiply and add is inline PTX
//     mul.rn.f32 / add.rn.f32, so the result is bit-identical to the Go oracle
//     MoeSumTopK regardless of -ftz / --use_fast_math.
//
// Contract (mirrored by the Go oracle MoeAlignBlockSizes in moe_align_sort.go):
//   - Input: flat int32 topk expert-id panel [num_tokens][top_k], length numel.
//     Entry i is valid iff 0 <= id < num_experts; anything else is skipped.
//   - Capacity: max_num_tokens_padded = numel + num_experts*(block_size-1) (each
//     expert pads by at most block_size-1), max_num_m_blocks = ceil(that/block_size).
//   - cumsum[e] is a multiple of block_size; expert e owns the padded region
//     [cumsum[e], cumsum[e+1]) of sorted_token_ids, length
//     ceil(count_e/block_size)*block_size. Pad slots hold the sentinel numel.
//   - Multiset guarantee: for each e the non-sentinel values in its region equal
//     the set of valid entry indices routed to e. Intra-region order is
//     UNSPECIFIED (the scatter assigns ranks by atomicAdd, exactly like
//     upstream's count_and_sort_expert_tokens_kernel), so consumers must treat a
//     region as a set; the Go oracle emits the canonical ascending form.
//   - expert_ids[b] = e iff block b (sorted positions [b*bs,(b+1)*bs)) lies
//     inside e's region, else -1. total_tokens_post_pad = cumsum[num_experts].
// Statuses: 0 ok, -1 null pointer or refused geometry, -2 graph capture open,
// -3 device allocation failure, positive cudaError_t for copy/launch/execution.

#include <limits.h>

#define MAE_MAX_EXPERTS 1024
#define MAE_THREADS 1024
#define MAE_FILL_THREADS 256
#define MAE_SCATTER_THREADS 256
#define MAE_MAX_SCATTER_BLOCKS 1024

static __device__ __forceinline__ float mae_mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

static __device__ __forceinline__ float mae_add_rn(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// Fill the whole padded capacity with the numel sentinel before the scatter
// overwrites the valid ranks.
__global__ void k_mae_fill_sentinel(int32_t *__restrict__ sorted, int32_t n, int32_t sentinel) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    sorted[i] = sentinel;
  }
}

// One 1024-thread block: count entries per expert, pad each count to the block
// boundary, exclusive-scan the padded counts into cumsum AND cursor (cursor is
// the device copy the scatter consumes), name every sorted block's expert, and
// publish total_tokens_post_pad.
__global__ void __launch_bounds__(MAE_THREADS) k_mae_align_block_size(
    const int32_t *__restrict__ topk_ids, int32_t numel,
    int32_t num_experts, int32_t block_size,
    int32_t *__restrict__ cumsum, int32_t *__restrict__ cursor,
    int32_t *__restrict__ expert_ids, int32_t *__restrict__ total_tokens_post_pad,
    int32_t max_num_m_blocks) {
  __shared__ int32_t s_counts[MAE_MAX_EXPERTS];
  __shared__ int32_t s_warp_total[32];
  const int tid = (int)threadIdx.x;

  if (tid < num_experts) s_counts[tid] = 0;
  __syncthreads();
  for (int i = tid; i < numel; i += MAE_THREADS) {
    const int32_t id = __ldg(topk_ids + i);
    if (id >= 0 && id < num_experts) atomicAdd(&s_counts[id], 1);
  }
  __syncthreads();

  int32_t val = 0;
  if (tid < num_experts) {
    const int32_t c = s_counts[tid];
    val = ((c + block_size - 1) / block_size) * block_size;
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  int32_t incl = val;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int32_t n = __shfl_up_sync(0xffffffffu, incl, d);
    if (lane >= d) incl += n;
  }
  if (lane == 31) s_warp_total[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    int32_t w = s_warp_total[lane];
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int32_t n = __shfl_up_sync(0xffffffffu, w, d);
      if (lane >= d) w += n;
    }
    s_warp_total[lane] = w;
  }
  __syncthreads();
  const int32_t carry = (warp == 0) ? 0 : s_warp_total[warp - 1];
  const int32_t excl = incl - val + carry;
  if (tid < num_experts) {
    cumsum[tid] = excl;
    cursor[tid] = excl;
  }
  if (tid == 0) total_tokens_post_pad[0] = s_warp_total[31];
  __syncthreads();

  if (tid < num_experts) {
    const int32_t lo = cumsum[tid];
    const int32_t hi = (tid + 1 < num_experts) ? cumsum[tid + 1] : total_tokens_post_pad[0];
    for (int32_t p = lo; p < hi; p += block_size) expert_ids[p / block_size] = tid;
  }
  __syncthreads();
  const int32_t filled = total_tokens_post_pad[0] / block_size;
  for (int64_t b = (int64_t)filled + tid; b < max_num_m_blocks; b += MAE_THREADS) {
    expert_ids[b] = -1;
  }
}

// Scatter half of the pair: assign each valid entry a rank inside its expert's
// padded region (atomicAdd on the cursor copy, upstream's scheme) and drop the
// input index there. The sentinel fill already covers every pad slot.
__global__ void k_mae_scatter_expert_tokens(
    const int32_t *__restrict__ topk_ids, int32_t numel, int32_t num_experts,
    int32_t *__restrict__ cursor, int32_t *__restrict__ sorted) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += stride) {
    const int32_t id = __ldg(topk_ids + (size_t)i);
    if (id < 0 || id >= num_experts) continue;
    const int32_t rank = atomicAdd(cursor + id, 1);
    sorted[rank] = (int32_t)i;
  }
}

// Weighted-sum finalize: out[t,j] = sum_k w[t,k]*in[t,k,j] over valid experts,
// fp32 accumulate in ascending k order, serial per output element.
__global__ void k_mae_sum_topk_f32(
    const float *__restrict__ expert_out, const float *__restrict__ weights,
    const int32_t *__restrict__ topk_ids, int32_t num_experts,
    float *__restrict__ out, int64_t num_tokens, int64_t topk, int64_t hidden) {
  const int64_t total = num_tokens * hidden;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    const int64_t t = idx / hidden;
    const int64_t j = idx % hidden;
    float acc = 0.0f;
    for (int64_t k = 0; k < topk; ++k) {
      const int32_t id = __ldg(topk_ids + (size_t)(t * topk + k));
      if (id < 0 || id >= num_experts) continue;
      const float w = __ldg(weights + (size_t)(t * topk + k));
      const float v = __ldg(expert_out + (size_t)(t * topk + k) * hidden + (size_t)j);
      acc = mae_add_rn(acc, mae_mul_rn(w, v));
    }
    out[idx] = acc;
  }
}

// mae_geom is the validated byte geometry of one align/sort call.
struct mae_geom {
  size_t ids, sorted, expert, cumsum;
  int32_t max_tokens_padded, max_m_blocks;
};

// mae_geometry validates shape and sizes every operand; false means status -1.
static bool mae_geometry(int64_t numel, int32_t num_experts, int32_t block_size, mae_geom *g) {
  if (numel < 0 || numel > INT32_MAX) return false;
  if (num_experts <= 0 || num_experts > MAE_MAX_EXPERTS) return false;
  if (block_size <= 0) return false;
  const int64_t max_padded = numel + (int64_t)num_experts * (int64_t)(block_size - 1);
  if (max_padded < 0 || max_padded > INT32_MAX) return false;
  const int64_t max_blocks = (max_padded + block_size - 1) / block_size;
  if (max_blocks < 0 || max_blocks > INT32_MAX) return false;
  g->ids = (size_t)numel * sizeof(int32_t);
  g->sorted = (size_t)max_padded * sizeof(int32_t);
  g->expert = (size_t)max_blocks * sizeof(int32_t);
  g->cumsum = (size_t)(num_experts + 1) * sizeof(int32_t);
  g->max_tokens_padded = (int32_t)max_padded;
  g->max_m_blocks = (int32_t)max_blocks;
  return true;
}

static size_t g_mae_align_launches = 0;
static size_t g_mae_sum_launches = 0;

// One synchronous align-and-sort: upload the ids, fill the sentinel, run the
// single-block align kernel, scatter the ranks, fence, read every output back.
// Host buffers must carry exactly the documented sizes (validated against the
// recomputed geometry). Statuses as the file header.
extern "C" int fcuda_moe_align_block_size(
    const int32_t *topk_ids, size_t topk_ids_bytes,
    int32_t num_experts, int32_t block_size,
    int32_t *sorted_token_ids, size_t sorted_bytes,
    int32_t *expert_ids, size_t expert_bytes,
    int32_t *cumsum, int32_t *total_tokens_post_pad) {
  if (!topk_ids || !sorted_token_ids || !expert_ids || !cumsum || !total_tokens_post_pad) return -1;
  mae_geom g;
  if (!mae_geometry((int64_t)(topk_ids_bytes / sizeof(int32_t)), num_experts, block_size, &g)) return -1;
  if (topk_ids_bytes != g.ids || sorted_bytes != g.sorted || expert_bytes != g.expert) return -1;
  if (g_capture_open) return -2;
  const int32_t numel = (int32_t)(topk_ids_bytes / sizeof(int32_t));
  void *dIds = fcuda_malloc(g.ids);
  void *dSorted = fcuda_malloc(g.sorted);
  void *dExpert = fcuda_malloc(g.expert);
  void *dCumsum = fcuda_malloc(g.cumsum);
  void *dCursor = fcuda_malloc(g.cumsum);
  void *dTotal = fcuda_malloc(sizeof(int32_t));
  int status = 0;
  cudaError_t e = cudaSuccess;
  if (!dIds || !dSorted || !dExpert || !dCumsum || !dCursor || !dTotal) {
    fcuda_free(dIds); fcuda_free(dSorted); fcuda_free(dExpert);
    fcuda_free(dCumsum); fcuda_free(dCursor); fcuda_free(dTotal);
    return -3;
  }
  (void)cudaGetLastError(); // discard stale status before attributing this call
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dIds, topk_ids, g.ids, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += g.ids;
  if (g.max_tokens_padded > 0) {
    const unsigned int fillGrid = (unsigned int)((g.max_tokens_padded + MAE_FILL_THREADS - 1) / MAE_FILL_THREADS);
    FCUDA_TS("k_mae_fill_sentinel", fillGrid,
             k_mae_fill_sentinel<<<fillGrid, MAE_FILL_THREADS, 0, g_stream>>>(
                 (int32_t *)dSorted, g.max_tokens_padded, numel));
    if ((e = cudaGetLastError()) != cudaSuccess) { status = (int)e; goto done; }
  }
  FCUDA_TS("k_mae_align_block_size", 1u,
           k_mae_align_block_size<<<1, MAE_THREADS, 0, g_stream>>>(
               (const int32_t *)dIds, numel, num_experts, block_size,
               (int32_t *)dCumsum, (int32_t *)dCursor, (int32_t *)dExpert,
               (int32_t *)dTotal, g.max_m_blocks));
  if ((e = cudaGetLastError()) != cudaSuccess) { status = (int)e; goto done; }
  if (numel > 0) {
    unsigned int scatterGrid = (unsigned int)((numel + MAE_SCATTER_THREADS - 1) / MAE_SCATTER_THREADS);
    if (scatterGrid > MAE_MAX_SCATTER_BLOCKS) scatterGrid = MAE_MAX_SCATTER_BLOCKS;
    FCUDA_TS("k_mae_scatter_expert_tokens", scatterGrid,
             k_mae_scatter_expert_tokens<<<scatterGrid, MAE_SCATTER_THREADS, 0, g_stream>>>(
                 (const int32_t *)dIds, numel, num_experts, (int32_t *)dCursor, (int32_t *)dSorted));
    if ((e = cudaGetLastError()) != cudaSuccess) { status = (int)e; goto done; }
  }
  if ((e = cudaStreamSynchronize(g_stream)) != cudaSuccess) { status = (int)e; goto done; }
  ++g_mae_align_launches;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(sorted_token_ids, dSorted, g.sorted, cudaMemcpyDeviceToHost)) != cudaSuccess) { status = (int)e; goto done; }
  g_host_bytes += g.sorted;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(expert_ids, dExpert, g.expert, cudaMemcpyDeviceToHost)) != cudaSuccess) { status = (int)e; goto done; }
  g_host_bytes += g.expert;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(cumsum, dCumsum, g.cumsum, cudaMemcpyDeviceToHost)) != cudaSuccess) { status = (int)e; goto done; }
  g_host_bytes += g.cumsum;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(total_tokens_post_pad, dTotal, sizeof(int32_t), cudaMemcpyDeviceToHost)) != cudaSuccess) { status = (int)e; goto done; }
  g_host_bytes += sizeof(int32_t);
done:
  fcuda_free(dIds);
  fcuda_free(dSorted);
  fcuda_free(dExpert);
  fcuda_free(dCumsum);
  fcuda_free(dCursor);
  fcuda_free(dTotal);
  return status;
}

// One synchronous weighted-sum finalize: upload the three operands, launch,
// fence, read out back. out is [num_tokens, hidden] float32. Statuses as above.
extern "C" int fcuda_moe_sum_topk(
    const float *expert_out, size_t expert_out_bytes,
    const float *weights, size_t weights_bytes,
    const int32_t *topk_ids, size_t topk_ids_bytes,
    int32_t num_experts,
    float *out, size_t out_bytes,
    int64_t num_tokens, int64_t topk, int64_t hidden) {
  if (!expert_out || !weights || !topk_ids || !out) return -1;
  if (num_tokens <= 0 || topk <= 0 || hidden <= 0) return -1;
  if (num_experts <= 0 || num_experts > MAE_MAX_EXPERTS) return -1;
  const int64_t tokensTopk = num_tokens * topk;
  if (tokensTopk <= 0 || tokensTopk > INT32_MAX) return -1;
  const int64_t elements = tokensTopk * hidden;
  if (elements <= 0 || elements > (int64_t)SIZE_MAX / sizeof(float)) return -1;
  const int64_t outElements = num_tokens * hidden;
  if (outElements <= 0 || outElements > (int64_t)SIZE_MAX / sizeof(float)) return -1;
  if (expert_out_bytes != (size_t)elements * sizeof(float)) return -1;
  if (weights_bytes != (size_t)tokensTopk * sizeof(float)) return -1;
  if (topk_ids_bytes != (size_t)tokensTopk * sizeof(int32_t)) return -1;
  if (out_bytes != (size_t)outElements * sizeof(float)) return -1;
  if (g_capture_open) return -2;
  void *dIn = fcuda_malloc((size_t)elements * sizeof(float));
  void *dW = fcuda_malloc((size_t)tokensTopk * sizeof(float));
  void *dIds = fcuda_malloc((size_t)tokensTopk * sizeof(int32_t));
  void *dOut = fcuda_malloc((size_t)outElements * sizeof(float));
  int status = 0;
  cudaError_t e = cudaSuccess;
  if (!dIn || !dW || !dIds || !dOut) {
    fcuda_free(dIn); fcuda_free(dW); fcuda_free(dIds); fcuda_free(dOut);
    return -3;
  }
  (void)cudaGetLastError();
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dIn, expert_out, (size_t)elements * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += (size_t)elements * sizeof(float);
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dW, weights, (size_t)tokensTopk * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += (size_t)tokensTopk * sizeof(float);
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dIds, topk_ids, (size_t)tokensTopk * sizeof(int32_t), cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += (size_t)tokensTopk * sizeof(int32_t);
  const unsigned int grid = (unsigned int)((outElements + MAE_SCATTER_THREADS - 1) / MAE_SCATTER_THREADS);
  FCUDA_TS("k_mae_sum_topk_f32", grid,
           k_mae_sum_topk_f32<<<grid, MAE_SCATTER_THREADS, 0, g_stream>>>(
               (const float *)dIn, (const float *)dW, (const int32_t *)dIds, num_experts,
               (float *)dOut, num_tokens, topk, hidden));
  if ((e = cudaGetLastError()) != cudaSuccess) { status = (int)e; goto done; }
  if ((e = cudaStreamSynchronize(g_stream)) != cudaSuccess) { status = (int)e; goto done; }
  ++g_mae_sum_launches;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(out, dOut, (size_t)outElements * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) { status = (int)e; goto done; }
  g_host_bytes += (size_t)outElements * sizeof(float);
done:
  fcuda_free(dIn);
  fcuda_free(dW);
  fcuda_free(dIds);
  fcuda_free(dOut);
  return status;
}

// Witnesses: cumulative successful operation completions.
extern "C" size_t fcuda_moe_align_launches(void) {
  return g_mae_align_launches;
}

extern "C" size_t fcuda_moe_sum_launches(void) {
  return g_mae_sum_launches;
}
