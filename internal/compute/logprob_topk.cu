// logprob_topk.cu — per-row top-k log-softmax receipt without materializing
// [rows, vocab].
//
// ADAPTED from vllm-project/vllm, vllm/v1/worker/gpu/sample/logprob.py at
// ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0): _topk_log_softmax_kernel
// (per-row max + log-sum-exp streamed over 1024-wide blocks, output emitted
// only at the gathered ids), _ranks_kernel (chosen-token rank counting ties
// as >=) and _fill_logprob_token_ids_kernel (gather-column layout: chosen
// token first, then the bounded top-k ids). Re-expressed against fak's
// libfakcuda conventions (caller stream, extern "C" fcuda_* entry, int status
// returns); no upstream code copied. Ref
// oss-port-vllmtop-logprob-device-topk-20261103.
//
// Not yet #included into cuda_kernels.cu: that one-line build wiring is a
// follow-on ticket beyond this candidate's path cap. The CPU reference in
// logprob_topk.go pins the observable contract (fp32 block-streamed max +
// log-sum-exp, gather-only emission, tie-worst chosen rank, non-finite row
// refusal via any_non_finite).
//
// Contract (mirrors logprob_topk.go):
//   - logits: row-major [num_rows][vocab_size] float32, row stride
//     logits_stride (>= vocab_size).
//   - ids: row-major [num_rows][num_ids] int32 gathered token ids; column 0
//     is conventionally the chosen token; duplicates are allowed.
//   - logprobs: caller-sized [num_rows][num_ids] float32, written only at the
//     gathered ids; no [num_rows][vocab_size] tensor is ever allocated.
//   - chosen_ranks[i]: count of j with logits[i,j] >= logits[i, chosen[i]].
//   - any_non_finite (nullable): caller-sized [num_rows] int32, 1 when row i
//     holds a NaN or infinite logit; the device twin of the host-side
//     ErrNonFiniteLogits sentinel. A flagged row's outputs are still written
//     (garbage-free consumers check the flag before reading them).

#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

#define LPTK_BLOCK_WIDTH 1024
#define LPTK_THREADS 128
#define LPTK_LANES_PER_THREAD (LPTK_BLOCK_WIDTH / LPTK_THREADS)

// Status codes returned by the fcuda_logprob_topk host entry.
#define LPTK_OK 0
#define LPTK_BAD_ARGS 1
#define LPTK_CUDA_ERROR 2

// Phase 1 (upstream _topk_log_softmax_kernel idea): one block per row streams
// the row in LPTK_BLOCK_WIDTH-wide tiles. Pass one reduces the row max and
// flags non-finite entries; pass two re-streams the tiles summing
// exp(x - max) so lse = log(se) is the fp32 log-sum-exp. Both passes keep
// every intermediate in float32, matching the upstream NOTE(woosuk) that all
// following operations use FP32.
__global__ void k_lptk_row_lse(const float* __restrict__ logits,
                               int64_t logits_stride, int vocab_size,
                               float* __restrict__ d_max, float* __restrict__ d_lse,
                               int32_t* __restrict__ d_any_non_finite) {
  const int row = blockIdx.x;
  const float* xrow = logits + (int64_t)row * logits_stride;
  extern __shared__ float red[];
  __shared__ int32_t bad_red;

  // Tile bounds helper: tiles stream the row in LPTK_BLOCK_WIDTH-wide blocks;
  // only the final tile may be short.
  auto tile_limit = [vocab_size](int tile) {
    const int remaining = vocab_size - tile * LPTK_BLOCK_WIDTH;
    return remaining < LPTK_BLOCK_WIDTH ? vocab_size
                                        : tile * LPTK_BLOCK_WIDTH + LPTK_BLOCK_WIDTH;
  };

  // Pass 1: row max + non-finite flag.
  float local_max = -INFINITY;
  int32_t local_bad = 0;
  for (int tile = 0; tile * LPTK_BLOCK_WIDTH < vocab_size; ++tile) {
    const int base = tile * LPTK_BLOCK_WIDTH + threadIdx.x * LPTK_LANES_PER_THREAD;
    const int limit = tile_limit(tile);
    for (int lane = 0; lane < LPTK_LANES_PER_THREAD; ++lane) {
      const int i = base + lane;
      if (i >= limit) break;
      const float x = xrow[i];
      if (!isfinite(x)) {
        local_bad = 1;
        continue;
      }
      if (x > local_max) local_max = x;
    }
  }
  red[threadIdx.x] = local_max;
  if (threadIdx.x == 0) bad_red = 0;
  __syncthreads();
  if (local_bad != 0) atomicOr(&bad_red, 1);
  for (int off = LPTK_THREADS / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off && red[threadIdx.x + off] > red[threadIdx.x]) {
      red[threadIdx.x] = red[threadIdx.x + off];
    }
    __syncthreads();
  }
  const float row_max = red[0];
  // Every thread has read the reduced max before any thread reuses red[] for
  // the pass-2 partial sums.
  __syncthreads();

  // Pass 2: se = sum(exp(x - row_max)), fp32 accumulation.
  float local_sum = 0.f;
  for (int tile = 0; tile * LPTK_BLOCK_WIDTH < vocab_size; ++tile) {
    const int base = tile * LPTK_BLOCK_WIDTH + threadIdx.x * LPTK_LANES_PER_THREAD;
    const int limit = tile_limit(tile);
    for (int lane = 0; lane < LPTK_LANES_PER_THREAD; ++lane) {
      const int i = base + lane;
      if (i >= limit) break;
      local_sum += expf(xrow[i] - row_max);
    }
  }
  red[threadIdx.x] = local_sum;
  __syncthreads();
  for (int off = LPTK_THREADS / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    d_max[row] = row_max;
    d_lse[row] = logf(red[0]);
    if (d_any_non_finite != nullptr) d_any_non_finite[row] = bad_red;
  }
}

// Phase 2 (upstream _topk_log_softmax_kernel gather loop): one thread per
// (row, gathered column) writes logits[id] - max - lse; the [rows, vocab]
// logprob matrix is never materialized. Grid-strided over the flat
// [rows, num_ids] output so very tall batches stay within gridDim limits.
__global__ void k_lptk_gather(const float* __restrict__ logits,
                              int64_t logits_stride,
                              const float* __restrict__ d_max,
                              const float* __restrict__ d_lse,
                              const int32_t* __restrict__ ids, int num_ids,
                              int num_rows, float* __restrict__ logprobs) {
  const int total = num_rows * num_ids;
  for (int flat = blockIdx.x * blockDim.x + threadIdx.x; flat < total;
       flat += gridDim.x * blockDim.x) {
    const int row = flat / num_ids;
    const int32_t id = ids[flat];
    const float x = logits[(int64_t)row * logits_stride + id];
    logprobs[flat] = x - d_max[row] - d_lse[row];
  }
}

// Phase 3 (upstream _ranks_kernel idea): one block per row counts the tokens
// whose logit is >= the chosen token's logit, streaming in LPTK_BLOCK_WIDTH
// tiles like phase 1. Ties therefore rank the chosen token worst among
// equals, exactly as the upstream (logits >= x) count.
__global__ void k_lptk_rank(const float* __restrict__ logits,
                            int64_t logits_stride,
                            const int32_t* __restrict__ chosen, int vocab_size,
                            int32_t* __restrict__ ranks) {
  const int row = blockIdx.x;
  const float* xrow = logits + (int64_t)row * logits_stride;
  const float x = xrow[chosen[row]];
  extern __shared__ int32_t red[];
  int32_t local = 0;
  for (int tile = 0; tile * LPTK_BLOCK_WIDTH < vocab_size; ++tile) {
    const int base = tile * LPTK_BLOCK_WIDTH + threadIdx.x * LPTK_LANES_PER_THREAD;
    const int remaining = vocab_size - tile * LPTK_BLOCK_WIDTH;
    const int limit = remaining < LPTK_BLOCK_WIDTH ? vocab_size
                                                   : tile * LPTK_BLOCK_WIDTH + LPTK_BLOCK_WIDTH;
    for (int lane = 0; lane < LPTK_LANES_PER_THREAD; ++lane) {
      const int i = base + lane;
      if (i >= limit) break;
      if (xrow[i] >= x) ++local;
    }
  }
  red[threadIdx.x] = local;
  __syncthreads();
  for (int off = LPTK_THREADS / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
    __syncthreads();
  }
  if (threadIdx.x == 0) ranks[row] = red[0];
}

extern "C" int fcuda_logprob_topk(const float* logits, int64_t logits_stride,
                                  const int32_t* ids, int32_t num_ids,
                                  const int32_t* chosen, float* logprobs,
                                  int32_t* chosen_ranks,
                                  int32_t* any_non_finite, int32_t num_rows,
                                  int32_t vocab_size, cudaStream_t stream) {
  if (logits == nullptr || ids == nullptr || chosen == nullptr ||
      logprobs == nullptr || chosen_ranks == nullptr || num_rows < 1 ||
      num_ids < 0 || vocab_size < 1 || logits_stride < vocab_size) {
    return LPTK_BAD_ARGS;
  }

  // Row scratch for the streamed max/lse pair; released after the last launch
  // (all three kernels are enqueued on the caller stream ahead of it).
  float* d_scratch = nullptr;
  cudaError_t err =
      cudaMalloc((void**)&d_scratch, (size_t)num_rows * 2 * sizeof(float));
  if (err != cudaSuccess) return LPTK_CUDA_ERROR;
  auto finish = [&](cudaError_t status) {
    cudaFree(d_scratch);
    return status == cudaSuccess ? LPTK_OK : LPTK_CUDA_ERROR;
  };
  float* d_max = d_scratch;
  float* d_lse = d_scratch + num_rows;

  dim3 block(LPTK_THREADS);
  const size_t smem = LPTK_THREADS * sizeof(float);
  err = k_lptk_row_lse<<<num_rows, block, smem, stream>>>(
      logits, logits_stride, vocab_size, d_max, d_lse, any_non_finite);
  if (err != cudaSuccess) return finish(err);

  {
    const int total = num_rows * num_ids;
    const int gather_threads = 256;
    const int gather_grid = total == 0
                                ? 1
                                : (total + gather_threads - 1) / gather_threads;
    err = k_lptk_gather<<<gather_grid, gather_threads, 0, stream>>>(
        logits, logits_stride, d_max, d_lse, ids, num_ids, num_rows, logprobs);
    if (err != cudaSuccess) return finish(err);
  }

  err = k_lptk_rank<<<num_rows, block, LPTK_THREADS * sizeof(int32_t), stream>>>(
      logits, logits_stride, chosen, vocab_size, chosen_ranks);
  return finish(err);
}
