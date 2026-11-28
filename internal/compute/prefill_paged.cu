// Paged chunked-prefill attention kernel over vLLM block-table KV arenas + fresh chunk.
// Copyright (c) 2023, The vLLM team. License and notice: THIRD_PARTY_NOTICES.md ("vLLM").
//
// Reads logical prefix positions [0, prefix) directly from paged KV blocks:
//   K_cache: [num_blocks, nKV, hd/4, block_size, 4]  (x=4 vectorized layout)
//   V_cache: [num_blocks, nKV, hd,   block_size]
// and fresh chunk positions [0, tokens) either from contiguous K_fresh/V_fresh [tokens, nKV, hd]
// or (when K_fresh/V_fresh are null) directly from the paged KV cache via block_table.
// No temporary contiguous KV concatenation buffer is allocated.

#include <limits.h>
#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
namespace fak_vllm_prefill {

constexpr int kPagedPrefillBlockSize = 16;
constexpr int kPagedPrefillVecX = 4;

__host__ __device__ __forceinline__ int64_t paged_k_elem_offset(
    int block_idx, int block_offset, int kvh, int d,
    int nKV, int hd, int block_size) {
  int hd_vec = hd / kPagedPrefillVecX;
  int d_vec = d / kPagedPrefillVecX;
  int d_sub = d % kPagedPrefillVecX;
  return ((((int64_t)block_idx * nKV + kvh) * hd_vec + d_vec) * block_size + block_offset) *
             kPagedPrefillVecX +
         d_sub;
}

__host__ __device__ __forceinline__ int64_t paged_v_elem_offset(
    int block_idx, int block_offset, int kvh, int d,
    int nKV, int hd, int block_size) {
  return (((int64_t)block_idx * nKV + kvh) * hd + d) * block_size + block_offset;
}

}  // namespace fak_vllm_prefill

// k_paged_chunked_prefill_attention mirrors k_prompt_causal_attention_panel's exact
// per-(token, head) online-softmax accumulation order so paged chunked prefill outputs
// remain numerically consistent with the contiguous prompt attention panel.
__global__ void k_paged_chunked_prefill_attention(
    const float *__restrict__ Q,
    const float *__restrict__ K_cache,
    const float *__restrict__ V_cache,
    const int *__restrict__ block_table,
    const float *__restrict__ K_fresh,
    const float *__restrict__ V_fresh,
    float *__restrict__ Out,
    int tokens, int prefix, int nH, int nKV, int hd, int block_size, float scale) {
  int flat = blockIdx.x;
  int h = flat % nH;
  int token = flat / nH;
  if (h >= nH || token >= tokens) return;

  int grp = nH / nKV;
  int kvh = h / grp;
  int kvWidth = nKV * hd;
  int nPos = prefix + token + 1;
  bool use_fresh = (K_fresh != nullptr && V_fresh != nullptr);

  const float *qh = Q + (size_t)token * nH * hd + (size_t)h * hd;
  extern __shared__ float smem[];
  float *qs = smem;
  float *red = smem + hd;
  int tid = threadIdx.x;
  for (int d = tid; d < hd; d += FLASH_THREADS) {
    qs[d] = qh[d];
  }
  __syncthreads();

  float m = -INFINITY;
  float l = 0.f;
  float acc[FLASH_ACC_MAX];
#pragma unroll
  for (int k = 0; k < FLASH_ACC_MAX; k++) acc[k] = 0.f;

  for (int j = 0; j < nPos; j++) {
    bool from_fresh = use_fresh && (j >= prefix);
    int phys_block = 0;
    int block_off = 0;
    const float *k_fresh_row = nullptr;
    const float *v_fresh_row = nullptr;
    if (from_fresh) {
      int cj = j - prefix;
      k_fresh_row = K_fresh + (size_t)cj * kvWidth + (size_t)kvh * hd;
      v_fresh_row = V_fresh + (size_t)cj * kvWidth + (size_t)kvh * hd;
    } else {
      phys_block = block_table[j / block_size];
      block_off = j % block_size;
    }

    float partial = 0.f;
    for (int d = tid; d < hd; d += FLASH_THREADS) {
      float kv = from_fresh
                     ? k_fresh_row[d]
                     : K_cache[fak_vllm_prefill::paged_k_elem_offset(
                           phys_block, block_off, kvh, d, nKV, hd, block_size)];
      partial += qs[d] * kv;
    }
    red[tid] = partial;
    __syncthreads();
    for (int s = FLASH_THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float sc = red[0] * scale;
    __syncthreads();

    float mNew = fmaxf(m, sc);
    float corr = expf(m - mNew);
    float p = expf(sc - mNew);
    l = l * corr + p;

    int k = 0;
    for (int d = tid; d < hd; d += FLASH_THREADS, k++) {
      float vv = from_fresh
                     ? v_fresh_row[d]
                     : V_cache[fak_vllm_prefill::paged_v_elem_offset(
                           phys_block, block_off, kvh, d, nKV, hd, block_size)];
      acc[k] = acc[k] * corr + p * vv;
    }
    m = mNew;
  }

  float invL = l > 0.f ? 1.f / l : 0.f;
  int k = 0;
  for (int d = tid; d < hd; d += FLASH_THREADS, k++) {
    Out[(size_t)token * nH * hd + (size_t)h * hd + d] = acc[k] * invL;
  }
}

extern "C" int fcuda_paged_chunked_prefill_attention_f32(
    const float *dQ, const float *dKCache, const float *dVCache,
    const int *dBlockTable, int maxBlocksPerSeq,
    const float *dKFresh, const float *dVFresh, float *dOut,
    int tokens, int prefix, int nH, int nKV, int hd, int block_size, float scale) {
  if (!dQ || !dOut || tokens <= 0 || prefix < 0 || nH <= 0 || nKV <= 0 ||
      nH % nKV != 0 || hd <= 0 || hd > FLASH_THREADS * FLASH_ACC_MAX ||
      hd % fak_vllm_prefill::kPagedPrefillVecX != 0 ||
      block_size != fak_vllm_prefill::kPagedPrefillBlockSize ||
      !isfinite(scale) || scale <= 0.f) {
    return -1;
  }
  bool use_fresh = (dKFresh != nullptr && dVFresh != nullptr);
  if ((dKFresh == nullptr) != (dVFresh == nullptr)) return -1;
  int pagedPositions = use_fresh ? prefix : (prefix + tokens);
  if (pagedPositions > 0) {
    if (!dKCache || !dVCache || !dBlockTable) return -1;
    int neededBlocks = (pagedPositions + block_size - 1) / block_size;
    if (maxBlocksPerSeq < neededBlocks) return -1;
  }
  if (tokens > INT_MAX / nH || prefix > INT_MAX - tokens) return -1;

  size_t shmem = ((size_t)hd + FLASH_THREADS) * sizeof(float);
  FCUDA_TS("k_paged_chunked_prefill_attention", tokens * nH,
           k_paged_chunked_prefill_attention<<<tokens * nH, FLASH_THREADS, shmem, g_stream>>>(
               dQ, dKCache, dVCache, dBlockTable, dKFresh, dVFresh, dOut,
               tokens, prefix, nH, nKV, hd, block_size, scale));
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : (int)err;
}
#endif  // __CUDACC__
