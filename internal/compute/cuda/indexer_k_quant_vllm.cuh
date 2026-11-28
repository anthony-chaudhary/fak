// Per-row symmetric INT8 quantization and paged cache write/gather helpers for the
// DSA lightning indexer key cache.
// Adapted from vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0).
// Copyright (c) 2023, The vLLM team. License and notice: THIRD_PARTY_NOTICES.md ("vLLM").
#ifndef FAK_INDEXER_K_QUANT_VLLM_CUH
#define FAK_INDEXER_K_QUANT_VLLM_CUH

#include <limits.h>
#include <math.h>
#include <stdint.h>

#include "paged_kv_write_map.h"

namespace fak_vllm_indexer {

constexpr int kIndexBlockSize = 16;

// Host-admissible bounds validation for paged INT8 indexer_k_quant_and_cache inserts.
inline bool indexer_k_write_map_valid(const fak_vllm_pa::PagedKVWriteMap &pages,
                                      int count, int num_blocks, int tokens,
                                      int index_dim) {
  if (count <= 0 || count > fak_vllm_pa::kWriteMaxPages || num_blocks <= 0 ||
      pages.first_offset < 0 || pages.first_offset >= kIndexBlockSize ||
      tokens <= 0 || index_dim <= 0) {
    return false;
  }
  const int64_t end = int64_t(pages.first_offset) + tokens;
  if ((end - 1) / kIndexBlockSize + 1 != count) return false;
  for (int i = 0; i < count; ++i) {
    if (pages.blocks[i] < 0 || pages.blocks[i] >= num_blocks) return false;
    for (int j = 0; j < i; ++j) {
      if (pages.blocks[j] == pages.blocks[i]) return false;
    }
  }
  return true;
}

#ifdef __CUDACC__
__device__ __forceinline__ int8_t indexer_k_quant_i8_val(float val, float inv_scale) {
  if (inv_scale <= 0.f) return 0;
  long q = llroundf(val * inv_scale);
  if (q > 127L) q = 127L;
  if (q < -127L) q = -127L;
  return (int8_t)q;
}

__device__ __forceinline__ float indexer_k_dequant_i8_val(int8_t q, float scale) {
  return (float)q * scale;
}

}  // namespace fak_vllm_indexer

#ifdef __CUDACC__
// k_dsa_index_quant_rows quantizes contiguous [nKeys, indexDim] FP32 indexer keys
// into per-row symmetric INT8 [nKeys, indexDim] + per-row FP32 scales [nKeys].
__global__ void k_dsa_index_quant_rows(const float *__restrict__ indexK,
                                       int8_t *__restrict__ outKQ,
                                       float *__restrict__ outScales,
                                       int nKeys, int indexDim) {
  int k = blockIdx.x;
  if (k >= nKeys) return;
  const float *src = indexK + (size_t)k * indexDim;
  int8_t *dst = outKQ + (size_t)k * indexDim;

  __shared__ float smax[128];
  float local_max = 0.f;
  for (int d = threadIdx.x; d < indexDim; d += blockDim.x) {
    float a = fabsf(src[d]);
    if (a > local_max) local_max = a;
  }
  smax[threadIdx.x] = local_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s && smax[threadIdx.x + s] > smax[threadIdx.x]) {
      smax[threadIdx.x] = smax[threadIdx.x + s];
    }
    __syncthreads();
  }

  float amax = smax[0];
  float scale = (amax > 0.f) ? (amax / 127.0f) : 0.f;
  float inv_scale = (scale > 0.f) ? (1.0f / scale) : 0.f;
  if (threadIdx.x == 0) {
    outScales[k] = scale;
  }
  for (int d = threadIdx.x; d < indexDim; d += blockDim.x) {
    dst[d] = fak_vllm_indexer::indexer_k_quant_i8_val(src[d], inv_scale);
  }
}

// k_dsa_index_quant_cache (vLLM indexer_k_quant_and_cache equivalent) quantizes each
// inserted indexer K token row on insert with a per-row FP32 scale and writes INT8 codes +
// scales into paged cache blocks addressed by fak_vllm_pa::PagedKVWriteMap:
//   k_cache:     [num_blocks, block_size, indexDim] int8_t
//   scale_cache: [num_blocks, block_size]           float
__global__ void k_dsa_index_quant_cache(const float *__restrict__ indexK,
                                        int8_t *__restrict__ k_cache,
                                        float *__restrict__ scale_cache,
                                        fak_vllm_pa::PagedKVWriteMap pages,
                                        int tokens, int indexDim, int block_size) {
  int token = blockIdx.x;
  if (token >= tokens) return;
  const float *src = indexK + (size_t)token * indexDim;

  __shared__ float smax[128];
  float local_max = 0.f;
  for (int d = threadIdx.x; d < indexDim; d += blockDim.x) {
    float a = fabsf(src[d]);
    if (a > local_max) local_max = a;
  }
  smax[threadIdx.x] = local_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s && smax[threadIdx.x + s] > smax[threadIdx.x]) {
      smax[threadIdx.x] = smax[threadIdx.x + s];
    }
    __syncthreads();
  }

  float amax = smax[0];
  float scale = (amax > 0.f) ? (amax / 127.0f) : 0.f;
  float inv_scale = (scale > 0.f) ? (1.0f / scale) : 0.f;

  int pos = pages.first_offset + token;
  int64_t block_idx = pages.blocks[pos / block_size];
  int block_offset = pos % block_size;
  int64_t scale_slot = block_idx * block_size + block_offset;
  int64_t key_base = scale_slot * indexDim;

  if (threadIdx.x == 0) {
    scale_cache[scale_slot] = scale;
  }
  for (int d = threadIdx.x; d < indexDim; d += blockDim.x) {
    k_cache[key_base + d] = fak_vllm_indexer::indexer_k_quant_i8_val(src[d], inv_scale);
  }
}

// k_dsa_index_quant_gather gathers logical positions [0, nKeys) from the paged INT8 key
// cache into contiguous quantized [nKeys, indexDim] + [nKeys] buffers without dequantizing
// to global memory so k_dsa_index_score can dequantize on the fly into f64 accumulators.
__global__ void k_dsa_index_quant_gather(const int8_t *__restrict__ k_cache,
                                         const float *__restrict__ scale_cache,
                                         const int *__restrict__ block_table,
                                         int8_t *__restrict__ outKQ,
                                         float *__restrict__ outScales,
                                         int nKeys, int indexDim, int block_size) {
  int k = blockIdx.x;
  if (k >= nKeys) return;
  int64_t block_idx = block_table[k / block_size];
  int block_offset = k % block_size;
  int64_t scale_slot = block_idx * block_size + block_offset;
  int64_t src_base = scale_slot * indexDim;
  int64_t dst_base = (int64_t)k * indexDim;

  if (threadIdx.x == 0) {
    outScales[k] = scale_cache[scale_slot];
  }
  for (int d = threadIdx.x; d < indexDim; d += blockDim.x) {
    outKQ[dst_base + d] = k_cache[src_base + d];
  }
}
#endif  // __CUDACC__

#endif  // FAK_INDEXER_K_QUANT_VLLM_CUH
