// ---- MLA concat-and-cache over a paged write map (fak cuda-mla-cache-insert) ----
// Adapted from vLLM csrc/cache_kernels.cu, concat_and_cache_mla_kernel (:424),
// concat_and_cache_mla_grouped_kernel (:472) and concat_and_cache_ds_mla_kernel
// (:526), at commit ae065918640a198a56f6f17eb45e251c3229fe41, Apache-2.0.
// Copyright (c) 2023, The vLLM team. License and notice: THIRD_PARTY_NOTICES.md
// ("vLLM multi-page CUDA KV append addressing" family; this port cites its own pin).
//
// The upstream bodies are kept; the deltas for Fak are:
//   * Only the scalar_t = cache_t = float lane is kept: the quantized-cache arms
//     of the upstream kernels are dropped. Fak serves the MLA cache as the
//     mlaKVLayout f32 row [latent | rope] (cacheStride = KVLatentDim + RopeDim,
//     internal/model/kvlayout.go:104); no quantized MLA arm is adopted yet.
//   * torch launchers and the device slot_mapping tensor are replaced by the
//     extern "C" adapters below, driven by the bounded by-value PagedKVWriteMap
//     (paged_kv_write_map.h) - same precedent as paged_attention_vllm.cuh.
//   * The grouped variant reads one shared write map instead of per-layer slot
//     rows: a physical block id addresses the same block in every layer's arena,
//     so the block table is per-session, not per-layer (existing invariant).
//   * The namespace is fak_vllm_mla; the shared entry-base/source scatter helpers
//     live in paged_kv_write_map.h so the host regression executes the exact
//     scatter the kernels run.
//
// Cache layout (upstream): kv_cache [num_blocks, block_size, (kv_lora_rank +
// pe_dim)] - one concatenated [latent | rope] entry per position. The ds family
// keeps upstream's fixed-row shape in the f32 lane: a reserved 64-wide rope tail
// that is always physically present and zeroed when pe_dim is 0.

#include <stdint.h>
#include "paged_kv_write_map.h"

namespace fak_vllm_mla {

using fak_vllm_pa::MlaLayerBases;
using fak_vllm_pa::PagedKVWriteMap;

// Grid: (tokens). One block per token; threads stride the entry width.
__global__ void mla_concat_cache_pages_kernel(
    const float* __restrict__ kv_c, const float* __restrict__ k_pe,
    float* __restrict__ kv_cache, const PagedKVWriteMap pages, int kv_lora_rank,
    int pe_dim, int block_size) {
  const int token = blockIdx.x;
  const int entry = kv_lora_rank + pe_dim;
  const int64_t base = fak_vllm_pa::mla_kv_entry_base(pages, token, entry, block_size);
  for (int i = threadIdx.x; i < entry; i += blockDim.x) {
    kv_cache[base + i] =
        fak_vllm_pa::mla_kv_entry_source(kv_c, k_pe, token, i, kv_lora_rank, pe_dim);
  }
}

// Grid: (tokens, layers). Layer y selects its arena base from the bounded
// by-value pointer array; the write map is shared across layers.
__global__ void mla_concat_cache_grouped_pages_kernel(
    const float* __restrict__ kv_c, const float* __restrict__ k_pe, MlaLayerBases bases,
    const PagedKVWriteMap pages, int kv_lora_rank, int pe_dim, int block_size,
    int c_layer_stride, int p_layer_stride) {
  const int token = blockIdx.x;
  const int layer = blockIdx.y;
  const float* c = kv_c + int64_t(layer) * c_layer_stride;
  const float* p = k_pe + int64_t(layer) * p_layer_stride;
  float* cache = bases.layers[layer];
  const int entry = kv_lora_rank + pe_dim;
  const int64_t base = fak_vllm_pa::mla_kv_entry_base(pages, token, entry, block_size);
  for (int i = threadIdx.x; i < entry; i += blockDim.x) {
    cache[base + i] =
        fak_vllm_pa::mla_kv_entry_source(c, p, token, i, kv_lora_rank, pe_dim);
  }
}

// Grid: (tokens); Block: (96). Upstream's fixed ds-family partition: 64 threads
// tile the 512-wide latent at 8 lanes each, 32 threads the reserved 64-wide rope
// tail at 2 lanes each (zeroed when pe_dim is 0).
__global__ void mla_concat_cache_ds_pages_kernel(
    const float* __restrict__ kv_c, const float* __restrict__ k_pe,
    float* __restrict__ kv_cache, const PagedKVWriteMap pages, int kv_lora_rank,
    int pe_dim, int block_size) {
  fak_vllm_pa::mla_kv_write_ds_thread(pages, blockIdx.x, threadIdx.x, kv_lora_rank,
                                      pe_dim, block_size, kv_c, k_pe, kv_cache);
}

}  // namespace fak_vllm_mla

extern "C" int fcuda_mla_concat_cache_f32(float* kv_cache, const float* kv_c,
                                          const float* k_pe, const int* block_ids,
                                          int num_blocks, int capacity_blocks,
                                          int block_offset, int tokens,
                                          int kv_lora_rank, int pe_dim, int block_size) {
  if (!kv_cache || !kv_c || !k_pe ||
      !fak_vllm_pa::mla_kv_write_map_valid(block_ids, num_blocks, capacity_blocks,
          block_offset, tokens, kv_lora_rank + pe_dim, block_size)) {
    return 1;
  }
  fak_vllm_pa::PagedKVWriteMap pages = {};
  pages.first_offset = block_offset;
  for (int i = 0; i < num_blocks; ++i) pages.blocks[i] = block_ids[i];
  const int entry = kv_lora_rank + pe_dim;
  const int threads = entry < 512 ? entry : 512;
  cudaError_t launch_status = cudaSuccess;
  FCUDA_TS("fak_mla_concat_cache", tokens,
           (fak_vllm_mla::mla_concat_cache_pages_kernel<<<tokens, threads, 0, g_stream>>>(
               kv_c, k_pe, kv_cache, pages, kv_lora_rank, pe_dim, block_size));
           launch_status = cudaGetLastError());
  return launch_status == cudaSuccess ? 0 : 4;
}

extern "C" int fcuda_mla_concat_cache_grouped_f32(const void* host_layer_bases, int layers,
                                                  const float* kv_c, const float* k_pe,
                                                  const int* block_ids, int num_blocks,
                                                  int capacity_blocks, int block_offset,
                                                  int tokens, int kv_lora_rank, int pe_dim,
                                                  int block_size, int c_layer_stride,
                                                  int p_layer_stride) {
  if (!host_layer_bases || layers <= 0 || layers > fak_vllm_pa::kMlaMaxLayers ||
      !kv_c || !k_pe ||
      !fak_vllm_pa::mla_kv_write_map_valid(block_ids, num_blocks, capacity_blocks,
          block_offset, tokens, kv_lora_rank + pe_dim, block_size)) {
    return 1;
  }
  fak_vllm_pa::MlaLayerBases bases = {};
  const void* const* host = (const void* const*)host_layer_bases;
  for (int i = 0; i < layers; ++i) {
    if (!host[i]) return 1;
    bases.layers[i] = (float*)host[i];
  }
  fak_vllm_pa::PagedKVWriteMap pages = {};
  pages.first_offset = block_offset;
  for (int i = 0; i < num_blocks; ++i) pages.blocks[i] = block_ids[i];
  const int entry = kv_lora_rank + pe_dim;
  const int threads = entry < 512 ? entry : 512;
  dim3 grid(tokens, layers);
  cudaError_t launch_status = cudaSuccess;
  FCUDA_TS("fak_mla_concat_cache_grouped", tokens,
           (fak_vllm_mla::mla_concat_cache_grouped_pages_kernel<<<grid, threads, 0, g_stream>>>(
               kv_c, k_pe, bases, pages, kv_lora_rank, pe_dim, block_size, c_layer_stride,
               p_layer_stride));
           launch_status = cudaGetLastError());
  return launch_status == cudaSuccess ? 0 : 4;
}

extern "C" int fcuda_mla_concat_cache_ds_f32(float* kv_cache, const float* kv_c,
                                             const float* k_pe, const int* block_ids,
                                             int num_blocks, int capacity_blocks,
                                             int block_offset, int tokens,
                                             int kv_lora_rank, int pe_dim, int block_size) {
  // ds family contract (upstream launcher checks): 512-wide latent, rope 0 or 64.
  if (kv_lora_rank != 512 || (pe_dim != 0 && pe_dim != 64)) return 2;
  if (!kv_cache || !kv_c || (pe_dim != 0 && !k_pe) ||
      !fak_vllm_pa::mla_kv_write_map_valid(block_ids, num_blocks, capacity_blocks,
          block_offset, tokens, kv_lora_rank + 64, block_size)) {
    return 1;
  }
  fak_vllm_pa::PagedKVWriteMap pages = {};
  pages.first_offset = block_offset;
  for (int i = 0; i < num_blocks; ++i) pages.blocks[i] = block_ids[i];
  cudaError_t launch_status = cudaSuccess;
  FCUDA_TS("fak_mla_concat_cache_ds", tokens,
           (fak_vllm_mla::mla_concat_cache_ds_pages_kernel<<<tokens, 96, 0, g_stream>>>(
               kv_c, k_pe, kv_cache, pages, kv_lora_rank, pe_dim, block_size));
           launch_status = cudaGetLastError());
  return launch_status == cudaSuccess ? 0 : 4;
}
