#ifndef FAK_PAGED_KV_WRITE_MAP_H
#define FAK_PAGED_KV_WRITE_MAP_H

// Slot/block addressing adapted from vLLM at
// e5196e2959a42908313510f55fa21f213a48802e (Apache-2.0).
// Copyright (c) 2023, The vLLM team. License and notice: THIRD_PARTY_NOTICES.md
// ("vLLM").
// Fak uses contiguous logical rows and a bounded by-value physical page map;
// F32 K/V layouts are unchanged.
#include <limits.h>
#include <stdint.h>

namespace fak_vllm_pa {
constexpr int kWriteMaxPages = 64;
static_assert(sizeof(int) == sizeof(int32_t), "Go page IDs require 32-bit C int");
struct PagedKVWriteMap {
  int blocks[kWriteMaxPages];
  int first_offset;
};
static_assert(sizeof(PagedKVWriteMap) + 6 * sizeof(void*) + 3 * sizeof(int) < 4096,
              "paged KV writer must fit the legacy CUDA parameter limit");

// Host admission before any launch. The arena extent is checked in bytes, and
// duplicate pages are refused because unordered blocks must not write one slot.
inline bool paged_kv_write_map_valid(const int* blocks, int count, int capacity,
                                     int first_offset, int tokens, int num_heads,
                                     int head_size, int block_size) {
  if (!blocks || count <= 0 || count > kWriteMaxPages || capacity <= 0 ||
      block_size != 16 || first_offset < 0 || first_offset >= block_size ||
      tokens <= 0 || num_heads <= 0 || head_size <= 0 || head_size % 4 != 0 ||
      num_heads > INT_MAX / head_size) return false;
  const int64_t end = int64_t(first_offset) + tokens;
  if ((end - 1) / block_size + 1 != count) return false;
  const int64_t page_bytes = int64_t(num_heads) * head_size * block_size * sizeof(float);
  if (int64_t(capacity) > INT64_MAX / page_bytes) return false;
  for (int i = 0; i < count; ++i) {
    if (blocks[i] < 0 || blocks[i] >= capacity) return false;
    for (int j = 0; j < i; ++j) if (blocks[j] == blocks[i]) return false;
  }
  return true;
}

#ifdef __CUDACC__
#define FAK_PAGED_WRITE_INLINE __host__ __device__ __forceinline__
#else
#define FAK_PAGED_WRITE_INLINE inline
#endif

// Shared verbatim by the real CUDA writer and its host address regression. Inputs
// have passed admission above; token and element are within the launched grid/row.
FAK_PAGED_WRITE_INLINE void paged_kv_write_offsets(
    const PagedKVWriteMap& pages, int token, int element, int num_heads,
    int head_size, int block_size, int64_t* key_offset, int64_t* value_offset) {
  const int pos = pages.first_offset + token;
  const int64_t block_idx = pages.blocks[pos / block_size];
  const int block_offset = pos % block_size;
  const int head_idx = element / head_size;
  const int head_offset = element % head_size;
  constexpr int x = 4;
  *key_offset = block_idx * num_heads * (head_size / x) * block_size * x +
      int64_t(head_idx) * (head_size / x) * block_size * x +
      int64_t(head_offset / x) * block_size * x + block_offset * x + head_offset % x;
  *value_offset = block_idx * num_heads * head_size * block_size +
      int64_t(head_idx) * head_size * block_size + int64_t(head_offset) * block_size + block_offset;
}

// ---- MLA concat-and-cache addressing ------------------------------------------
// Adapted from vLLM at ae065918640a198a56f6f17eb45e251c3229fe41
// (Apache-2.0): the MLA concat-and-cache entry geometry. The MLA cache holds
// one concatenated [latent | rope] entry per position - entry width
// kv_lora_rank + pe_dim floats - laid out [num_blocks, block_size, entry]. A physical
// page id addresses the same block in every layer's arena, so one write map drives
// every layer (the per-session block table invariant of the standard writer).
constexpr int kMlaMaxLayers = 64;
struct MlaLayerBases {
  float* layers[kMlaMaxLayers];
};
static_assert(sizeof(MlaLayerBases) + sizeof(PagedKVWriteMap) + 8 * sizeof(int) < 4096,
              "MLA grouped writer must fit the legacy CUDA parameter limit");

// Host admission before any launch: the standard page-map rules with the arena
// extent checked against the concatenated entry width instead of the K/V head
// geometry. Duplicate pages are refused because unordered blocks must not write
// one slot.
inline bool mla_kv_write_map_valid(const int* blocks, int count, int capacity,
                                   int first_offset, int tokens, int entry_width,
                                   int block_size) {
  if (!blocks || count <= 0 || count > kWriteMaxPages || capacity <= 0 ||
      block_size != 16 || first_offset < 0 || first_offset >= block_size ||
      tokens <= 0 || entry_width <= 0) return false;
  const int64_t end = int64_t(first_offset) + tokens;
  if ((end - 1) / block_size + 1 != count) return false;
  if (int64_t(entry_width) > INT64_MAX / block_size) return false;
  const int64_t page_floats = int64_t(block_size) * entry_width;
  if (int64_t(capacity) > INT64_MAX / page_floats) return false;
  for (int i = 0; i < count; ++i) {
    if (blocks[i] < 0 || blocks[i] >= capacity) return false;
    for (int j = 0; j < i; ++j) if (blocks[j] == blocks[i]) return false;
  }
  return true;
}

// Shared verbatim by the MLA CUDA writers and their host regression: the arena
// offset of one token's entry start. Inputs have passed admission above.
FAK_PAGED_WRITE_INLINE int64_t mla_kv_entry_base(const PagedKVWriteMap& pages, int token,
                                                 int entry_width, int block_size) {
  const int pos = pages.first_offset + token;
  return int64_t(pages.blocks[pos / block_size]) * int64_t(block_size) * entry_width +
         int64_t(pos % block_size) * entry_width;
}

// Shared verbatim by the MLA CUDA writers and their host regression: the source
// lane of one entry element - latent lanes first, then rope lanes - exactly the
// upstream concat order into the cache slot.
FAK_PAGED_WRITE_INLINE float mla_kv_entry_source(const float* kv_c, const float* k_pe,
                                                 int token, int element, int kv_lora_rank,
                                                 int pe_dim) {
  if (element < kv_lora_rank) return kv_c[int64_t(token) * kv_lora_rank + element];
  return k_pe[int64_t(token) * pe_dim + (element - kv_lora_rank)];
}

// Shared verbatim by the ds-family CUDA writer and its host regression: one
// thread's fixed share of one token's ds entry. Upstream's partition is kept:
// 64 threads tile the 512-wide latent at 8 lanes each, 32 threads tile the
// reserved 64-wide rope tail at 2 lanes each. The rope tail is always physically
// present in the ds entry (entry width kv_lora_rank + 64) and is zeroed when
// pe_dim is 0; the quantized-cache arms of the upstream kernel are dropped.
FAK_PAGED_WRITE_INLINE void mla_kv_write_ds_thread(const PagedKVWriteMap& pages, int token,
                                                   int thread, int kv_lora_rank, int pe_dim,
                                                   int block_size, const float* kv_c,
                                                   const float* k_pe, float* kv_cache) {
  const int entry = kv_lora_rank + 64;
  const int64_t base = mla_kv_entry_base(pages, token, entry, block_size);
  if (thread < 64) {
    const int64_t src = int64_t(token) * kv_lora_rank + thread * 8;
    for (int i = 0; i < 8; ++i) kv_cache[base + thread * 8 + i] = kv_c[src + i];
  } else {
    const int rope_idx = (thread - 64) * 2;
    const int64_t dst = base + kv_lora_rank + rope_idx;
    if (pe_dim == 0) {
      kv_cache[dst] = 0.f;
      kv_cache[dst + 1] = 0.f;
    } else {
      const int64_t src = int64_t(token) * pe_dim + rope_idx;
      kv_cache[dst] = k_pe[src];
      kv_cache[dst + 1] = k_pe[src + 1];
    }
  }
}
#undef FAK_PAGED_WRITE_INLINE
}  // namespace fak_vllm_pa
#endif
