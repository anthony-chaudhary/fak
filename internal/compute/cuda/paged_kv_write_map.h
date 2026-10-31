#ifndef FAK_PAGED_KV_WRITE_MAP_H
#define FAK_PAGED_KV_WRITE_MAP_H

// Slot/block addressing adapted from vLLM csrc/cache_kernels.cu at
// e5196e2959a42908313510f55fa21f213a48802e (Apache-2.0).
// Copyright (c) 2023, The vLLM team. See THIRD_PARTY_NOTICES.md.
// Fak uses contiguous logical rows and a bounded by-value physical page map,
// rather than vLLM's device slot_mapping tensor; F32 K/V layouts are unchanged.
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
#undef FAK_PAGED_WRITE_INLINE
}  // namespace fak_vllm_pa
#endif
