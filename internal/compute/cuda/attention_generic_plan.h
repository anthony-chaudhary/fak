#pragma once

// Host-only admission and launch planning for the generic tiled f32 attention
// (attention_generic.cuh). Kept free of CUDA types so CPU tests compile it directly.

#include <cstddef>
#include <cstring>

#define ATTN_GENERIC_WARPS 4
#define ATTN_GENERIC_TILE 32 // keys per shared tile: one key per lane in the score phase
#define ATTN_GENERIC_ACC 8   // per-lane output dims (d = lane + 32*i), so hd <= 256
#define ATTN_GENERIC_MAX_HD (32 * ATTN_GENERIC_ACC)
#define ATTN_GENERIC_MAX_SPLITS 128
#define ATTN_GENERIC_CHUNK 64 // keys per softmax chunk below MAX_SPLITS chunks
#define ATTN_GENERIC_DEFAULT_SMEM (48 * 1024)

// FAK_CUDA_ATTN_GENERIC=legacy restores the per-key block-reduction kernels.
static inline bool attn_generic_mode_legacy(const char *mode) {
  return mode != nullptr && std::strcmp(mode, "legacy") == 0;
}

static inline bool attn_generic_admits(int tokens, int prefix, int nH, int nKV, int hd) {
  if (tokens <= 0 || prefix < 0 || nH <= 0 || nKV <= 0 || nH % nKV != 0) return false;
  if (hd <= 0 || hd > ATTN_GENERIC_MAX_HD) return false;
  return tokens <= (int)(0x7fffffff / nH) && prefix <= 0x7fffffff - tokens;
}

// Query rows per warp. A block holds WARPS*slots (token, head) rows of one KV
// head group so every K/V tile staged in shared memory is reused by all of them;
// decode-sized groups use fewer slots so no warp idles on padding rows.
static inline int attn_generic_slots(int groupRows) {
  if (groupRows <= ATTN_GENERIC_WARPS) return 1;
  if (groupRows <= 2 * ATTN_GENERIC_WARPS) return 2;
  return 4;
}

static inline size_t attn_generic_smem_bytes(int hd, int rows) {
  size_t h = (size_t)hd;
  return ((size_t)rows * h                    // query rows
          + (size_t)ATTN_GENERIC_TILE * (h | 1) // K tile, odd stride keeps lane rows bank-disjoint
          + (size_t)ATTN_GENERIC_TILE * h       // V tile
          + (size_t)rows * ATTN_GENERIC_TILE) * // per-row probabilities
         sizeof(float);
}

#if defined(__CUDACC__)
#define ATTN_GENERIC_HD __host__ __device__
#else
#define ATTN_GENERIC_HD
#endif

// Keys per KV chunk for a row that sees n keys. A row's softmax is computed per
// chunk and the chunk states are folded in order, so the result depends only on
// n: a decode step (chunks spread over grid.z) and the matching prompt-panel row
// (chunks folded in-kernel) stay bitwise identical. 64 keys keeps a long decode
// window spread over many blocks; beyond MAX_SPLITS chunks the chunk widens.
static inline ATTN_GENERIC_HD int attn_generic_chunk(int n) {
  int chunk = ATTN_GENERIC_CHUNK;
  if (n > chunk * ATTN_GENERIC_MAX_SPLITS) {
    chunk = (n - 1) / ATTN_GENERIC_MAX_SPLITS + 1;
    chunk = (chunk + ATTN_GENERIC_TILE - 1) / ATTN_GENERIC_TILE * ATTN_GENERIC_TILE;
  }
  return chunk;
}

static inline ATTN_GENERIC_HD int attn_generic_splits(int n) {
  int chunk = attn_generic_chunk(n);
  return n <= 0 ? 1 : (n - 1) / chunk + 1;
}
