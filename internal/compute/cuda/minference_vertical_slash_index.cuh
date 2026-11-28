// ---- Vertical/slash -> block-sparse attention metadata builder ----
// Adapted from microsoft/MInference csrc/vertical_slash_index.cu at commit
// Copyright (c) Microsoft Corporation.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Port contract: upstream save_blocks and convert_vertical_slash_indexes_kernel
// are kept line for line and renamed into the fak_minference namespace:
//   save_blocks                           -> vs_save_blocks
//   convert_vertical_slash_indexes_kernel -> vertical_slash_index_kernel
//   convert_vertical_slash_indexes_64x64  -> fcuda_vertical_slash_index_64x64
// The pybind/torch wrapper is dropped; its shape derivation and zero-filled
// outputs are the caller's contract (see the Go host mirror
// BuildVerticalSlashIndexHost in vertical_slash_index.go, which executes this
// same merge on the CPU).
//
// Caller contract, unchanged from upstream: per head, vertical_indexes is
// ascending and slash_indexes is descending and non-negative; NNZ_V >= 1 and
// NNZ_S >= 1; all four outputs are zero-filled before launch; seqlens holds one
// entry per batch row.
//
// One deviation: upstream reads past the slash row when no slash diagonal lies
// below a row block's end. Here that row block is left empty instead.
//
// Staged kernel: not included from cuda_kernels.cu and not launched by any
// route. Device compilation and execution are unrun.

#include <stdint.h>

namespace fak_minference {

__device__ void vs_save_blocks(int* block_offset, int range_start, int range_end,
                               int block_size, int& block_count) {
  for (int idx = range_start; idx < range_end; idx += block_size) {
    block_offset[block_count++] = idx;
  }
}

// Grid: (heads, batch, ceil(rows / threads)). One thread per 64-row block.
__global__ void vertical_slash_index_kernel(
    const int* seqlens,           // [BATCH, ]
    const int* vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    const int* slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int* block_count,             // [BATCH, N_HEADS, N_ROWS]
    int* block_offset,            // [BATCH, N_HEADS, N_ROWS, NNZ_S]
    int* column_count,            // [BATCH, N_HEADS, N_ROWS]
    int* column_index,            // [BATCH, N_HEADS, N_ROWS, NNZ_V]
    int N_HEADS, int N_ROWS, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int NNZ_V, int NNZ_S) {
  const int batch_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int group_idx = blockIdx.z;

  int seqlen = seqlens[batch_idx];
  int block_idx_m = group_idx * blockDim.x + threadIdx.x;
  int start_m = block_idx_m * BLOCK_SIZE_M;
  if (start_m >= seqlen) {
    return;
  }
  int end_m = start_m + BLOCK_SIZE_M;
  vertical_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_V;
  slash_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_S;
  int row_offset = (batch_idx * N_HEADS + head_idx) * N_ROWS + block_idx_m;
  block_count += row_offset;
  block_offset += row_offset * NNZ_S;
  column_count += row_offset;
  column_index += row_offset * NNZ_V;

  int tmp_col_cnt = 0, tmp_blk_cnt = 0;
  int s = 0, v = 0;
  int v_idx = vertical_indexes[v++];
  int s_idx = slash_indexes[s++];
  while (s_idx >= end_m) {
    if (s >= NNZ_S) {
      return;
    }
    s_idx = slash_indexes[s++];
  }
  s_idx = max(end_m - s_idx, BLOCK_SIZE_M);
  int range_start = s_idx - BLOCK_SIZE_M, range_end = s_idx;
  while (1) {
    if (v_idx < range_end) {
      if (v_idx < range_start) {
        column_index[tmp_col_cnt++] = v_idx;
      }
      if (v < NNZ_V) {
        v_idx = vertical_indexes[v++];
      } else {
        v_idx = end_m + BLOCK_SIZE_M;
      }
    } else {
      if (s < NNZ_S) {
        s_idx = max(end_m - slash_indexes[s++], BLOCK_SIZE_M);
      } else {
        vs_save_blocks(block_offset, range_start, range_end, BLOCK_SIZE_N, tmp_blk_cnt);
        break;
      }
      if (s_idx > range_end + BLOCK_SIZE_M) {
        vs_save_blocks(block_offset, range_start, range_end, BLOCK_SIZE_N, tmp_blk_cnt);
        range_start = s_idx - BLOCK_SIZE_M;
        range_end = s_idx;
      } else if (s_idx > range_end) {
        range_end += BLOCK_SIZE_M;
      }
    }
  }

  block_count[0] = tmp_blk_cnt;
  column_count[0] = tmp_col_cnt;
}

}  // namespace fak_minference

// Returns 0 on a clean launch, 1 on a refused shape, 4 on a launch error.
extern "C" int fcuda_vertical_slash_index_64x64(
    const int* seqlens, const int* vertical_indexes, const int* slash_indexes,
    int* block_count, int* block_offset, int* column_count, int* column_index,
    int batch_size, int n_heads, int n_rows, int nnz_v, int nnz_s) {
  if (!seqlens || !vertical_indexes || !slash_indexes || !block_count ||
      !block_offset || !column_count || !column_index || batch_size <= 0 ||
      n_heads <= 0 || n_rows <= 0 || nnz_v <= 0 || nnz_s <= 0) {
    return 1;
  }
  const int BLOCK_SIZE_M = 64;
  const int BLOCK_SIZE_N = 64;
  const int N_THREADS = 64;
  const dim3 dimBlock(N_THREADS);
  const dim3 dimGrid(n_heads, batch_size, (n_rows + N_THREADS - 1) / N_THREADS);
  fak_minference::vertical_slash_index_kernel<<<dimGrid, dimBlock>>>(
      seqlens, vertical_indexes, slash_indexes, block_count, block_offset,
      column_count, column_index, n_heads, n_rows, BLOCK_SIZE_M, BLOCK_SIZE_N,
      nnz_v, nnz_s);
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}
