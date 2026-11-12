// dsv41_k_quant.cu — DSv4.1 mxfp8 K-plane quantize-and-insert +
// dequantize-and-gather.
//
// Adapted from vLLM at ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0).
// License and notice: THIRD_PARTY_NOTICES.md ("vLLM").
// The V4.1 all-dims MXFP8 record (512 e4m3 payload bytes + 16 ue8m0 scales of
// 32 dims each) is quantized at paged-cache insert and dequantized during
// decode-attention gather.
// Ref oss-port-vllmtop-dsv41-k-quant-cache-20261103.
//
// Not yet #included into cuda_kernels.cu: that one-line build wiring is a
// follow-on ticket beyond this candidate's path cap. The Go oracle in
// dsv41_k_quant.go pins the observable contract (record bytes, ue8m0 scale
// encoding, paged slot addressing, one-pass/per-block gather parity).
//
// Local behavior:
//   - Paged pages hold 16 tokens (the fak paged writer admits block_size ==
//     16 only). Record layout is unchanged.
//   - The gather grid width is workers = clamp(ceil(max_gather_len/8), 1,
//     128); no GPU or throughput claim attaches to this candidate.
//   - f32 K remains the default plane and the oracle; this candidate adds the
//     quantized K-plane dtype option without touching the f32 write map.

#include <cuda_runtime.h>
#include <cstdint>
#include "cuda/dsv41_k_quant_vllm.cuh"

using namespace fak_dsv41_kq;

#define DSV41_KQ_OK 0
#define DSV41_KQ_BAD_ARGS 1
#define DSV41_KQ_CUDA_ERROR 4

// One thread owns four consecutive payload dims; threads 8*t..8*t+7 share
// tile t, so the 8-lane xor reduction yields the tile absmax without shared
// memory. All 128 lanes reach every shuffle.
__global__ void k_dsv41_kq_quant_insert_mxfp8(const float* __restrict__ k,
                                              uint8_t* __restrict__ arena,
                                              const PagedKVWriteMap pages) {
  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int dim0 = tid * 4;
  const int tile = dim0 / kQuantBlock;
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = k[(size_t)token * kHeadDim + dim0 + i];
  float amax = fmaxf(fabsf(v[0]), fmaxf(fabsf(v[1]), fmaxf(fabsf(v[2]), fabsf(v[3]))));
#pragma unroll
  for (int mask = 4; mask > 0; mask >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask));
  const int exp = dsv41_kq_ue8m0_exponent(amax);
  const float inv = exp2f(-((float)exp));  // 2^-exp, exact
  uint8_t* rec = arena + dsv41_kq_insert_slot(pages, token);
#pragma unroll
  for (int i = 0; i < 4; ++i) rec[dim0 + i] = dsv41_kq_e4m3_encode(v[i] * inv);
  if ((tid & 7) == 0) rec[kHeadDim + tile] = dsv41_kq_encode_scale(exp);
}

// One-pass form: the whole 528-byte record is pulled before any tile is
// decoded; each thread decodes its four payload bytes with its tile's scale.
__global__ void k_dsv41_kq_dequant_gather_mxfp8_one_pass(
    float* __restrict__ out, const uint8_t* __restrict__ arena,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    const int* __restrict__ gather_lens, int max_blocks_per_seq, int out_rows,
    int offset) {
  const int seq = blockIdx.x;
  const int worker = blockIdx.y;
  const int nworkers = gridDim.y;
  const int seq_len = seq_lens[seq];
  const int gather_len = gather_lens ? gather_lens[seq] : seq_len;
  const int start = seq_len - gather_len;
  const int tid = threadIdx.x;
  const int dim0 = tid * 4;
  const int tile = dim0 / kQuantBlock;
  const int* table = block_tables + (size_t)seq * max_blocks_per_seq;
  for (int i = worker; i < gather_len; i += nworkers) {
    const int pos = start + i;
    const uint8_t* rec = arena + dsv41_kq_gather_slot(table, pos);
    const float scale = dsv41_kq_decode_scale(rec[kHeadDim + tile]);
    float* orow = out + ((size_t)seq * out_rows + offset + i) * kHeadDim;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      orow[dim0 + j] = dsv41_kq_e4m3_decode(rec[dim0 + j]) * scale;
  }
}

// Per-tile form: one load/store pair per 32-dim tile, mirroring upstream's
// per-quant-block loop. It must agree element-wise with the one-pass form.
__global__ void k_dsv41_kq_dequant_gather_mxfp8_per_block(
    float* __restrict__ out, const uint8_t* __restrict__ arena,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    const int* __restrict__ gather_lens, int max_blocks_per_seq, int out_rows,
    int offset) {
  const int seq = blockIdx.x;
  const int worker = blockIdx.y;
  const int nworkers = gridDim.y;
  const int seq_len = seq_lens[seq];
  const int gather_len = gather_lens ? gather_lens[seq] : seq_len;
  const int start = seq_len - gather_len;
  const int tid = threadIdx.x;
  const int* table = block_tables + (size_t)seq * max_blocks_per_seq;
  for (int i = worker; i < gather_len; i += nworkers) {
    const int pos = start + i;
    const uint8_t* rec = arena + dsv41_kq_gather_slot(table, pos);
    float* orow = out + ((size_t)seq * out_rows + offset + i) * kHeadDim;
#pragma unroll
    for (int tile = 0; tile < kNumScales; ++tile) {
      if (tid < kQuantBlock) {
        const int dim = tile * kQuantBlock + tid;
        const float scale = dsv41_kq_decode_scale(rec[kHeadDim + tile]);
        orow[dim] = dsv41_kq_e4m3_decode(rec[dim]) * scale;
      }
    }
  }
}

// Admission mirrors paged_kv_write_map_valid for the mxfp8 plane: the f32
// head geometry terms are dropped because the record carries no head split,
// everything else (count cap, range, distinctness, contiguous coverage) is
// identical.
extern "C" int fcuda_dsv41_kq_quant_insert_mxfp8(
    const float* k, const int32_t* blocks, int num_blocks, int capacity_blocks,
    int first_offset, int tokens, uint8_t* arena, cudaStream_t stream) {
  if (!k || !arena || !blocks || num_blocks < 1 ||
      num_blocks > fak_vllm_pa::kWriteMaxPages || capacity_blocks <= 0 ||
      first_offset < 0 || first_offset >= kPageSize || tokens <= 0) {
    return DSV41_KQ_BAD_ARGS;
  }
  const int64_t end = int64_t(first_offset) + tokens;
  if ((end - 1) / kPageSize + 1 != num_blocks) return DSV41_KQ_BAD_ARGS;
  for (int i = 0; i < num_blocks; ++i) {
    if (blocks[i] < 0 || blocks[i] >= capacity_blocks) return DSV41_KQ_BAD_ARGS;
    for (int j = 0; j < i; ++j)
      if (blocks[j] == blocks[i]) return DSV41_KQ_BAD_ARGS;
  }
  PagedKVWriteMap pages = {};
  pages.first_offset = first_offset;
  for (int i = 0; i < num_blocks; ++i) pages.blocks[i] = blocks[i];
  cudaError_t err =
      k_dsv41_kq_quant_insert_mxfp8<<<tokens, 128, 0, stream>>>(k, arena, pages);
  return err == cudaSuccess ? DSV41_KQ_OK : DSV41_KQ_CUDA_ERROR;
}

// gather_lens may be null, which gathers every token of each sequence like
// upstream's absent-pointer arm. max_gather_len is the host-side upper bound
// on any gather length and sizes the worker grid only.
extern "C" int fcuda_dsv41_kq_dequant_gather_mxfp8(
    float* out, const uint8_t* arena, const int32_t* block_tables,
    const int32_t* seq_lens, const int32_t* gather_lens, int seqs,
    int max_blocks_per_seq, int out_rows, int offset, int max_gather_len,
    int one_pass, cudaStream_t stream) {
  if (!out || !arena || !block_tables || !seq_lens || seqs < 1 ||
      max_blocks_per_seq < 1 || out_rows < 1 || offset < 0 ||
      offset >= out_rows || max_gather_len < 0) {
    return DSV41_KQ_BAD_ARGS;
  }
  int workers = (max_gather_len + 7) / 8;
  if (workers < 1) workers = 1;
  if (workers > 128) workers = 128;
  dim3 grid(seqs, workers);
  cudaError_t err;
  if (one_pass) {
    err = k_dsv41_kq_dequant_gather_mxfp8_one_pass<<<grid, 128, 0, stream>>>(
        out, arena, block_tables, seq_lens, gather_lens, max_blocks_per_seq,
        out_rows, offset);
  } else {
    err = k_dsv41_kq_dequant_gather_mxfp8_per_block<<<grid, 128, 0, stream>>>(
        out, arena, block_tables, seq_lens, gather_lens, max_blocks_per_seq,
        out_rows, offset);
  }
  return err == cudaSuccess ? DSV41_KQ_OK : DSV41_KQ_CUDA_ERROR;
}

#undef DSV41_KQ_OK
#undef DSV41_KQ_BAD_ARGS
#undef DSV41_KQ_CUDA_ERROR
