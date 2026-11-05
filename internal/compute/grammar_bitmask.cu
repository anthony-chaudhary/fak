// grammar_bitmask.cu — grammar bitmask device application.
//
// ADAPTED from vllm-project/vllm,
// vllm/v1/worker/gpu/structured_outputs.py:133 (_apply_grammar_bitmask_kernel)
// at ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0; attribution in
// THIRD_PARTY_NOTICES.md), lineage xgrammar
// apply_token_bitmask_inplace_triton (mlc-ai/xgrammar
// python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py): a packed
// int32 bitmask (32 tokens per word, bit=0 => forbidden) applied to
// device-resident f32 logits rows addressed by an explicit row-index vector.
// Re-expressed against fak conventions (extern "C" fcuda_* entry, int status
// returns, explicit stream argument); no upstream code copied.
// Ref oss-port-vllmtop-grammar-bitmask-device-20261104.
//
// Not yet #included into cuda_kernels.cu: that one-line build wiring is a
// follow-on ticket beyond this candidate's path cap. The Go oracle in
// grammar_bitmask.go pins the observable contract (packing, tail-word guard,
// row-index addressing, -inf placement).
//
// Deltas from upstream:
//   - The triton program grid (num_masks, cdiv(vocab_size, BLOCK_SIZE=8192))
//     becomes a CUDA grid (num_rows, cdiv(words_per_row, 128)): one thread
//     owns one packed word and its 32 tokens, mirroring upstream's one-word
//     load per 32-token chunk.
//   - Upstream decomposes the flat-logit mapping via MASK_STRIDE +
//     cu_num_logits to skip adaptively-trimmed draft positions; here the
//     caller pre-folds that decomposition into the row-index vector, so the
//     kernel receives target row indices directly.
//   - Empty-row collapse to a forced stop token happens on the host at pack
//     time (PackGrammarBitmask) before H2D; the kernel applies the packed
//     words verbatim.
//   - No GPU or throughput claim attaches to this candidate.

#include <cuda_runtime.h>
#include <cstdint>

#define GB_OK 0
#define GB_BAD_ARGS 1
#define GB_CUDA_ERROR 4

// One thread owns one packed word and its 32 consecutive tokens; the column
// guard keeps the tail word from writing past vocab_size.
__global__ void k_grammar_bitmask_apply(float* __restrict__ logits,
                                        const int32_t* __restrict__ rows,
                                        const int32_t* __restrict__ bitmask,
                                        int vocab_size, int words_per_row) {
  const int row = blockIdx.x;
  const int word_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (word_idx >= words_per_row) return;
  const int col0 = word_idx * 32;
  if (col0 >= vocab_size) return;
  const int32_t word = bitmask[(size_t)row * words_per_row + word_idx];
  float* orow = logits + (size_t)rows[row] * vocab_size;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int col = col0 + i;
    if (col < vocab_size && (((word >> i) & 1) == 0))
      orow[col] = __int_as_float(0xff800000);  // -inf, exact
  }
}

// rows addresses num_rows target rows of the flat logits buffer; bitmask holds
// num_rows * ceil(vocab_size/32) packed words. Forbidden tokens are stored as
// -inf in place; allowed tokens are never touched.
extern "C" int fcuda_grammar_bitmask_apply(float* logits,
                                           const int32_t* rows,
                                           const int32_t* bitmask,
                                           int num_rows, int vocab_size,
                                           cudaStream_t stream) {
  if (!logits || !rows || !bitmask || num_rows <= 0 || vocab_size <= 0)
    return GB_BAD_ARGS;
  const int words_per_row = (vocab_size + 31) / 32;
  dim3 grid(num_rows, (words_per_row + 127) / 128);
  cudaError_t err = k_grammar_bitmask_apply<<<grid, 128, 0, stream>>>(
      logits, rows, bitmask, vocab_size, words_per_row);
  if (err != cudaSuccess) return GB_CUDA_ERROR;
  return cudaPeekAtLastError() == cudaSuccess ? GB_OK : GB_CUDA_ERROR;
}

#undef GB_OK
#undef GB_BAD_ARGS
#undef GB_CUDA_ERROR
