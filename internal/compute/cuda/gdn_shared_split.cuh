// Shared activation split for the retained Q8 GDN sequence producer (#14097).
// The surrounding sequence adapts llama.cpp gated_delta_net.cu at
// 0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd (MIT; existing notices retained).
// This is original orchestration of Fak's existing compensated kernels, not a
// new upstream arithmetic port. Include after q8_dequant_sgemm.
#include <stdint.h>
#include <limits.h>

namespace fak_gdn_shared_split {
struct span { uintptr_t begin, end; };
static bool extent(const void *p, size_t bytes, size_t alignment, span *s) {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(p);
  if (!p || !bytes || begin % alignment || bytes > UINTPTR_MAX - begin) return false;
  *s = {begin, begin + bytes};
  return true;
}
static bool overlap(span a, span b) { return a.begin < b.end && b.begin < a.end; }

static bool selected(const char *mode, const char *sequence_arm, bool capture, int tokens) {
  if (capture || tokens < 128 || tokens > 65535 || !mode || strcmp(mode, "shared-split") != 0)
    return false;
#ifdef FAK_CUDA_Q8_MMQ_PREFILL
  if (sequence_arm && strcmp(sequence_arm, "mmq") == 0) return false;
#else
  (void)sequence_arm;
#endif
  return true;
}

// No tensor is touched before every external read/write interval is checked.
// Read-only inputs may overlap. Outputs may overlap neither reads nor outputs.
static bool ranges(const void *const codes[4], const float *const scales[4],
    const float *x, float *const y[4], const int rows[4], int cols, int tokens,
    span reads[9], span writes[4], size_t *maxW, size_t *xn) {
  if (!codes || !scales || !y || !rows || !maxW || !xn || cols <= 0 || cols % 32 ||
      tokens < 128 || tokens > 65535 || tokens > INT_MAX / cols) return false;
  *maxW = 0;
  *xn = (size_t)tokens * cols;
  if (*xn > SIZE_MAX / sizeof(float) ||
      !extent(x, *xn * sizeof(float), alignof(float), &reads[8])) return false;
  for (int i = 0; i < 4; ++i) {
    if (rows[i] <= 0 || rows[i] > INT_MAX / cols || tokens > INT_MAX / rows[i]) return false;
    const size_t wn = (size_t)rows[i] * cols, yn = (size_t)tokens * rows[i];
    if (wn > SIZE_MAX / sizeof(__half) || wn / 32 > SIZE_MAX / sizeof(float) ||
        yn > SIZE_MAX / sizeof(float) ||
        !extent(codes[i], wn, 1, &reads[2*i]) ||
        !extent(scales[i], wn / 32 * sizeof(float), alignof(float), &reads[2*i+1]) ||
        !extent(y[i], yn * sizeof(float), alignof(float), &writes[i])) return false;
    if (wn > *maxW) *maxW = wn;
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 9; ++j) if (overlap(writes[i], reads[j])) return false;
    for (int j = 0; j < i; ++j) if (overlap(writes[i], writes[j])) return false;
  }
  return true;
}
} // namespace fak_gdn_shared_split

// One scratch acquisition covers all four producers. The activation halves are
// immutable after their sole split; weight halves are overwritten only after the
// preceding three GEMMs in the same stream. No cache, extra owner or growth rule.
static int qwen35_gdn_q8_shared_split(const void *const codes[4],
    const float *const scales[4], const float *x, float *const y[4],
    const int rows[4], int cols, int tokens) {
  using namespace fak_gdn_shared_split;
  if (g_cuda_cleanup_status != 0) return 70000 + g_cuda_cleanup_status;
#ifdef FAK_CUDA_V41_CHECKED_STREAM
  if (v41_candidates_stream_uncertain()) return 70000 + (int)cudaErrorNotReady;
#endif
  if (g_capture_open || !g_stream ||
      (g_sequence_split.phase != SequenceSplitPhase::Idle &&
       g_sequence_split.phase != SequenceSplitPhase::Active)) return -1;
  span reads[9], writes[4];
  size_t maxW = 0, xn = 0;
  if (!ranges(codes, scales, x, y, rows, cols, tokens, reads, writes, &maxW, &xn)) return -1;
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return 20000 + (int)error;
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  error = cudaStreamIsCapturing(g_stream, &capture);
  if (error != cudaSuccess) return 20000 + (int)error;
  if (capture != cudaStreamCaptureStatusNone) return -1;
  FP16CompensatedScratch scratch;
  int status = fp16_compensated_scratch(maxW, xn, &scratch);
  if (status != 0) return status;
  // The allocator/loan remains authoritative. These additional interval checks
  // reject a malformed owner before any split can corrupt an operand.
  span owned[4];
  for (int i = 0; i < 4 && status == 0; ++i) {
    if (!extent(scratch.ptr[i], (i < 2 ? maxW : xn) * sizeof(__half), alignof(__half), &owned[i])) {
      status = -1;
      break;
    }
    for (int j = 0; j < 9; ++j) if (overlap(owned[i], reads[j])) status = -1;
    for (int j = 0; j < 4; ++j) if (overlap(owned[i], writes[j])) status = -1;
    for (int j = 0; j < i; ++j) if (overlap(owned[i], owned[j])) status = -1;
  }
  __half *wHi = scratch.ptr[0], *wLo = scratch.ptr[1], *xHi = scratch.ptr[2], *xLo = scratch.ptr[3];
  if (status == 0) {
    FCUDA_TS("k_fp16_split", (xn + 255) / 256, k_fp16_split<<<(xn + 255) / 256, 256, 0, g_stream>>>(x, xHi, xLo, xn));
    error = cudaGetLastError();
    if (error != cudaSuccess) status = 20000 + (int)error;
  }
  for (int i = 0; i < 4 && status == 0; ++i) {
    const size_t wn = (size_t)rows[i] * cols;
    FCUDA_TS("k_q8_dequant_split", (wn + 255) / 256, k_q8_dequant_split<<<(wn + 255) / 256, 256, 0, g_stream>>>(
        (const signed char *)codes[i], scales[i], wHi, wLo, wn, cols, 32));
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      status = 20000 + (int)error;
      break;
    }
    status = fp16_compensated_gemm(wHi, wLo, xHi, xLo, y[i], rows[i], cols, tokens);
  }
  const int release_status = fp16_compensated_release(&scratch);
  // No post-submit fallback. The caller drains all submitted work and invalidates
  // the request through its existing error path; preserve the primary failure.
  return status != 0 ? status : release_status;
}
