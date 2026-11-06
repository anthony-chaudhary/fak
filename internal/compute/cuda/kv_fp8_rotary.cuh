// SPDX-License-Identifier: Apache-2.0
// Reuses Fak k_qwen35_partial_rope_panel at 9b8ea46b, transformer_ops.cuh.
// vLLM FP8 quantization lineage remains in kv_fp8.cuh; this rotary operation
// is NOT copied from vLLM and must not use its scaled-half conversion helpers.
// Unlinked proposal: no TU inclusion or runtime activation in this patch.
#ifndef FAK_CUDA_KV_FP8_ROTARY_CUH
#define FAK_CUDA_KV_FP8_ROTARY_CUH
#include "kv_fp8.cuh"
#include "../cuda_kv_fp8_rotary.h"

namespace fak_kv_fp8 {
__global__ void partial_rope_stage(float *out, const float *raw,
    int tail, int from, int end, int heads, int hd, int rotary,
    double theta, int *status) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t width = (size_t)heads * hd;
  if (i >= (size_t)tail * width) return;
  const int token = (int)(i / width);
  const int within = (int)(i - (size_t)token * width);
  const int head = within / hd, dim = within - head * hd;
  const size_t base = ((size_t)end + token) * width + (size_t)head * hd;
  if (dim >= rotary) {
    const float value = raw[base + dim];
    if (!finite(value)) atomicOr(status, FCUDA_KV_FP8_NONFINITE);
    out[i] = value; // retain the unrotated tail exactly
    return;
  }
  const int half = rotary / 2;
  const int j = dim < half ? dim : dim - half;
  // Keep the producer's pow(theta, negative exponent), double angle/trig,
  // F32 coefficients and rotate-half expression. 1/pow is a DIFFERENT path.
  const double freq = pow(theta, -(2.0 * j) / rotary);
  const double angle = (double)(from + token) * freq;
  const float cs = (float)cos(angle), sn = (float)sin(angle);
  const float a = raw[base + j], b = raw[base + j + half];
  const float value = dim < half ? a * cs - b * sn : b * cs + a * sn;
  if (!finite(a) || !finite(b)) atomicOr(status, FCUDA_KV_FP8_NONFINITE);
  if (!finite(value)) atomicOr(status, FCUDA_KV_FP8_BAD_RESULT);
  out[i] = value;
}
} // namespace fak_kv_fp8

extern "C" int fcuda_kv_fp8_partial_rope_stage_async(
    float *out, const float *raw, int old_positions, int from, int end,
    int heads, int hd, int rotary, double theta, int producer, int *status) {
  if (int closed = fak_kv_fp8::admit()) return closed;
  if (!out || !raw || !status || heads != 4 || hd != 256 || rotary != 64 ||
      producer != FCUDA_KV_FP8_QWEN_PARTIAL_PANEL_V1 ||
      old_positions <= 0 || old_positions > INT_MAX/(heads*hd) ||
      from < 0 || end <= from || end > old_positions ||
      !(theta > 0.0) || !isfinite(theta))
    return (int)cudaErrorInvalidValue;
  const int tail = old_positions-end;
  if (tail == 0) return 0;
  const size_t n = (size_t)tail*heads*hd;
  fak_kv_fp8::partial_rope_stage<<<(n+255)/256,256,0,g_stream>>>(
      out,raw,tail,from,end,heads,hd,rotary,theta,status);
  return fak_kv_fp8::launch_status(cudaGetLastError());
}
#endif
