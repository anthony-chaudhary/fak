// Persistent [max_position, rot_dim] RoPE cos/sin table and position-id array rotation kernel.
// Adapted from vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0).
// Copyright (c) 2023, The vLLM team. License and notice: THIRD_PARTY_NOTICES.md ("vLLM").
//
// Table layout: [max_position, rot_dim], where each row pos stores
//   cos[0 .. rot_dim/2 - 1] followed by sin[0 .. rot_dim/2 - 1].
// Supports both NeoX (split-half: x[j], x[j + half]) and interleaved (adjacent: x[2*j], x[2*j+1])
// pairings, partial RoPE offsets (rope_dim_offset), and forward or inverse rotation.

#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
__global__ void k_rope_build_cos_sin_cache(float *__restrict__ cos_sin_cache,
                                           int max_position, int rot_dim, double theta) {
  int pos = blockIdx.x;
  if (pos >= max_position) return;
  int half = rot_dim / 2;
  float *row = cos_sin_cache + (size_t)pos * rot_dim;
  for (int j = threadIdx.x; j < half; j += blockDim.x) {
    double inv = 1.0 / pow(theta, (double)(2 * j) / (double)rot_dim);
    double ang = (double)pos * inv;
    row[j] = (float)cos(ang);
    row[half + j] = (float)sin(ang);
  }
}

__global__ void k_rope_position_array(float *__restrict__ x,
                                      const int32_t *__restrict__ positions,
                                      const float *__restrict__ cos_sin_cache,
                                      int tokens, int nHeads, int headDim,
                                      int rot_dim, int rope_dim_offset,
                                      int max_position, int is_neox, int inverse) {
  int flat = blockIdx.x;
  int half = rot_dim / 2;
  int total_heads = tokens * nHeads;
  if (flat >= total_heads) return;
  int tok = flat / nHeads;
  int pos = positions[tok];
  if (pos < 0 || pos >= max_position) return;

  const float *row = cos_sin_cache + (size_t)pos * rot_dim;
  float *hv = x + (size_t)flat * headDim + rope_dim_offset;
  float sign = (inverse != 0) ? -1.0f : 1.0f;

  for (int j = threadIdx.x; j < half; j += blockDim.x) {
    float c = row[j];
    float s = row[half + j] * sign;
    if (is_neox != 0) {
      float a = hv[j];
      float b = hv[j + half];
      hv[j] = (float)(a * c) - (float)(b * s);
      hv[j + half] = (float)(b * c) + (float)(a * s);
    } else {
      float a = hv[2 * j];
      float b = hv[2 * j + 1];
      hv[2 * j] = (float)(a * c) - (float)(b * s);
      hv[2 * j + 1] = (float)(b * c) + (float)(a * s);
    }
  }
}

extern "C" int fcuda_rope_cos_sin_cache_build_f32(float *dCosSinCache, int max_position,
                                                  int rot_dim, double theta) {
  if (!dCosSinCache || max_position <= 0 || rot_dim <= 0 || (rot_dim % 2) != 0 ||
      !(theta > 0.0)) {
    return -1;
  }
  int threads = (rot_dim / 2 < 256) ? (rot_dim / 2) : 256;
  if (threads < 32) threads = 32;
  k_rope_build_cos_sin_cache<<<max_position, threads>>>(dCosSinCache, max_position, rot_dim, theta);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : (int)err;
}

extern "C" int fcuda_rope_position_array_f32(float *dX, const int32_t *dPositions,
                                             const float *dCosSinCache, int tokens,
                                             int nHeads, int headDim, int rot_dim,
                                             int rope_dim_offset, int max_position,
                                             int is_neox, int inverse) {
  if (!dX || !dPositions || !dCosSinCache || tokens <= 0 || nHeads <= 0 || headDim <= 0 ||
      rot_dim <= 0 || (rot_dim % 2) != 0 || rope_dim_offset < 0 ||
      rope_dim_offset + rot_dim > headDim || max_position <= 0) {
    return -1;
  }
  int half = rot_dim / 2;
  int threads = (half < 256) ? half : 256;
  if (threads < 32) threads = 32;
  k_rope_position_array<<<tokens * nHeads, threads>>>(
      dX, dPositions, dCosSinCache, tokens, nHeads, headDim,
      rot_dim, rope_dim_offset, max_position, is_neox, inverse);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : (int)err;
}
#endif  // __CUDACC__
