//go:build linux && cgo && rocm

#include "rocm_backend.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

char last_error[512] = "";
hipStream_t stream = nullptr;
int active_device = -1;
char active_gfx[64] = "";
size_t host_transfer_bytes = 0;
size_t kernel_launches = 0;

int failf(int code, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(last_error, sizeof(last_error), fmt, args);
  va_end(args);
  return code ? code : -1;
}

int hip_fail(const char *what, hipError_t status) {
  return failf(1000 + static_cast<int>(status), "%s: %s", what,
               hipGetErrorString(status));
}

int ready(const char *what) {
  if (active_device < 0 || stream == nullptr)
    return failf(-2, "%s: ROCm backend is not initialized", what);
  last_error[0] = '\0';
  hipError_t status = hipSetDevice(active_device);
  if (status != hipSuccess) return hip_fail("hipSetDevice", status);
  return 0;
}

int launch_status(const char *what) {
  hipError_t status = hipGetLastError();
  if (status != hipSuccess) return hip_fail(what, status);
  status = hipStreamSynchronize(stream);
  if (status != hipSuccess) return hip_fail(what, status);
  ++kernel_launches;
  return 0;
}

__device__ float fp16_at(const uint8_t *p) {
  return __half2float(*reinterpret_cast<const __half *>(p));
}

__device__ void q4_scale_min(int group, const uint8_t *q, uint8_t *scale,
                             uint8_t *minimum) {
  if (group < 4) {
    *scale = q[group] & 63;
    *minimum = q[group + 4] & 63;
  } else {
    *scale = (q[group + 4] & 0x0f) | ((q[group - 4] >> 6) << 4);
    *minimum = (q[group + 4] >> 4) | ((q[group] >> 6) << 4);
  }
}

__global__ void f32_matmul(const float *w, const float *x, float *y,
                           int out, int in, int rows) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (o >= out || row >= rows) return;
  const float *wr = w + static_cast<size_t>(o) * in;
  const float *xr = x + static_cast<size_t>(row) * in;
  float sum = 0.0f;
  for (int i = 0; i < in; ++i) sum = fmaf(wr[i], xr[i], sum);
  y[static_cast<size_t>(row) * out + o] = sum;
}

__global__ void q8_matmul(const int8_t *codes, const float *scales,
                          const float *x, float *y, int out, int in, int rows,
                          int block) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (o >= out || row >= rows) return;
  int blocks = in / block;
  const int8_t *wr = codes + static_cast<size_t>(o) * in;
  const float *ws = scales + static_cast<size_t>(o) * blocks;
  const float *xr = x + static_cast<size_t>(row) * in;
  float sum = 0.0f;
  for (int b = 0; b < blocks; ++b) {
    float partial = 0.0f;
    int base = b * block;
    for (int i = 0; i < block; ++i)
      partial = fmaf(static_cast<float>(wr[base + i]), xr[base + i], partial);
    sum = fmaf(ws[b], partial, sum);
  }
  y[static_cast<size_t>(row) * out + o] = sum;
}

__device__ float q4k_value(const uint8_t *blk, int elem) {
  int group = elem / 32;
  int lane = elem & 31;
  uint8_t scale, minimum;
  q4_scale_min(group, blk + 4, &scale, &minimum);
  uint8_t packed = blk[16 + (group >> 1) * 32 + lane];
  int code = (group & 1) ? packed >> 4 : packed & 15;
  return fp16_at(blk) * static_cast<float>(scale * code) -
         fp16_at(blk + 2) * static_cast<float>(minimum);
}

__device__ float q5k_value(const uint8_t *blk, int elem) {
  int group = elem / 32;
  int lane = elem & 31;
  uint8_t scale, minimum;
  q4_scale_min(group, blk + 4, &scale, &minimum);
  const uint8_t *qh = blk + 16;
  const uint8_t *ql = blk + 48;
  int code = (group & 1) ? (ql[(group >> 1) * 32 + lane] >> 4)
                         : (ql[(group >> 1) * 32 + lane] & 15);
  if (qh[lane] & (1u << group)) code += 16;
  return fp16_at(blk) * static_cast<float>(scale * code) -
         fp16_at(blk + 2) * static_cast<float>(minimum);
}

__device__ float q6k_value(const uint8_t *blk, int elem) {
  const uint8_t *ql = blk;
  const uint8_t *qh = blk + 128;
  const int8_t *scales = reinterpret_cast<const int8_t *>(blk + 192);
  int half = elem / 128;
  int within = elem & 127;
  int lane = within & 31;
  int quarter = within / 32;
  int qoff = half * 64;
  int hoff = half * 32;
  int code;
  if (quarter == 0)
    code = (ql[qoff + lane] & 15) | (((qh[hoff + lane] >> 0) & 3) << 4);
  else if (quarter == 1)
    code = (ql[qoff + lane + 32] & 15) | (((qh[hoff + lane] >> 2) & 3) << 4);
  else if (quarter == 2)
    code = (ql[qoff + lane] >> 4) | (((qh[hoff + lane] >> 4) & 3) << 4);
  else
    code = (ql[qoff + lane + 32] >> 4) | (((qh[hoff + lane] >> 6) & 3) << 4);
  int scale_index = half * 8 + (lane >> 4) + quarter * 2;
  return fp16_at(blk + 208) * static_cast<float>(scales[scale_index]) *
         static_cast<float>(code - 32);
}

enum KQuantKind { KQ4 = 4, KQ5 = 5, KQ6 = 6 };

template <KQuantKind kind>
__global__ void kquant_matmul(const uint8_t *weights, const float *x, float *y,
                              int out, int in, int rows) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (o >= out || row >= rows) return;
  constexpr int block_bytes = kind == KQ4 ? 144 : (kind == KQ5 ? 176 : 210);
  int superblocks = in / 256;
  const uint8_t *wr = weights + static_cast<size_t>(o) * superblocks * block_bytes;
  const float *xr = x + static_cast<size_t>(row) * in;
  float sum = 0.0f;
  for (int sb = 0; sb < superblocks; ++sb) {
    const uint8_t *blk = wr + static_cast<size_t>(sb) * block_bytes;
    for (int i = 0; i < 256; ++i) {
      float w = kind == KQ4 ? q4k_value(blk, i)
                            : (kind == KQ5 ? q5k_value(blk, i)
                                           : q6k_value(blk, i));
      sum = fmaf(w, xr[sb * 256 + i], sum);
    }
  }
  y[static_cast<size_t>(row) * out + o] = sum;
}

__global__ void rmsnorm(const float *x, const float *weight, float *y,
                        int rows, int width, float eps) {
  int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float reduction[256];
  const float *xr = x + static_cast<size_t>(row) * width;
  float local = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    local = fmaf(xr[i], xr[i], local);
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (int step = blockDim.x / 2; step > 0; step >>= 1) {
    if (threadIdx.x < step) reduction[threadIdx.x] += reduction[threadIdx.x + step];
    __syncthreads();
  }
  float inv = rsqrtf(reduction[0] / static_cast<float>(width) + eps);
  float *yr = y + static_cast<size_t>(row) * width;
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    yr[i] = xr[i] * inv * weight[i];
}

__global__ void rope(const float *src, float *dst, int pos, int heads,
                     int head_dim, double theta) {
  int half = head_dim / 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= heads * half) return;
  int head = index / half;
  int d = index % half;
  double angle = static_cast<double>(pos) /
                 pow(theta, static_cast<double>(2 * d) / head_dim);
  float c = static_cast<float>(cos(angle));
  float s = static_cast<float>(sin(angle));
  size_t base = static_cast<size_t>(head) * head_dim;
  float a = src[base + d];
  float b = src[base + d + half];
  dst[base + d] = a * c - b * s;
  dst[base + d + half] = b * c + a * s;
}

__global__ void swiglu(const float *gate, const float *up, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
  }
}

// Adapted finite-input clamp arithmetic from ROCmFPX PR6, unary.cuh/unary.cu,
// revision9fb229123f3ee1847e32c29445c8cb0fa1e5fb36 (ggml/llama.cpp upstream).
// Modification: contiguous separate F32 operands, Fak stream/error ABI, and
// ordered comparisons to preserve NaN operands rather than fminf masking them.
// Canonical creator attribution: Carlo Pasquale (Charlie12345), ROCmFPX founder.
// https://github.com/ROCmFPX/ROCmFPX/blob/9fb229123f3ee1847e32c29445c8cb0fa1e5fb36/NOTICE
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2025-2026 Carlo Pasquale (Charlie12345), ROCmFPX additions
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
__global__ void swiglu_limit(const float *gate, const float *up, float *out, int n, float limit) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= static_cast<unsigned int>(n)) return;
  float g = gate[i];
  float u = up[i];
  if (g > limit) g = limit;
  if (u > limit) u = limit;
  if (u < -limit) u = -limit;
  out[i] = (g / (1.0f + expf(-g))) * u;
}

__global__ void add(float *dst, const float *src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

__global__ void add_bias(float *dst, const float *bias, int n, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += bias[i % width];
}

/* Correctness-first native attention: one GPU thread owns one query head and
 * performs online softmax. It keeps all math and KV traffic device-resident;
 * later tuning can parallelize within the head without changing the ABI. */
__global__ void attention(const float *q, const float *k, const float *v,
                          float *out, int q_heads, int kv_heads, int positions,
                          int head_dim, int group_size, float scale) {
  int qh = blockIdx.x * blockDim.x + threadIdx.x;
  if (qh >= q_heads) return;
  int kvh = qh / group_size;
  int stride = kv_heads * head_dim;
  const float *qr = q + static_cast<size_t>(qh) * head_dim;
  float max_score = -INFINITY;
  float denom = 0.0f;
  for (int d = 0; d < head_dim; ++d)
    out[static_cast<size_t>(qh) * head_dim + d] = 0.0f;
  for (int p = 0; p < positions; ++p) {
    const float *kr = k + static_cast<size_t>(p) * stride + kvh * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) dot = fmaf(qr[d], kr[d], dot);
    float score = dot * scale;
    float next_max = fmaxf(max_score, score);
    float correction = expf(max_score - next_max);
    float probability = expf(score - next_max);
    denom = denom * correction + probability;
    const float *vr = v + static_cast<size_t>(p) * stride + kvh * head_dim;
    float *yr = out + static_cast<size_t>(qh) * head_dim;
    for (int d = 0; d < head_dim; ++d)
      yr[d] = yr[d] * correction + probability * vr[d];
    max_score = next_max;
  }
  float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
  for (int d = 0; d < head_dim; ++d)
    out[static_cast<size_t>(qh) * head_dim + d] *= inv;
}

__global__ void argmax(const float *values, int n, int *result) {
  if (isnan(values[0])) {
    if (threadIdx.x == 0) *result = 0;
    return;
  }
  __shared__ float best_value[256];
  __shared__ int best_index[256];
  float value = values[0];
  int index = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float candidate = values[i];
    if (candidate > value || (candidate == value && i < index)) {
      value = candidate;
      index = i;
    }
  }
  best_value[threadIdx.x] = value;
  best_index[threadIdx.x] = index;
  __syncthreads();
  for (int step = blockDim.x / 2; step > 0; step >>= 1) {
    if (threadIdx.x < step) {
      float other = best_value[threadIdx.x + step];
      int other_index = best_index[threadIdx.x + step];
      if (other > best_value[threadIdx.x] ||
          (other == best_value[threadIdx.x] && other_index < best_index[threadIdx.x])) {
        best_value[threadIdx.x] = other;
        best_index[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *result = best_index[0];
}

__global__ void kv_rerope(const float *raw_keys, float *keys, int start,
                          int positions, int kv_heads, int head_dim,
                          double theta) {
  int half = head_dim / 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (positions - start) * kv_heads * half;
  if (index >= total) return;
  int d = index % half;
  int pair = index / half;
  int head = pair % kv_heads;
  int pos = start + pair / kv_heads;
  size_t base = (static_cast<size_t>(pos) * kv_heads + head) * head_dim;
  double angle = static_cast<double>(pos) /
                 pow(theta, static_cast<double>(2 * d) / head_dim);
  float c = static_cast<float>(cos(angle));
  float s = static_cast<float>(sin(angle));
  float a = raw_keys[base + d];
  float b = raw_keys[base + d + half];
  keys[base + d] = a * c - b * s;
  keys[base + d + half] = b * c + a * s;
}

}  // namespace

// ============================================================================
// Masked WMMA attention for gfx1151 wave32 (direct port, MIT).
//
// Source: gufo-org/gufo, src/models/qwen/hip/kernels/attention_wmma.hip at
// commit d0f7c3a023a11af81161264eeb24e9a97aa7498a.
//
// MIT License
//
// Copyright (c) 2026 gufo contributors
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
// Upstream design notice, retained:
//
//   Masked prefill attention on the WMMA matrix cores. One pass covers the
//   visible prefix and causal diagonal. Wave32 fragment layout, verified
//   against a CPU double-precision reference: A holds row L%16 and 16
//   contiguous k, B holds column L%16 and 16 contiguous k (so B is fed a
//   transposed tile), and C element i is row 2i + L/16, column L%16.
//
//   Global request count dominates, not bytes: every block re-reads K and V
//   for its key range, so the cost per query row falls as the block covers
//   more rows. V has to reach LDS transposed, because both WMMA operands want
//   16 contiguous values of the contracted dimension; giving each lane one key
//   keeps those LDS writes conflict-free. Tile i+1's K and V are loaded into a
//   second register set right after tile i's staging barrier so the load sits
//   behind tile i's S, softmax and PV phases.
//
// Upstream reports a 1.12-1.30x ablation for this kernel alone at batch 2048.
// That figure is upstream's, was not reproduced here, and is not a whole-model
// gain.
//
// Changes from upstream:
//   * K and V are read from the resident FP32 cache and rounded to FP16 with
//     __float2half_rn at the global load, which is the value upstream's
//     PackTiledAttentionKvKernel stores. No FP16 cache or per-call copy exists,
//     so cache ownership is unchanged; staged bytes and the WMMA sequence are
//     upstream's.
//   * The packed head-major variant and its FFN-scratch workspace are omitted.
//   * Q, K and V use scalar global loads, so no 16-byte alignment is assumed
//     of the caller's tensors.
//
// Compiled only with -DFAK_ROCM_WMMA_GFX1151 and --offload-arch=gfx1151: the
// _w32 builtin does not exist for other targets. Selected at run time only by
// wmma_attention_selected below. The CPU mirror of this tiling is
// HIPAttentionWMMAReference in hip_attention.go; the constants there and here
// must stay equal.
// ============================================================================
#if defined(FAK_ROCM_WMMA_GFX1151)
namespace {

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kQueryHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kGqa = kQueryHeads / kKvHeads;
constexpr std::uint32_t kAttnWidth = kQueryHeads * kHeadDim;
constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// A fragment is 16 contiguous halves = 32 bytes. Reading it element by element
// costs 16 ds_read_u16 plus 16 __half -> _Float16 conversions; the two formats
// are bit-identical, so two 16-byte loads and a bitcast do the same work in two
// instructions. Every fragment base here is 16-byte aligned by construction.
__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

// Eight resident FP32 cache values as the eight FP16 values upstream's packed
// cache would hold for them.
__device__ __forceinline__ uint4 PackHalves(const float* p) {
  union {
    uint4 u;
    __half h[8];
  } cvt;
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    cvt.h[i] = __float2half_rn(p[i]);
  }
  return cvt.u;
}

constexpr std::uint32_t kWmmaHeads =
    2;  // query heads per block, must divide GQA
// Measured tile: 32 queries x 2 heads = 64 rows per block against 16 keys.
constexpr std::uint32_t kWmmaQueryRows = 32;
constexpr std::uint32_t kWmmaKeys = 16;
constexpr std::uint32_t kWmmaKSteps = kHeadDim / 16;

// Row padding keeps a 16-byte-per-lane fragment read off a single bank group.
constexpr std::uint32_t kWmmaKStride = kHeadDim + 8;

}  // namespace

// External linkage on purpose: an anonymous-namespace kernel is dispatched with
// an empty name, and rocprofv3 then merges it into an unattributed bucket.
//
// kQueryRows queries x kWmmaHeads heads form (kQueryRows / 16) * kWmmaHeads
// 16-row blocks, and kKeys keys form kKeys / 16 key blocks. Their product is
// pinned to 4 so the eight waves split as four S tiles x two halves of the head
// dimension, which is what keeps Q in registers at 64 VGPRs.
template<std::uint32_t kQueryRows, std::uint32_t kKeys>
__launch_bounds__(256, 2) __global__
    void WmmaCausalAttention(const float* __restrict__ q,
                             const float* __restrict__ gate,
                             const float* __restrict__ k_cache,
                             const float* __restrict__ v_cache,
                             float* __restrict__ out_context,
                             std::uint32_t start_pos, std::size_t batch_size,
                             float* __restrict__ lse_out,
                             std::uint32_t key_begin) {
  constexpr std::uint32_t kRowBlocks = (kQueryRows / 16) * kWmmaHeads;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / (8 / kSTiles);
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kOTilesPerWave = (kRowBlocks * kWmmaKSteps) / 8;
  constexpr std::uint32_t kSoftmaxLanes = 256 / kRows;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  static_assert(kSTiles == 4, "eight waves cover four S tiles in two halves");
  static_assert(kOTilesPerWave == 2 * kRowBlocks, "O tiles per wave");
  static_assert(kKeys % 16 == 0 && kQueryRows % 16 == 0, "16-row WMMA tiles");
  static_assert(kKStepsPerWave * 8 / kSTiles == kWmmaKSteps, "k split");
  static_assert(kWmmaKStride % 8 == 0 && kVtStride % 8 == 0,
                "fragment rows must start on a 16-byte boundary");
  static_assert(kSoftmaxLanes * (kKeys / kSoftmaxLanes) == kKeys, "softmax");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  const std::uint32_t query_start = blockIdx.x * kQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kGqa / kWmmaHeads);
  const std::uint32_t pair_in_group = head_pair % (kGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      (kv_head * kGqa) + (pair_in_group * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  // Row block -> (query head, 16-query sub-block of this block's rows).
  const auto row_block_head = [&](std::uint32_t rb) {
    return first_query_head + (rb % kWmmaHeads);
  };
  const auto row_block_offset = [&](std::uint32_t rb) {
    return (rb / kWmmaHeads) * 16;
  };

  constexpr std::uint32_t kKvLdsHalves =
      kKeys * kWmmaKStride > kHeadDim * kVtStride ? kKeys * kWmmaKStride
                                                  : kHeadDim * kVtStride;
  __shared__ __half kv_lds[kKvLdsHalves];
  // Padding avoids repeated banks in softmax reads and probability fragments.
  __shared__ float s_lds[2][kSTiles][16][17];
  __shared__ __half p_lds[kRows][kKeys + 8];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t local_query =
        query_start + row_block_offset(s_rb) + sub;
    const bool live = local_query < batch_size;
    const float* q_row =
        q + (static_cast<std::size_t>(local_query) * kAttnWidth) +
        (static_cast<std::size_t>(row_block_head(s_rb)) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
#pragma unroll
      for (std::uint32_t e = 0; e < 16; ++e) {
        const float f = live ? q_row[d0 + e] : 0.0F;
        q_frag[ks][e] = static_cast<_Float16>(f * attention_scale);
      }
    }
  }

  // This wave owns dim tiles `wave` and `wave + 8` for every row block, so both
  // halves of the head dimension stay evenly loaded.
  v8f o_acc[kRowBlocks][2] = {};
  // Each softmax row keeps the same four owner lanes across the key sweep.
  // Only the final statistics and each tile's rescale factor cross waves.
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  // One key per lane for V, a 16-dim slice per thread. This costs a strided
  // global read but makes the transpose's LDS writes conflict-free.
  constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
  constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
  static_assert(kKRegs * 256 == kKeys * (kHeadDim / 8), "K stages evenly");
  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);
  constexpr std::size_t kv_stride = kKvWidth;
  const std::size_t head_offset = static_cast<std::size_t>(kv_head) * kHeadDim;
  const float* v_base = v_cache + head_offset + v_slice;
  const float* k_base = k_cache + head_offset;

  const auto load_v = [&](std::uint32_t key_start, uint4* dst) {
    const std::uint32_t key_position = key_start + v_key;
    const float* src =
        v_base + (static_cast<std::size_t>(key_position) * kv_stride);
    const bool live = key_position < context_end;
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      dst[j] = live ? PackHalves(src + (j * 8)) : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  // Coalesced: a wave reads one key row's contiguous values.
  const auto load_k = [&](std::uint32_t key_start, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_position = key_start + (idx / (kHeadDim / 8));
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      dst[n] =
          (key_position < context_end)
              ? PackHalves(k_base +
                           (static_cast<std::size_t>(key_position) *
                            kv_stride) +
                           d8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  uint4 k_cur[kKRegs];
  uint4 v_cur[kVRegs];
  uint4 k_pre[kKRegs] = {};
  uint4 v_pre[kVRegs] = {};
  load_k(key_begin, k_cur);
  load_v(key_begin, v_cur);

  for (std::uint32_t key_start = key_begin; key_start < max_visible;
       key_start += kKeys) {
    // --- stage K from the registers the previous iteration prefetched
    __syncthreads();
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&kv_lds[(key_row * kWmmaKStride) + d8]) =
          k_cur[n];
    }
    __syncthreads();

    // --- prefetch the next key tile. Everything below covers its latency.
    const std::uint32_t next_start = key_start + kKeys;
    if (next_start < max_visible) {
      load_k(next_start, k_pre);
      load_v(next_start, v_pre);
    }

    // --- S = Q K^T ---
    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&kv_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        s_acc = Wmma(q_frag[ks], k_frag, s_acc);
      }
      // Each half writes its own slot, so the reader sums them instead of a
      // read-modify-write that would need a barrier between the halves.
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
    __syncthreads();

    // --- online softmax: kSoftmaxLanes threads per query row ---
    {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + row;
      const std::uint32_t absolute_query = start_pos + local_query;
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = key_start + col;
        const bool valid = local_query < batch_size &&
                           key_position <= absolute_query &&
                           key_position < context_end;
        const std::uint32_t tile = ((col / 16) * kRowBlocks) + rb;
        vals[m] = valid ? (s_lds[0][tile][row][col % 16] +
                           s_lds[1][tile][row][col % 16])
                        : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      }
      const float prev_max = running_max;
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][(seg * kPerLane) + m] = static_cast<__half>(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_sum += __shfl_xor(part_sum, off);
      }
      running_max = next_max;
      running_sum = (running_sum * prior_scale) + part_sum;
      if (seg == 0)
        row_scale[rg] = prior_scale;
    }
    __syncthreads();

    // Each scale is shared across both dimension tiles.
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      float scale[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale[i];
        }
      }
    }

    // --- stage V transposed. No barrier is needed here: the barrier after the
    // softmax already separates every read of K in the S phase from these
    // writes to the same buffer.
    //
    // Lane L owns key L % kKeys and each thread a dim slice, so consecutive
    // lanes write consecutive *columns* of V^T: two 16-bit writes share a dword
    // and the wave covers 16 banks.
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      const auto* packed = reinterpret_cast<const __half*>(&v_cur[j]);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        kv_lds[((v_slice + (j * 8) + i) * kVtStride) + v_key] = packed[i];
      }
    }
    __syncthreads();

    // --- O += P V ---
    // Dimension tiles are visited first so the compiler can share their
    // fragments across row blocks.
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
        const std::uint32_t dim_tile = wave + t * 8;
        v16h v_frag[kKeyBlocks];
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          v_frag[kb] =
              LoadFrag(&kv_lds[(dim_tile * 16 + sub) * kVtStride + kb * 16]);
        }
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          v8f next = Wmma(p_frag, v_frag[kb], o_acc[rb][t]);
          const auto first_key = key_start + kb * 16;
          // Like Flash-Next, sum a query's final partial tile over visible
          // keys only. WMMA can round differently when a zero probability
          // multiplies a populated future value instead of chunk padding.
          // Complete tiles keep the matrix-core path and its original order.
          if (first_key + 15 > start_pos + query_start) {
            v8f tail = o_acc[rb][t];
            // Keep the position difference uniform across the wave; only the
            // even/odd row owner varies between its two half-waves.
            const int relative_key = static_cast<int>(first_key) -
                                     static_cast<int>(start_pos + query_start +
                                                      row_block_offset(rb));
#pragma unroll
            for (std::uint32_t key = 0; key < 16; ++key) {
              const float value = static_cast<float>(v_frag[kb][key]);
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                if (static_cast<int>(half_id) <
                        relative_key + 15 - static_cast<int>(2 * i) &&
                    static_cast<int>(half_id) >= relative_key +
                                                     static_cast<int>(key) -
                                                     static_cast<int>(2 * i)) {
                  tail[i] =
                      fmaf(__half2float(
                               p_lds[rb * 16 + 2 * i + half_id][kb * 16 + key]),
                           value, tail[i]);
                }
              }
            }
#pragma unroll
            for (std::uint32_t i = 0; i < 8; ++i)
              if (static_cast<int>(half_id) <
                  relative_key + 15 - static_cast<int>(2 * i))
                next[i] = tail[i];
          }
          o_acc[rb][t] = next;
        }
      }
    }

#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      k_cur[n] = k_pre[n];
    }
#pragma unroll
    for (std::uint32_t n = 0; n < kVRegs; ++n) {
      v_cur[n] = v_pre[n];
    }
  }
  if (tid % kSoftmaxLanes == 0) {
    row_max[tid / kSoftmaxLanes] = running_max;
    row_sum[tid / kSoftmaxLanes] = running_sum;
  }
  __syncthreads();

  // --- epilogue ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
    const std::uint32_t query_head = row_block_head(rb);
    const std::uint32_t row_offset = row_block_offset(rb);
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const std::uint32_t local_query = query_start + row_offset + row;
        if (local_query >= batch_size) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(local_query) * kAttnWidth) +
            (static_cast<std::size_t>(query_head) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr && lse_out == nullptr) {
          value *= 1.0F / (1.0F + __expf(-gate[offset]));
        }
        out_context[offset] = value;
      }
    }
  }
  if (lse_out != nullptr && wave == 0) {
    for (std::uint32_t rg = lane; rg < kRows; rg += 32) {
      const std::uint32_t rb = rg / 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + (rg % 16);
      if (local_query >= batch_size) {
        continue;
      }
      const float denominator = row_sum[rg];
      lse_out[(static_cast<std::size_t>(row_block_head(rb)) * batch_size) +
              local_query] =
          (denominator > 0.0F) ? (row_max[rg] + logf(denominator)) : -INFINITY;
    }
  }
}

namespace {

/* Every clause must hold or the generic kernel keeps the call: the build
 * carries the WMMA kernel, the operator opted in, the device is gfx1151, and
 * the model geometry is the one the kernel is compiled for. */
bool wmma_attention_selected(int q_heads, int kv_heads, int positions,
                             int head_dim, int group_size, float scale) {
  const char *opt_in = getenv("FAK_ROCM_WMMA_ATTENTION");
  if (opt_in == nullptr || strcmp(opt_in, "1") != 0) return false;
  if (strncmp(active_gfx, "gfx1151", 7) != 0 ||
      (active_gfx[7] != '\0' && active_gfx[7] != ':'))
    return false;
  return q_heads == static_cast<int>(kQueryHeads) &&
         kv_heads == static_cast<int>(kKvHeads) &&
         head_dim == static_cast<int>(kHeadDim) &&
         group_size == static_cast<int>(kGqa) && positions >= 1 &&
         scale == 1.0F / 16.0F;
}

}  // namespace
#endif  // defined(FAK_ROCM_WMMA_GFX1151)

extern "C" const char *frocm_last_error(void) { return last_error; }

extern "C" int frocm_init(int device, char *name, int name_len, char *gfx,
                           int gfx_len, size_t *total_mem) {
  last_error[0] = '\0';
  int count = 0;
  hipError_t hs = hipGetDeviceCount(&count);
  if (hs != hipSuccess) return hip_fail("hipGetDeviceCount", hs);
  if (device < 0 || device >= count)
    return failf(-1, "device %d outside [0,%d)", device, count);
  if ((hs = hipSetDevice(device)) != hipSuccess) return hip_fail("hipSetDevice", hs);
  hipDeviceProp_t prop{};
  if ((hs = hipGetDeviceProperties(&prop, device)) != hipSuccess)
    return hip_fail("hipGetDeviceProperties", hs);
  if (stream == nullptr && (hs = hipStreamCreate(&stream)) != hipSuccess)
    return hip_fail("hipStreamCreate", hs);
  active_device = device;
  strncpy(active_gfx, prop.gcnArchName, sizeof(active_gfx) - 1);
  active_gfx[sizeof(active_gfx) - 1] = '\0';
  if (name && name_len > 0) {
    strncpy(name, prop.name, static_cast<size_t>(name_len - 1));
    name[name_len - 1] = '\0';
  }
  if (gfx && gfx_len > 0) {
    strncpy(gfx, prop.gcnArchName, static_cast<size_t>(gfx_len - 1));
    gfx[gfx_len - 1] = '\0';
  }
  if (total_mem) *total_mem = prop.totalGlobalMem;
  return 0;
}

extern "C" int frocm_mem_info(size_t *free_mem, size_t *total_mem) {
  if (int rc = ready("mem_info")) return rc;
  hipError_t status = hipMemGetInfo(free_mem, total_mem);
  return status == hipSuccess ? 0 : hip_fail("hipMemGetInfo", status);
}

extern "C" int frocm_malloc(void **ptr, size_t bytes) {
  if (int rc = ready("malloc")) return rc;
  if (!ptr || bytes == 0) return failf(-1, "malloc: invalid output or zero bytes");
  hipError_t status = hipMalloc(ptr, bytes);
  return status == hipSuccess ? 0 : hip_fail("hipMalloc", status);
}

extern "C" int frocm_free(void *ptr) {
  if (!ptr) return 0;
  if (int rc = ready("free")) return rc;
  hipError_t status = hipFree(ptr);
  return status == hipSuccess ? 0 : hip_fail("hipFree", status);
}

extern "C" int frocm_h2d(void *dst, const void *src, size_t bytes) {
  if (int rc = ready("h2d")) return rc;
  if (!dst || (!src && bytes)) return failf(-1, "h2d: invalid pointer");
  hipError_t status = hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
  if (status != hipSuccess) return hip_fail("hipMemcpy H2D", status);
  host_transfer_bytes += bytes;
  return 0;
}

extern "C" int frocm_d2h(void *dst, const void *src, size_t bytes) {
  if (int rc = ready("d2h")) return rc;
  if ((!dst && bytes) || !src) return failf(-1, "d2h: invalid pointer");
  hipError_t status = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
  if (status != hipSuccess) return hip_fail("hipMemcpy D2H", status);
  host_transfer_bytes += bytes;
  return 0;
}

extern "C" int frocm_d2d(void *dst, const void *src, size_t bytes) {
  if (int rc = ready("d2d")) return rc;
  if ((!dst || !src) && bytes) return failf(-1, "d2d: invalid pointer");
  hipError_t status = hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream);
  if (status != hipSuccess) return hip_fail("hipMemcpyAsync D2D", status);
  status = hipStreamSynchronize(stream);
  return status == hipSuccess ? 0 : hip_fail("hipMemcpyAsync D2D completion", status);
}

extern "C" int frocm_sync(void) {
  if (int rc = ready("sync")) return rc;
  hipError_t status = hipStreamSynchronize(stream);
  return status == hipSuccess ? 0 : hip_fail("hipStreamSynchronize", status);
}

extern "C" size_t frocm_host_transfer_bytes(void) { return host_transfer_bytes; }
extern "C" void frocm_host_transfer_reset(void) { host_transfer_bytes = 0; }
extern "C" size_t frocm_kernel_launches(void) { return kernel_launches; }
extern "C" void frocm_kernel_launches_reset(void) { kernel_launches = 0; }

extern "C" int frocm_matmul_f32(const float *w, const float *x, float *y,
                                 int out, int in, int rows) {
  if (int rc = ready("matmul_f32")) return rc;
  if (!w || !x || !y || out <= 0 || in <= 0 || rows <= 0)
    return failf(-1, "matmul_f32: invalid pointer or geometry");
  f32_matmul<<<dim3((out + 127) / 128, rows), 128, 0, stream>>>(
      w, x, y, out, in, rows);
  return launch_status("matmul_f32 launch");
}

extern "C" int frocm_q8_matmul_f32(const int8_t *codes, const float *scales,
                                    const float *x, float *y, int out, int in,
                                    int rows, int block) {
  if (int rc = ready("q8_matmul_f32")) return rc;
  if (!codes || !scales || !x || !y || out <= 0 || in <= 0 || rows <= 0 ||
      block <= 0 || in % block != 0)
    return failf(-1, "q8_matmul_f32: invalid pointer or geometry");
  q8_matmul<<<dim3((out + 127) / 128, rows), 128, 0, stream>>>(
      codes, scales, x, y, out, in, rows, block);
  return launch_status("q8_matmul_f32 launch");
}

template <KQuantKind kind>
int launch_kquant(const char *what, const uint8_t *weights, const float *x,
                  float *y, int out, int in, int rows) {
  if (int rc = ready(what)) return rc;
  if (!weights || !x || !y || out <= 0 || in <= 0 || rows <= 0 || in % 256 != 0)
    return failf(-1, "%s: invalid pointer or geometry", what);
  kquant_matmul<kind><<<dim3((out + 127) / 128, rows), 128, 0, stream>>>(
      weights, x, y, out, in, rows);
  return launch_status(what);
}

extern "C" int frocm_q4k_matmul_f32(const uint8_t *w, const float *x, float *y,
                                     int out, int in, int rows) {
  return launch_kquant<KQ4>("q4k_matmul_f32 launch", w, x, y, out, in, rows);
}
extern "C" int frocm_q5k_matmul_f32(const uint8_t *w, const float *x, float *y,
                                     int out, int in, int rows) {
  return launch_kquant<KQ5>("q5k_matmul_f32 launch", w, x, y, out, in, rows);
}
extern "C" int frocm_q6k_matmul_f32(const uint8_t *w, const float *x, float *y,
                                     int out, int in, int rows) {
  return launch_kquant<KQ6>("q6k_matmul_f32 launch", w, x, y, out, in, rows);
}

extern "C" int frocm_rmsnorm_f32(const float *x, const float *weight, float *y,
                                  int rows, int width, float eps) {
  if (int rc = ready("rmsnorm_f32")) return rc;
  if (!x || !weight || !y || rows <= 0 || width <= 0 || eps < 0)
    return failf(-1, "rmsnorm_f32: invalid pointer or geometry");
  rmsnorm<<<rows, 256, 0, stream>>>(x, weight, y, rows, width, eps);
  return launch_status("rmsnorm_f32 launch");
}

extern "C" int frocm_rope_f32(const float *src, float *dst, int pos, int heads,
                               int head_dim, double theta) {
  if (int rc = ready("rope_f32")) return rc;
  if (!src || !dst || pos < 0 || heads <= 0 || head_dim <= 0 || head_dim % 2 || theta <= 0)
    return failf(-1, "rope_f32: invalid pointer or geometry");
  int pairs = heads * (head_dim / 2);
  rope<<<(pairs + 127) / 128, 128, 0, stream>>>(src, dst, pos, heads, head_dim, theta);
  return launch_status("rope_f32 launch");
}

extern "C" int frocm_swiglu_f32(const float *gate, const float *up, float *out, int n) {
  if (int rc = ready("swiglu_f32")) return rc;
  if (!gate || !up || !out || n <= 0) return failf(-1, "swiglu_f32: invalid input");
  swiglu<<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
  return launch_status("swiglu_f32 launch");
}

extern "C" int frocm_swiglu_limit_f32(const float *gate, const float *up, float *out, int n, float limit) {
  if (int rc = ready("swiglu_limit_f32")) return rc;
  if (!gate || !up || !out || n <= 0 || !isfinite(limit) || limit <= 0.0f)
    return failf(-1, "swiglu_limit_f32: invalid input");
  swiglu_limit<<<(n - 1) / 256 + 1, 256, 0, stream>>>(gate, up, out, n, limit);
  return launch_status("swiglu_limit_f32 launch");
}

extern "C" int frocm_add_f32(float *dst, const float *src, int n) {
  if (int rc = ready("add_f32")) return rc;
  if (!dst || !src || n <= 0) return failf(-1, "add_f32: invalid input");
  add<<<(n + 255) / 256, 256, 0, stream>>>(dst, src, n);
  return launch_status("add_f32 launch");
}

extern "C" int frocm_add_bias_f32(float *dst, const float *bias, int rows, int width) {
  if (int rc = ready("add_bias_f32")) return rc;
  if (!dst || !bias || rows <= 0 || width <= 0) return failf(-1, "add_bias_f32: invalid input");
  int n = rows * width;
  add_bias<<<(n + 255) / 256, 256, 0, stream>>>(dst, bias, n, width);
  return launch_status("add_bias_f32 launch");
}

extern "C" int frocm_attention_f32(const float *q, const float *k, const float *v,
                                    float *out, int q_heads, int kv_heads, int positions,
                                    int head_dim, int group_size, float scale) {
  if (int rc = ready("attention_f32")) return rc;
  if (!q || !k || !v || !out || q_heads <= 0 || kv_heads <= 0 || positions <= 0 ||
      head_dim <= 0 || group_size <= 0 || q_heads != kv_heads * group_size || !isfinite(scale))
    return failf(-1, "attention_f32: invalid pointer or geometry");
#if defined(FAK_ROCM_WMMA_GFX1151)
  if (wmma_attention_selected(q_heads, kv_heads, positions, head_dim,
                              group_size, scale)) {
    /* One decode row is a one-query chunk whose prefix is every earlier
     * position, so the whole resident cache is visible to it. */
    const std::size_t batch_size = 1;
    const dim3 grid(static_cast<unsigned int>(
                        (batch_size + kWmmaQueryRows - 1) / kWmmaQueryRows),
                    kKvHeads * (kGqa / kWmmaHeads));
    WmmaCausalAttention<kWmmaQueryRows, kWmmaKeys><<<grid, 256, 0, stream>>>(
        q, nullptr, k, v, out, static_cast<std::uint32_t>(positions - 1),
        batch_size, nullptr, 0);
    return launch_status("attention_wmma launch");
  }
#endif
  attention<<<(q_heads + 63) / 64, 64, 0, stream>>>(
      q, k, v, out, q_heads, kv_heads, positions, head_dim, group_size, scale);
  return launch_status("attention_f32 launch");
}

extern "C" int frocm_argmax_f32(const float *values, int n, int *index) {
  if (int rc = ready("argmax_f32")) return rc;
  if (!values || !index || n <= 0) return failf(-1, "argmax_f32: invalid input");
  int *device_index = nullptr;
  hipError_t hs = hipMalloc(reinterpret_cast<void **>(&device_index), sizeof(int));
  if (hs != hipSuccess) return hip_fail("argmax hipMalloc", hs);
  argmax<<<1, 256, 0, stream>>>(values, n, device_index);
  int rc = launch_status("argmax_f32 launch");
  if (!rc) {
    hs = hipMemcpy(index, device_index, sizeof(int), hipMemcpyDeviceToHost);
    if (hs != hipSuccess) rc = hip_fail("argmax D2H", hs);
  }
  hipError_t free_status = hipFree(device_index);
  if (!rc && free_status != hipSuccess) rc = hip_fail("argmax hipFree", free_status);
  return rc;
}

extern "C" int frocm_kv_append_f32(float *keys, float *raw_keys, float *values,
                                    const float *key, const float *raw_key,
                                    const float *value, int slot, int row_width) {
  if (int rc = ready("kv_append_f32")) return rc;
  if (!keys || !raw_keys || !values || !key || !raw_key || !value || slot < 0 || row_width <= 0)
    return failf(-1, "kv_append_f32: invalid pointer or geometry");
  size_t bytes = static_cast<size_t>(row_width) * sizeof(float);
  size_t offset = static_cast<size_t>(slot) * row_width;
  hipError_t hs = hipMemcpyAsync(keys + offset, key, bytes, hipMemcpyDeviceToDevice, stream);
  if (hs != hipSuccess) return hip_fail("kv append key", hs);
  hs = hipMemcpyAsync(raw_keys + offset, raw_key, bytes, hipMemcpyDeviceToDevice, stream);
  if (hs != hipSuccess) return hip_fail("kv append raw key", hs);
  hs = hipMemcpyAsync(values + offset, value, bytes, hipMemcpyDeviceToDevice, stream);
  if (hs != hipSuccess) return hip_fail("kv append value", hs);
  hs = hipStreamSynchronize(stream);
  return hs == hipSuccess ? 0 : hip_fail("kv append completion", hs);
}

extern "C" int frocm_kv_evict_f32(float *keys, float *raw_keys, float *values,
                                   int positions, int from, int count, int kv_heads,
                                   int head_dim, double theta) {
  if (int rc = ready("kv_evict_f32")) return rc;
  if (!keys || !raw_keys || !values || positions <= 0 || from < 0 || count <= 0 ||
      from >= positions || kv_heads <= 0 || head_dim <= 0 || head_dim % 2 || theta <= 0)
    return failf(-1, "kv_evict_f32: invalid pointer or geometry");
  int end = from + count < positions ? from + count : positions;
  int survivors = positions - end;
  int new_positions = positions - (end - from);
  size_t row_bytes = static_cast<size_t>(kv_heads) * head_dim * sizeof(float);
  if (survivors > 0) {
    size_t bytes = static_cast<size_t>(survivors) * row_bytes;
    void *tmp = nullptr;
    hipError_t hs = hipMalloc(&tmp, bytes * 3);
    if (hs != hipSuccess) return hip_fail("kv evict hipMalloc", hs);
    uint8_t *base = static_cast<uint8_t *>(tmp);
    float *tmp_keys = reinterpret_cast<float *>(base);
    float *tmp_raw = reinterpret_cast<float *>(base + bytes);
    float *tmp_values = reinterpret_cast<float *>(base + bytes * 2);
    hs = hipMemcpyAsync(tmp_keys, keys + static_cast<size_t>(end) * kv_heads * head_dim,
                        bytes, hipMemcpyDeviceToDevice, stream);
    if (hs == hipSuccess)
      hs = hipMemcpyAsync(tmp_raw, raw_keys + static_cast<size_t>(end) * kv_heads * head_dim,
                          bytes, hipMemcpyDeviceToDevice, stream);
    if (hs == hipSuccess)
      hs = hipMemcpyAsync(tmp_values, values + static_cast<size_t>(end) * kv_heads * head_dim,
                          bytes, hipMemcpyDeviceToDevice, stream);
    if (hs == hipSuccess)
      hs = hipMemcpyAsync(keys + static_cast<size_t>(from) * kv_heads * head_dim,
                          tmp_keys, bytes, hipMemcpyDeviceToDevice, stream);
    if (hs == hipSuccess)
      hs = hipMemcpyAsync(raw_keys + static_cast<size_t>(from) * kv_heads * head_dim,
                          tmp_raw, bytes, hipMemcpyDeviceToDevice, stream);
    if (hs == hipSuccess)
      hs = hipMemcpyAsync(values + static_cast<size_t>(from) * kv_heads * head_dim,
                          tmp_values, bytes, hipMemcpyDeviceToDevice, stream);
    if (hs == hipSuccess) hs = hipStreamSynchronize(stream);
    hipError_t free_status = hipFree(tmp);
    if (hs != hipSuccess) return hip_fail("kv evict compact", hs);
    if (free_status != hipSuccess) return hip_fail("kv evict hipFree", free_status);
  }
  int pairs = (new_positions - from) * kv_heads * (head_dim / 2);
  if (pairs > 0) {
    kv_rerope<<<(pairs + 127) / 128, 128, 0, stream>>>(
        raw_keys, keys, from, new_positions, kv_heads, head_dim, theta);
    return launch_status("kv evict re-RoPE launch");
  }
  return 0;
}
