// SPDX-License-Identifier: Apache-2.0
//
// FP8 KV storage mechanism adapted from vllm-project/vllm
// fa3b9d1377219247060d3c8b49eae199c9e7e154:
// csrc/quantization/w8a8/fp8/common.cu,
// csrc/quantization/w8a8/fp8/common.cuh, and
// csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh.
// Row maximum reduction and native saturating E4M3 conversion are retained.
// Fak uses amax/448 with zero/underflow handling, NOT the donor 1/(448*512)
// floor. PyTorch/CUB dispatch, ROCm, paged layout and scaled-half vector helpers
// are not imported. F32 attention recurrence follows Fak kv_precision.cuh.
//
// This fragment is intentionally not included in cuda_kernels.cu until the
// five-plane owner/status/view/rotary integration and strict build admission
// land coherently. Source presence is not a realized KV tier.
#ifndef FAK_CUDA_KV_FP8_CUH
#define FAK_CUDA_KV_FP8_CUH
#ifndef FAK_CUDA_KV_FP8_STRICT_ARITHMETIC
#error "FP8 KV requires admitted no-FTZ, precise division/multiply build flags"
#endif
#include "../cuda_kv_fp8.h"
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <limits.h>
#include <math.h>

namespace fak_kv_fp8 {
constexpr int kThreads = 256;
constexpr int kAttentionThreads = 128;
constexpr int kAttentionAcc = 8;

static int admit() {
  if (v41_candidates_stream_uncertain())
    return g_cuda_cleanup_status != 0 ? g_cuda_cleanup_status : (int)cudaErrorNotReady;
  if (g_capture_open) return (int)cudaErrorStreamCaptureUnsupported;
  return 0;
}
static int launch_status(cudaError_t status) {
  if (status != cudaSuccess && g_cuda_cleanup_status == 0)
    g_cuda_cleanup_status = (int)status;
  return (int)status;
}
// Quantize/dequantize accept hd up to INT_MAX. Their dimension induction
// variables must be size_t: the final increment can exceed INT_MAX even
// though every visited element and rows*hd remain within the int ABI bound.
static bool geometry(int rows, int hd) {
  return rows > 0 && hd > 0 && rows <= INT_MAX / hd;
}
__device__ __forceinline__ bool finite(float x) {
  return (__float_as_uint(x) & 0x7f800000U) != 0x7f800000U;
}
__device__ __forceinline__ float widen(uint8_t code, float scale) {
  // Only the UNscaled FP8-to-half conversion is used. Every finite E4M3
  // value is exactly representable in half. Scaling must happen in F32.
  __half_raw h = __nv_cvt_fp8_to_halfraw(code, __NV_E4M3);
  return __fmul_rn(__half2float(__ushort_as_half(h.x)), scale);
}
__device__ __forceinline__ uint8_t encode(float x, float scale) {
  float normalized = __fdiv_rn(x, scale);
  normalized = fmaxf(-448.0f, fminf(448.0f, normalized));
  return __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3);
}

__global__ void validate3(const float *raw, const float *key, const float *value,
                         int n, int *status) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n && (!finite(raw[i]) || !finite(key[i]) || !finite(value[i])))
    atomicOr(status, FCUDA_KV_FP8_NONFINITE);
}

__global__ void quantize(uint8_t *codes, float *scales, const float *src,
                        int rows, int hd, int *status) {
  int row = blockIdx.x, tid = threadIdx.x;
  if (row >= rows) return;
  __shared__ float maxima[kThreads];
  __shared__ unsigned invalid[kThreads];
  size_t base = (size_t)row * hd;
  float maximum = 0.0f;
  unsigned bad = 0;
  for (size_t d = (size_t)tid; d < (size_t)hd; d += (size_t)kThreads) {
    float x = src[base + d];
    if (!finite(x)) bad |= FCUDA_KV_FP8_NONFINITE;
    else maximum = fmaxf(maximum, __uint_as_float(__float_as_uint(x) & 0x7fffffffU));
  }
  maxima[tid] = maximum;
  invalid[tid] = bad;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      maxima[tid] = fmaxf(maxima[tid], maxima[tid + stride]);
      invalid[tid] |= invalid[tid + stride];
    }
    __syncthreads();
  }
  if (invalid[0] != 0) {
    if (tid == 0) atomicOr(status, (int)invalid[0]);
    return;
  }
  // All-zero rows preserve signed-zero codes under positive scale one.
  float scale = maxima[0] == 0.0f ? 1.0f : __fdiv_rn(maxima[0], 448.0f);
  if (scale == 0.0f) scale = __uint_as_float(1U);
  if (tid == 0) scales[row] = scale;
  for (size_t d = (size_t)tid; d < (size_t)hd; d += (size_t)kThreads) {
    uint8_t code = encode(src[base + d], scale);
    if (!finite(widen(code, scale))) atomicOr(status, FCUDA_KV_FP8_BAD_RESULT);
    codes[base + d] = code;
  }
}

__global__ void dequantize(float *dst, const uint8_t *codes, const float *scales,
                          int rows, int hd, int *status) {
  int row = blockIdx.x, tid = threadIdx.x;
  if (row >= rows) return;
  __shared__ unsigned invalid[kThreads];
  size_t base = (size_t)row * hd;
  float scale = scales[row];
  unsigned bad = (!(scale > 0.0f) || !finite(scale)) ? FCUDA_KV_FP8_BAD_SCALE : 0;
  for (size_t d = (size_t)tid; d < (size_t)hd; d += (size_t)kThreads) {
    uint8_t code = codes[base + d];
    if ((code & 0x7fU) == 0x7fU) bad |= FCUDA_KV_FP8_BAD_CODE;
    else if (!finite(widen(code, scale))) bad |= FCUDA_KV_FP8_BAD_RESULT;
  }
  invalid[tid] = bad;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) invalid[tid] |= invalid[tid + stride];
    __syncthreads();
  }
  if (invalid[0] != 0) {
    if (tid == 0) atomicOr(status, (int)invalid[0]);
    return;
  }
  for (size_t d = (size_t)tid; d < (size_t)hd; d += (size_t)kThreads)
    dst[base + d] = widen(codes[base + d], scale);
}

// One CTA per query/token: ordinary F32 online-softmax recurrence, not a
// tensor-core kernel and not a claim of GQA-shared HBM traffic. Scalar loads
// intentionally avoid the donor's lossy scaled-half2 vector specializations.
__global__ void attention(const float *q, const uint8_t *k, const float *ks,
                          const uint8_t *v, const float *vs, float *out,
                          int tokens, int prefix, int heads, int kvheads,
                          int hd, float scale, int *status) {
  int flat = blockIdx.x, h = flat % heads, token = flat / heads;
  if (token >= tokens) return;
  int tid = threadIdx.x, kh = h / (heads / kvheads);
  int count = prefix + token + 1;
  extern __shared__ float smem[];
  float *query = smem, *reduction = smem + hd;
  size_t qbase = ((size_t)token * heads + h) * hd;
  for (int d = tid; d < hd; d += kAttentionThreads) query[d] = q[qbase + d];
  __syncthreads();
  float acc[kAttentionAcc] = {};
  float maximum = 0.0f, denominator = 0.0f;
  for (int pos = 0; pos < count; ++pos) {
    size_t scale_index = (size_t)pos * kvheads + kh;
    size_t base = scale_index * hd;
    float key_scale = ks[scale_index], value_scale = vs[scale_index];
    if (!(key_scale > 0.0f) || !(value_scale > 0.0f) ||
        !finite(key_scale) || !finite(value_scale)) {
      if (tid == 0) atomicOr(status, FCUDA_KV_FP8_BAD_SCALE);
      return;
    }
    float partial = 0.0f;
    for (int d = tid; d < hd; d += kAttentionThreads)
      partial += query[d] * widen(k[base + d], key_scale);
    reduction[tid] = partial;
    __syncthreads();
    for (int stride = kAttentionThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) reduction[tid] += reduction[tid + stride];
      __syncthreads();
    }
    float score = reduction[0] * scale;
    if (!finite(score)) {
      if (tid == 0) atomicOr(status, FCUDA_KV_FP8_BAD_RESULT);
      return;
    }
    __syncthreads();
    if (pos == 0) {
      maximum = score;
      denominator = 1.0f;
      int slot = 0;
      for (int d = tid; d < hd; d += kAttentionThreads, ++slot)
        acc[slot] = widen(v[base + d], value_scale);
    } else {
      float next = fmaxf(maximum, score);
      float correction = expf(maximum - next), probability = expf(score - next);
      denominator = denominator * correction + probability;
      int slot = 0;
      for (int d = tid; d < hd; d += kAttentionThreads, ++slot)
        acc[slot] = acc[slot] * correction + probability * widen(v[base + d], value_scale);
      maximum = next;
    }
  }
  float inverse = __fdiv_rn(1.0f, denominator);
  int slot = 0;
  for (int d = tid; d < hd; d += kAttentionThreads, ++slot) {
    float value = acc[slot] * inverse;
    if (!finite(value)) atomicOr(status, FCUDA_KV_FP8_BAD_RESULT);
    out[qbase + d] = value;
  }
}
} // namespace fak_kv_fp8

extern "C" int fcuda_kv_fp8_begin_async(int *status) {
  if (int closed = fak_kv_fp8::admit()) return closed;
  if (!status) return (int)cudaErrorInvalidValue;
  return fak_kv_fp8::launch_status(cudaMemsetAsync(status, 0, sizeof(int), g_stream));
}
extern "C" int fcuda_kv_fp8_validate_async(const float *raw, const float *key,
    const float *value, int n, int *status) {
  if (int closed = fak_kv_fp8::admit()) return closed;
  if (!raw || !key || !value || !status || n <= 0) return (int)cudaErrorInvalidValue;
  fak_kv_fp8::validate3<<<((size_t)n + 255)/256,256,0,g_stream>>>(raw,key,value,n,status);
  return fak_kv_fp8::launch_status(cudaGetLastError());
}
extern "C" int fcuda_kv_fp8_quantize_async(uint8_t *codes, float *scales,
    const float *src, int rows, int hd, int *status) {
  if (int closed = fak_kv_fp8::admit()) return closed;
  if (!codes || !scales || !src || !status || !fak_kv_fp8::geometry(rows,hd))
    return (int)cudaErrorInvalidValue;
  fak_kv_fp8::quantize<<<rows,256,0,g_stream>>>(codes,scales,src,rows,hd,status);
  return fak_kv_fp8::launch_status(cudaGetLastError());
}
extern "C" int fcuda_kv_fp8_widen_async(float *dst, const uint8_t *codes,
    const float *scales, int rows, int hd, int *status) {
  if (int closed = fak_kv_fp8::admit()) return closed;
  if (!dst || !codes || !scales || !status || !fak_kv_fp8::geometry(rows,hd))
    return (int)cudaErrorInvalidValue;
  fak_kv_fp8::dequantize<<<rows,256,0,g_stream>>>(dst,codes,scales,rows,hd,status);
  return fak_kv_fp8::launch_status(cudaGetLastError());
}
extern "C" int fcuda_kv_fp8_attention_async(const float *q,
    const uint8_t *k, const float *ks, const uint8_t *v, const float *vs,
    float *out, int tokens, int prefix, int heads, int kvheads, int hd,
    float scale, int *status) {
  if (int closed = fak_kv_fp8::admit()) return closed;
  if (!q || !k || !ks || !v || !vs || !out || !status ||
      tokens <= 0 || prefix < 0 || tokens > INT_MAX-prefix ||
      heads <= 0 || kvheads <= 0 || heads%kvheads != 0 ||
      hd <= 0 || hd > fak_kv_fp8::kAttentionThreads*fak_kv_fp8::kAttentionAcc ||
      heads > INT_MAX/hd || tokens > INT_MAX/(heads*hd) ||
      kvheads > INT_MAX/hd || prefix+tokens > INT_MAX/(kvheads*hd) ||
      !(scale > 0.0f) || !isfinite(scale))
    return (int)cudaErrorInvalidValue;
  size_t shared = (size_t)(hd + fak_kv_fp8::kAttentionThreads) * sizeof(float);
  fak_kv_fp8::attention<<<tokens*heads,128,shared,g_stream>>>(
      q,k,ks,v,vs,out,tokens,prefix,heads,kvheads,hd,scale,status);
  return fak_kv_fp8::launch_status(cudaGetLastError());
}
#endif
