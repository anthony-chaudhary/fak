// SPDX-License-Identifier: Apache-2.0
// Original Fak exact-ownership wrapper. Numerical attention remains the existing
// F32 online-softmax kernel in kv_fp8.cuh, not a tensor-core or graph port.
// Include after kv_fp8_append.cuh; intentionally unlinked from production.
#ifndef FAK_CUDA_KV_FP8_ATTENTION_OWNER_CUH
#define FAK_CUDA_KV_FP8_ATTENTION_OWNER_CUH
#include "../cuda_kv_fp8_attention_owner.h"
#include "kv_fp8_append.cuh"

extern "C" fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_attention_owned(
    const fcuda_kv_fp8_append_destination *k,
    const float *q,size_t q_bytes,float *out,size_t out_bytes,
    int tokens,int prefix,int heads,float scale,int *status) {
  using namespace fak_kv_fp8_lifecycle;
  result r=admit(FCUDA_KV_FP8_PHASE_ATTENTION);
  if (r.status) return r;
  const int kvheads=4,hd=256;
  int rows=0,elements=0,old_rows=0;
  if (!k || !fak_kv_fp8_append::geometry(prefix,tokens,kvheads,hd,&rows,&elements,&old_rows) ||
      heads<=0 || heads%kvheads || heads>INT_MAX/hd || tokens>INT_MAX/(heads*hd) ||
      (size_t)(tokens*heads*hd)>SIZE_MAX/sizeof(float) ||
      !(scale>0.0f) || !isfinite(scale))
    return refusal((int)cudaErrorInvalidValue,r.phase);
  const size_t expected=(size_t)tokens*heads*hd*sizeof(float);
  if (q_bytes!=expected || out_bytes!=expected || !fak_kv_fp8_append::input_live(q,expected) ||
      !live(out,expected,FCUDA_KV_FP8_OWNER_WORKSPACE) || (uintptr_t)out%alignof(float) ||
      !live(status,sizeof(int),FCUDA_KV_FP8_OWNER_WORKSPACE) || (uintptr_t)status%alignof(int) ||
      !fak_kv_fp8_append::disjoint(q,expected,out,expected) ||
      !fak_kv_fp8_append::disjoint(q,expected,status,sizeof(int)) ||
      !fak_kv_fp8_append::disjoint(out,expected,status,sizeof(int)))
    return refusal((int)cudaErrorInvalidValue,r.phase);
  const size_t total_rows=(size_t)old_rows+rows,total=total_rows*hd;
  const size_t sizes[5]={total*4,total,total,total_rows*4,total_rows*4};
  for (int i=0;i<5;++i) {
    if (k->bytes[i]!=sizes[i] || !live(k->plane[i],sizes[i],FCUDA_KV_FP8_OWNER_PLANE) ||
        ((i==0 || i>=3) && (uintptr_t)k->plane[i]%alignof(float)) ||
        !fak_kv_fp8_append::disjoint(k->plane[i],sizes[i],q,expected) ||
        !fak_kv_fp8_append::disjoint(k->plane[i],sizes[i],out,expected) ||
        !fak_kv_fp8_append::disjoint(k->plane[i],sizes[i],status,sizeof(int)))
      return refusal((int)cudaErrorInvalidValue,r.phase);
    for (int j=0;j<i;++j)
      if (!fak_kv_fp8_append::disjoint(k->plane[i],sizes[i],k->plane[j],sizes[j]))
        return refusal((int)cudaErrorInvalidValue,r.phase);
  }
  int code=fcuda_kv_fp8_attention_async(q,(const uint8_t *)k->plane[1],(const float *)k->plane[3],
      (const uint8_t *)k->plane[2],(const float *)k->plane[4],out,
      tokens,prefix,heads,kvheads,hd,scale,status);
  return code ? failed(code,r.phase) : r;
}

#ifdef __CUDACC__
// ============================================================================
// UNWIRED group-32 asymmetric K8V4 append + decode attention CUDA kernels.
//
// ADAPTED from magnitudedev/magnitude (Apache-2.0):
//   inference/engine/kernels/kernels/cuda/attention_decode_k8v4.cu:46@8d7dbcc1ba
//   inference/engine/kernels/kernels/cuda/attention_append_k8v4.cu:11@8d7dbcc1ba
//
// Clean-room CUDA C++ expression (no upstream bytes copied):
//   - Group size 32 with IEEE-754 binary16 (scale, zero) coefficients.
//   - Keys: signed int8 affine quantization in [-127, 127].
//   - Values: uint8 nibble-packed 4-bit codes (low=even, high=odd).
//   - Decode attention: F32 online-softmax recurrence across prefix + new tokens.
//
// Status: [SW-VERIFIED] source contract only — guarded under #ifdef __CUDACC__
// so host C++ admission tests compile without <cuda_runtime.h>, and
// intentionally UNWIRED from cuda_kernels.cu (token_time: unmeasured /
// HW-qualification-pending).
// ============================================================================
#include <cuda_fp16.h>

namespace fak_kv_k8v4 {
constexpr int kGroupSize = 32;
constexpr int kAttentionThreads = 128;
constexpr int kAttentionAccMax = 8;

__device__ __forceinline__ bool finite_f32(float x) {
  return (__float_as_uint(x) & 0x7f800000U) != 0x7f800000U;
}

__device__ __forceinline__ uint16_t encode_f16_rn(float x) {
  __half h = __float2half_rn(x);
  return __half_as_ushort(h);
}

__device__ __forceinline__ float decode_f16(uint16_t bits) {
  return __half2float(__ushort_as_half(bits));
}

// One 32-thread warp per (row, group_of_32) quantizes K (int8) and V (packed uint8 int4)
// and stores IEEE-754 binary16 (scale, zero) pairs into kMeta / vMeta.
__global__ void k_kv_k8v4_append(
    int8_t *__restrict__ kCodes, uint16_t *__restrict__ kMeta,
    uint8_t *__restrict__ vCodes, uint16_t *__restrict__ vMeta,
    const float *__restrict__ kSrc, const float *__restrict__ vSrc,
    int prefix, int rows, int hd, int *__restrict__ status) {
  const int groupsPerHead = hd / kGroupSize;
  const int row = (int)blockIdx.x / groupsPerHead;
  const int g = (int)blockIdx.x % groupsPerHead;
  const int lane = (int)threadIdx.x;
  if (row >= rows || lane >= kGroupSize) return;

  const size_t srcIdx = (size_t)row * (size_t)hd + (size_t)g * kGroupSize + (size_t)lane;
  const float kv = kSrc[srcIdx];
  const float vv = vSrc[srcIdx];
  const unsigned bad = (!finite_f32(kv) || !finite_f32(vv)) ? FCUDA_KV_FP8_NONFINITE : 0U;
  if (__any_sync(0xffffffffu, bad != 0U)) {
    if (lane == 0) atomicOr(status, (int)FCUDA_KV_FP8_NONFINITE);
    return;
  }

  float kMin = kv, kMax = kv, vMin = vv, vMax = vv;
  for (int off = 16; off > 0; off >>= 1) {
    kMin = fminf(kMin, __shfl_xor_sync(0xffffffffu, kMin, off));
    kMax = fmaxf(kMax, __shfl_xor_sync(0xffffffffu, kMax, off));
    vMin = fminf(vMin, __shfl_xor_sync(0xffffffffu, vMin, off));
    vMax = fmaxf(vMax, __shfl_xor_sync(0xffffffffu, vMax, off));
  }

  const uint16_t kZeroH = encode_f16_rn(0.5f * (kMin + kMax));
  const float kZero = decode_f16(kZeroH);
  uint16_t kScaleH = 0;
  if (kMax > kMin) {
    const float kSpan = fmaxf(fabsf(kMax - kZero), fabsf(kZero - kMin));
    kScaleH = encode_f16_rn(kSpan / 127.0f);
    if ((kScaleH & 0x7fffU) == 0U && kSpan > 0.0f) kScaleH = 1U;
    if (127.0f * decode_f16(kScaleH) < kSpan && (kScaleH & 0x7fffU) < 0x7bffU) kScaleH++;
  }
  const float kScale = decode_f16(kScaleH);

  uint16_t vZeroH = encode_f16_rn(vMin);
  float vZero = decode_f16(vZeroH);
  if (vZero > vMin && finite_f32(vZero)) {
    if ((vZeroH & 0x7fffU) == 0U) vZeroH = 0x8001U;
    else if ((vZeroH & 0x8000U) == 0U) vZeroH--;
    else if ((vZeroH & 0x7fffU) < 0x7bffU) vZeroH++;
    vZero = decode_f16(vZeroH);
  }
  uint16_t vScaleH = 0;
  if (vMax > vMin) {
    const float vSpan = vMax - vZero;
    vScaleH = encode_f16_rn(vSpan / 15.0f);
    if ((vScaleH & 0x7fffU) == 0U && vSpan > 0.0f) vScaleH = 1U;
    if (15.0f * decode_f16(vScaleH) < vSpan && (vScaleH & 0x7fffU) < 0x7bffU) vScaleH++;
  }
  const float vScale = decode_f16(vScaleH);

  if (!finite_f32(kZero) || !finite_f32(kScale) || !finite_f32(vZero) || !finite_f32(vScale)) {
    if (lane == 0) atomicOr(status, FCUDA_KV_FP8_BAD_SCALE);
    return;
  }

  int kq = 0;
  if (kScale > 0.0f) {
    kq = (int)llroundf((kv - kZero) / kScale);
    if (kq < -127) kq = -127;
    if (kq > 127) kq = 127;
  }
  const int dstRow = prefix + row;
  kCodes[(size_t)dstRow * (size_t)hd + (size_t)g * kGroupSize + (size_t)lane] = (int8_t)kq;

  int vq = 0;
  if (vScale > 0.0f) {
    vq = (int)llroundf((vv - vZero) / vScale);
    if (vq < 0) vq = 0;
    if (vq > 15) vq = 15;
  }
  const int vqNeighbor = __shfl_xor_sync(0xffffffffu, vq, 1);
  if ((lane & 1) == 0) {
    vCodes[(size_t)dstRow * (size_t)(hd / 2) + (size_t)g * 16 + (size_t)(lane >> 1)] =
        (uint8_t)((vq & 0x0f) | ((vqNeighbor & 0x0f) << 4));
  }
  if (lane == 0) {
    const size_t metaIdx = ((size_t)dstRow * (size_t)groupsPerHead + (size_t)g) * 2;
    kMeta[metaIdx] = kScaleH;
    kMeta[metaIdx + 1] = kZeroH;
    vMeta[metaIdx] = vScaleH;
    vMeta[metaIdx + 1] = vZeroH;
  }
}

// One CTA per (token, query_head): online-softmax F32 decode attention over
// group-32 int8 Keys and nibble-packed int4 Values with f16 (scale, zero).
__global__ void k_kv_k8v4_decode_attention(
    const float *__restrict__ q,
    const int8_t *__restrict__ kCodes, const uint16_t *__restrict__ kMeta,
    const uint8_t *__restrict__ vCodes, const uint16_t *__restrict__ vMeta,
    float *__restrict__ out, int tokens, int prefix, int heads, int kvheads,
    int hd, float scale, int *__restrict__ status) {
  const int flat = (int)blockIdx.x;
  const int h = flat % heads;
  const int token = flat / heads;
  if (token >= tokens) return;

  const int tid = (int)threadIdx.x;
  const int kh = h / (heads / kvheads);
  const int count = prefix + token + 1;
  const int groups = hd / kGroupSize;

  extern __shared__ float smem[];
  float *query = smem;
  float *reduction = smem + hd;
  const size_t qbase = ((size_t)token * (size_t)heads + (size_t)h) * (size_t)hd;
  int qBad = 0;
  for (int d = tid; d < hd; d += kAttentionThreads) {
    const float qv = q[qbase + (size_t)d];
    if (!finite_f32(qv)) qBad = 1;
    query[d] = qv;
  }
  if (__syncthreads_or(qBad)) {
    if (tid == 0) atomicOr(status, FCUDA_KV_FP8_NONFINITE);
    return;
  }

  float acc[kAttentionAccMax] = {};
  float maximum = 0.0f;
  float denominator = 0.0f;
  for (int pos = 0; pos < count; ++pos) {
    const size_t row = (size_t)pos * (size_t)kvheads + (size_t)kh;
    float partial = 0.0f;
    int badMeta = 0;
    for (int d = tid; d < hd; d += kAttentionThreads) {
      const int g = d >> 5;
      const size_t metaIdx = (row * (size_t)groups + (size_t)g) * 2;
      const float kScale = decode_f16(kMeta[metaIdx]);
      const float kZero = decode_f16(kMeta[metaIdx + 1]);
      if (!finite_f32(kScale) || kScale < 0.0f || !finite_f32(kZero)) {
        badMeta = 1;
      } else {
        const float kVal = kZero + (float)kCodes[row * (size_t)hd + (size_t)d] * kScale;
        partial += query[d] * kVal;
      }
    }
    if (__syncthreads_or(badMeta)) {
      if (tid == 0) atomicOr(status, FCUDA_KV_FP8_BAD_SCALE);
      return;
    }
    reduction[tid] = partial;
    __syncthreads();
    for (int stride = kAttentionThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) reduction[tid] += reduction[tid + stride];
      __syncthreads();
    }
    const float score = reduction[0] * scale;
    if (!finite_f32(score)) {
      if (tid == 0) atomicOr(status, FCUDA_KV_FP8_BAD_RESULT);
      return;
    }
    __syncthreads();

    float correction = 1.0f;
    float probability = 1.0f;
    if (pos == 0) {
      maximum = score;
      denominator = 1.0f;
    } else {
      const float next = fmaxf(maximum, score);
      correction = expf(maximum - next);
      probability = expf(score - next);
      denominator = denominator * correction + probability;
      maximum = next;
    }
    int slot = 0;
    int badVMeta = 0;
    for (int d = tid; d < hd; d += kAttentionThreads, ++slot) {
      const int g = d >> 5;
      const size_t metaIdx = (row * (size_t)groups + (size_t)g) * 2;
      const float vScale = decode_f16(vMeta[metaIdx]);
      const float vZero = decode_f16(vMeta[metaIdx + 1]);
      if (!finite_f32(vScale) || vScale < 0.0f || !finite_f32(vZero)) {
        badVMeta = 1;
      } else {
        const uint8_t packed = vCodes[row * (size_t)(hd / 2) + (size_t)(d >> 1)];
        const uint8_t nibble = (d & 1) ? (uint8_t)(packed >> 4) : (uint8_t)(packed & 0x0fU);
        const float vVal = vZero + (float)nibble * vScale;
        acc[slot] = (pos == 0) ? vVal : (acc[slot] * correction + probability * vVal);
      }
    }
    if (__syncthreads_or(badVMeta)) {
      if (tid == 0) atomicOr(status, FCUDA_KV_FP8_BAD_SCALE);
      return;
    }
  }

  if (!(denominator > 0.0f) || !finite_f32(denominator)) {
    if (tid == 0) atomicOr(status, FCUDA_KV_FP8_BAD_RESULT);
    return;
  }
  const float inverse = __fdiv_rn(1.0f, denominator);
  int slot = 0;
  for (int d = tid; d < hd; d += kAttentionThreads, ++slot) {
    const float value = acc[slot] * inverse;
    if (!finite_f32(value)) atomicOr(status, FCUDA_KV_FP8_BAD_RESULT);
    out[qbase + (size_t)d] = value;
  }
}
} // namespace fak_kv_k8v4

extern "C" int fcuda_kv_k8v4_append_async(
    int8_t *kCodes, uint16_t *kMeta, uint8_t *vCodes, uint16_t *vMeta,
    const float *kSrc, const float *vSrc,
    int prefix, int tokens, int kvheads, int hd, int *status) {
  using namespace fak_kv_fp8_lifecycle;
  result r = admit(FCUDA_KV_FP8_PHASE_APPEND_INGEST);
  if (r.status) return r.status;
  int rows = 0, elements = 0, old_rows = 0;
  if (!kCodes || !kMeta || !vCodes || !vMeta || !kSrc || !vSrc || !status ||
      hd <= 0 || (hd % fak_kv_k8v4::kGroupSize) != 0 ||
      !fak_kv_fp8_append::geometry(prefix, tokens, kvheads, hd, &rows, &elements, &old_rows)) {
    return (int)cudaErrorInvalidValue;
  }
  const int groupsPerHead = hd / fak_kv_k8v4::kGroupSize;
  const int blocks = rows * groupsPerHead;
  fak_kv_k8v4::k_kv_k8v4_append<<<blocks, fak_kv_k8v4::kGroupSize, 0, g_stream>>>(
      kCodes, kMeta, vCodes, vMeta, kSrc, vSrc, old_rows, rows, hd, status);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess && !g_cuda_cleanup_status) g_cuda_cleanup_status = (int)e;
  return (int)e;
}

extern "C" int fcuda_kv_k8v4_decode_attention_async(
    const float *q, const int8_t *kCodes, const uint16_t *kMeta,
    const uint8_t *vCodes, const uint16_t *vMeta, float *out,
    int tokens, int prefix, int heads, int kvheads, int hd, float scale, int *status) {
  using namespace fak_kv_fp8_lifecycle;
  result r = admit(FCUDA_KV_FP8_PHASE_ATTENTION);
  if (r.status) return r.status;
  if (!q || !kCodes || !kMeta || !vCodes || !vMeta || !out || !status ||
      tokens <= 0 || prefix < 0 || tokens > INT_MAX - prefix ||
      heads <= 0 || kvheads <= 0 || (heads % kvheads) != 0 ||
      hd <= 0 || (hd % fak_kv_k8v4::kGroupSize) != 0 ||
      hd > fak_kv_k8v4::kAttentionThreads * fak_kv_k8v4::kAttentionAccMax ||
      heads > INT_MAX / hd || tokens > INT_MAX / (heads * hd) ||
      kvheads > INT_MAX / hd || (prefix + tokens) > INT_MAX / (kvheads * hd) ||
      !(scale > 0.0f) || !isfinite(scale)) {
    return (int)cudaErrorInvalidValue;
  }
  const size_t shared = (size_t)(hd + fak_kv_k8v4::kAttentionThreads) * sizeof(float);
  fak_kv_k8v4::k_kv_k8v4_decode_attention<<<tokens * heads, fak_kv_k8v4::kAttentionThreads, shared, g_stream>>>(
      q, kCodes, kMeta, vCodes, vMeta, out, tokens, prefix, heads, kvheads, hd, scale, status);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess && !g_cuda_cleanup_status) g_cuda_cleanup_status = (int)e;
  return (int)e;
}
#endif // __CUDACC__
#endif
