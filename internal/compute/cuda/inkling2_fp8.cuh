#ifndef FAK_INKLING2_FP8_CUH
#define FAK_INKLING2_FP8_CUH

// inkling2_fp8.cuh: OCP FP8 E4M3FN encode/decode and per-(token, kvHead) KV
// quantize helpers for Inkling-2-Mini global KV cache.
//
// Format: OCP E4M3FN (1 sign, 4 exponent bias=7, 3 mantissa, no Inf, NaN=0x7F/0xFF).
// Max finite magnitude: 448.0f (0x7E / 0xFE). Smallest normal: 2^-6 (0x08).
// Subnormals: m * 2^-9 (m in 1..7). Writer never emits 0x7F/0xFF.
//
// Per-(token, kvHead) scaling over headDim:
//   scale = absmax / 448.0f (0.0f when absmax < 1e-30f, storing all-zero codes)
//   dequant(pos, kvh, d) = decode_e4m3fn(code[pos, kvh, d]) * scale[pos, kvh]
//
// Overflow detection: if any input element is non-finite (!isfinite) or scale/inv
// is non-finite, atomicOr(overflow, 1) is set and the row is zeroed cleanly.

#include <cuda_fp16.h>
#include <stdint.h>

static constexpr float kInk2FP8Max = 448.0f;
static constexpr float kInk2FP8MinAbsmax = 1e-30f;

static __device__ __forceinline__ float ink2_warp_max(float v) {
  for (int d = 16; d > 0; d >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, d));
  return v;
}

// OCP E4M3FN round-to-nearest-even encode with signed saturation at +/-448.0f.
static __device__ __forceinline__ uint8_t ink2_e4m3fn_encode(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  uint16_t packed;
  // PTX cvt.rn.satfinite.e4m3x2.f32 d, a, b puts b in bits[7:0], a in bits[15:8].
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(0.0f), "f"(x));
  return (uint8_t)(packed & 0xffu);
#else
  if (!(x == x)) return 0x00u;
  if (x >= kInk2FP8Max) return 0x7Eu;
  if (x <= -kInk2FP8Max) return 0xFEu;
  const uint32_t bits = __float_as_uint(x);
  const uint32_t sign = (bits >> 24) & 0x80u;
  const uint32_t mag = bits & 0x7FFFFFFFu;
  if (mag == 0u) return (uint8_t)sign;
  const int exp = (int)((mag >> 23) & 0xFFu) - 127;
  if (exp >= -6) {
    const uint32_t frac = mag & 0x7FFFFFu;
    uint32_t m = frac >> 20;
    const uint32_t rem = frac & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) ++m;
    int e = exp + 7;
    if (m == 8u) {
      m = 0u;
      ++e;
    }
    if (e > 15 || (e == 15 && m > 6u)) return (uint8_t)(sign | 0x7Eu);
    return (uint8_t)(sign | ((uint32_t)e << 3) | m);
  }
  // Subnormal: value = m * 2^-9 => q = |x| * 512.0f in [0, 8].
  const float q = __uint_as_float(mag) * 512.0f;
  uint32_t m = (uint32_t)q;
  const float r = q - (float)m;
  if (r > 0.5f || (r == 0.5f && (m & 1u))) ++m;
  if (m >= 8u) return (uint8_t)(sign | 0x08u);
  return (uint8_t)(sign | m);
#endif
}

// OCP E4M3FN decode to float32. Every finite E4M3FN value is exact in FP16 and FP32.
static __device__ __forceinline__ float ink2_e4m3fn_decode(uint8_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  uint32_t h2;
  const uint16_t packed = (uint16_t)b;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(packed));
  return __low2float(*reinterpret_cast<const __half2 *>(&h2));
#else
  const uint32_t sign = (uint32_t)(b & 0x80u) << 24;
  const uint32_t e = (b >> 3) & 0xFu;
  const uint32_t m = b & 0x7u;
  if (e == 0u) {
    const float v = (float)m * (1.0f / 512.0f);
    return sign ? -v : v;
  }
  return __uint_as_float(sign | ((e + 120u) << 23) | (m << 20));
#endif
}

// Decode two packed E4M3FN bytes (lo = bits[7:0], hi = bits[15:8]), scale in FP32,
// and round once to packed __half2 (for prefill tensor-core smem tile staging).
static __device__ __forceinline__ uint32_t ink2_e4m3fn_decode2_scaled_h2(uint16_t twoBytes, float sc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(twoBytes));
  const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&h2));
  const __half2 out = __floats2half2_rn(f.x * sc, f.y * sc);
  return *reinterpret_cast<const uint32_t *>(&out);
#else
  const float lo = ink2_e4m3fn_decode((uint8_t)(twoBytes & 0xffu)) * sc;
  const float hi = ink2_e4m3fn_decode((uint8_t)(twoBytes >> 8)) * sc;
  const __half2 out = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t *>(&out);
#endif
}

static __device__ __forceinline__ float ink2_kv_f(uint8_t x) { return ink2_e4m3fn_decode(x); }

template <class T>
static __device__ __forceinline__ float ink2_kv_val(T x, const float *sc, size_t slot, int kvHeads, int kvh) {
  float v = ink2_kv_f(x);
  if constexpr (sizeof(T) == 1) v = __fmul_rn(v, sc[slot * (size_t)kvHeads + (size_t)kvh]);
  return v;
}

#endif // FAK_INKLING2_FP8_CUH
