#ifndef FAK_INKLING2_NVFP4_KV_CUH
#define FAK_INKLING2_NVFP4_KV_CUH

// inkling2_nvfp4_kv.cuh: block-16 NVFP4 (OCP E2M1 codes + E4M3FN block scales)
// encode/decode helpers for Inkling-2-Mini global KV cache.
//
// Format (headDim % 16 == 0, 8 blocks of 16 elements for headDim = 128):
//   - Codes: packed uint8_t E2M1 [cap][kvHeads][headDim/2]
//       low nibble (bits[3:0]) = even dim 2*i, high nibble (bits[7:4]) = odd dim 2*i+1
//   - Scales: uint8_t OCP E4M3FN [cap][kvHeads][headDim/16]
//       positive finite E4M3FN byte (0x00..0x7E); 0x00 for zero/sub-threshold block
//   - Dequant(pos, kvh, d) = decode_e2m1(nibble[pos, kvh, d]) * decode_e4m3fn(scale[pos, kvh, d/16])
//
// Uses native Blackwell sm_120f/sm_120a/sm_100+ PTX (cvt.rn.satfinite.e2m1x2.f32,
// cvt.rn.f16x2.e2m1x2, cvt.rn.satfinite.e4m3x2.f32) on the hot path, with an exact
// scalar reference fallback for < sm_100 or generic PTX targets.

#include <cuda_fp16.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__) &&                                                      \
    (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) || \
     defined(__CUDA_ARCH_FEAT_SM100_ALL) ||                                        \
     (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000))
#define INK2_NVFP4_KV_NATIVE 1
#else
#define INK2_NVFP4_KV_NATIVE 0
#endif

static constexpr float kInk2NVFP4Max = 6.0f;            // max representable E2M1 magnitude
static constexpr float kInk2NVFP4MaxBlockAbsmax = 6.0f * 448.0f; // 2688.0f (E4M3FN max scale * E2M1 max)
static constexpr float kInk2NVFP4MinAbsmax = 1e-30f;

// 16-lane half-warp max reduction (lanes 0..15 and 16..31 stay strictly disjoint).
static __device__ __forceinline__ float ink2_halfwarp_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return v;
}

// Exact scalar E2M1 round-to-nearest-even encode (matches cvt.rn.satfinite.e2m1x2.f32 bit-for-bit).
static __device__ __forceinline__ uint8_t ink2_e2m1_encode_scalar(float x) {
  const uint32_t bits = __float_as_uint(x);
  const uint8_t sign = (uint8_t)((bits >> 28) & 0x8u);
  const float ax = fabsf(x);
  uint8_t mag;
  if (ax <= 0.25f) mag = 0u;
  else if (ax < 0.75f) mag = 1u;
  else if (ax <= 1.25f) mag = 2u;
  else if (ax < 1.75f) mag = 3u;
  else if (ax <= 2.5f) mag = 4u;
  else if (ax < 3.5f) mag = 5u;
  else if (ax <= 5.0f) mag = 6u;
  else mag = 7u;
  return (uint8_t)(sign | mag);
}

// Exact scalar E2M1 decode (4-bit nibble -> float32).
static __device__ __forceinline__ float ink2_e2m1_decode_scalar(uint8_t nib) {
  const uint32_t mag = nib & 0x7u;
  const uint32_t e = (mag >> 1) & 3u;
  const uint32_t m = mag & 1u;
  const float v = __uint_as_float(e ? (((e + 126u) << 23) | (m << 22)) : (m ? 0x3f000000u : 0u));
  return (nib & 0x8u) ? -v : v;
}

// Pack two floats (lo -> low nibble bits[3:0], hi -> high nibble bits[7:4]) into one E2M1 byte.
static __device__ __forceinline__ uint8_t ink2_e2m1x2_encode(float lo, float hi) {
#if INK2_NVFP4_KV_NATIVE
  uint16_t r;
  // PTX cvt.rn.satfinite.e2m1x2.f32 t, a, b puts a in high nibble and b in low nibble.
  asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }"
      : "=h"(r) : "f"(hi), "f"(lo));
  return (uint8_t)(r & 0xffu);
#else
  return (uint8_t)(ink2_e2m1_encode_scalar(lo) | (ink2_e2m1_encode_scalar(hi) << 4));
#endif
}

// Decode one packed E2M1 byte (low nibble -> .x, high nibble -> .y) to float2.
static __device__ __forceinline__ float2 ink2_e2m1x2_decode_f2(uint8_t byte) {
#if INK2_NVFP4_KV_NATIVE
  uint32_t h2;
  const uint16_t u16 = (uint16_t)byte;
  asm("{ .reg .b8 t; cvt.u8.u16 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(h2) : "h"(u16));
  return __half22float2(*reinterpret_cast<const __half2 *>(&h2));
#else
  return make_float2(ink2_e2m1_decode_scalar(byte & 0x0fu), ink2_e2m1_decode_scalar(byte >> 4));
#endif
}

// Decode one packed E2M1 byte, multiply by block scale in FP32, and pack to __half2.
static __device__ __forceinline__ uint32_t ink2_e2m1x2_decode_scaled_h2(uint8_t byte, float sc) {
  const float2 f = ink2_e2m1x2_decode_f2(byte);
  const __half2 out = __floats2half2_rn(f.x * sc, f.y * sc);
  return *reinterpret_cast<const uint32_t *>(&out);
}

#endif // FAK_INKLING2_NVFP4_KV_CUH
