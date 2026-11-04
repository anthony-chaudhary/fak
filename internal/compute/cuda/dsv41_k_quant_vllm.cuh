#ifndef FAK_DSV41_K_QUANT_VLLM_CUH
#define FAK_DSV41_K_QUANT_VLLM_CUH

// V4.1 mxfp8 K-plane record helpers shared by the quantize-and-insert and
// dequantize-and-gather kernels in ../dsv41_k_quant.cu.
//
// ADAPTED from vllm-project/vllm,
// vllm/models/deepseek_v41/common/ops/cache_utils.py:184,528
// (_quantize_and_insert_k_mxfp8_kernel, _dequantize_and_gather_k_mxfp8_kernel)
// at ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0; attribution in
// THIRD_PARTY_NOTICES.md). Re-expressed against fak conventions; no upstream
// code copied. Ref oss-port-vllmtop-dsv41-k-quant-cache-20261103.
//
// Record (V4.1 mxfp8, 528 B/token, interleaved per token inside a page):
//   [0,512)   e4m3 payload, one byte per dim (RoPE dims included)
//   [512,528) 16 ue8m0 scale bytes, one per 32-dim tile
// Delta from upstream: the paged page holds 16 tokens (the fak paged writer,
// paged_kv_write_map.h, admits block_size == 16 only), not upstream's
// 64/128-token pages. The slot decomposition below is the same
// pos -> (block, offset) pair that paged_kv_write_offsets uses for the f32
// planes.

#include <stdint.h>
#include "paged_kv_write_map.h"

namespace fak_dsv41_kq {

constexpr int kHeadDim = 512;
constexpr int kQuantBlock = 32;
constexpr int kNumScales = 16;
constexpr int kBytesPerToken = kHeadDim + kNumScales;  // 528
constexpr int kPageSize = 16;                          // fak paged page
constexpr int kPageBytes = kPageSize * kBytesPerToken;  // 8448
constexpr float kFP8Max = 448.0f;                       // OCP e4m3fn max
constexpr float kScaleFloor = 1e-4f;                    // upstream absmax floor

#ifdef __CUDACC__
#define FAK_DSV41_KQ_INLINE __host__ __device__ __forceinline__
#else
#define FAK_DSV41_KQ_INLINE inline
#endif

union Dsv41KBits {
  float f;
  uint32_t u;
};

FAK_DSV41_KQ_INLINE uint32_t dsv41_kq_float_bits(float x) {
  Dsv41KBits b;
  b.f = x;
  return b.u;
}

FAK_DSV41_KQ_INLINE float dsv41_kq_bits_float(uint32_t u) {
  Dsv41KBits b;
  b.u = u;
  return b.f;
}

// UE8M0 tile exponent: the smallest e with 2^e >= amax/kFP8Max, amax floored
// at kScaleFloor so an all-zero tile still lands on a finite scale. raw is a
// positive normal f32, so ceil(log2(raw)) resolves from its bits without a
// transcendental: unbiased exponent field, plus one when the mantissa is
// nonzero. Matches the Go reference dsv41kTileExponent bit for bit.
FAK_DSV41_KQ_INLINE int dsv41_kq_ue8m0_exponent(float amax) {
  if (amax < kScaleFloor) amax = kScaleFloor;
  const float raw = amax / kFP8Max;
  const uint32_t bits = dsv41_kq_float_bits(raw);
  int exp = int((bits >> 23) & 0xFFu) - 127;
  if (bits & 0x7FFFFFu) ++exp;
  return exp;
}

// Bias the exponent by 127 and clamp to the uint8 range. Upstream clamps to
// 255 (not 254), so the e8m0 NaN byte is a valid stored scale; the reachable
// exponent range makes it unemitted in practice.
FAK_DSV41_KQ_INLINE uint8_t dsv41_kq_encode_scale(int exp) {
  int enc = exp + 127;
  if (enc < 0) enc = 0;
  if (enc > 255) enc = 255;
  return (uint8_t)enc;
}

FAK_DSV41_KQ_INLINE float dsv41_kq_decode_scale(uint8_t enc) {
  return exp2f((float)enc - 127.0f);
}

// OCP e4m3fn encode: round-to-nearest-even with signed saturation at +/-448
// (codes 0x7E/0xFE). The 0x7F NaN code is never emitted. Mirrors the Go
// reference dsv41kEncodeE4M3 bit for bit.
FAK_DSV41_KQ_INLINE uint8_t dsv41_kq_e4m3_encode(float x) {
  if (x >= kFP8Max) return 0x7Eu;
  if (x <= -kFP8Max) return 0xFEu;
  const uint32_t bits = dsv41_kq_float_bits(x);
  const uint32_t sign = bits >> 31;
  const uint32_t mag = bits & 0x7FFFFFFFu;
  if (mag == 0) return (uint8_t)(sign << 7);
  const int exp = int((mag >> 23) & 0xFFu) - 127;
  if (exp >= -6) {
    const uint32_t frac = mag & 0x7FFFFFu;
    uint32_t m = frac >> 20;
    const uint32_t rem = frac & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) ++m;
    int e = exp + 7;
    if (m == 8u) {
      m = 0;
      ++e;
    }
    if (e > 15) return (uint8_t)((sign << 7) | 0x7Eu);
    return (uint8_t)((sign << 7) | ((uint32_t)e << 3) | m);
  }
  // Subnormal target: q = |x| * 2^13 is exact in f32 and lies in [0, 8).
  const float q = dsv41_kq_bits_float(mag) * 8192.0f;
  uint32_t m = (uint32_t)q;
  const float r = q - (float)m;
  if (r > 0.5f || (r == 0.5f && (m & 1u))) ++m;
  if (m > 7u) m = 7u;
  return (uint8_t)((sign << 7) | m);
}

// OCP e4m3fn decode. Normals rebuild the f32 bits directly (exact); the
// subnormal value m*2^-13 is an exact power-of-two scaling. The writer never
// emits the 0x7F NaN code; decoding it as the finite 480 keeps the kernel
// free of NaN propagation paths.
FAK_DSV41_KQ_INLINE float dsv41_kq_e4m3_decode(uint8_t b) {
  const uint32_t sign = (uint32_t)(b >> 7);
  const uint32_t e = (b >> 3) & 0xFu;
  const uint32_t m = b & 0x7u;
  if (e == 0) return ldexpf((float)m, -13);
  return dsv41_kq_bits_float((sign << 31) | ((e + 7u) << 23) | (m << 20));
}

// Byte base of one token record inside the paged mxfp8 K arena:
// page*8448 + inPageOffset*528. The (page, offset) split is verbatim the
// paged_kv_write_offsets decomposition (pos = first_offset + token;
// page = blocks[pos/page]; offset = pos%page), so both planes address the
// same physical slot.
FAK_DSV41_KQ_INLINE int64_t dsv41_kq_record_slot(int block_idx, int pos) {
  return int64_t(block_idx) * kPageBytes +
      int64_t(pos % kPageSize) * kBytesPerToken;
}

FAK_DSV41_KQ_INLINE int64_t dsv41_kq_insert_slot(const PagedKVWriteMap& pages,
                                                 int token) {
  const int pos = pages.first_offset + token;
  return dsv41_kq_record_slot(pages.blocks[pos / kPageSize], pos);
}

FAK_DSV41_KQ_INLINE int64_t dsv41_kq_gather_slot(
    const int* block_table_row, int pos) {
  return dsv41_kq_record_slot(block_table_row[pos / kPageSize], pos);
}

#undef FAK_DSV41_KQ_INLINE
}  // namespace fak_dsv41_kq
#endif  // FAK_DSV41_K_QUANT_VLLM_CUH
