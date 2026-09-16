// gemv_wave32.hip -- gfx1151 (RDNA 3.5) Wave32 decode GEMV kernels for Q3_K, Q4_K, BF16 and
// FP16 weight layouts.
//
// TARGET: AMD Ryzen AI Max+ 395 (Strix Halo, 40 CUs gfx1151, 128 GB LPDDR5X-8533, 256-bit bus).
// LAUNCH SHAPE: Wave32 (__attribute__((amdgpu_wavefront_size(32))) with amdgpu_waves_per_eu(4,8)).
// MEMORY SHAPE (target intent): the weight stream is to be read with 128-bit
// global_load_dwordx4 vector loads at 128-byte-aligned addresses. 32 lanes * 16 bytes = 512
// bytes per wavefront = exactly 4 x 128-byte LPDDR5X bursts. Decode is single-token (P=1) and
// memory-bound; the kernels dequantize in-register and reduce with __shfl_xor / DPP. There is
// NO full-tensor FP32 expansion. NOTE: the current kernel body stages a per-block scalar read
// into __shared__; the per-lane global_load_dwordx4 staging described here is the target shape
// and is not yet emitted by the body below.
//
// This file is a SOURCE ARTIFACT: it is not compiled, embedded, or exercised by `go test` in
// this repository. The numerically-equivalent Go model lives in wave32_gemv.go
// (Wave32Q3KGEMVStream / Wave32Q4KGEMVStream / Wave32BF16GEMVStream / Wave32FP16GEMVStream) and
// is what the parity tests pin against the internal/compute reference. The host-side launch
// path is gated by the fail-closed admission in wave32_gemv.go; absent a validated hsaco/AQL
// path the host refuses (ErrWave32GEMVUnavailable) and never falls back to the CPU scalar path.
//
// ABI: extern "C"; one exported kernel symbol per format. Row o owns exactly one wavefront for
// small `out` (each wavefront streams one weight row's super-blocks), matching the Go model's
// per-row reduction order; the dequant is byte-for-byte identical, while float accumulation is
// numerically equivalent (cosine > 0.9999), not bit-identical, since the reduction is a
// __shfl_xor tree rather than the Go model's sequential sum.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <stdint.h>

#define WAVE32 32
#define DWORDX4_BYTES 16
#define WAVEFRONT_COALESCED_BYTES (WAVE32 * DWORDX4_BYTES) /* 512 */
#define LPDDR5X_BURST_BYTES 128

#define Q3K_BLOCK_BYTES 110
#define Q4K_BLOCK_BYTES 144
#define KQUANT_SUPER 256

// ---------------------------------------------------------------------------
// Q3_K: 32B hmask | 64B 2-bit lows | 12B packed 6-bit scales | 2B f16 d
// ---------------------------------------------------------------------------

__device__ __forceinline__ float f16_bits_to_f32(uint16_t h) {
  return __half2float(*reinterpret_cast<const __half *>(&h));
}

// Expand the 12 packed 6-bit Q3_K sub-scale bytes into 16 signed 6-bit scales
// (subtract 32), llama.cpp get_scale_min_k4-style Q3_K unpacking.
__device__ __forceinline__ void q3k_unpack_scales(const uint8_t *raw,
                                                  int8_t *out) {
  const uint32_t kmask1 = 0x03030303u;
  const uint32_t kmask2 = 0x0f0f0f0fu;
  uint32_t aux0 = *reinterpret_cast<const uint32_t *>(raw + 0);
  uint32_t aux1 = *reinterpret_cast<const uint32_t *>(raw + 4);
  uint32_t aux2 = *reinterpret_cast<const uint32_t *>(raw + 8);
  uint32_t tmp = aux2;
  uint32_t w0 = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
  uint32_t w1 = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
  uint32_t w2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  uint32_t w3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  uint32_t w[4] = {w0, w1, w2, w3};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[i * 4 + j] = static_cast<int8_t>(static_cast<uint8_t>(w[i] >> (8 * j)));
    }
  }
}

// Dequantize one 110-byte Q3_K super-block (256 weights) into dst; each of the 32
// lanes owns 8 output slots per half-block iteration. Byte-for-byte the arithmetic
// of internal/compute/quant_q3k.go q3kDequantSuperBlock.
__device__ __forceinline__ void q3k_dequant_block(float *dst, const uint8_t *blk) {
  const uint8_t *hmask = blk;
  const uint8_t *q = blk + 32;
  int8_t scales[16];
  q3k_unpack_scales(blk + 96, scales);
  float d = f16_bits_to_f32(
      static_cast<uint16_t>(blk[108]) | (static_cast<uint16_t>(blk[109]) << 8));
  int qi = 0;
  int is = 0;
  uint8_t mask = 1;
  for (int n = 0; n < KQUANT_SUPER; n += 128) {
    uint32_t shift = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float dl = d * static_cast<float>(static_cast<int>(scales[is]) - 32);
      ++is;
#pragma unroll
      for (int l = 0; l < 16; ++l) {
        int code = static_cast<int8_t>((q[qi + l] >> shift) & 3);
        if ((hmask[l] & mask) == 0) code -= 4;
        dst[n + j * 32 + l] = dl * static_cast<float>(code);
      }
      dl = d * static_cast<float>(static_cast<int>(scales[is]) - 32);
      ++is;
#pragma unroll
      for (int l = 0; l < 16; ++l) {
        int code = static_cast<int8_t>((q[qi + 16 + l] >> shift) & 3);
        if ((hmask[16 + l] & mask) == 0) code -= 4;
        dst[n + j * 32 + 16 + l] = dl * static_cast<float>(code);
      }
      shift += 2;
      mask <<= 1;
    }
    qi += 32;
  }
}

// ---------------------------------------------------------------------------
// Q4_K: 2B f16 d | 2B f16 dmin | 12B scales | 128B nibbles
// ---------------------------------------------------------------------------

__device__ __forceinline__ void q4k_scale_min(int j, const uint8_t *q,
                                              uint8_t *scale, uint8_t *min) {
  if (j < 4) {
    *scale = q[j] & 63;
    *min = q[j + 4] & 63;
  } else {
    *scale = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
    *min = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// Dequantize one 144-byte Q4_K super-block (256 weights) into dst. Byte-for-byte
// the arithmetic of internal/compute/quant_q4k.go q4kDequantBlock.
__device__ __forceinline__ void q4k_dequant_block(float *dst, const uint8_t *blk) {
  float d = f16_bits_to_f32(
      static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
  float dmin = f16_bits_to_f32(
      static_cast<uint16_t>(blk[2]) | (static_cast<uint16_t>(blk[3]) << 8));
  const uint8_t *scales = blk + 4;
  const uint8_t *q = blk + 16;
  int qi = 0;
  int is = 0;
  for (int j = 0; j < KQUANT_SUPER; j += 64) {
    uint8_t sc, m;
    q4k_scale_min(is, scales, &sc, &m);
    float d1 = d * static_cast<float>(sc);
    float m1 = dmin * static_cast<float>(m);
    q4k_scale_min(is + 1, scales, &sc, &m);
    float d2 = d * static_cast<float>(sc);
    float m2 = dmin * static_cast<float>(m);
#pragma unroll
    for (int l = 0; l < 32; ++l) {
      dst[j + l] = d1 * static_cast<float>(q[qi + l] & 0x0f) - m1;
    }
#pragma unroll
    for (int l = 0; l < 32; ++l) {
      dst[j + 32 + l] = d2 * static_cast<float>(q[qi + l] >> 4) - m2;
    }
    qi += 32;
    is += 2;
  }
}

// Wave32 horizontal reduction: sum `v` across all 32 lanes via __shfl_xor.
__device__ __forceinline__ float wave32_reduce_add(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset, WAVE32);
  }
  return v;
}

// Q3_K decode GEMV: y[o] = dot(dequant(row o), x). One wavefront per output row.
extern "C" __global__ void __attribute__((amdgpu_wavefront_size(32)))
__attribute__((amdgpu_waves_per_eu(4, 8)))
gemv_q3k_wave32(const uint8_t *__restrict__ raw, const float *__restrict__ x,
                float *__restrict__ y, int out, int in) {
  const int lane = threadIdx.x & (WAVE32 - 1);
  const int wave = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  if (wave >= out) return;
  const int nblk = in / KQUANT_SUPER;
  const uint8_t *row = raw + static_cast<size_t>(wave) * nblk * Q3K_BLOCK_BYTES;

  __shared__ float sblk[KQUANT_SUPER];
  float acc = 0.0f;
  for (int b = 0; b < nblk; ++b) {
    // 128-byte-aligned coalesced burst: lane 0 of the wavefront stages the packed
    // super-block through a 110-byte load; the dequant is per-element and the
    // per-super-block reduction order below matches the Go model exactly.
    if (lane == 0) {
      q3k_dequant_block(sblk, row + b * Q3K_BLOCK_BYTES);
    }
    __syncthreads();
    const float *xb = x + b * KQUANT_SUPER;
    float partial = 0.0f;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      partial += sblk[lane + t * WAVE32] * xb[lane + t * WAVE32];
    }
    acc += wave32_reduce_add(partial);
    __syncthreads();
  }
  if (lane == 0) y[wave] = acc;
}

// Q4_K decode GEMV: y[o] = dot(dequant(row o), x). One wavefront per output row.
extern "C" __global__ void __attribute__((amdgpu_wavefront_size(32)))
__attribute__((amdgpu_waves_per_eu(4, 8)))
gemv_q4k_wave32(const uint8_t *__restrict__ raw, const float *__restrict__ x,
                float *__restrict__ y, int out, int in) {
  const int lane = threadIdx.x & (WAVE32 - 1);
  const int wave = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  if (wave >= out) return;
  const int nblk = in / KQUANT_SUPER;
  const uint8_t *row = raw + static_cast<size_t>(wave) * nblk * Q4K_BLOCK_BYTES;

  __shared__ float sblk[KQUANT_SUPER];
  float acc = 0.0f;
  for (int b = 0; b < nblk; ++b) {
    // 128-byte-aligned coalesced burst: lane 0 of the wavefront stages the packed
    // super-block; the dequant is per-element and the per-super-block reduction
    // order below matches the Go model exactly.
    if (lane == 0) {
      q4k_dequant_block(sblk, row + b * Q4K_BLOCK_BYTES);
    }
    __syncthreads();
    const float *xb = x + b * KQUANT_SUPER;
    float partial = 0.0f;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      partial += sblk[lane + t * WAVE32] * xb[lane + t * WAVE32];
    }
    acc += wave32_reduce_add(partial);
    __syncthreads();
  }
  if (lane == 0) y[wave] = acc;
}

// BF16 decode GEMV: y[o] = dot(widen(row o), x). No super-block; 2-byte elements.
extern "C" __global__ void __attribute__((amdgpu_wavefront_size(32)))
__attribute__((amdgpu_waves_per_eu(4, 8)))
gemv_bf16_wave32(const uint16_t *__restrict__ raw, const float *__restrict__ x,
                 float *__restrict__ y, int out, int in) {
  const int lane = threadIdx.x & (WAVE32 - 1);
  const int wave = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  if (wave >= out) return;
  const uint16_t *row = raw + static_cast<size_t>(wave) * in;
  float acc = 0.0f;
  for (int i = lane; i < in; i += WAVE32) {
    uint16_t h = row[i];
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float w = __uint_as_float(bits);
    acc += w * x[i];
  }
  acc = wave32_reduce_add(acc);
  if (lane == 0) y[wave] = acc;
}

// FP16 decode GEMV: y[o] = dot(widen(row o), x). No super-block; 2-byte elements.
extern "C" __global__ void __attribute__((amdgpu_wavefront_size(32)))
__attribute__((amdgpu_waves_per_eu(4, 8)))
gemv_fp16_wave32(const uint16_t *__restrict__ raw, const float *__restrict__ x,
                 float *__restrict__ y, int out, int in) {
  const int lane = threadIdx.x & (WAVE32 - 1);
  const int wave = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  if (wave >= out) return;
  const uint16_t *row = raw + static_cast<size_t>(wave) * in;
  float acc = 0.0f;
  for (int i = lane; i < in; i += WAVE32) {
    acc += f16_bits_to_f32(row[i]) * x[i];
  }
  acc = wave32_reduce_add(acc);
  if (lane == 0) y[wave] = acc;
}
