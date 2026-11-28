// nvfp4_w4a4_gemm.cuh: native sm_120f NVFP4 W4A4 Tensor Core GEMM for Inkling-2
// batched prefill (T >= 16) and multi-row speculative verify (K >= 4).
//
// Computes Y[M][N] = X[M][K] * W[N][K]^T where W is resident NVFP4 (packed E2M1,
// per-16 UE4M3FN block scales, FP32 global scale) and X is dynamically quantized
// per row to block-16 E2M1 + UE4M3FN scales + FP32 row scale.
//
// Uses Blackwell sm_120f block-scaled MMA:
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__) &&                                                    \
    (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) || \
     (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))
#define INK2_W4A4_NATIVE 1
#else
#define INK2_W4A4_NATIVE 0
#endif

namespace ink2_w4a4 {

constexpr int kBK = 64;
constexpr int kQuantThreads = 128;
constexpr int kGemmWarps = 4;
constexpr int kGemmThreads = kGemmWarps * 32;

#if INK2_W4A4_NATIVE
static __device__ __forceinline__ uint32_t cvt_e2m1x2(float lo, float hi) {
  uint16_t r;
  asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

static __device__ __forceinline__ uint32_t cvt_e4m3(float x) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"(x));
  return r & 0xffu;
}

static __device__ __forceinline__ float e4m3_to_f32(uint32_t code) {
  uint32_t h2;
  asm("{.reg .b16 r; cvt.u16.u32 r, %1; cvt.rn.f16x2.e4m3x2 %0, r;}\n" : "=r"(h2) : "r"(code & 0xffu));
  return __half22float2(*reinterpret_cast<__half2 *>(&h2)).x;
}

static __device__ __forceinline__ float e2m1_abs(uint32_t c) {
  const uint32_t e = (c >> 1) & 3u, m = c & 1u;
  return __uint_as_float(e ? (((e + 126u) << 23) | (m << 22)) : (m ? 0x3f000000u : 0u));
}

static __device__ __forceinline__ float block16_err(const float (&v)[16], float sc) {
  const float inv = sc > 0.f ? __frcp_rn(sc) : 0.f;
  float err = 0.f;
#pragma unroll
  for (int k = 0; k < 16; k += 2) {
    const uint32_t q = cvt_e2m1x2(__fmul_rn(v[k], inv), __fmul_rn(v[k + 1], inv));
    const float e0 = __fsub_rn(fabsf(v[k]), __fmul_rn(e2m1_abs(q & 0xfu), sc));
    const float e1 = __fsub_rn(fabsf(v[k + 1]), __fmul_rn(e2m1_abs(q >> 4), sc));
    err = fmaf(e0, e0, err);
    err = fmaf(e1, e1, err);
  }
  return err;
}

static __device__ __forceinline__ void mma_m16n8k64_nvfp4(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                                         uint32_t a3, uint32_t b0, uint32_t b1, uint32_t sfa,
                                                         uint32_t sfb) {
  asm("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

static __device__ __forceinline__ int swz64(int r, int c) { return r * 64 + ((c ^ ((r >> 1) & 3)) << 4); }
#endif

__device__ int g_ink2_w4a4_flag = 0;
__global__ void k_ink2_w4a4_probe() {
  if (threadIdx.x == 0) g_ink2_w4a4_flag = INK2_W4A4_NATIVE;
}

// Dynamic block-16 activation quantizer for dense projections.
// X is [M, in], output Q is [M, inPad/2], QS is [M, inPad/16], RS is [M].
// Tail columns [in, inPad) are zero-filled so K-tiles of 64 read valid zeros.
__global__ void __launch_bounds__(kQuantThreads) k_ink2_w4a4_quant(const float *__restrict__ X,
                                                                   uint8_t *__restrict__ Q,
                                                                   uint8_t *__restrict__ QS,
                                                                   float *__restrict__ RS,
                                                                   int M, int in, int inPad) {
#if INK2_W4A4_NATIVE
  __shared__ float wmax[kQuantThreads / 32];
  const int r = blockIdx.x;
  if (r >= M) return;
  const float *x = X + (size_t)r * in;
  float amax = 0.f;
  for (int i = threadIdx.x; i < in; i += kQuantThreads) amax = fmaxf(amax, fabsf(x[i]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if ((threadIdx.x & 31) == 0) wmax[threadIdx.x >> 5] = amax;
  __syncthreads();
  amax = wmax[0];
#pragma unroll
  for (int w = 1; w < kQuantThreads / 32; ++w) amax = fmaxf(amax, wmax[w]);
  __syncthreads();

  float rs = __fdiv_rn(amax, 6.f * 448.f);
  if (!isfinite(rs)) rs = 0.f;
  if (threadIdx.x == 0) RS[r] = rs;
  const float ic = rs > 0.f ? __frcp_rn(rs) : 0.f;

  const int nb = in / 16;
  const int nbPad = inPad / 16;
  for (int b = threadIdx.x; b < nbPad; b += kQuantThreads) {
    if (b >= nb) {
      *reinterpret_cast<uint2 *>(Q + (size_t)r * (inPad / 2) + b * 8) = make_uint2(0u, 0u);
      QS[(size_t)r * nbPad + b] = 0u;
      continue;
    }
    float vals[16], v[16], amax16 = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      vals[k] = x[b * 16 + k];
      v[k] = __fmul_rn(vals[k], ic);
      amax16 = fmaxf(amax16, fabsf(v[k]));
    }
    const float a6 = __fdiv_rn(amax16, 6.f);
    const int first = a6 > 0.f ? (int)cvt_e4m3(a6) : 0;
    int code = first > 0x7e ? 0x7e : first;
    float sc = e4m3_to_f32((uint32_t)code);
    float best = block16_err(v, sc);
#pragma unroll
    for (int i = 1; i < 5; ++i) {
      const int tc = first + ((i & 1) ? -((i + 1) >> 1) : (i >> 1));
      if (tc < 0 || tc > 0x7e) continue;
      const float tsc = e4m3_to_f32((uint32_t)tc);
      const float cur = block16_err(v, tsc);
      if (cur < best) {
        best = cur;
        code = tc;
        sc = tsc;
      }
    }
    const float s = __fmul_rn(ic, sc > 0.f ? __frcp_rn(sc) : 0.f);
    uint32_t q0 = 0, q1 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q0 |= cvt_e2m1x2(__fmul_rn(vals[2 * i], s), __fmul_rn(vals[2 * i + 1], s)) << (8 * i);
      q1 |= cvt_e2m1x2(__fmul_rn(vals[8 + 2 * i], s), __fmul_rn(vals[9 + 2 * i], s)) << (8 * i);
    }
    *reinterpret_cast<uint2 *>(Q + (size_t)r * (inPad / 2) + b * 8) = make_uint2(q0, q1);
    QS[(size_t)r * nbPad + b] = (uint8_t)code;
  }
#endif
}

// Dense NVFP4 W4A4 GEMM: Y[M][N] = X[M][in] * W[N][in]^T.
// Block tile: BM = 64 output rows (16 per warp x 4 warps), BN = 16 token rows (2 x m16n8k64), BK = 64.
__global__ void __launch_bounds__(kGemmThreads) k_ink2_w4a4_gemm(
    const uint8_t *__restrict__ W, const uint8_t *__restrict__ WScales, float wGlobal,
    const uint8_t *__restrict__ Q, const uint8_t *__restrict__ QS, const float *__restrict__ RS,
    float *__restrict__ Y, int M, int N, int in, int inPad) {
#if INK2_W4A4_NATIVE
  constexpr int BM = 64, BN = 16, BK = 64, KB = BK / 16;
  __shared__ __align__(16) uint8_t smem_w[BM * (BK / 2)];
  __shared__ __align__(16) uint8_t smem_ws[BM * KB];
  __shared__ __align__(16) uint8_t smem_x[BN * (BK / 2)];
  __shared__ __align__(16) uint8_t smem_xs[BN * KB];

  const int rb = blockIdx.x * BM;
  const int cb = blockIdx.y * BN;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wRowBytes = in / 2, wNb = in / 16;
  const int xRowBytes = inPad / 2, xNb = inPad / 16;
  const int nk = inPad / BK;

  float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

  for (int kc = 0; kc < nk; ++kc) {
    const int k0 = kc * BK;
    for (int i = tid; i < BM * 2; i += kGemmThreads) {
      const int r = i >> 1, c = i & 1;
      const int gr = rb + r;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (gr < N && (k0 / 2 + c * 16) < wRowBytes) {
        const int avail = wRowBytes - (k0 / 2 + c * 16);
        if (avail >= 16) {
          v = *reinterpret_cast<const uint4 *>(W + (size_t)gr * wRowBytes + k0 / 2 + c * 16);
        } else {
          uint8_t tmp[16] = {0};
          for (int b = 0; b < avail; ++b) tmp[b] = W[(size_t)gr * wRowBytes + k0 / 2 + c * 16 + b];
          v = *reinterpret_cast<const uint4 *>(tmp);
        }
      }
      *reinterpret_cast<uint4 *>(smem_w + swz64(r, c)) = v;
    }
    for (int i = tid; i < BM; i += kGemmThreads) {
      const int gr = rb + i;
      uint32_t sc = 0u;
      if (gr < N) {
        for (int b = 0; b < KB && (k0 / 16 + b) < wNb; ++b) {
          sc |= (uint32_t)WScales[(size_t)gr * wNb + k0 / 16 + b] << (8 * b);
        }
      }
      *reinterpret_cast<uint32_t *>(smem_ws + i * KB) = sc;
    }
    for (int i = tid; i < BN * 2; i += kGemmThreads) {
      const int r = i >> 1, c = i & 1;
      const int gm = cb + r;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (gm < M) v = *reinterpret_cast<const uint4 *>(Q + (size_t)gm * xRowBytes + k0 / 2 + c * 16);
      *reinterpret_cast<uint4 *>(smem_x + swz64(r, c)) = v;
    }
    for (int i = tid; i < BN; i += kGemmThreads) {
      const int gm = cb + i;
      uint32_t sc = 0u;
      if (gm < M) sc = *reinterpret_cast<const uint32_t *>(QS + (size_t)gm * xNb + k0 / 16);
      *reinterpret_cast<uint32_t *>(smem_xs + i * KB) = sc;
    }
    __syncthreads();

    const int r0 = warp * 16 + g;
    const uint32_t a0 = *reinterpret_cast<const uint32_t *>(smem_w + swz64(r0, 0) + 4 * t);
    const uint32_t a1 = *reinterpret_cast<const uint32_t *>(smem_w + swz64(r0 + 8, 0) + 4 * t);
    const uint32_t a2 = *reinterpret_cast<const uint32_t *>(smem_w + swz64(r0, 1) + 4 * t);
    const uint32_t a3 = *reinterpret_cast<const uint32_t *>(smem_w + swz64(r0 + 8, 1) + 4 * t);
    const uint32_t sfa = *reinterpret_cast<const uint32_t *>(smem_ws + (r0 + 8 * (t & 1)) * KB);

#pragma unroll
    for (int n = 0; n < 2; ++n) {
      const int c = n * 8 + g;
      const uint32_t b0 = *reinterpret_cast<const uint32_t *>(smem_x + swz64(c, 0) + 4 * t);
      const uint32_t b1 = *reinterpret_cast<const uint32_t *>(smem_x + swz64(c, 1) + 4 * t);
      const uint32_t sfb = *reinterpret_cast<const uint32_t *>(smem_xs + c * KB);
      mma_m16n8k64_nvfp4(acc[n], a0, a1, a2, a3, b0, b1, sfa, sfb);
    }
    __syncthreads();
  }

#pragma unroll
  for (int n = 0; n < 2; ++n) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int row = rb + warp * 16 + g + (c >= 2 ? 8 : 0);
      const int col = cb + n * 8 + 2 * t + (c & 1);
      if (row < N && col < M) {
        Y[(size_t)col * N + row] = acc[n][c] * RS[col] * wGlobal;
      }
    }
  }
#endif
}

}  // namespace ink2_w4a4

extern "C" int fcuda_ink2_w4a4_gemm_available(void) {
  static int avail = -1;
  if (avail >= 0) return avail;
  int flag = 0;
  ink2_w4a4::k_ink2_w4a4_probe<<<1, 32, 0, g_stream>>>();
  cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) {
    e = cudaMemcpyFromSymbolAsync(&flag, ink2_w4a4::g_ink2_w4a4_flag, sizeof(flag), 0, cudaMemcpyDeviceToHost, g_stream);
  }
  if (e == cudaSuccess) e = cudaStreamSynchronize(g_stream);
  if (e != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  avail = flag == 1 ? 1 : 0;
  return avail;
}
