// Inkling-2 prefill dense projections Y[T][out] = X[T][in] * W^T for the resident GGUF weight formats.
// Quantized weights run one int8 tensor-core GEMM (mma.sync s8) against per-32 q8 activations, the scheme
// llama.cpp MMQ uses: every 32-element (16 for Q6_K) weight sub-block is unpacked to int8 in shared memory,
// its integer dot is taken on the tensor cores, and the f32 block scales (and the K-quant min term) are
// folded per sub-block. F32 weights run a register-blocked f32 GEMM.
#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

namespace ink2d {

constexpr int kThreads = 256;
constexpr int kBN = 64;         // tokens per block tile
constexpr int kBK = 128;        // k elements per stage (half a K-quant super-block)
constexpr int kLds = kBK + 16;  // smem row stride in bytes; 36 words keeps fragment loads conflict-free

enum Fmt { Q8_0 = 0, Q4_K = 4, Q5_K = 5, Q6_K = 6 };

// Activation workspace: int8 codes [Tpad][in] and per-32 (d, d*sum(q)) [Tpad][in/32]; rows >= T are zero.
static int8_t *g_ws_q = nullptr;
static float2 *g_ws_s = nullptr;
static size_t g_ws_qcap = 0, g_ws_scap = 0;

static int workspace(size_t qbytes, size_t sbytes) {
  if (qbytes > g_ws_qcap) {
    if (g_ws_q) {
      cudaStreamSynchronize(g_stream);
      cudaFree(g_ws_q);
      g_ws_q = nullptr;
      g_ws_qcap = 0;
    }
    cudaError_t e = fcuda_alloc_status(cudaMalloc(&g_ws_q, qbytes));
    if (e != cudaSuccess) return (int)e;
    g_ws_qcap = qbytes;
  }
  if (sbytes > g_ws_scap) {
    if (g_ws_s) {
      cudaStreamSynchronize(g_stream);
      cudaFree(g_ws_s);
      g_ws_s = nullptr;
      g_ws_scap = 0;
    }
    cudaError_t e = fcuda_alloc_status(cudaMalloc(&g_ws_s, sbytes));
    if (e != cudaSuccess) return (int)e;
    g_ws_scap = sbytes;
  }
  return 0;
}

// One thread per 4 elements, 8 threads per 32-block; rows t >= T write zeros.
__global__ void k_ink2p_dense_quant(const float *__restrict__ X, int8_t *__restrict__ qx, float2 *__restrict__ xs,
                                    int in, int T) {
  const int t = blockIdx.y;
  const int i4 = blockIdx.x * blockDim.x + threadIdx.x;  // float4 index within the row
  if (i4 * 4 >= in) return;
  float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
  if (t < T) v = reinterpret_cast<const float4 *>(X + (size_t)t * in)[i4];
  float a = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
  for (int o = 4; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  const float d = a / 127.f;
  const float id = a > 0.f ? 127.f / a : 0.f;
  const int q0 = __float2int_rn(v.x * id), q1 = __float2int_rn(v.y * id);
  const int q2 = __float2int_rn(v.z * id), q3 = __float2int_rn(v.w * id);
  int s = q0 + q1 + q2 + q3;
#pragma unroll
  for (int o = 4; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const uint32_t packed = (uint32_t)(q0 & 0xff) | ((uint32_t)(q1 & 0xff) << 8) | ((uint32_t)(q2 & 0xff) << 16) |
                          ((uint32_t)(q3 & 0xff) << 24);
  reinterpret_cast<uint32_t *>(qx + (size_t)t * in)[i4] = packed;
  if ((threadIdx.x & 7) == 0) xs[(size_t)t * (in / 32) + i4 / 8] = make_float2(d, d * (float)s);
}

__device__ __forceinline__ void mma_k32(int (&c)[4], const int (&a)[4], const int (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_k16(int (&c)[4], int a0, int a1, int b0) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void k4_scale_min(int j, const uint8_t *q, int &sc, int &mn) {
  if (j < 4) {
    sc = q[j] & 63;
    mn = q[j + 4] & 63;
  } else {
    sc = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
    mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// Raw weight registers for one (row, 64-element part) of a stage, then unpacked to int8 + scales in smem.
// sc[] holds 4 per-16 scales (Q6_K) or 2 per-32 scales; mn[] the per-32 min terms (K-quants with mins).
template <int F>
struct WRaw;

template <>
struct WRaw<Q8_0> {
  uint4 c[4];
  float s[2];
  __device__ __forceinline__ void load(const uint8_t *W, const float *S, int row, int in, int k0) {
    const uint4 *p = reinterpret_cast<const uint4 *>(W + (size_t)row * in + k0);
#pragma unroll
    for (int i = 0; i < 4; ++i) c[i] = __ldg(p + i);
    const float *sp = S + (size_t)row * (in / 32) + k0 / 32;
    s[0] = __ldg(sp);
    s[1] = __ldg(sp + 1);
  }
  __device__ __forceinline__ void store(int8_t *dst, float *sc, float *mn) const {
#pragma unroll
    for (int i = 0; i < 4; ++i) reinterpret_cast<uint4 *>(dst)[i] = c[i];
    sc[0] = s[0];
    sc[1] = s[1];
    mn[0] = mn[1] = 0.f;
  }
};

template <>
struct WRaw<Q4_K> {
  uint4 hdr, q[2];
  int j;
  __device__ __forceinline__ void load(const uint8_t *W, const float *, int row, int in, int k0) {
    const int sb = k0 >> 8;
    j = (k0 & 255) >> 6;  // 64-element chunk in the super-block
    const uint8_t *b = W + ((size_t)row * (in >> 8) + sb) * 144;
    hdr = __ldg(reinterpret_cast<const uint4 *>(b));
    const uint4 *qp = reinterpret_cast<const uint4 *>(b + 16 + 32 * j);
    q[0] = __ldg(qp);
    q[1] = __ldg(qp + 1);
  }
  __device__ __forceinline__ void store(int8_t *dst, float *sc, float *mn) const {
    const uint32_t *w = reinterpret_cast<const uint32_t *>(q);
    uint32_t *o = reinterpret_cast<uint32_t *>(dst);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      o[i] = w[i] & 0x0f0f0f0fu;
      o[8 + i] = (w[i] >> 4) & 0x0f0f0f0fu;
    }
    const uint8_t *h = reinterpret_cast<const uint8_t *>(&hdr);
    const float d = __half2float(*reinterpret_cast<const __half *>(h));
    const float dm = __half2float(*reinterpret_cast<const __half *>(h + 2));
    int s0, m0, s1, m1;
    k4_scale_min(2 * j, h + 4, s0, m0);
    k4_scale_min(2 * j + 1, h + 4, s1, m1);
    sc[0] = d * s0;
    sc[1] = d * s1;
    mn[0] = dm * m0;
    mn[1] = dm * m1;
  }
};

template <>
struct WRaw<Q5_K> {
  uint4 hdr, q[2], qh[2];
  int j;
  __device__ __forceinline__ void load(const uint8_t *W, const float *, int row, int in, int k0) {
    const int sb = k0 >> 8;
    j = (k0 & 255) >> 6;
    const uint8_t *b = W + ((size_t)row * (in >> 8) + sb) * 176;
    hdr = __ldg(reinterpret_cast<const uint4 *>(b));
    qh[0] = __ldg(reinterpret_cast<const uint4 *>(b + 16));
    qh[1] = __ldg(reinterpret_cast<const uint4 *>(b + 32));
    const uint4 *qp = reinterpret_cast<const uint4 *>(b + 48 + 32 * j);
    q[0] = __ldg(qp);
    q[1] = __ldg(qp + 1);
  }
  __device__ __forceinline__ void store(int8_t *dst, float *sc, float *mn) const {
    const uint32_t *w = reinterpret_cast<const uint32_t *>(q);
    const uint32_t *hb = reinterpret_cast<const uint32_t *>(qh);
    uint32_t *o = reinterpret_cast<uint32_t *>(dst);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      o[i] = (w[i] & 0x0f0f0f0fu) | (((hb[i] >> (2 * j)) & 0x01010101u) << 4);
      o[8 + i] = ((w[i] >> 4) & 0x0f0f0f0fu) | (((hb[i] >> (2 * j + 1)) & 0x01010101u) << 4);
    }
    const uint8_t *h = reinterpret_cast<const uint8_t *>(&hdr);
    const float d = __half2float(*reinterpret_cast<const __half *>(h));
    const float dm = __half2float(*reinterpret_cast<const __half *>(h + 2));
    int s0, m0, s1, m1;
    k4_scale_min(2 * j, h + 4, s0, m0);
    k4_scale_min(2 * j + 1, h + 4, s1, m1);
    sc[0] = d * s0;
    sc[1] = d * s1;
    mn[0] = dm * m0;
    mn[1] = dm * m1;
  }
};

// Q6_K blocks are 210 bytes, so only 2-byte aligned: aligned words are loaded and funnel-shifted.
template <>
struct WRaw<Q6_K> {
  uint32_t l[17], h[9];
  uint32_t sw[2];
  uint32_t dw;
  int part, off;
  __device__ __forceinline__ void load(const uint8_t *W, const float *, int row, int in, int k0) {
    const int sb = k0 >> 8;
    const int half = (k0 & 255) >> 7;
    part = (k0 & 127) >> 6;
    const uint8_t *b = W + ((size_t)row * (in >> 8) + sb) * 210;
    off = (int)((uintptr_t)b & 3);
    const uint32_t *base = reinterpret_cast<const uint32_t *>(b - off);
    // ql half at byte 64*half (64 bytes), qh half at 128 + 32*half (32 bytes), scales at 192 + 8*half + 4*part,
    // d at 208.
    const uint32_t *pl = base + (64 * half) / 4;
#pragma unroll
    for (int i = 0; i < 17; ++i) l[i] = __ldg(pl + i);
    const uint32_t *ph = base + (128 + 32 * half) / 4;
#pragma unroll
    for (int i = 0; i < 9; ++i) h[i] = __ldg(ph + i);
    const uint32_t *ps = base + (192 + 8 * half + 4 * part) / 4;
    sw[0] = __ldg(ps);
    // the second word is only funneled in when the block is misaligned; at off 0 it would
    // reach past the 210-byte block
    sw[1] = off ? __ldg(ps + 1) : 0u;
    dw = __ldg(reinterpret_cast<const unsigned short *>(b + 208));
  }
  __device__ __forceinline__ void store(int8_t *dst, float *sc, float *) const {
    const int sh = off * 8;
    uint32_t *o = reinterpret_cast<uint32_t *>(dst);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t qa = __funnelshift_r(l[i], l[i + 1], sh);          // ql[4i..4i+3]
      const uint32_t qb = __funnelshift_r(l[i + 8], l[i + 9], sh);      // ql[32+4i..]
      const uint32_t hh = __funnelshift_r(h[i], h[i + 1], sh);          // qh[4i..]
      uint32_t lo_a, lo_b, hi_a, hi_b;
      if (part == 0) {
        lo_a = qa & 0x0f0f0f0fu;
        lo_b = qb & 0x0f0f0f0fu;
        hi_a = (hh & 0x03030303u) << 4;
        hi_b = ((hh >> 2) & 0x03030303u) << 4;
      } else {
        lo_a = (qa >> 4) & 0x0f0f0f0fu;
        lo_b = (qb >> 4) & 0x0f0f0f0fu;
        hi_a = ((hh >> 4) & 0x03030303u) << 4;
        hi_b = ((hh >> 6) & 0x03030303u) << 4;
      }
      // q - 32 per byte: values are 0..63, subtract 32 with a SIMD byte subtract.
      o[i] = __vsub4(lo_a | hi_a, 0x20202020u);
      o[8 + i] = __vsub4(lo_b | hi_b, 0x20202020u);
    }
    const uint32_t s4 = __funnelshift_r(sw[0], sw[1], sh);
    const float d = __half2float(__ushort_as_half((unsigned short)dw));
#pragma unroll
    for (int g = 0; g < 4; ++g) sc[g] = d * (float)(int8_t)((s4 >> (8 * g)) & 0xff);
  }
};

template <int F>
struct FmtTraits {
  static constexpr bool kSub16 = F == Q6_K;
  static constexpr bool kMin = F == Q4_K || F == Q5_K;
};

// Block tile BM outputs x kBN tokens, 8 warps laid WM x WN, each warp 32 rows x (kBN/WN) cols.
template <int F, int BM>
__global__ void __launch_bounds__(kThreads, 2)
    k_ink2d_mmq(const uint8_t *__restrict__ W, const float *__restrict__ S, const int8_t *__restrict__ qx,
                const float2 *__restrict__ xs, float *__restrict__ Y, int out, int in, int T) {
  ink2_pdl_enter();
  constexpr bool kSub16 = FmtTraits<F>::kSub16;
  constexpr bool kMin = FmtTraits<F>::kMin;
  constexpr int WM = BM / 32, WN = 8 / WM;
  constexpr int NTL = kBN / WN / 8;  // n8 tiles per warp
  constexpr int NSC = kSub16 ? 8 : 4;  // weight scales per row per stage
  constexpr int AITER = (BM * 2 + kThreads - 1) / kThreads;

  __shared__ __align__(16) int8_t sA[BM * kLds];
  __shared__ __align__(16) int8_t sB[kBN * kLds];
  __shared__ float sAs[BM][NSC];
  __shared__ float sAm[BM][4];
  __shared__ float2 sBs[kBN][4];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, tig = lane & 3;
  const int wm = warp % WM, wn = warp / WM;
  const int o0 = blockIdx.x * BM, t0 = blockIdx.y * kBN;
  const int nk = in / kBK;
  const int nsub = in / 32;

  WRaw<F> ra[AITER];
  uint4 rb[2];
  float2 rbs;

  auto load = [&](int kk) {
    const int k0 = kk * kBK;
#pragma unroll
    for (int it = 0; it < AITER; ++it) {
      const int idx = tid + it * kThreads;
      if (idx < BM * 2) {
        const int row = min(o0 + (idx >> 1), out - 1);
        ra[it].load(W, S, row, in, k0 + (idx & 1) * 64);
      }
    }
    const int bt = tid >> 2, bp = tid & 3;
    const uint4 *bp4 = reinterpret_cast<const uint4 *>(qx + (size_t)(t0 + bt) * in + k0 + bp * 32);
    rb[0] = __ldg(bp4);
    rb[1] = __ldg(bp4 + 1);
    rbs = __ldg(xs + (size_t)(t0 + bt) * nsub + k0 / 32 + bp);
  };
  auto store = [&]() {
#pragma unroll
    for (int it = 0; it < AITER; ++it) {
      const int idx = tid + it * kThreads;
      if (idx < BM * 2) {
        const int r = idx >> 1, p = idx & 1;
        ra[it].store(sA + r * kLds + p * 64, &sAs[r][p * (NSC / 2)], &sAm[r][p * 2]);
      }
    }
    const int bt = tid >> 2, bp = tid & 3;
    uint4 *d = reinterpret_cast<uint4 *>(sB + bt * kLds + bp * 32);
    d[0] = rb[0];
    d[1] = rb[1];
    sBs[bt][bp] = rbs;
  };

  float acc[2][NTL][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < NTL; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

  load(0);
  for (int kk = 0; kk < nk; ++kk) {
    __syncthreads();
    store();
    __syncthreads();
    if (kk + 1 < nk) load(kk + 1);
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
      const int kb = sub * 32 + tig * 4;
      int a[2][4], b[NTL][2];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int8_t *ap = sA + (wm * 32 + mt * 16 + g) * kLds + kb;
        a[mt][0] = *reinterpret_cast<const int *>(ap);
        a[mt][1] = *reinterpret_cast<const int *>(ap + 8 * kLds);
        a[mt][2] = *reinterpret_cast<const int *>(ap + 16);
        a[mt][3] = *reinterpret_cast<const int *>(ap + 8 * kLds + 16);
      }
#pragma unroll
      for (int nt = 0; nt < NTL; ++nt) {
        const int8_t *bp = sB + (wn * (kBN / WN) + nt * 8 + g) * kLds + kb;
        b[nt][0] = *reinterpret_cast<const int *>(bp);
        b[nt][1] = *reinterpret_cast<const int *>(bp + 16);
      }
      float xd[NTL][2], xm[NTL][2];
#pragma unroll
      for (int nt = 0; nt < NTL; ++nt)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const float2 v = sBs[wn * (kBN / WN) + nt * 8 + tig * 2 + c][sub];
          xd[nt][c] = v.x;
          xm[nt][c] = v.y;
        }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int r0 = wm * 32 + mt * 16 + g;
        if constexpr (kSub16) {
          const float s00 = sAs[r0][sub * 2], s01 = sAs[r0][sub * 2 + 1];
          const float s10 = sAs[r0 + 8][sub * 2], s11 = sAs[r0 + 8][sub * 2 + 1];
#pragma unroll
          for (int nt = 0; nt < NTL; ++nt) {
            int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
            mma_k16(c0, a[mt][0], a[mt][1], b[nt][0]);
            mma_k16(c1, a[mt][2], a[mt][3], b[nt][1]);
            acc[mt][nt][0] += (s00 * (float)c0[0] + s01 * (float)c1[0]) * xd[nt][0];
            acc[mt][nt][1] += (s00 * (float)c0[1] + s01 * (float)c1[1]) * xd[nt][1];
            acc[mt][nt][2] += (s10 * (float)c0[2] + s11 * (float)c1[2]) * xd[nt][0];
            acc[mt][nt][3] += (s10 * (float)c0[3] + s11 * (float)c1[3]) * xd[nt][1];
          }
        } else {
          const float s0 = sAs[r0][sub], s1 = sAs[r0 + 8][sub];
          float m0 = 0.f, m1 = 0.f;
          if constexpr (kMin) {
            m0 = sAm[r0][sub];
            m1 = sAm[r0 + 8][sub];
          }
#pragma unroll
          for (int nt = 0; nt < NTL; ++nt) {
            int c[4] = {0, 0, 0, 0};
            mma_k32(c, a[mt], b[nt]);
            acc[mt][nt][0] = fmaf((float)c[0], s0 * xd[nt][0], acc[mt][nt][0]);
            acc[mt][nt][1] = fmaf((float)c[1], s0 * xd[nt][1], acc[mt][nt][1]);
            acc[mt][nt][2] = fmaf((float)c[2], s1 * xd[nt][0], acc[mt][nt][2]);
            acc[mt][nt][3] = fmaf((float)c[3], s1 * xd[nt][1], acc[mt][nt][3]);
            if constexpr (kMin) {
              acc[mt][nt][0] = fmaf(-m0, xm[nt][0], acc[mt][nt][0]);
              acc[mt][nt][1] = fmaf(-m0, xm[nt][1], acc[mt][nt][1]);
              acc[mt][nt][2] = fmaf(-m1, xm[nt][0], acc[mt][nt][2]);
              acc[mt][nt][3] = fmaf(-m1, xm[nt][1], acc[mt][nt][3]);
            }
          }
        }
      }
    }
  }

#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < NTL; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int o = o0 + wm * 32 + mt * 16 + g + (i >= 2 ? 8 : 0);
        const int t = t0 + wn * (kBN / WN) + nt * 8 + tig * 2 + (i & 1);
        if (o < out && t < T) Y[(size_t)t * out + o] = acc[mt][nt][i];
      }
}

// Register-blocked f32 GEMM: TM x TM tile, 256 threads x (TM/16 outs x TM/16 tokens), k step 16.
template <int TM>
__global__ void __launch_bounds__(256) k_ink2d_f32(const float *__restrict__ W, const float *__restrict__ X,
                                                   float *__restrict__ Y, int out, int in, int T) {
  ink2_pdl_enter();
  constexpr int R = TM / 16;
  __shared__ float sW[16][TM + 4];
  __shared__ float sX[16][TM + 4];
  const int tid = threadIdx.x;
  const int o0 = blockIdx.x * TM, t0 = blockIdx.y * TM;
  const int tx = tid & 15, ty = tid >> 4;  // tx -> outs, ty -> tokens
  float acc[R][R] = {};
  const int lr = tid >> 2, lc = (tid & 3) * 4;  // loader: row 0..63, k col 0,4,8,12
  const bool loader = lr < TM;
  const bool vec = (in & 3) == 0 && ((uintptr_t)X & 15) == 0 && ((uintptr_t)W & 15) == 0;
  for (int k0 = 0; k0 < in; k0 += 16) {
    float4 wv = make_float4(0.f, 0.f, 0.f, 0.f), xv = wv;
    const int o = o0 + lr, t = t0 + lr, k = k0 + lc;
    if (loader && o < out) {
      if (vec && k + 3 < in) {
        wv = __ldg(reinterpret_cast<const float4 *>(W + (size_t)o * in + k));
      } else {
        float *pv = &wv.x;
        for (int q = 0; q < 4; ++q) pv[q] = k + q < in ? W[(size_t)o * in + k + q] : 0.f;
      }
    }
    if (loader && t < T) {
      if (vec && k + 3 < in) {
        xv = __ldg(reinterpret_cast<const float4 *>(X + (size_t)t * in + k));
      } else {
        float *pv = &xv.x;
        for (int q = 0; q < 4; ++q) pv[q] = k + q < in ? X[(size_t)t * in + k + q] : 0.f;
      }
    }
    __syncthreads();
    if (loader) {
      sW[lc + 0][lr] = wv.x;
      sW[lc + 1][lr] = wv.y;
      sW[lc + 2][lr] = wv.z;
      sW[lc + 3][lr] = wv.w;
      sX[lc + 0][lr] = xv.x;
      sX[lc + 1][lr] = xv.y;
      sX[lc + 2][lr] = xv.z;
      sX[lc + 3][lr] = xv.w;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
      float w[R], x[R];
#pragma unroll
      for (int i = 0; i < R; ++i) {
        w[i] = sW[kk][tx + 16 * i];
        x[i] = sX[kk][ty + 16 * i];
      }
#pragma unroll
      for (int i = 0; i < R; ++i)
#pragma unroll
        for (int j = 0; j < R; ++j) acc[i][j] = fmaf(w[i], x[j], acc[i][j]);
    }
  }
#pragma unroll
  for (int i = 0; i < R; ++i)
#pragma unroll
    for (int j = 0; j < R; ++j) {
      const int o = o0 + tx + 16 * i, t = t0 + ty + 16 * j;
      if (o < out && t < T) Y[(size_t)t * out + o] = acc[i][j];
    }
}

template <int F, int BM>
static int launch_mmq(const uint8_t *W, const float *S, float *Y, int out, int in, int T, int Tpad) {
  const dim3 grid((out + BM - 1) / BM, Tpad / kBN);
  FCUDA_TS("k_ink2d_mmq", grid,
           k_ink2d_mmq<F, BM><<<grid, kThreads, 0, g_stream>>>(W, S, g_ws_q, g_ws_s, Y, out, in, T));
  return (int)cudaGetLastError();
}

// bm: 0 = heuristic, 64 or 128 forces the output tile.
template <int F>
static int matmul(const uint8_t *W, const float *S, const float *X, float *Y, int out, int in, int T, int bm) {
  if (!W || !X || !Y || out <= 0 || T <= 0 || in <= 0 || in % kBK != 0 || (F != Q8_0 && in % 256 != 0) ||
      (F == Q8_0 && !S) || ((uintptr_t)X & 15) != 0 || (F != Q6_K && ((uintptr_t)W & 15) != 0) ||
      ((uintptr_t)W & 1) != 0)
    return (int)cudaErrorInvalidValue;
  const int Tpad = (T + kBN - 1) / kBN * kBN;
  if (Tpad / kBN > 65535) return (int)cudaErrorInvalidValue;
  int rc = workspace((size_t)Tpad * in, (size_t)Tpad * (in / 32) * sizeof(float2));
  if (rc) return rc;
  const dim3 qg((in / 4 + 255) / 256, Tpad);
  FCUDA_TS("k_ink2p_dense_quant", qg, k_ink2p_dense_quant<<<qg, 256, 0, g_stream>>>(X, g_ws_q, g_ws_s, in, T));
  rc = (int)cudaGetLastError();
  if (rc) return rc;
  if (bm == 0) {
    int dev = 0, nsm = 84;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    // Measured on RTX 5080: the K-quant unpack favours the 64-row tile at every shape; Q8_0 gains from the
    // 128-row tile once there are enough output tiles to fill the SMs.
    const long tiles128 = (long)((out + 127) / 128) * (Tpad / kBN);
    bm = F == Q8_0 && tiles128 >= nsm / 2 ? 128 : 64;
  }
  return bm == 128 ? launch_mmq<F, 128>(W, S, Y, out, in, T, Tpad) : launch_mmq<F, 64>(W, S, Y, out, in, T, Tpad);
}

}  // namespace ink2d

// fcuda_ink2p_dense_mmq: Y[T][out] = X[T][in] * W^T on the int8 tensor cores. kind is the GGUF dtype
// family: 8 = Q8_0 (W = int8 codes [out][in], S = f32 per-32 scales [out][in/32], block 32 only),
// 4 = Q4_K, 5 = Q5_K, 6 = Q6_K (W = raw GGUF super-blocks, S unused). Returns a CUDA status;
// cudaErrorInvalidValue (1) before any enqueue means the shape/alignment is outside this kernel, so the
// caller falls back to its legacy launcher.
extern "C" int fcuda_ink2p_dense_mmq(const void *W, const float *S, const float *X, float *Y, int out, int in,
                                     int T, int kind) {
  const uint8_t *w = (const uint8_t *)W;
  switch (kind) {
    case 8: return ink2d::matmul<ink2d::Q8_0>(w, S, X, Y, out, in, T, 0);
    case 4: return ink2d::matmul<ink2d::Q4_K>(w, S, X, Y, out, in, T, 0);
    case 5: return ink2d::matmul<ink2d::Q5_K>(w, S, X, Y, out, in, T, 0);
    case 6: return ink2d::matmul<ink2d::Q6_K>(w, S, X, Y, out, in, T, 0);
  }
  return (int)cudaErrorInvalidValue;
}

// fcuda_ink2p_gemm_f32_v2 has the signature of fcuda_ink2p_gemm_f32 (W [rows][cols], X [T][cols], Y [T][rows]).
extern "C" int fcuda_ink2p_gemm_f32_v2(const float *W, const float *X, float *Y, int rows, int cols, int T) {
  if (!W || !X || !Y || rows <= 0 || cols <= 0 || T <= 0) return (int)cudaErrorInvalidValue;
  // Narrow projections (sink, conv proj) and small-T batches have too few output tiles to fill
  // the SMs; the warp-per-dot kernel spreads them over rows x T instead.
  if (rows < 64 || (rows < 512 && T <= 64)) return fcuda_ink2p_gemm_f32(W, X, Y, rows, cols, T);
  if (rows < 512) {
    const dim3 grid((rows + 31) / 32, (T + 31) / 32);
    FCUDA_TS("k_ink2d_f32", grid, ink2d::k_ink2d_f32<32><<<grid, 256, 0, g_stream>>>(W, X, Y, rows, cols, T));
  } else {
    const dim3 grid((rows + 63) / 64, (T + 63) / 64);
    FCUDA_TS("k_ink2d_f32", grid, ink2d::k_ink2d_f32<64><<<grid, 256, 0, g_stream>>>(W, X, Y, rows, cols, T));
  }
  return (int)cudaGetLastError();
}
