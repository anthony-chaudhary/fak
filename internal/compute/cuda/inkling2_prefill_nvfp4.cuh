// inkling2_prefill_nvfp4.cuh: int8 tensor-core grouped NVFP4 expert GEMM for prefill.
//
// Same bucketing contract as k_ink2p_expert_nvfp4 (offsets/order/tiles/ntiles from
// fcuda_ink2p_bucket, INK2P_NV_PAIRS-pair tiles, pair i reads X row (i / xDiv) * xStride
// and writes Y row i), but the activations are quantized once per X row to int8 with
// one fp32 scale per 16 columns (q8_1-style) and the weights decode to exact int8
// (2 * E2M1 is an integer in [-12, 12]). Each 16-wide K block is one
// mma.m16n8k16.s8 whose int32 result is rescaled by (E4M3 weight scale x activation
// scale) in fp32, so only the activation rounding differs from the BF16 kernel.
// Weight, scale and activation tiles stream through a cp.async ring of shared stages; the
// copies prefetch 256 B into L2 because a stage takes only 32 B of each 1152 B weight row.
// ink2p_nv3_launch is a lossy per-32 requant variant (k32 MMA), not the default.
// Included after inkling2_prefill.cuh (the launcher falls back to fcuda_ink2p_expert_nvfp4).

#define INK2P_NV2_BK 128
#define INK2P_NV2_PAIRS 32

static __device__ __forceinline__ void ink2p_nv2_cp16(void *s, const void *g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(s)), "l"(g));
}
template <int PF>
static __device__ __forceinline__ void ink2p_nv2_cp16w(void *s, const void *g) {
  if (PF)
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(s)),
                 "l"(g));
  else
    ink2p_nv2_cp16(s, g);
}
static __device__ __forceinline__ void ink2p_nv2_cp8(void *s, const void *g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"((unsigned)__cvta_generic_to_shared(s)), "l"(g));
}
static __device__ __forceinline__ void ink2p_nv2_cp4(void *s, const void *g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned)__cvta_generic_to_shared(s)), "l"(g));
}
static __device__ __forceinline__ void ink2p_nv2_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
static __device__ __forceinline__ void ink2p_nv2_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// Four E2M1 codes (one per nibble, element i in nibble i) to four int8 values 2*e2m1.
static __device__ __forceinline__ uint32_t ink2p_nv2_dec4(uint32_t codes) {
  const uint32_t mag = codes & 0x7777u;
  const uint32_t pos = __byte_perm(0x03020100u, 0x0C080604u, mag);
  const uint32_t neg = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, mag);
  return __byte_perm(pos, neg, 0x3210u | ((codes >> 1) & 0x4444u));
}

// Two E4M3 bytes (low byte first) to two floats.
static __device__ __forceinline__ float2 ink2p_nv2_e4m3x2(uint32_t raw16) {
  uint32_t h2;
  asm("{.reg .b16 r; cvt.u16.u32 r, %1; cvt.rn.f16x2.e4m3x2 %0, r;}\n" : "=r"(h2) : "r"(raw16));
  return __half22float2(*reinterpret_cast<__half2 *>(&h2));
}

// Rows of X that the bucketed pairs read: (npairs - 1) / xDiv + 1, npairs from the tile list.
static __device__ __forceinline__ int ink2p_nv2_xrows(const int *offsets, const int *tiles, const int *ntiles,
                                                       int xDiv) {
  const int nt = ntiles[0];
  if (nt <= 0) return 0;
  const int npairs = offsets[tiles[2 * (nt - 1)] + 1];
  return (npairs + xDiv - 1) / xDiv;
}

// One G-lane group per G-column block (G = 16 or 32): Q = rint(x * 127 / amax), QS = amax / 127.
template <int G>
__global__ void k_ink2p_nv2_quant(const int *offsets, const int *tiles, const int *ntiles, const float *X, int xDiv,
                                  int xStride, int8_t *Q, float *QS, int in) {
  const int nrows = ink2p_nv2_xrows(offsets, tiles, ntiles, xDiv);
  const size_t total = (size_t)nrows * in;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < ((total + 31) & ~(size_t)31);
       i += (size_t)gridDim.x * blockDim.x) {
    const bool ok = i < total;
    const int r = ok ? (int)(i / in) : 0, c = ok ? (int)(i - (size_t)r * in) : 0;
    const float x = ok ? X[(size_t)r * xStride + c] : 0.f;
    float amax = fabsf(x);
#pragma unroll
    for (int o = G / 2; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (!ok) continue;
    const float inv = amax > 1e-30f ? 127.f / amax : 0.f;  // 127/amax overflows for denormal amax
    Q[i] = (int8_t)__float2int_rn(x * inv);
    if ((c & (G - 1)) == 0) QS[(size_t)r * (in / G) + c / G] = amax * (1.f / 127.f);
  }
}

// Block = (BM-row slab, NTILE consecutive pair tiles of one expert). A tile whose
// predecessor of the same expert starts an NTILE group exits, so every weight slab is
// streamed and decoded once per NTILE*INK2P_NV2_PAIRS pairs. Warp w owns rows
// [w*16*MT, (w+1)*16*MT) against every n8 tile of the pairs; K advances BK columns per
// cp.async stage.
template <int MT, int WARPS, int STAGES, int BK, int NTILE, int PF>
__global__ void __launch_bounds__(WARPS * 32) k_ink2p_expert_nvfp4_v2(
    const uint8_t *Packed, const uint8_t *Scales, const int *offsets, const int *order, const int *tiles,
    const int *ntiles, const int8_t *Q, const float *QS, int xDiv, float *Y, int rows, int in, const float *escale) {
  constexpr int KB = BK / 16, BM = WARPS * 16 * MT, P = INK2P_NV2_PAIRS * NTILE, NT = P / 8;
  constexpr int T = WARPS * 32;
  constexpr int WS = BK / 2 + 16, XS = BK + 16;
  constexpr int SW = BM * WS, SWS = BM * KB, SX = P * XS, SXS = KB * P * 4;
  constexpr int STAGE = SW + SWS + SX + SXS;
  extern __shared__ __align__(16) uint8_t ink2p_nv2_smem[];
  __shared__ int pairs[P], xrows[P];
  if ((int)blockIdx.y >= ntiles[0]) return;
  const int e = tiles[2 * blockIdx.y], start = tiles[2 * blockIdx.y + 1];
  if (((start - offsets[e]) / INK2P_NV2_PAIRS) % NTILE != 0) return;
  const int np = min(P, offsets[e + 1] - start);
  const int tid = threadIdx.x;
  for (int i = tid; i < P; i += T) {
    const int p = order[start + (i < np ? i : 0)];
    pairs[i] = i < np ? p : -1;
    xrows[i] = p / xDiv;
  }
  __syncthreads();
  const int nta = (np + 7) >> 3;
  const int rb = blockIdx.x * BM;
  const size_t wrow0 = (size_t)e * rows + rb;
  const int rowBytes = in / 2, nb = in / 16, nk = in / BK;

  auto load = [&](int stage, int kc) {
    uint8_t *s = ink2p_nv2_smem + stage * STAGE;
    const int k0 = kc * BK;
    for (int i = tid; i < BM * (BK / 32); i += T) {
      const int r = i / (BK / 32), c = i % (BK / 32);
      ink2p_nv2_cp16w<PF>(s + r * WS + c * 16, Packed + (wrow0 + r) * rowBytes + k0 / 2 + c * 16);
    }
    for (int i = tid; i < BM; i += T) {
      if (KB == 8) ink2p_nv2_cp8(s + SW + i * KB, Scales + (wrow0 + i) * nb + k0 / 16);
      else ink2p_nv2_cp4(s + SW + i * KB, Scales + (wrow0 + i) * nb + k0 / 16);
    }
    for (int i = tid; i < nta * 8 * (BK / 16); i += T) {
      const int c = i / (BK / 16), j = i % (BK / 16);
      ink2p_nv2_cp16w<PF>(s + SW + SWS + c * XS + j * 16, Q + (size_t)xrows[c] * in + k0 + j * 16);
    }
    for (int i = tid; i < KB * P; i += T) {
      const int kb = i / P, c = i % P;
      if (c < nta * 8) ink2p_nv2_cp4(s + SW + SWS + SX + i * 4, QS + (size_t)xrows[c] * nb + k0 / 16 + kb);
    }
  };

  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wr = warp * 16 * MT;
  const float kMagic = 12582912.f; // 1.5 * 2^23: int32 accumulator seeded with its bits reads as kMagic + dot
  const int kMagicBits = 0x4B400000;
  float acc[MT][NT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nk) load(s, s);
    ink2p_nv2_commit();
  }
  for (int kc = 0; kc < nk; ++kc) {
    ink2p_nv2_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nx = kc + STAGES - 1;
      if (nx < nk) load(nx % STAGES, nx);
      ink2p_nv2_commit();
    }
    const uint8_t *s = ink2p_nv2_smem + (kc % STAGES) * STAGE;
    const uint8_t *sw = s, *sws = s + SW, *sx = s + SW + SWS;
    const float *sxs = reinterpret_cast<const float *>(s + SW + SWS + SX);
#pragma unroll
    for (int kl = 0; kl < KB; kl += 2) {
      float2 wsc[MT][2];
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          wsc[m][h] = ink2p_nv2_e4m3x2(*reinterpret_cast<const uint16_t *>(sws + (wr + m * 16 + g + 8 * h) * KB + kl));
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int kb = kl + kk;
        uint32_t b[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n)
          if (n < nta) b[n] = *reinterpret_cast<const uint32_t *>(sx + (n * 8 + g) * XS + kb * 16 + 4 * t);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          const int r0 = wr + m * 16 + g;
          const uint32_t a0 = ink2p_nv2_dec4(*reinterpret_cast<const uint16_t *>(sw + r0 * WS + kb * 8 + 2 * t));
          const uint32_t a1 = ink2p_nv2_dec4(*reinterpret_cast<const uint16_t *>(sw + (r0 + 8) * WS + kb * 8 + 2 * t));
          int d[NT][4];
#pragma unroll
          for (int n = 0; n < NT; ++n)
            if (n < nta)
              asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
                  : "=r"(d[n][0]), "=r"(d[n][1]), "=r"(d[n][2]), "=r"(d[n][3])
                  : "r"(a0), "r"(a1), "r"(b[n]), "r"(kMagicBits));
          const float w0 = kk ? wsc[m][0].y : wsc[m][0].x, w1 = kk ? wsc[m][1].y : wsc[m][1].x;
          const float n0 = -kMagic * w0, n1 = -kMagic * w1; // exact: w has <= 4 significant bits
#pragma unroll
          for (int n = 0; n < NT; ++n) {
            if (n < nta) {
              const float2 xs = *reinterpret_cast<const float2 *>(sxs + kb * P + n * 8 + 2 * t);
              acc[m][n][0] = fmaf(fmaf(__int_as_float(d[n][0]), w0, n0), xs.x, acc[m][n][0]);
              acc[m][n][1] = fmaf(fmaf(__int_as_float(d[n][1]), w0, n0), xs.y, acc[m][n][1]);
              acc[m][n][2] = fmaf(fmaf(__int_as_float(d[n][2]), w1, n1), xs.x, acc[m][n][2]);
              acc[m][n][3] = fmaf(fmaf(__int_as_float(d[n][3]), w1, n1), xs.y, acc[m][n][3]);
            }
          }
        }
      }
    }
  }
  ink2p_nv2_wait<0>();
  const float f = 0.5f * (escale ? escale[e] : 1.f);
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int row = rb + wr + m * 16 + g + (c >= 2 ? 8 : 0), col = n * 8 + 2 * t + (c & 1);
        if (n < nta && col < np) Y[(size_t)pairs[col] * rows + row] = acc[m][n][c] * f;
      }
}

// Grow-only int8 activation scratch (rows x in bytes, then rows x in/16 floats). Sized
// from the host tile bound, so it is stable for a fixed chunk size and allocates only
// on the first call (outside graph capture).
static int8_t *g_ink2p_nv2_q = nullptr;
static size_t g_ink2p_nv2_q_cap = 0;

template <int MT, int WARPS, int STAGES, int BK = INK2P_NV2_BK, int NTILE = 1, int PF = 1>
static int ink2p_nv2_launch(const uint8_t *W, const uint8_t *scales, const int *offsets, const int *order,
                            const int *tiles, const int *ntiles, int maxTiles, const float *X, int xDiv, int xStride,
                            float *Y, int rows, int in, const float *escale) {
  constexpr int BM = WARPS * 16 * MT, P = INK2P_NV2_PAIRS * NTILE;
  constexpr int STAGE = BM * (BK / 2 + 16) + BM * (BK / 16) + P * (BK + 16) + (BK / 16) * P * 4;
  constexpr int SMEM = STAGES * STAGE;
  const size_t maxRows = ((size_t)INK2P_NV2_PAIRS * maxTiles + xDiv - 1) / xDiv;
  const size_t qBytes = (maxRows * in + 15) & ~(size_t)15, need = qBytes + maxRows * (in / 16) * sizeof(float);
  if (need > g_ink2p_nv2_q_cap) {
    if (g_ink2p_nv2_q) cudaFree(g_ink2p_nv2_q);
    g_ink2p_nv2_q = nullptr;
    g_ink2p_nv2_q_cap = 0;
    const cudaError_t e = cudaMalloc(&g_ink2p_nv2_q, need);
    if (e != cudaSuccess) return (int)e;
    g_ink2p_nv2_q_cap = need;
  }
  int8_t *q = g_ink2p_nv2_q;
  float *qs = reinterpret_cast<float *>(q + qBytes);
  static bool attr = false;
  if (!attr) {
    const cudaError_t e = cudaFuncSetAttribute(k_ink2p_expert_nvfp4_v2<MT, WARPS, STAGES, BK, NTILE, PF>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    if (e != cudaSuccess) return (int)e;
    attr = true;
  }
  const int qgrid = (int)((maxRows * in + 255) / 256 < 4096 ? (maxRows * in + 255) / 256 : 4096);
  FCUDA_TS("k_ink2p_nv2_quant", qgrid,
           k_ink2p_nv2_quant<16><<<qgrid, 256, 0, g_stream>>>(offsets, tiles, ntiles, X, xDiv, xStride, q, qs, in));
  const dim3 grid(rows / BM, maxTiles);
  FCUDA_TS("k_ink2p_expert_nvfp4_v2", grid,
           (k_ink2p_expert_nvfp4_v2<MT, WARPS, STAGES, BK, NTILE, PF><<<grid, WARPS * 32, SMEM, g_stream>>>(
               W, scales, offsets, order, tiles, ntiles, q, qs, xDiv, Y, rows, in, escale)));
  return (int)cudaGetLastError();
}

// Per-32 requant path. Each 32-wide K block of a weight row (two E4M3 scales s0, s1)
// is re-expressed as int8 with one scale S = max|s| / 16: the larger-scale half is exact
// (16 * e2m1 is an integer in [-96, 96]) and the other half rounds to the nearest
// multiple of S. That allows one mma.m16n8k32.s8 and one rescale per 32 K against q8
// activations quantized per 32 columns. The requant runs once per staged tile and is
// shared by every pair of the block.
static __device__ __forceinline__ uint32_t ink2p_nv3_pack4(int a, int b, int c, int d) {
  return __byte_perm(__byte_perm(a, b, 0x0040), __byte_perm(c, d, 0x0040), 0x5410);
}

// Four E2M1 codes to int8 through an 8-entry signed magnitude table (pos/neg halves).
static __device__ __forceinline__ uint32_t ink2p_nv3_dec4(uint32_t codes, uint32_t p0, uint32_t p1, uint32_t n0,
                                                          uint32_t n1) {
  const uint32_t mag = codes & 0x7777u;
  return __byte_perm(__byte_perm(p0, p1, mag), __byte_perm(n0, n1, mag), 0x3210u | ((codes >> 1) & 0x4444u));
}

// Sixteen codes (two words) of one half at ratio r = s_i / max|s| into four int8 words.
static __device__ __forceinline__ uint4 ink2p_nv3_half(uint32_t lo, uint32_t hi, float r) {
  const float m = 12582912.f; // low byte of the bits of m + v is rint(v) for |v| < 2^22
  const int b1 = __float_as_int(fmaf(8.f, r, m)), b2 = __float_as_int(fmaf(16.f, r, m));
  const int b3 = __float_as_int(fmaf(24.f, r, m)), b4 = __float_as_int(fmaf(32.f, r, m));
  const int b5 = __float_as_int(fmaf(48.f, r, m)), b6 = __float_as_int(fmaf(64.f, r, m));
  const int b7 = __float_as_int(fmaf(96.f, r, m));
  const uint32_t p0 = ink2p_nv3_pack4(0, b1, b2, b3), p1 = ink2p_nv3_pack4(b4, b5, b6, b7);
  const uint32_t n0 = __vsub4(0u, p0), n1 = __vsub4(0u, p1);
  return make_uint4(ink2p_nv3_dec4(lo & 0xffffu, p0, p1, n0, n1), ink2p_nv3_dec4(lo >> 16, p0, p1, n0, n1),
                    ink2p_nv3_dec4(hi & 0xffffu, p0, p1, n0, n1), ink2p_nv3_dec4(hi >> 16, p0, p1, n0, n1));
}

template <int MT, int WARPS, int STAGES, int BK, int NTILE, int MINB, int PF>
__global__ void __launch_bounds__(WARPS * 32, MINB) k_ink2p_expert_nvfp4_v3(
    const uint8_t *Packed, const uint8_t *Scales, const int *offsets, const int *order, const int *tiles,
    const int *ntiles, const int8_t *Q, const float *QS, int xDiv, float *Y, int rows, int in, const float *escale) {
  constexpr int KB = BK / 16, KB32 = BK / 32, BM = WARPS * 16 * MT, P = INK2P_NV2_PAIRS * NTILE, NT = P / 8;
  constexpr int T = WARPS * 32;
  constexpr int WS = BK / 2 + 16, XS = BK + 16, WDS = BK + 16;
  constexpr int SW = BM * WS, SWS = BM * KB, SX = P * XS, SXS = KB32 * P * 4;
  constexpr int STAGE = SW + SWS + SX + SXS;
  extern __shared__ __align__(16) uint8_t ink2p_nv3_smem[];
  int8_t *wd = reinterpret_cast<int8_t *>(ink2p_nv3_smem + STAGES * STAGE);
  float *swd = reinterpret_cast<float *>(ink2p_nv3_smem + STAGES * STAGE + BM * WDS);
  __shared__ int pairs[P], xrows[P];
  if ((int)blockIdx.y >= ntiles[0]) return;
  const int e = tiles[2 * blockIdx.y], start = tiles[2 * blockIdx.y + 1];
  if (((start - offsets[e]) / INK2P_NV2_PAIRS) % NTILE != 0) return;
  const int np = min(P, offsets[e + 1] - start);
  const int tid = threadIdx.x;
  for (int i = tid; i < P; i += T) {
    const int p = order[start + (i < np ? i : 0)];
    pairs[i] = i < np ? p : -1;
    xrows[i] = p / xDiv;
  }
  __syncthreads();
  const int nta = (np + 7) >> 3;
  const int rb = blockIdx.x * BM;
  const size_t wrow0 = (size_t)e * rows + rb;
  const int rowBytes = in / 2, nb = in / 16, nb32 = in / 32, nk = in / BK;

  auto load = [&](int stage, int kc) {
    uint8_t *s = ink2p_nv3_smem + stage * STAGE;
    const int k0 = kc * BK;
    for (int i = tid; i < BM * (BK / 32); i += T) {
      const int r = i / (BK / 32), c = i % (BK / 32);
      ink2p_nv2_cp16w<PF>(s + r * WS + c * 16, Packed + (wrow0 + r) * rowBytes + k0 / 2 + c * 16);
    }
    for (int i = tid; i < BM; i += T) {
      if (KB == 8) ink2p_nv2_cp8(s + SW + i * KB, Scales + (wrow0 + i) * nb + k0 / 16);
      else ink2p_nv2_cp4(s + SW + i * KB, Scales + (wrow0 + i) * nb + k0 / 16);
    }
    for (int i = tid; i < nta * 8 * (BK / 16); i += T) {
      const int c = i / (BK / 16), j = i % (BK / 16);
      ink2p_nv2_cp16w<PF>(s + SW + SWS + c * XS + j * 16, Q + (size_t)xrows[c] * in + k0 + j * 16);
    }
    for (int i = tid; i < KB32 * P; i += T) {
      const int j = i / P, c = i % P;
      if (c < nta * 8) ink2p_nv2_cp4(s + SW + SWS + SX + i * 4, QS + (size_t)xrows[c] * nb32 + k0 / 32 + j);
    }
  };

  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wr = warp * 16 * MT;
  const float kMagic = 12582912.f; // int32 accumulator seeded with its bits reads as kMagic + dot
  const int kMagicBits = 0x4B400000;
  float acc[MT][NT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nk) load(s, s);
    ink2p_nv2_commit();
  }
  for (int kc = 0; kc < nk; ++kc) {
    ink2p_nv2_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nx = kc + STAGES - 1;
      if (nx < nk) load(nx % STAGES, nx);
      ink2p_nv2_commit();
    }
    const uint8_t *s = ink2p_nv3_smem + (kc % STAGES) * STAGE;
    const uint8_t *sw = s, *sws = s + SW, *sx = s + SW + SWS;
    const float *sxs = reinterpret_cast<const float *>(s + SW + SWS + SX);
    for (int u = tid; u < BM * KB32; u += T) {
      const int r = u / KB32, j = u % KB32;
      const uint4 raw = *reinterpret_cast<const uint4 *>(sw + r * WS + j * 16);
      const float2 sc = ink2p_nv2_e4m3x2(*reinterpret_cast<const uint16_t *>(sws + r * KB + 2 * j));
      const float smax = fmaxf(fabsf(sc.x), fabsf(sc.y));
      const float r0 = smax > 0.f ? sc.x / smax : 0.f, r1 = smax > 0.f ? sc.y / smax : 0.f;
      uint4 *dst = reinterpret_cast<uint4 *>(wd + r * WDS + j * 32);
      dst[0] = ink2p_nv3_half(raw.x, raw.y, r0);
      dst[1] = ink2p_nv3_half(raw.z, raw.w, r1);
      swd[r * KB32 + j] = smax * 0.0625f;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < KB32; ++j) {
      uint32_t b0[NT], b1[NT];
#pragma unroll
      for (int n = 0; n < NT; ++n)
        if (n < nta) {
          const uint8_t *xb = sx + (n * 8 + g) * XS + j * 32 + 4 * t;
          b0[n] = *reinterpret_cast<const uint32_t *>(xb);
          b1[n] = *reinterpret_cast<const uint32_t *>(xb + 16);
        }
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const int r0 = wr + m * 16 + g;
        const int8_t *wa = wd + r0 * WDS + j * 32 + 4 * t;
        const uint32_t a0 = *reinterpret_cast<const uint32_t *>(wa);
        const uint32_t a1 = *reinterpret_cast<const uint32_t *>(wa + 8 * WDS);
        const uint32_t a2 = *reinterpret_cast<const uint32_t *>(wa + 16);
        const uint32_t a3 = *reinterpret_cast<const uint32_t *>(wa + 8 * WDS + 16);
        int d[NT][4];
#pragma unroll
        for (int n = 0; n < NT; ++n)
          if (n < nta)
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                "{%10,%10,%10,%10};\n"
                : "=r"(d[n][0]), "=r"(d[n][1]), "=r"(d[n][2]), "=r"(d[n][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[n]), "r"(b1[n]), "r"(kMagicBits));
        const float w0 = swd[r0 * KB32 + j], w1 = swd[(r0 + 8) * KB32 + j];
        const float n0 = -kMagic * w0, n1 = -kMagic * w1; // exact: w has <= 4 significant bits
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          if (n < nta) {
            const float2 xs = *reinterpret_cast<const float2 *>(sxs + j * P + n * 8 + 2 * t);
            acc[m][n][0] = fmaf(fmaf(__int_as_float(d[n][0]), w0, n0), xs.x, acc[m][n][0]);
            acc[m][n][1] = fmaf(fmaf(__int_as_float(d[n][1]), w0, n0), xs.y, acc[m][n][1]);
            acc[m][n][2] = fmaf(fmaf(__int_as_float(d[n][2]), w1, n1), xs.x, acc[m][n][2]);
            acc[m][n][3] = fmaf(fmaf(__int_as_float(d[n][3]), w1, n1), xs.y, acc[m][n][3]);
          }
        }
      }
    }
  }
  ink2p_nv2_wait<0>();
  const float f = escale ? escale[e] : 1.f;
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int row = rb + wr + m * 16 + g + (c >= 2 ? 8 : 0), col = n * 8 + 2 * t + (c & 1);
        if (n < nta && col < np) Y[(size_t)pairs[col] * rows + row] = acc[m][n][c] * f;
      }
}

template <int MT, int WARPS, int STAGES, int BK, int NTILE, int MINB = 1, int PF = 1>
static int ink2p_nv3_launch(const uint8_t *W, const uint8_t *scales, const int *offsets, const int *order,
                            const int *tiles, const int *ntiles, int maxTiles, const float *X, int xDiv, int xStride,
                            float *Y, int rows, int in, const float *escale) {
  constexpr int BM = WARPS * 16 * MT, P = INK2P_NV2_PAIRS * NTILE;
  constexpr int STAGE = BM * (BK / 2 + 16) + BM * (BK / 16) + P * (BK + 16) + (BK / 32) * P * 4;
  constexpr int SMEM = STAGES * STAGE + BM * (BK + 16) + BM * (BK / 32) * 4;
  const size_t maxRows = ((size_t)INK2P_NV2_PAIRS * maxTiles + xDiv - 1) / xDiv;
  const size_t qBytes = (maxRows * in + 15) & ~(size_t)15, need = qBytes + maxRows * (in / 16) * sizeof(float);
  if (need > g_ink2p_nv2_q_cap) {
    if (g_ink2p_nv2_q) cudaFree(g_ink2p_nv2_q);
    g_ink2p_nv2_q = nullptr;
    g_ink2p_nv2_q_cap = 0;
    const cudaError_t e = cudaMalloc(&g_ink2p_nv2_q, need);
    if (e != cudaSuccess) return (int)e;
    g_ink2p_nv2_q_cap = need;
  }
  int8_t *q = g_ink2p_nv2_q;
  float *qs = reinterpret_cast<float *>(q + qBytes);
  static bool attr = false;
  if (!attr) {
    const cudaError_t e = cudaFuncSetAttribute(k_ink2p_expert_nvfp4_v3<MT, WARPS, STAGES, BK, NTILE, MINB, PF>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    if (e != cudaSuccess) return (int)e;
    attr = true;
  }
  const int qgrid = (int)((maxRows * in + 255) / 256 < 4096 ? (maxRows * in + 255) / 256 : 4096);
  FCUDA_TS("k_ink2p_nv2_quant", qgrid,
           k_ink2p_nv2_quant<32><<<qgrid, 256, 0, g_stream>>>(offsets, tiles, ntiles, X, xDiv, xStride, q, qs, in));
  const dim3 grid(rows / BM, maxTiles);
  FCUDA_TS("k_ink2p_expert_nvfp4_v3", grid,
           (k_ink2p_expert_nvfp4_v3<MT, WARPS, STAGES, BK, NTILE, MINB, PF><<<grid, WARPS * 32, SMEM, g_stream>>>(
               W, scales, offsets, order, tiles, ntiles, q, qs, xDiv, Y, rows, in, escale)));
  return (int)cudaGetLastError();
}

// Drop-in for fcuda_ink2p_expert_nvfp4 (same arguments and bucket scratch); shapes the
// int8 path does not tile fall back to the BF16 kernel.
extern "C" int fcuda_ink2p_expert_nvfp4_v2(const uint8_t *W, const uint8_t *scales, const int *offsets,
                                           const int *order, const int *tiles, const int *ntiles, int maxTiles,
                                           const float *X, int xDiv, int xStride, float *Y, int rows, int in,
                                           const float *escale) {
  constexpr int MT = 2, WARPS = 8, STAGES = 3, BK = 64, NTILE = 1, PF = 1;
  if (!scales || in % BK != 0 || rows % (WARPS * 16 * MT) != 0 || INK2P_NV2_PAIRS != INK2P_NV_PAIRS)
    return fcuda_ink2p_expert_nvfp4(W, scales, offsets, order, tiles, ntiles, maxTiles, X, xDiv, xStride, Y, rows,
                                    in, escale);
  return ink2p_nv2_launch<MT, WARPS, STAGES, BK, NTILE, PF>(W, scales, offsets, order, tiles, ntiles, maxTiles, X,
                                                            xDiv, xStride, Y, rows, in, escale);
}
