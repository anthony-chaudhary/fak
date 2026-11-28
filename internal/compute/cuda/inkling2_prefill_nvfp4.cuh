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

// One E4M3 byte to float, exact; the fallback for targets below sm_89, which lack the cvt.
static __device__ __forceinline__ float ink2p_nv2_e4m3(uint32_t b) {
  const uint32_t sign = (b & 0x80u) << 24, e = (b >> 3) & 0xFu, m = b & 7u;
  if (e == 0xFu && m == 7u) return __uint_as_float(sign | 0x7fc00000u);
  if (e == 0u) return __uint_as_float(sign | __float_as_uint((float)m * 0x1p-9f));
  return __uint_as_float(sign | ((e + 120u) << 23) | (m << 20));
}

// Two E4M3 bytes (low byte first) to two floats.
static __device__ __forceinline__ float2 ink2p_nv2_e4m3x2(uint32_t raw16) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  uint32_t h2;
  asm("{.reg .b16 r; cvt.u16.u32 r, %1; cvt.rn.f16x2.e4m3x2 %0, r;}\n" : "=r"(h2) : "r"(raw16));
  return __half22float2(*reinterpret_cast<__half2 *>(&h2));
#else
  return make_float2(ink2p_nv2_e4m3(raw16 & 0xFFu), ink2p_nv2_e4m3((raw16 >> 8) & 0xFFu));
#endif
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
#include <unordered_set>

static std::unordered_set<const void *> g_ink2p_nvfp4_slab_set;

static bool ink2p_nvfp4_is_slab(const void *p) {
  return !g_ink2p_nvfp4_slab_set.empty() && g_ink2p_nvfp4_slab_set.count(p) != 0;
}

static void ink2p_nvfp4_slab_evict(const void *p) {
  if (!g_ink2p_nvfp4_slab_set.empty()) g_ink2p_nvfp4_slab_set.erase(p);
}

__global__ static void k_ink2p_nvfp4_repack_packed_slab128(
    const uint4 *__restrict__ src, uint4 *__restrict__ dst, int batchE, int rows, int nk) {
  const size_t total = (size_t)batchE * rows * nk * 4u;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int col16 = (int)(idx % (size_t)(nk * 4));
  const size_t er = idx / (size_t)(nk * 4);
  const int row = (int)(er % (size_t)rows);
  const int e = (int)(er / (size_t)rows);
  const int slab = row >> 7, r = row & 127;
  const int kc = col16 >> 2, c = col16 & 3;
  const size_t slabOff = ((size_t)e * (rows >> 7) + slab) * nk + kc;
  dst[slabOff * (128u * 4u) + (size_t)r * 4u + c] = src[idx];
}

__global__ static void k_ink2p_nvfp4_repack_scales_slab128(
    const uint64_t *__restrict__ src, uint64_t *__restrict__ dst, int batchE, int rows, int nk) {
  const size_t total = (size_t)batchE * rows * nk;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int kc = (int)(idx % (size_t)nk);
  const size_t er = idx / (size_t)nk;
  const int row = (int)(er % (size_t)rows);
  const int e = (int)(er / (size_t)rows);
  const int slab = row >> 7, r = row & 127;
  const size_t slabOff = ((size_t)e * (rows >> 7) + slab) * nk + kc;
  dst[slabOff * 128u + r] = src[idx];
}

extern "C" int fcuda_ink2p_nvfp4_ensure_slab128(uint8_t *W, uint8_t *scales, int nexp, int rows, int in) {
  if (!W || !scales || nexp <= 0 || rows <= 0 || in <= 0 || (rows & 127) != 0 || (in & 127) != 0) {
    return (int)cudaErrorInvalidValue;
  }
  if (ink2p_nvfp4_is_slab(W)) return 0;
  if (g_capture_open) return (int)cudaErrorStreamCaptureUnsupported;
  const int nk = in >> 7;
  const int batchCap = nexp < 8 ? nexp : 8;
  const size_t pExpBytes = (size_t)rows * (size_t)(in >> 1);
  const size_t sExpBytes = (size_t)rows * (size_t)(in >> 4);
  uint8_t *tmpP = nullptr, *tmpS = nullptr;
  cudaError_t ce = cudaMalloc(&tmpP, (size_t)batchCap * pExpBytes);
  if (ce != cudaSuccess) return (int)ce;
  ce = cudaMalloc(&tmpS, (size_t)batchCap * sExpBytes);
  if (ce != cudaSuccess) {
    cudaFree(tmpP);
    return (int)ce;
  }
  for (int e0 = 0; e0 < nexp; e0 += batchCap) {
    const int be = (nexp - e0 < batchCap) ? (nexp - e0) : batchCap;
    uint8_t *wPtr = W + (size_t)e0 * pExpBytes;
    uint8_t *sPtr = scales + (size_t)e0 * sExpBytes;
    const size_t pTotal = (size_t)be * rows * nk * 4u;
    const size_t sTotal = (size_t)be * rows * nk;
    k_ink2p_nvfp4_repack_packed_slab128<<<(unsigned)((pTotal + 255) / 256), 256, 0, g_stream>>>(
        (const uint4 *)wPtr, (uint4 *)tmpP, be, rows, nk);
    k_ink2p_nvfp4_repack_scales_slab128<<<(unsigned)((sTotal + 255) / 256), 256, 0, g_stream>>>(
        (const uint64_t *)sPtr, (uint64_t *)tmpS, be, rows, nk);
    ce = cudaGetLastError();
    if (ce != cudaSuccess) break;
    ce = cudaMemcpyAsync(wPtr, tmpP, (size_t)be * pExpBytes, cudaMemcpyDeviceToDevice, g_stream);
    if (ce != cudaSuccess) break;
    ce = cudaMemcpyAsync(sPtr, tmpS, (size_t)be * sExpBytes, cudaMemcpyDeviceToDevice, g_stream);
    if (ce != cudaSuccess) break;
  }
  if (ce == cudaSuccess) ce = cudaStreamSynchronize(g_stream);
  cudaFree(tmpS);
  cudaFree(tmpP);
  if (ce != cudaSuccess) return (int)ce;
  g_ink2p_nvfp4_slab_set.insert(W);
  return 0;
}

template <int MT, int WARPS, int STAGES, int BK, int NTILE, int PF, bool SLAB = false>
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
    if (SLAB) {
      const size_t slabOff = ((size_t)e * (rows / BM) + blockIdx.x) * nk + kc;
      const uint8_t *pSlab = Packed + slabOff * (size_t)(BM * (BK / 2));
      const uint8_t *sSlab = Scales + slabOff * (size_t)(BM * KB);
      for (int i = tid; i < BM * (BK / 32); i += T) {
        const int r = i / (BK / 32), c = i % (BK / 32);
        ink2p_nv2_cp16w<PF>(s + r * WS + c * 16, pSlab + (size_t)i * 16);
      }
      for (int i = tid; i < BM; i += T) {
        if (KB == 8) ink2p_nv2_cp8(s + SW + i * KB, sSlab + (size_t)i * 8);
        else ink2p_nv2_cp4(s + SW + i * KB, sSlab + (size_t)i * 4);
      }
    } else {
      for (int i = tid; i < BM * (BK / 32); i += T) {
        const int r = i / (BK / 32), c = i % (BK / 32);
        ink2p_nv2_cp16w<PF>(s + r * WS + c * 16, Packed + (wrow0 + r) * rowBytes + k0 / 2 + c * 16);
      }
      for (int i = tid; i < BM; i += T) {
        if (KB == 8) ink2p_nv2_cp8(s + SW + i * KB, Scales + (wrow0 + i) * nb + k0 / 16);
        else ink2p_nv2_cp4(s + SW + i * KB, Scales + (wrow0 + i) * nb + k0 / 16);
      }
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

template <int MT, int WARPS, int STAGES, int BK, int NTILE, int PF, bool SLAB = false>
static cudaError_t ink2p_nv2_prepare_smem() {
  constexpr int BM = WARPS * 16 * MT, P = INK2P_NV2_PAIRS * NTILE;
  constexpr int STAGE = BM * (BK / 2 + 16) + BM * (BK / 16) + P * (BK + 16) + (BK / 16) * P * 4;
  static bool attr = false;
  if (attr) return cudaSuccess;
  const cudaError_t e = cudaFuncSetAttribute(k_ink2p_expert_nvfp4_v2<MT, WARPS, STAGES, BK, NTILE, PF, SLAB>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, STAGES * STAGE);
  attr = e == cudaSuccess;
  return e;
}

template <int MT, int WARPS, int STAGES, int BK = INK2P_NV2_BK, int NTILE = 1, int PF = 1, bool SLAB = false>
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
    const cudaError_t e = fcuda_alloc_status(cudaMalloc(&g_ink2p_nv2_q, need));
    if (e != cudaSuccess) return (int)e;
    g_ink2p_nv2_q_cap = need;
  }
  int8_t *q = g_ink2p_nv2_q;
  float *qs = reinterpret_cast<float *>(q + qBytes);
  const cudaError_t admitted = ink2p_nv2_prepare_smem<MT, WARPS, STAGES, BK, NTILE, PF, SLAB>();
  if (admitted != cudaSuccess) return (int)admitted;
  const int qgrid = (int)((maxRows * in + 255) / 256 < 4096 ? (maxRows * in + 255) / 256 : 4096);
  FCUDA_TS("k_ink2p_nv2_quant", qgrid,
           k_ink2p_nv2_quant<16><<<qgrid, 256, 0, g_stream>>>(offsets, tiles, ntiles, X, xDiv, xStride, q, qs, in));
  const dim3 grid(rows / BM, maxTiles);
  FCUDA_TS(SLAB ? "k_ink2p_expert_nvfp4_v2_slab128" : (BM == 128 && BK == 128 ? "k_ink2p_expert_nvfp4_v2_bm128" : "k_ink2p_expert_nvfp4_v2"), grid,
           (k_ink2p_expert_nvfp4_v2<MT, WARPS, STAGES, BK, NTILE, PF, SLAB><<<grid, WARPS * 32, SMEM, g_stream>>>(
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
    const cudaError_t e = fcuda_alloc_status(cudaMalloc(&g_ink2p_nv2_q, need));
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

// Native W4A4 path: activations are quantized to NVFP4 with the algorithm of
// llama.cpp's quantize_mmq_nvfp4 (fp32 row scale amax / (6 * 448), one UE4M3 code per
// 16 columns picked by a +-2 code squared-error search, RN-satfinite E2M1 values) but
// IEEE-rounded at every step; llama.cpp builds with fast math, so near-tie codes can
// differ from it, while the Go port in the tests matches this kernel bit for bit. And
// each 64-wide K step is one block-scaled mma.m16n8k64 e2m1 x e2m1 with the
// repacked E4M3 weight scales and the activation codes applied in hardware. The
// repacked planes are already in MMA order (byte i = K 2i low, 2i+1 high), so the
// output needs no factor beyond the row scale and escale. The instructions exist
// only in sm_120a / sm_120f (and sm_121a) images; other images compile empty
// kernels and the host learns which one loaded from k_ink2p_nvf4_probe.
#if defined(__CUDA_ARCH__) &&                                                    \
    (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) || \
     (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))
#define INK2P_NVF4_NATIVE 1
#else
#define INK2P_NVF4_NATIVE 0
#endif

#define INK2P_NVF4_BK 128

#if INK2P_NVF4_NATIVE
// Two floats to one byte of E2M1 codes; the first cvt source lands in the high nibble.
static __device__ __forceinline__ uint32_t ink2p_nvf4_e2m1x2(float lo, float hi) {
  uint16_t r;
  asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

static __device__ __forceinline__ uint32_t ink2p_nvf4_e4m3(float x) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"(x));
  return r & 0xffu;
}

static __device__ __forceinline__ float ink2p_nvf4_e2m1abs(uint32_t c) {
  const uint32_t e = (c >> 1) & 3u, m = c & 1u;
  return __uint_as_float(e ? (((e + 126u) << 23) | (m << 22)) : (m ? 0x3f000000u : 0u));
}

// Squared reconstruction error of one 16-value block (v already divided by the row
// scale) under block scale sc, summed in llama.cpp's element order.
static __device__ __forceinline__ float ink2p_nvf4_err(const float (&v)[16], float sc) {
  const float inv = sc > 0.f ? __frcp_rn(sc) : 0.f;
  float err = 0.f;
#pragma unroll
  for (int k = 0; k < 16; k += 2) {
    const uint32_t q = ink2p_nvf4_e2m1x2(__fmul_rn(v[k], inv), __fmul_rn(v[k + 1], inv));
    const float e0 = __fsub_rn(fabsf(v[k]), __fmul_rn(ink2p_nvf4_e2m1abs(q & 0xfu), sc));
    const float e1 = __fsub_rn(fabsf(v[k + 1]), __fmul_rn(ink2p_nvf4_e2m1abs(q >> 4), sc));
    err = fmaf(e0, e0, err);
    err = fmaf(e1, e1, err);
  }
  return err;
}

static __device__ __forceinline__ void ink2p_nvf4_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                                      uint32_t a3, uint32_t b0, uint32_t b1, uint32_t sfa,
                                                      uint32_t sfb) {
  // {byte-id, thread-id} = {0, 0}: all four scale bytes are used; scale_A comes from
  // lanes %4 == 0 (row g) and 1 (row g+8), scale_B from lane %4 == 0 (column g).
  asm("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

// Byte offset of 16-byte chunk c of 64-byte row r. XOR by (r >> 1) & 3 puts the
// eight rows a fragment load touches on distinct bank quads.
static __device__ __forceinline__ int ink2p_nvf4_swz(int r, int c) { return r * 64 + ((c ^ ((r >> 1) & 3)) << 4); }
#endif

__device__ int g_ink2p_nvf4_flag;
__global__ void k_ink2p_nvf4_probe() { g_ink2p_nvf4_flag = INK2P_NVF4_NATIVE; }

// One block per X row: Q[r] holds in/2 bytes of E2M1 pairs (low nibble = even column),
// QS[r] in/16 UE4M3 codes, RS[r] the fp32 row scale. Requires in % 16 == 0.
template <int T>
__global__ void __launch_bounds__(T) k_ink2p_nvf4_quant(const int *offsets, const int *tiles, const int *ntiles,
                                                        const float *X, int xDiv, int xStride, uint8_t *Q,
                                                        uint8_t *QS, float *RS, int in) {
#if INK2P_NVF4_NATIVE
  __shared__ float wmax[T / 32];
  const int nrows = ink2p_nv2_xrows(offsets, tiles, ntiles, xDiv);
  const int nb = in / 16;
  for (int r = blockIdx.x; r < nrows; r += gridDim.x) {
    const float *x = X + (size_t)r * xStride;
    float amax = 0.f;
    for (int i = threadIdx.x; i < in; i += T) amax = fmaxf(amax, fabsf(x[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((threadIdx.x & 31) == 0) wmax[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = wmax[0];
#pragma unroll
    for (int w = 1; w < T / 32; ++w) amax = fmaxf(amax, wmax[w]);
    __syncthreads();
    const float rs = __fdiv_rn(amax, 6.f * 448.f);
    if (threadIdx.x == 0) RS[r] = rs;
    const float ic = rs > 0.f ? __frcp_rn(rs) : 0.f;
    for (int b = threadIdx.x; b < nb; b += T) {
      float vals[16], v[16], amax16 = 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        vals[k] = x[b * 16 + k];
        v[k] = __fmul_rn(vals[k], ic);
        amax16 = fmaxf(amax16, fabsf(v[k]));
      }
      const float a6 = __fdiv_rn(amax16, 6.f);
      const int first = a6 > 0.f ? (int)ink2p_nvf4_e4m3(a6) : 0;
      int code = first;
      float sc = ink2p_nv2_e4m3x2((uint32_t)first).x;
      float best = ink2p_nvf4_err(v, sc);
#pragma unroll
      for (int i = 1; i < 5; ++i) {
        const int tc = first + ((i & 1) ? -((i + 1) >> 1) : (i >> 1));  // -1, +1, -2, +2
        if (tc < 0 || tc > 0x7e) continue;
        const float tsc = ink2p_nv2_e4m3x2((uint32_t)tc).x;
        const float cur = ink2p_nvf4_err(v, tsc);
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
        q0 |= ink2p_nvf4_e2m1x2(__fmul_rn(vals[2 * i], s), __fmul_rn(vals[2 * i + 1], s)) << (8 * i);
        q1 |= ink2p_nvf4_e2m1x2(__fmul_rn(vals[8 + 2 * i], s), __fmul_rn(vals[9 + 2 * i], s)) << (8 * i);
      }
      *reinterpret_cast<uint2 *>(Q + (size_t)r * (in / 2) + b * 8) = make_uint2(q0, q1);
      QS[(size_t)r * nb + b] = (uint8_t)code;
    }
  }
#endif
}

// Same bucketing contract, grid and epilogue placement as k_ink2p_expert_nvfp4_v2 with
// NTILE = 1. Warp w owns rows [w*16*MT, (w+1)*16*MT) against every n8 tile of the 32
// pairs; a stage carries INK2P_NVF4_BK columns of weights, weight scales, activation
// nibbles and activation codes.
template <int MT, int WARPS, int STAGES, int PF>
__global__ void __launch_bounds__(WARPS * 32) k_ink2p_expert_nvfp4_w4a4(
    const uint8_t *Packed, const uint8_t *Scales, const int *offsets, const int *order, const int *tiles,
    const int *ntiles, const uint8_t *Q, const uint8_t *QS, const float *RS, int xDiv, float *Y, int rows, int in,
    const float *escale) {
#if INK2P_NVF4_NATIVE
  constexpr int BK = INK2P_NVF4_BK, KB = BK / 16, BM = WARPS * 16 * MT, P = INK2P_NV2_PAIRS, NT = P / 8;
  constexpr int T = WARPS * 32;
  static_assert(BK / 2 == 64, "ink2p_nvf4_swz assumes 64-byte stage rows");
  constexpr int SW = BM * (BK / 2), SWS = BM * KB, SX = P * (BK / 2), SXS = P * KB;
  constexpr int STAGE = SW + SWS + SX + SXS;
  extern __shared__ __align__(16) uint8_t ink2p_nvf4_smem[];
  __shared__ int pairs[P], xrows[P];
  __shared__ float prs[P];
  if ((int)blockIdx.y >= ntiles[0]) return;
  const int e = tiles[2 * blockIdx.y], start = tiles[2 * blockIdx.y + 1];
  const int np = min(P, offsets[e + 1] - start);
  const int tid = threadIdx.x;
  for (int i = tid; i < P; i += T) {
    const int p = order[start + (i < np ? i : 0)];
    pairs[i] = i < np ? p : -1;
    xrows[i] = p / xDiv;
    prs[i] = RS[p / xDiv];
  }
  __syncthreads();
  const int nta = (np + 7) >> 3;
  const int rb = blockIdx.x * BM;
  const size_t wrow0 = (size_t)e * rows + rb;
  const int rowBytes = in / 2, nb = in / 16, nk = in / BK;

  auto load = [&](int stage, int kc) {
    uint8_t *s = ink2p_nvf4_smem + stage * STAGE;
    const int k0 = kc * BK;
    for (int i = tid; i < BM * 4; i += T) {
      const int r = i >> 2, c = i & 3;
      ink2p_nv2_cp16w<PF>(s + ink2p_nvf4_swz(r, c), Packed + (wrow0 + r) * rowBytes + k0 / 2 + c * 16);
    }
    for (int i = tid; i < BM; i += T) ink2p_nv2_cp8(s + SW + i * KB, Scales + (wrow0 + i) * nb + k0 / 16);
    for (int i = tid; i < nta * 8 * 4; i += T) {
      const int c = i >> 2, j = i & 3;
      ink2p_nv2_cp16w<PF>(s + SW + SWS + ink2p_nvf4_swz(c, j), Q + (size_t)xrows[c] * rowBytes + k0 / 2 + j * 16);
    }
    for (int i = tid; i < nta * 8; i += T)
      ink2p_nv2_cp8(s + SW + SWS + SX + i * KB, QS + (size_t)xrows[i] * nb + k0 / 16);
  };

  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wr = warp * 16 * MT;
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
    const uint8_t *s = ink2p_nvf4_smem + (kc % STAGES) * STAGE;
    const uint8_t *sw = s, *sws = s + SW, *sx = s + SW + SWS, *sxs = s + SW + SWS + SX;
#pragma unroll
    for (int ks = 0; ks < BK / 64; ++ks) {
      uint32_t b0[NT], b1[NT], sb[NT];
#pragma unroll
      for (int n = 0; n < NT; ++n)
        if (n < nta) {
          const int c = n * 8 + g;
          b0[n] = *reinterpret_cast<const uint32_t *>(sx + ink2p_nvf4_swz(c, 2 * ks) + 4 * t);
          b1[n] = *reinterpret_cast<const uint32_t *>(sx + ink2p_nvf4_swz(c, 2 * ks + 1) + 4 * t);
          sb[n] = *reinterpret_cast<const uint32_t *>(sxs + c * KB + 4 * ks);
        }
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const int r0 = wr + m * 16 + g;
        const uint32_t a0 = *reinterpret_cast<const uint32_t *>(sw + ink2p_nvf4_swz(r0, 2 * ks) + 4 * t);
        const uint32_t a1 = *reinterpret_cast<const uint32_t *>(sw + ink2p_nvf4_swz(r0 + 8, 2 * ks) + 4 * t);
        const uint32_t a2 = *reinterpret_cast<const uint32_t *>(sw + ink2p_nvf4_swz(r0, 2 * ks + 1) + 4 * t);
        const uint32_t a3 = *reinterpret_cast<const uint32_t *>(sw + ink2p_nvf4_swz(r0 + 8, 2 * ks + 1) + 4 * t);
        const uint32_t sa = *reinterpret_cast<const uint32_t *>(sws + (r0 + 8 * (t & 1)) * KB + 4 * ks);
#pragma unroll
        for (int n = 0; n < NT; ++n)
          if (n < nta) ink2p_nvf4_mma(acc[m][n], a0, a1, a2, a3, b0[n], b1[n], sa, sb[n]);
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
        if (n < nta && col < np) Y[(size_t)pairs[col] * rows + row] = acc[m][n][c] * prs[col] * f;
      }
#endif
}

// 1 when the loaded image carries the native kernels. Backend init probes eagerly so
// capture and eager calls agree; only a successful probe is cached, and a probe is
// skipped while g_stream captures or another call's error is still pending so that
// error reaches its own status check.
static int g_ink2p_nvf4_avail = -1;
static int ink2p_nvf4_available() {
  if (g_ink2p_nvf4_avail >= 0) return g_ink2p_nvf4_avail;
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(g_stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone) return 0;
  if (cudaPeekAtLastError() != cudaSuccess) return 0;
  int flag = 0;
  FCUDA_TS("k_ink2p_nvf4_probe", 1, k_ink2p_nvf4_probe<<<1, 1, 0, g_stream>>>());
  cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess)
    e = cudaMemcpyFromSymbolAsync(&flag, g_ink2p_nvf4_flag, sizeof(flag), 0, cudaMemcpyDeviceToHost, g_stream);
  if (e == cudaSuccess) e = cudaStreamSynchronize(g_stream);
  if (e != cudaSuccess) {
    static bool logged = false;
    if (!logged) {
      fprintf(stderr, "fak-cuda: NVFP4 W4A4 image probe failed (%s); int8 path until a probe succeeds\n",
              cudaGetErrorString(e));
      logged = true;
    }
    (void)cudaGetLastError();
    return 0;
  }
  g_ink2p_nvf4_avail = flag == 1 ? 1 : 0;
  return g_ink2p_nvf4_avail;
}

extern "C" int fcuda_ink2p_nvfp4_w4a4_available(void) { return ink2p_nvf4_available(); }

// Grow-only NVFP4 activation scratch: nibbles, then UE4M3 codes, then fp32 row scales.
static uint8_t *g_ink2p_nvf4_q = nullptr;
static size_t g_ink2p_nvf4_q_cap = 0;
// Layout of the last quantize, for the test readback.
static size_t g_ink2p_nvf4_q_rows = 0, g_ink2p_nvf4_q_bytes = 0, g_ink2p_nvf4_s_bytes = 0;
static int g_ink2p_nvf4_q_in = 0;

template <int MT, int WARPS, int STAGES, int PF = 1>
static int ink2p_nvf4_launch(const uint8_t *W, const uint8_t *scales, const int *offsets, const int *order,
                             const int *tiles, const int *ntiles, int maxTiles, const float *X, int xDiv, int xStride,
                             float *Y, int rows, int in, const float *escale) {
  constexpr int BK = INK2P_NVF4_BK, BM = WARPS * 16 * MT, P = INK2P_NV2_PAIRS, QT = 128;
  constexpr int STAGE = BM * (BK / 2) + BM * (BK / 16) + P * (BK / 2) + P * (BK / 16);
  constexpr int SMEM = STAGES * STAGE;
  const size_t maxRows = ((size_t)P * maxTiles + xDiv - 1) / xDiv;
  const size_t qBytes = (maxRows * (in / 2) + 15) & ~(size_t)15, sBytes = (maxRows * (in / 16) + 15) & ~(size_t)15;
  const size_t need = qBytes + sBytes + maxRows * sizeof(float);
  if (need > g_ink2p_nvf4_q_cap) {
    // cudaMalloc/cudaFree inside a capture invalidate it; the caller must size the
    // scratch with an eager call of the largest shape before capturing.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(g_stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone)
      return (int)cudaErrorStreamCaptureUnsupported;
    if (g_ink2p_nvf4_q) cudaFree(g_ink2p_nvf4_q);
    g_ink2p_nvf4_q = nullptr;
    g_ink2p_nvf4_q_cap = 0;
    const cudaError_t e = fcuda_alloc_status(cudaMalloc(&g_ink2p_nvf4_q, need));
    if (e != cudaSuccess) return (int)e;
    g_ink2p_nvf4_q_cap = need;
  }
  uint8_t *q = g_ink2p_nvf4_q, *qs = q + qBytes;
  float *rs = reinterpret_cast<float *>(qs + sBytes);
  g_ink2p_nvf4_q_rows = maxRows;
  g_ink2p_nvf4_q_bytes = qBytes;
  g_ink2p_nvf4_s_bytes = sBytes;
  g_ink2p_nvf4_q_in = in;
  if (SMEM > 48 * 1024) {
    static bool attr = false;
    if (!attr) {
      const cudaError_t e = cudaFuncSetAttribute(k_ink2p_expert_nvfp4_w4a4<MT, WARPS, STAGES, PF>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
      if (e != cudaSuccess) return (int)e;
      attr = true;
    }
  }
  const int qgrid = (int)(maxRows < 1 ? 1 : maxRows < 4096 ? maxRows : 4096);
  FCUDA_TS("k_ink2p_nvf4_quant", qgrid,
           k_ink2p_nvf4_quant<QT><<<qgrid, QT, 0, g_stream>>>(offsets, tiles, ntiles, X, xDiv, xStride, q, qs, rs, in));
  const dim3 grid(rows / BM, maxTiles);
  FCUDA_TS("k_ink2p_expert_nvfp4_w4a4", grid,
           (k_ink2p_expert_nvfp4_w4a4<MT, WARPS, STAGES, PF><<<grid, WARPS * 32, SMEM, g_stream>>>(
               W, scales, offsets, order, tiles, ntiles, q, qs, rs, xDiv, Y, rows, in, escale)));
  return (int)cudaGetLastError();
}

// Copies the first nrows rows of the last W4A4 activation quantize (nibbles, UE4M3
// codes, row scales) to the host so a test can compare them with the host quantizer.
extern "C" int fcuda_ink2p_nvfp4_w4a4_read_quant(int nrows, int in, uint8_t *q, uint8_t *qs, float *rs) {
  if (!g_ink2p_nvf4_q || nrows < 0 || (size_t)nrows > g_ink2p_nvf4_q_rows || in != g_ink2p_nvf4_q_in)
    return (int)cudaErrorInvalidValue;
  const uint8_t *dq = g_ink2p_nvf4_q, *dqs = dq + g_ink2p_nvf4_q_bytes;
  const uint8_t *drs = dqs + g_ink2p_nvf4_s_bytes;
  cudaError_t e = cudaSuccess;
  FCUDA_TS("readback", dim3(0, 0),
           e = cudaMemcpyAsync(q, dq, (size_t)nrows * (in / 2), cudaMemcpyDeviceToHost, g_stream));
  if (e == cudaSuccess)
    FCUDA_TS("readback", dim3(0, 0),
             e = cudaMemcpyAsync(qs, dqs, (size_t)nrows * (in / 16), cudaMemcpyDeviceToHost, g_stream));
  if (e == cudaSuccess)
    FCUDA_TS("readback", dim3(0, 0),
             e = cudaMemcpyAsync(rs, drs, (size_t)nrows * sizeof(float), cudaMemcpyDeviceToHost, g_stream));
  if (e == cudaSuccess) e = cudaStreamSynchronize(g_stream);
  return (int)e;
}

// W4A4 activations run wherever the image carries the native kernel (sm_120f/a);
// FAK_INKLING2_PREFILL_NVFP4_W4A4=0 keeps int8 activations.
static bool ink2p_nvf4_selected() {
  const char *v = getenv("FAK_INKLING2_PREFILL_NVFP4_W4A4");
  if (v && strcmp(v, "0") == 0) return false;
  return ink2p_nvf4_available() == 1;
}

// Drop-in for fcuda_ink2p_expert_nvfp4 (same arguments and bucket scratch); shapes the
// int8 path does not tile fall back to the BF16 kernel.
extern "C" int fcuda_ink2p_expert_nvfp4_v2(const uint8_t *W, const uint8_t *scales, const int *offsets,
                                           const int *order, const int *tiles, const int *ntiles, int maxTiles,
                                           const float *X, int xDiv, int xStride, float *Y, int rows, int in,
                                           const float *escale) {
  if (scales && in % 128 == 0 && rows % 128 == 0 && INK2P_NV2_PAIRS == INK2P_NV_PAIRS && ink2p_nvfp4_is_slab(W)) {
    const cudaError_t admitted = ink2p_nv2_prepare_smem<1, 8, 2, 128, 1, 1, true>();
    if (admitted == cudaSuccess)
      return ink2p_nv2_launch<1, 8, 2, 128, 1, 1, true>(W, scales, offsets, order, tiles, ntiles, maxTiles, X, xDiv,
                                                        xStride, Y, rows, in, escale);
    return (int)admitted;
  }
  if (scales && in % INK2P_NVF4_BK == 0 && rows % 256 == 0 && ink2p_nvf4_selected())
    return ink2p_nvf4_launch<2, 8, 2>(W, scales, offsets, order, tiles, ntiles, maxTiles, X, xDiv, xStride, Y, rows,
                                      in, escale);
  constexpr int MT = 2, WARPS = 8, STAGES = 3, BK = 64, NTILE = 1, PF = 1;
  if (!scales || in % BK != 0 || rows % (WARPS * 16 * MT) != 0 || INK2P_NV2_PAIRS != INK2P_NV_PAIRS)
    return fcuda_ink2p_expert_nvfp4(W, scales, offsets, order, tiles, ntiles, maxTiles, X, xDiv, xStride, Y, rows,
                                    in, escale);
  static bool bm128_refused = false;
  const char *bm128 = getenv("FAK_INKLING2_NVFP4_BM128");
  if (!bm128_refused && in % 128 == 0 && bm128 && strcmp(bm128, "1") == 0) {
    const cudaError_t admitted = ink2p_nv2_prepare_smem<1, 8, 2, 128, 1, 1, false>();
    if (admitted == cudaSuccess)
      return ink2p_nv2_launch<1, 8, 2, 128, 1, 1, false>(W, scales, offsets, order, tiles, ntiles, maxTiles, X, xDiv,
                                                         xStride, Y, rows, in, escale);
    if (admitted != cudaErrorInvalidValue && admitted != cudaErrorNotSupported) return (int)admitted;
    cudaGetLastError();
    bm128_refused = true;
  }
  return ink2p_nv2_launch<MT, WARPS, STAGES, BK, NTILE, PF>(W, scales, offsets, order, tiles, ntiles, maxTiles, X,
                                                            xDiv, xStride, Y, rows, in, escale);
}
