// inkling2_expert_gemv.cuh: bandwidth-shaped NVFP4 expert GEMV for the resident
// inkling2_loop decode. Included by inkling2_loop.cuh after k_ink2_expert_nvfp4,
// whose arithmetic it keeps bit for bit: lane l still owns 16-value blocks
// l, l+32, ... of every row, each block is the same 16-step fmaf chain scaled
// once, and the lane sums meet in the same shuffle tree. Only the schedule moves:
//   - the grid is sized to about 4 blocks per SM and every warp walks rows with a
//     grid stride, so the old 576-block grid's second, nearly empty wave is gone;
//   - a row's KB packed words and scale bytes are all issued before its math, the
//     next row's loads are issued before the current row's reduction, and the
//     first row's loads overlap the activation staging;
//   - the staged activation is swizzled per 16-value block so the float4 reads of
//     32 lanes (block stride 64 B) hit distinct shared-memory bank groups.
// FAK_INKLING2_NVFP4_GEMV=0 keeps the reference kernel.

#ifndef INK2_NV2_WARPS
#define INK2_NV2_WARPS 8
#endif
#ifndef INK2_NV2_ROWS
#define INK2_NV2_ROWS 1
#endif
#define INK2_NV2_MAX_KB 8
#ifndef INK2_NV2_BLOCKS_PER_SM
#define INK2_NV2_BLOCKS_PER_SM 4
#endif
// A verify chunk routes T*k slots through one launch; past INK2_WIDE_SLOTS slots a
// sparser grid (fewer, longer blocks per SM) is the faster schedule.
#define INK2_WIDE_SLOTS 16
#ifndef INK2_NV2_BLOCKS_PER_SM_WIDE
#define INK2_NV2_BLOCKS_PER_SM_WIDE 2
#endif

// Physical float4 slot of activation float4 j (0..3) of 16-value block b.
static __device__ __forceinline__ int ink2_nv2_slot(int b, int j) { return b * 4 + ((j + (b >> 1)) & 3); }

template <int KB, int R, bool SLAB = false>
__global__ void __launch_bounds__(INK2_NV2_WARPS * 32)
    k_ink2_expert_nvfp4_v2(const uint8_t *__restrict__ Packed, const uint8_t *__restrict__ Scales,
                           const int *ids, const float *X, int xStride, float *Y, int rows, int in,
                           const float *escale, int situW, double beta, double lb) {
  ink2_pdl_enter();
  extern __shared__ float4 xs4[];
  __shared__ float lut[16];
  const int slot = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int nb = in >> 4;
  const size_t prow = (size_t)(in >> 1);
  const size_t ebase = (size_t)ids[slot] * rows;
  int o0 = (blockIdx.x * INK2_NV2_WARPS + (threadIdx.x >> 5)) * R;
  const int ostride = gridDim.x * INK2_NV2_WARPS * R;

  uint2 pk[KB][R];
  uint32_t sb[KB][R];
  auto load = [&](int base) {
#pragma unroll
    for (int k = 0; k < KB; ++k) {
      const int b = lane + 32 * k;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int localRow = base + r;
        const bool ok = b < nb && localRow < rows;
        if (SLAB) {
          const size_t slabIdx = ((size_t)ids[slot] * (size_t)(rows >> 7) + (size_t)(localRow >> 7)) * (size_t)(nb >> 3) + (size_t)(b >> 3);
          pk[k][r] = ok ? __ldcs((const uint2 *)(Packed + slabIdx * (128u * 64u) + (size_t)(localRow & 127) * 64u + (size_t)(b & 7) * 8u)) : make_uint2(0u, 0u);
          sb[k][r] = ok ? (uint32_t)__ldcs(Scales + slabIdx * (128u * 8u) + (size_t)(localRow & 127) * 8u + (size_t)(b & 7)) : 0u;
        } else {
          const size_t row = ebase + (size_t)localRow;
          pk[k][r] = ok ? __ldcs((const uint2 *)(Packed + row * prow + (size_t)b * 8)) : make_uint2(0u, 0u);
          sb[k][r] = ok ? (uint32_t)__ldcs(Scales + row * (size_t)nb + b) : 0u;
        }
      }
    }
  };
  if (o0 < rows) load(o0);

  const float *x = X + (size_t)slot * xStride;
  float *xsf = (float *)xs4;
  for (int i = threadIdx.x; i < in; i += blockDim.x) {
    const int b = i >> 4;
    xsf[ink2_nv2_slot(b, (i >> 2) & 3) * 4 + (i & 3)] = nv4_bf16_rne(ink2_stage(x, i, situW, beta, lb));
  }
  if (threadIdx.x < 16) lut[threadIdx.x] = nv4_e2m1(threadIdx.x);
  __syncthreads();

  const float es = escale ? escale[ids[slot]] : 0.f;
  while (o0 < rows) {
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0.f;
#pragma unroll
    for (int k = 0; k < KB; ++k) {
      const int b = lane + 32 * k;
      if (b < nb) {
        float xb[16];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float4 v = xs4[ink2_nv2_slot(b, j)];
          xb[4 * j] = v.x;
          xb[4 * j + 1] = v.y;
          xb[4 * j + 2] = v.z;
          xb[4 * j + 3] = v.w;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
          float part = 0.f;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const uint32_t word = j < 4 ? pk[k][r].x : pk[k][r].y;
            const uint32_t byte = (word >> (8 * (j & 3))) & 0xffu;
            part = fmaf(lut[byte & 0x0fu], xb[2 * j], part);
            part = fmaf(lut[byte >> 4], xb[2 * j + 1], part);
          }
          sum[r] = fmaf(part, nv4_e4m3fn(sb[k][r]), sum[r]);
        }
      }
    }
    const int cur = o0;
    o0 += ostride;
    if (o0 < rows) load(o0);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float v = sum[r];
      for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
      const int o = cur + r;
      if (lane == 0 && o < rows) Y[(size_t)slot * rows + o] = escale ? __fmul_rn(v, es) : v;
    }
  }
}

// Blocks per routed slot so the whole k-slot grid is about one resident wave.
static int ink2_nv2_blocks_per_slot(int rows, int k) {
  static int sms = 0;
  if (sms == 0) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0) sms = 84;
  }
  const int groups = (rows + INK2_NV2_ROWS - 1) / INK2_NV2_ROWS;
  const int needed = (groups + INK2_NV2_WARPS - 1) / INK2_NV2_WARPS;
  const int perSM = k > INK2_WIDE_SLOTS ? INK2_NV2_BLOCKS_PER_SM_WIDE : INK2_NV2_BLOCKS_PER_SM;
  // a wide grid rounds down: a block past sms * perSM waits for a second wave
  int g = k > INK2_WIDE_SLOTS ? (sms * perSM) / k : (sms * perSM + k - 1) / k;
  if (g > needed) g = needed;
  return g < 1 ? 1 : g;
}

static int g_ink2_nv2_mode = -1;

static bool ink2_nv2_enabled(void) {
  if (g_ink2_nv2_mode < 0) {
    const char *v = getenv("FAK_INKLING2_NVFP4_GEMV");
    g_ink2_nv2_mode = (v && v[0] == '0' && v[1] == 0) ? 0 : 1;
  }
  return g_ink2_nv2_mode == 1;
}

// Test hook: 1 selects the v2 kernel, 0 the reference; returns the previous mode.
extern "C" int fcuda_ink2_nvfp4_gemv_select(int mode) {
  const int prev = ink2_nv2_enabled() ? 1 : 0;
  g_ink2_nv2_mode = mode ? 1 : 0;
  return prev;
}

// Launches the v2 kernel; returns false (nothing launched) when the shape needs the
// reference kernel.
static bool ink2_expert_nvfp4_v2(const uint8_t *W, const uint8_t *scales, const int *ids, const float *X,
                                 int xStride, float *Y, int k, int rows, int in, const float *escale, int situW,
                                 double beta, double lb) {
  const int kb = ((in >> 4) + 31) / 32;
  if (!ink2_nv2_enabled() || kb < 1 || kb > INK2_NV2_MAX_KB) return false;
  const dim3 grid(ink2_nv2_blocks_per_slot(rows, k), k);
  const size_t smem = (size_t)in * sizeof(float);
  const bool slab = (rows % 128 == 0 && in % 128 == 0 && ink2p_nvfp4_is_slab(W));
#define INK2_NV2_CASE(KB)                                                                                     \
  case KB:                                                                                                    \
    if (slab) {                                                                                               \
      FCUDA_TS("k_ink2_expert_nvfp4_v2_slab", grid,                                                           \
               (k_ink2_expert_nvfp4_v2<KB, INK2_NV2_ROWS, true><<<grid, INK2_NV2_WARPS * 32, smem, g_stream>>>( \
                   W, scales, ids, X, xStride, Y, rows, in, escale, situW, beta, lb)));                       \
    } else {                                                                                                  \
      FCUDA_TS("k_ink2_expert_nvfp4_v2", grid,                                                                \
               (k_ink2_expert_nvfp4_v2<KB, INK2_NV2_ROWS, false><<<grid, INK2_NV2_WARPS * 32, smem, g_stream>>>( \
                   W, scales, ids, X, xStride, Y, rows, in, escale, situW, beta, lb)));                       \
    }                                                                                                         \
    break;
  switch (kb) {
    INK2_NV2_CASE(1)
    INK2_NV2_CASE(2)
    INK2_NV2_CASE(3)
    INK2_NV2_CASE(4)
    INK2_NV2_CASE(5)
    INK2_NV2_CASE(6)
    INK2_NV2_CASE(7)
    INK2_NV2_CASE(8)
  }
#undef INK2_NV2_CASE
  return true;
}

// IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S: k_ink2_expert_iq's per-lane term (lane l covers
// values 8l..8l+7 of each 256-value super-block, sum = fmaf(db, dot8, sum) in block
// order, same shuffle tree), so outputs are bit-identical. The reference is bound by
// converting every grid byte from a 64-bit integer (I2F.U64); here each byte converts
// exactly through 32 bits, the grid and sign tables live in shared memory, R rows
// advance together per super-block and the grid is about 6 blocks per SM.
// FAK_INKLING2_IQ_GEMV=0 keeps the reference kernel.
#ifndef INK2_IQ3_ROWS
#define INK2_IQ3_ROWS 2
#endif
#ifndef INK2_IQ3_BLOCKS_PER_SM
#define INK2_IQ3_BLOCKS_PER_SM 6
#endif
#ifndef INK2_IQ3_BLOCKS_PER_SM_WIDE
#define INK2_IQ3_BLOCKS_PER_SM_WIDE 4
#endif

// Blocks per SM of the v3 IQ grids for k routed slots.
static int ink2_iq3_blocks_per_sm(int k) {
  return k > INK2_WIDE_SLOTS ? INK2_IQ3_BLOCKS_PER_SM_WIDE : INK2_IQ3_BLOCKS_PER_SM;
}

// Blocks per routed slot of the v3 IQ grids; a wide grid rounds down so no block waits
// for a second wave.
static int ink2_iq3_blocks_per_slot(int sms, int k) {
  const int total = sms * ink2_iq3_blocks_per_sm(k);
  return k > INK2_WIDE_SLOTS ? total / k : (total + k - 1) / k;
}

template <int FMT> struct Ink2IQShape {
  static constexpr int SB = FMT == INK2_IQ2XS ? 74 : FMT == INK2_IQ2S ? 82 : FMT == INK2_IQ3XXS ? 98 : 110;
  static constexpr int GRID_BYTES = FMT == INK2_IQ2XS ? 512 * 8 : FMT == INK2_IQ2S ? 1024 * 8 : FMT == INK2_IQ3XXS ? 256 * 4 : 512 * 4;
};

template <int FMT>
static __device__ __forceinline__ const void *ink2_iq_grid_src(void) {
  if (FMT == INK2_IQ2XS) return fcuda_iq2xs_grid;
  if (FMT == INK2_IQ2S) return fcuda_iq2s_grid;
  if (FMT == INK2_IQ3XXS) return fcuda_iq3xxs_grid;
  return fcuda_iq3s_grid;
}

// fcuda_iq_dot8 with the grid byte converted through 32 bits: (float) of a 64-bit
// integer compiles to I2F.U64, the binding cost of the reference IQ kernels. Each
// byte is 0..255, so the magic-number conversion is exact and every fmaf operand,
// hence the result, is unchanged.
static __device__ __forceinline__ float ink2_iq_dot8(unsigned int lo, unsigned int hi, unsigned char signs,
                                                     const float *xs) {
  const float4 a = *(const float4 *)xs;
  const float4 b = *(const float4 *)(xs + 4);
  const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  float part = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const unsigned int byte = ((j < 4 ? lo : hi) >> (8 * (j & 3))) & 0xffu;
    float v = __uint_as_float(0x4b000000u | byte) - 8388608.0f;
    v = (signs & (1u << j)) ? -v : v;
    part = fmaf(v, xv[j], part);
  }
  return part;
}

template <int FMT>
static __device__ __forceinline__ void ink2_iq_term_tab(const unsigned char *blk, int lane, const float *xs,
                                                        const void *gridTab, const unsigned char *ksigns, float &db,
                                                        float &dot) {
  const int ib32 = lane >> 2, l = lane & 3;
  const float d = __half2float(*(const __half *)blk);
  if (FMT == INK2_IQ2XS) {
    const unsigned short qv = ((const unsigned short *)(blk + 2))[4 * ib32 + l];
    const unsigned char sc = blk[66 + ib32];
    db = d * (0.5f + (float)(l < 2 ? (sc & 0x0f) : (sc >> 4))) * 0.25f;
    const uint2 g = ((const uint2 *)gridTab)[qv & 511];
    dot = ink2_iq_dot8(g.x, g.y, ksigns[qv >> 9], xs);
  } else if (FMT == INK2_IQ2S) {
    const unsigned int idx = blk[2 + 4 * ib32 + l] | (((unsigned int)blk[66 + ib32] << (8 - 2 * l)) & 0x300);
    const unsigned char sign = blk[34 + 4 * ib32 + l];
    const unsigned char sc = blk[74 + ib32];
    db = d * (0.5f + (float)(l < 2 ? (sc & 0x0f) : (sc >> 4))) * 0.25f;
    const uint2 g = ((const uint2 *)gridTab)[idx];
    dot = ink2_iq_dot8(g.x, g.y, sign, xs);
  } else if (FMT == INK2_IQ3XXS) {
    const unsigned int aux32 = fcuda_iq_u32_at(blk + 66 + 4 * ib32);
    db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const unsigned char signs = ksigns[(aux32 >> (7 * l)) & 127];
    const unsigned int *g = (const unsigned int *)gridTab;
    dot = ink2_iq_dot8(g[blk[2 + 8 * ib32 + 2 * l]], g[blk[2 + 8 * ib32 + 2 * l + 1]], signs, xs);
  } else {
    const unsigned int high = blk[66 + ib32];
    const unsigned int idx1 = blk[2 + 8 * ib32 + 2 * l] | ((high << (8 - 2 * l)) & 0x100);
    const unsigned int idx2 = blk[2 + 8 * ib32 + 2 * l + 1] | ((high << (7 - 2 * l)) & 0x100);
    const unsigned char sign = blk[74 + 4 * ib32 + l];
    const unsigned int nibble = (blk[106 + ib32 / 2] >> (4 * (ib32 & 1))) & 0x0f;
    db = d * (float)(1 + 2 * (int)nibble);
    const unsigned int *g = (const unsigned int *)gridTab;
    dot = ink2_iq_dot8(g[idx1], g[idx2], sign, xs);
  }
}

template <int FMT, int R>
__global__ void __launch_bounds__(256)
    k_ink2_expert_iq_v3(const unsigned char *__restrict__ W, const int *ids, const float *X, int xStride, float *Y,
                        int rows, int in, const float *escale, int situW, double beta, double lb) {
  ink2_pdl_enter();
  extern __shared__ __align__(16) unsigned char ink2_iq3_sm[];
  constexpr int SB = Ink2IQShape<FMT>::SB, GBYTES = Ink2IQShape<FMT>::GRID_BYTES;
  float *xsh = (float *)ink2_iq3_sm;
  unsigned int *gsh = (unsigned int *)(ink2_iq3_sm + (size_t)in * 4);
  unsigned char *ssh = (unsigned char *)gsh + GBYTES;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int slot = blockIdx.y;
  const float *xg = X + (size_t)slot * xStride;
  for (int i = threadIdx.x; i < in; i += blockDim.x) xsh[i] = ink2_stage(xg, i, situW, beta, lb);
  const unsigned int *gsrc = (const unsigned int *)ink2_iq_grid_src<FMT>();
  for (int i = threadIdx.x; i < GBYTES / 4; i += blockDim.x) gsh[i] = gsrc[i];
  if (threadIdx.x < 128) ssh[threadIdx.x] = fcuda_ksigns_iq2xs[threadIdx.x];
  __syncthreads();
  const int nsb = in / 256;
  const size_t rowBytes = (size_t)nsb * SB;
  const size_t ebase = (size_t)ids[slot] * rows;
  const float es = escale ? escale[ids[slot]] : 0.f;
  const float *xl = xsh + lane * 8;
  for (int o0 = (blockIdx.x * (blockDim.x >> 5) + warp) * R; o0 < rows; o0 += gridDim.x * (blockDim.x >> 5) * R) {
    const unsigned char *wrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wrow[r] = W + (ebase + (size_t)min(o0 + r, rows - 1)) * rowBytes;
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0.f;
    for (int sb = 0; sb < nsb; ++sb) {
      float db[R], dot[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        ink2_iq_term_tab<FMT>(wrow[r] + (size_t)sb * SB, lane, xl + (size_t)sb * 256, gsh, ssh, db[r], dot[r]);
      }
#pragma unroll
      for (int r = 0; r < R; ++r) sum[r] = fmaf(db[r], dot[r], sum[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float v = sum[r];
#pragma unroll
      for (int delta = 16; delta > 0; delta >>= 1) v += __shfl_down_sync(0xffffffff, v, delta);
      const int o = o0 + r;
      if (lane == 0 && o < rows) Y[(size_t)slot * rows + o] = escale ? __fmul_rn(v, es) : v;
    }
  }
}

static int g_ink2_iq3_mode = -1;

static bool ink2_iq3_enabled(void) {
  if (g_ink2_iq3_mode < 0) {
    const char *v = getenv("FAK_INKLING2_IQ_GEMV");
    g_ink2_iq3_mode = (v && v[0] == '0' && v[1] == 0) ? 0 : 1;
  }
  return g_ink2_iq3_mode == 1;
}

// Test hook: 1 selects the v3 IQ kernels, 0 the reference; returns the previous mode.
extern "C" int fcuda_ink2_iq_gemv_select(int mode) {
  const int prev = ink2_iq3_enabled() ? 1 : 0;
  g_ink2_iq3_mode = mode ? 1 : 0;
  return prev;
}

// Launches the v3 IQ kernel for fmt; returns false (nothing launched) for a format,
// shape or mode that keeps the reference kernel.
static bool ink2_expert_iq_v3(int fmt, const uint8_t *W, const int *ids, const float *X, int xStride, float *Y, int k,
                              int rows, int in, const float *escale, int situW, double beta, double lb) {
  if (!ink2_iq3_enabled() || (in & 255) != 0 || in <= 0) return false;
  static int sms = 0;
  if (sms == 0) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0) sms = 84;
  }
  const int needed = ((rows + INK2_IQ3_ROWS - 1) / INK2_IQ3_ROWS + 7) / 8;
  int g = ink2_iq3_blocks_per_slot(sms, k);
  if (g > needed) g = needed;
  const dim3 grid(g < 1 ? 1 : g, k);
#define INK2_IQ3_CASE(F)                                                                                          \
  case F: {                                                                                                       \
    const size_t smem = (size_t)in * 4 + Ink2IQShape<F>::GRID_BYTES + 128;                                        \
    if (smem > 48 * 1024) return false;                                                                           \
    FCUDA_TS("k_ink2_expert_iq_v3", grid,                                                                         \
             k_ink2_expert_iq_v3<F, INK2_IQ3_ROWS><<<grid, 256, smem, g_stream>>>(                                \
                 W, ids, X, xStride, Y, rows, in, escale, situW, beta, lb));                                      \
    return true;                                                                                                  \
  }
  switch (fmt) {
    INK2_IQ3_CASE(INK2_IQ2XS)
    INK2_IQ3_CASE(INK2_IQ2S)
    INK2_IQ3_CASE(INK2_IQ3XXS)
    INK2_IQ3_CASE(INK2_IQ3S)
  }
#undef INK2_IQ3_CASE
  return false;
}

// IQ4_NL: k_ink2_expert_iq's per-lane term and block order (bit-identical), with the
// codebook staged once as floats instead of an int->float conversion per weight, the
// 4 code bytes read as two 16-bit words, R rows advancing together per super-block
// and the IQ grid sizing above. Shares FAK_INKLING2_IQ_GEMV with the IQ2/IQ3 kernels.
template <int R>
__global__ void __launch_bounds__(256)
    k_ink2_expert_iq4nl_v3(const unsigned char *__restrict__ W, const int *ids, const float *X, int xStride,
                           float *Y, int rows, int in, const float *escale, int situW, double beta, double lb) {
  ink2_pdl_enter();
  extern __shared__ __align__(16) float ink2_kq_sm[];
  __shared__ float kv[16];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int slot = blockIdx.y;
  const float *xg = X + (size_t)slot * xStride;
  for (int i = threadIdx.x; i < in; i += blockDim.x) ink2_kq_sm[i] = ink2_stage(xg, i, situW, beta, lb);
  if (threadIdx.x < 16) kv[threadIdx.x] = (float)fcuda_kvalues_iq4nl[threadIdx.x];
  __syncthreads();
  const int nsb = in / 256;
  const size_t rowBytes = (size_t)nsb * 144;
  const size_t ebase = (size_t)ids[slot] * rows;
  const float es = escale ? escale[ids[slot]] : 0.f;
  const int ib = lane >> 2, il = lane & 3;
  for (int o0 = (blockIdx.x * (blockDim.x >> 5) + warp) * R; o0 < rows; o0 += gridDim.x * (blockDim.x >> 5) * R) {
    const unsigned char *wrow[R];
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      wrow[r] = W + (ebase + (size_t)min(o0 + r, rows - 1)) * rowBytes;
      sum[r] = 0.f;
    }
    for (int sb = 0; sb < nsb; ++sb) {
      const float *xs = ink2_kq_sm + (size_t)sb * 256 + 32 * ib + 4 * il;
      const float4 lo = *(const float4 *)xs;
      const float4 hi = *(const float4 *)(xs + 16);
      const float xl[4] = {lo.x, lo.y, lo.z, lo.w};
      const float xh[4] = {hi.x, hi.y, hi.z, hi.w};
      float d[R], part[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const unsigned char *b = wrow[r] + (size_t)sb * 144 + 18 * ib;
        const unsigned short *q2 = (const unsigned short *)(b + 2 + 4 * il);
        const unsigned int q = (unsigned int)q2[0] | ((unsigned int)q2[1] << 16);
        d[r] = __half2float(*(const __half *)b);
        part[r] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const unsigned int byte = (q >> (8 * j)) & 0xffu;
          part[r] = fmaf(kv[byte & 0xf], xl[j], part[r]);
          part[r] = fmaf(kv[byte >> 4], xh[j], part[r]);
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) sum[r] = fmaf(d[r], part[r], sum[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float v = sum[r];
#pragma unroll
      for (int delta = 16; delta > 0; delta >>= 1) v += __shfl_down_sync(0xffffffff, v, delta);
      const int o = o0 + r;
      if (lane == 0 && o < rows) Y[(size_t)slot * rows + o] = escale ? __fmul_rn(v, es) : v;
    }
  }
}

// Launches the IQ4_NL v3 kernel; returns false (nothing launched) for a shape or mode
// that keeps the reference kernel.
static bool ink2_expert_iq4nl_v3(const uint8_t *W, const int *ids, const float *X, int xStride, float *Y, int k,
                                 int rows, int in, const float *escale, int situW, double beta, double lb) {
  if (!ink2_iq3_enabled() || (in & 255) != 0 || in <= 0 || (size_t)in * 4 > 48 * 1024) return false;
  int sms = 0, dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0) sms = 84;
  const int needed = ((rows + INK2_IQ3_ROWS - 1) / INK2_IQ3_ROWS + 7) / 8;
  int g = ink2_iq3_blocks_per_slot(sms, k);
  if (g > needed) g = needed;
  const dim3 grid(g < 1 ? 1 : g, k);
  FCUDA_TS("k_ink2_expert_iq4nl_v3", grid, k_ink2_expert_iq4nl_v3<INK2_IQ3_ROWS><<<grid, 256, (size_t)in * 4, g_stream>>>(
                                               W, ids, X, xStride, Y, rows, in, escale, situW, beta, lb));
  return true;
}
