// inkling2_prefill_iq.cuh: grouped IQ / Q3_K expert GEMM for the batched prefill on
// int8 tensor cores, the MMQ scheme of llama.cpp (MIT, Copyright (c) 2023-2026 The ggml
// authors; THIRD_PARTY_NOTICES.md "llama.cpp") applied to the per-expert pair tiles of
// fcuda_ink2p_bucket.
//
// Each routed pair's activation row is quantized once to int8 with one f32 scale per 32
// values (q8_1 without the sum). A block owns 128 weight rows of one expert and up to NB
// of its pairs. Per 256-wide super-block the block copies the raw weight rows into shared
// memory with coalesced 16-byte loads (prefetched one super-block ahead in registers) and
// every warp decodes the A fragments of its own 16 rows straight into registers: the K
// order inside each 32-wide block is permuted so a lane's 8 decoded weights are exactly
// its fragment bytes (grid formats and Q3_K: MMA lane t of k-block kb holds super-block
// lane 4*kb+t, weights 8t..8t+7, against activations x[8t..8t+7]; IQ4_NL's nibble halves
// already sit at 4t and 16+4t). Each m16n8k32 s8 MMA partial is folded into f32 with
// weight-scale x activation-scale; scales that change every 16 values (IQ2_XS, IQ2_S,
// Q3_K) run the MMA twice with the other half's activations zeroed and combine the two
// partials with their small integer sub-scales first. Grid tables live in shared memory.
//
// Included after inkling2_prefill.cuh: uses INK2_* formats, INK2P_NV_PAIRS, the IQ grids
// of iq_gemv.cuh / iq23s_gemv.cuh and ink2_last_error.

#define INK2Q_THREADS 256

template <int FMT> struct Ink2qFmt {
  static constexpr int SB = FMT == INK2_IQ2XS ? 74 : FMT == INK2_IQ2S ? 82 : FMT == INK2_IQ3XXS ? 98 : FMT == INK2_IQ4NL ? 144 : 110;
  static constexpr bool PER16 = FMT == INK2_IQ2XS || FMT == INK2_IQ2S || FMT == INK2_Q3K;
};

// sign bits b0..b3 -> bytes 0x00 / 0xff
static __device__ __forceinline__ uint32_t ink2q_signmask4(uint32_t s4) {
  return ((s4 * 0x00204081u) & 0x01010101u) * 0xffu;
}

// bytes 0..7 -> signed bytes -4..3 (biased by 0x80 so no byte borrows)
static __device__ __forceinline__ uint32_t ink2q_sub4(uint32_t x) {
  return ((x | 0x80808080u) - 0x04040404u) ^ 0x80808080u;
}

static __device__ __forceinline__ void ink2q_signed8(unsigned long long grid, uint32_t signs, uint32_t &lo, uint32_t &hi) {
  const uint32_t m0 = ink2q_signmask4(signs & 15u), m1 = ink2q_signmask4((signs >> 4) & 15u);
  // grid bytes are 1..62, so negating a byte as (g ^ 0xff) + 1 never carries
  lo = ((uint32_t)grid ^ m0) + (m0 & 0x01010101u);
  hi = ((uint32_t)(grid >> 32) ^ m1) + (m1 & 0x01010101u);
}

// 4 IQ4_NL nibble indices (one per byte, 0..15) -> the 4 kvalues bytes
static __device__ __forceinline__ uint32_t ink2q_iq4nl4(uint32_t idx) {
  const uint32_t t0 = 0xbfad9881u, t1 = 0xf6eaddcfu, t2 = 0x26190d01u, t3 = 0x71594535u;
  const uint32_t sel = (idx & 0x07u) | ((idx >> 4) & 0x70u) | ((idx >> 8) & 0x700u) | ((idx >> 12) & 0x7000u);
  const uint32_t lo = __byte_perm(t0, t1, sel), hi = __byte_perm(t2, t3, sel);
  const uint32_t pick = ((idx & 0x08080808u) >> 3) * 4u; // 4 where the index is >= 8
  const uint32_t psel = 0x3210u + ((pick & 0xffu) | ((pick >> 4) & 0xf0u) | ((pick >> 8) & 0xf00u) | ((pick >> 12) & 0xf000u));
  return __byte_perm(lo, hi, psel);
}

static __device__ __forceinline__ float ink2q_h2f(uint32_t bits) {
  return __half2float(__ushort_as_half((unsigned short)bits));
}

// Quantize the activation row of every bucketed pair: position pos (index into order)
// gets Xq[pos*in ..] int8 and Xd[pos*(in/32) ..] f32 scales.
__global__ void __launch_bounds__(256) k_ink2q_quant(const int *offsets, const int *order, const int *tiles,
                                                     const int *ntiles, const float *X, int xDiv, int xStride,
                                                     int8_t *Xq, float *Xd, int in) {
  if ((int)blockIdx.x >= ntiles[0]) return;
  const int e = tiles[2 * blockIdx.x], start = tiles[2 * blockIdx.x + 1];
  const int np = min(INK2P_NV_PAIRS, offsets[e + 1] - start);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int p = warp; p < np; p += INK2Q_THREADS / 32) {
    const size_t pos = (size_t)start + p;
    const float *src = X + (size_t)(order[pos] / xDiv) * xStride;
    for (int k = lane * 4; k < in; k += 128) {
      const float v0 = src[k], v1 = src[k + 1], v2 = src[k + 2], v3 = src[k + 3];
      float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4));
      const float d = amax / 127.f, inv = amax > 0.f ? 127.f / amax : 0.f;
      char4 q;
      q.x = (signed char)__float2int_rn(v0 * inv);
      q.y = (signed char)__float2int_rn(v1 * inv);
      q.z = (signed char)__float2int_rn(v2 * inv);
      q.w = (signed char)__float2int_rn(v3 * inv);
      *(char4 *)(Xq + pos * in + k) = q;
      if ((lane & 7) == 0) Xd[pos * (in / 32) + k / 32] = d;
    }
  }
}

static __device__ __forceinline__ void ink2q_mma32(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                   uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
               : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}

template <int FMT> struct Ink2qTab {
  static constexpr int BYTES = FMT == INK2_IQ2XS ? 512 * 8 : FMT == INK2_IQ2S ? 1024 * 8
                               : FMT == INK2_IQ3XXS ? 256 * 4 : FMT == INK2_IQ3S ? 512 * 4 : 16;
};

template <int FMT> static __device__ __forceinline__ void ink2q_tab_fill(uint8_t *tab) {
  const int tid = threadIdx.x;
  if (FMT == INK2_IQ2XS) {
    for (int i = tid; i < 512; i += blockDim.x) ((unsigned long long *)tab)[i] = fcuda_iq2xs_grid[i];
  } else if (FMT == INK2_IQ2S) {
    for (int i = tid; i < 1024; i += blockDim.x) ((unsigned long long *)tab)[i] = fcuda_iq2s_grid[i];
  } else if (FMT == INK2_IQ3XXS) {
    for (int i = tid; i < 256; i += blockDim.x) ((uint32_t *)tab)[i] = fcuda_iq3xxs_grid[i];
  } else if (FMT == INK2_IQ3S) {
    for (int i = tid; i < 512; i += blockDim.x) ((uint32_t *)tab)[i] = fcuda_iq3s_grid[i];
  }
}

// ksigns_iq2xs: 7 sign bits plus an even-parity eighth
static __device__ __forceinline__ uint32_t ink2q_ksigns(uint32_t s7) { return s7 | ((__popc(s7) & 1u) << 7); }

static __device__ __forceinline__ uint32_t ink2q_s16(const uint8_t *p) { return *(const unsigned short *)p; }

// one super-block lane decoded from shared memory: lo / hi are its two 4-weight fragment
// words, sc the f32 scale (PER16: super-block factor), isc the integer 16-wide sub-scale
template <int FMT>
static __device__ __forceinline__ void ink2q_dec_s(const uint8_t *blk, int lane, const uint8_t *tab, float d,
                                                   uint32_t &lo, uint32_t &hi, float &sc, int &isc) {
  isc = 1;
  if (FMT == INK2_Q3K) {
    const int e0 = lane * 8, half = e0 >> 7, j = (e0 & 127) >> 5, sub = (e0 & 31) >> 4, l0 = e0 & 15;
    const int is = half * 8 + j * 2 + sub, li = sub * 16 + l0;
    const uint8_t *q = blk + 32 + half * 32 + li, *h = blk + li;
    const uint32_t q0 = ink2q_s16(q) | (ink2q_s16(q + 2) << 16), q1 = ink2q_s16(q + 4) | (ink2q_s16(q + 6) << 16);
    const uint32_t h0 = ink2q_s16(h) | (ink2q_s16(h + 2) << 16), h1 = ink2q_s16(h + 4) | (ink2q_s16(h + 6) << 16);
    const uint32_t s0 = blk[96 + (is & 7)], s1 = blk[96 + 8 + (is & 3)];
    const int scv = (int)((is < 8 ? (s0 & 0x0f) : (s0 >> 4)) | (((s1 >> (2 * (is >> 2))) & 3) << 4));
    const int hs = half * 4 + j;
    lo = ink2q_sub4(((q0 >> (2 * j)) & 0x03030303u) | (((h0 >> hs) & 0x01010101u) << 2));
    hi = ink2q_sub4(((q1 >> (2 * j)) & 0x03030303u) | (((h1 >> hs) & 0x01010101u) << 2));
    sc = d;
    isc = scv - 32;
    return;
  }
  if (FMT == INK2_IQ4NL) {
    const uint8_t *b = blk + 18 * (lane >> 2);
    const uint32_t q4 = ink2q_s16(b + 2 + 4 * (lane & 3)) | (ink2q_s16(b + 4 + 4 * (lane & 3)) << 16);
    lo = ink2q_iq4nl4(q4 & 0x0f0f0f0fu);
    hi = ink2q_iq4nl4((q4 >> 4) & 0x0f0f0f0fu);
    sc = ink2q_h2f(ink2q_s16(b));
    return;
  }
  const int ib32 = lane >> 2, l = lane & 3;
  unsigned long long grid;
  uint32_t signs;
  if (FMT == INK2_IQ2XS) {
    const uint32_t qv = ink2q_s16(blk + 2 + 2 * (4 * ib32 + l));
    const uint32_t s = blk[66 + ib32];
    isc = 2 * (int)(l < 2 ? (s & 0x0f) : (s >> 4)) + 1;
    sc = d * 0.125f;
    grid = ((const unsigned long long *)tab)[qv & 511];
    signs = ink2q_ksigns(qv >> 9);
  } else if (FMT == INK2_IQ2S) {
    const uint32_t idx = blk[2 + 4 * ib32 + l] | (((uint32_t)blk[66 + ib32] << (8 - 2 * l)) & 0x300);
    const uint32_t s = blk[74 + ib32];
    isc = 2 * (int)(l < 2 ? (s & 0x0f) : (s >> 4)) + 1;
    sc = d * 0.125f;
    grid = ((const unsigned long long *)tab)[idx];
    signs = blk[34 + 4 * ib32 + l];
  } else if (FMT == INK2_IQ3XXS) {
    const uint32_t aux32 = ink2q_s16(blk + 66 + 4 * ib32) | (ink2q_s16(blk + 68 + 4 * ib32) << 16);
    const uint32_t q2 = ink2q_s16(blk + 2 + 8 * ib32 + 2 * l);
    sc = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    signs = ink2q_ksigns((aux32 >> (7 * l)) & 127);
    grid = (unsigned long long)((const uint32_t *)tab)[q2 & 0xff] |
           ((unsigned long long)((const uint32_t *)tab)[q2 >> 8] << 32);
  } else {
    const uint32_t high = blk[66 + ib32];
    const uint32_t q2 = ink2q_s16(blk + 2 + 8 * ib32 + 2 * l);
    const uint32_t idx1 = (q2 & 0xff) | ((high << (8 - 2 * l)) & 0x100);
    const uint32_t idx2 = (q2 >> 8) | ((high << (7 - 2 * l)) & 0x100);
    signs = blk[74 + 4 * ib32 + l];
    sc = d * (float)(1 + 2 * (int)((blk[106 + ib32 / 2] >> (4 * (ib32 & 1))) & 0x0f));
    grid = (unsigned long long)((const uint32_t *)tab)[idx1] | ((unsigned long long)((const uint32_t *)tab)[idx2] << 32);
  }
  ink2q_signed8(grid, signs, lo, hi);
}

// int32 MMA partial -> f32. A 32-wide block partial stays below 2^22 in magnitude, so
// the float-bias trick (two full-rate ops) replaces the quarter-rate I2F; the 16-wide
// scale combination of the PER16 formats can exceed that and keeps the conversion.
template <bool WIDE> static __device__ __forceinline__ float ink2q_i2f(int c) {
  if (WIDE) return (float)c;
  return __int_as_float(c + 0x4b400000) - 12582912.f;
}

template <int FMT, int NB, int MINB>
__global__ void __launch_bounds__(INK2Q_THREADS, MINB)
    k_ink2p_expert_iq_v2(const uint8_t *W, const int *offsets, const int *tiles, const int *ntiles, const int *order,
                         const int8_t *Xq, const float *Xd, float *Y, int rows, int in, const float *escale) {
  constexpr int SB = Ink2qFmt<FMT>::SB, BM = 128, NCH = (SB + 14 + 15) / 16, RS = 16 * NCH + 16;
  constexpr int RW = (BM * NCH + INK2Q_THREADS - 1) / INK2Q_THREADS, NT8 = NB / 8, XS = 256 + 32;
  constexpr int XCH = NB * 16 / INK2Q_THREADS, DCH = NB * 8 / INK2Q_THREADS;
  constexpr bool PER16 = Ink2qFmt<FMT>::PER16;
  __shared__ __align__(16) uint8_t rawsm[BM * RS];
  __shared__ __align__(16) int8_t xq[NB * XS];
  __shared__ __align__(16) float xd[8][NB];
  __shared__ __align__(16) uint8_t tab[Ink2qTab<FMT>::BYTES];
  __shared__ int pairs[NB];
  if ((int)blockIdx.y >= ntiles[0]) return;
  const int e = tiles[2 * blockIdx.y], start = tiles[2 * blockIdx.y + 1];
  if (NB > INK2P_NV_PAIRS && (((start - offsets[e]) / INK2P_NV_PAIRS) & 1)) return;
  const int np = min(NB, offsets[e + 1] - start);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  if (tid < NB) pairs[tid] = tid < np ? order[start + tid] : 0;
  ink2q_tab_fill<FMT>(tab);
  const int row0 = blockIdx.x * BM;
  const int nsb = in / 256, nkd = in / 32;
  const int nt = (np + 7) >> 3;
  const size_t rowBytes = (size_t)nsb * SB;
  const uint8_t *wexp = W + (size_t)e * rows * rowBytes;
  float acc[NT8][4];
#pragma unroll
  for (int n = 0; n < NT8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
  uint4 rv[RW];
  int4 xv[XCH];
  float dv[DCH];
  auto fetch = [&](int sb) {
#pragma unroll
    for (int i = 0; i < RW; ++i) {
      // super-blocks are only 2-byte aligned: copy the aligned 16-byte chunks that cover
      // the row's super-block, keeping its offset inside the first chunk
      const int f = tid + INK2Q_THREADS * i, rr = f / NCH, c = f % NCH;
      rv[i] = make_uint4(0, 0, 0, 0);
      if (f < BM * NCH) {
        const uint8_t *p = wexp + (size_t)min(row0 + rr, rows - 1) * rowBytes + (size_t)sb * SB;
        const uint8_t *a = (const uint8_t *)((uintptr_t)p & ~(uintptr_t)15) + 16 * c;
        if (a < p + SB) rv[i] = __ldg((const uint4 *)a);
      }
    }
#pragma unroll
    for (int c = 0; c < XCH; ++c) {
      const int idx = tid + INK2Q_THREADS * c, p = idx >> 4, ch = idx & 15;
      xv[c] = *(const int4 *)(Xq + (size_t)(start + min(p, np - 1)) * in + sb * 256 + ch * 16);
    }
#pragma unroll
    for (int c = 0; c < DCH; ++c) {
      const int idx = tid + INK2Q_THREADS * c, p = idx >> 3, kb = idx & 7;
      dv[c] = p < np ? __ldg(Xd + (size_t)(start + p) * nkd + sb * 8 + kb) : 0.f;
    }
  };
  fetch(0);
  const int r0 = warp * 16 + g, r1 = r0 + 8;
  for (int sb = 0; sb < nsb; ++sb) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < RW; ++i) {
      const int f = tid + INK2Q_THREADS * i;
      if (f < BM * NCH) *(uint4 *)(rawsm + (f / NCH) * RS + 16 * (f % NCH)) = rv[i];
    }
#pragma unroll
    for (int c = 0; c < XCH; ++c) {
      const int idx = tid + INK2Q_THREADS * c, p = idx >> 4, ch = idx & 15;
      *(int4 *)(xq + p * XS + ch * 16) = xv[c];
    }
#pragma unroll
    for (int c = 0; c < DCH; ++c) {
      const int idx = tid + INK2Q_THREADS * c, p = idx >> 3, kb = idx & 7;
      xd[kb][p] = dv[c];
    }
    __syncthreads();
    if (sb + 1 < nsb) fetch(sb + 1);
    const uint8_t *b0p = rawsm + r0 * RS + (int)((uintptr_t)(wexp + (size_t)min(row0 + r0, rows - 1) * rowBytes + (size_t)sb * SB) & 15);
    const uint8_t *b1p = rawsm + r1 * RS + (int)((uintptr_t)(wexp + (size_t)min(row0 + r1, rows - 1) * rowBytes + (size_t)sb * SB) & 15);
    const float d0 = FMT == INK2_IQ4NL ? 0.f : ink2q_h2f(ink2q_s16(b0p + (FMT == INK2_Q3K ? 108 : 0)));
    const float d1 = FMT == INK2_IQ4NL ? 0.f : ink2q_h2f(ink2q_s16(b1p + (FMT == INK2_Q3K ? 108 : 0)));
#pragma unroll 2
    for (int kb = 0; kb < 8; ++kb) {
      uint32_t a0, a1, a2, a3;
      float s0, s1;
      int is0, is1;
      ink2q_dec_s<FMT>(b0p, 4 * kb + t, tab, d0, a0, a2, s0, is0);
      ink2q_dec_s<FMT>(b1p, 4 * kb + t, tab, d1, a1, a3, s1, is1);
      int il0 = 0, ih0 = 0, il1 = 0, ih1 = 0;
      if (PER16) {
        const int pk = (is0 & 0xff) | ((is1 & 0xff) << 8);
        const int plo = __shfl_sync(0xffffffffu, pk, lane & ~3), phi = __shfl_sync(0xffffffffu, pk, (lane & ~3) | 2);
        il0 = (signed char)(plo & 0xff), il1 = (signed char)((plo >> 8) & 0xff);
        ih0 = (signed char)(phi & 0xff), ih1 = (signed char)((phi >> 8) & 0xff);
      }
#pragma unroll
      for (int n = 0; n < NT8; ++n) {
        if (n >= nt) break;
        const int8_t *xr = xq + (n * 8 + g) * XS + kb * 32;
        uint32_t b0, b1;
        if (FMT == INK2_IQ4NL) {
          b0 = *(const uint32_t *)(xr + 4 * t);
          b1 = *(const uint32_t *)(xr + 16 + 4 * t);
        } else {
          const uint2 bb = *(const uint2 *)(xr + 8 * t);
          b0 = bb.x;
          b1 = bb.y;
        }
        const float2 x = *(const float2 *)&xd[kb][n * 8 + 2 * t];
        int c[4];
        if (PER16) {
          int cl[4], ch[4];
          const bool lowHalf = t < 2;
          ink2q_mma32(cl, a0, a1, a2, a3, lowHalf ? b0 : 0u, lowHalf ? b1 : 0u);
          ink2q_mma32(ch, a0, a1, a2, a3, lowHalf ? 0u : b0, lowHalf ? 0u : b1);
          c[0] = cl[0] * il0 + ch[0] * ih0;
          c[1] = cl[1] * il0 + ch[1] * ih0;
          c[2] = cl[2] * il1 + ch[2] * ih1;
          c[3] = cl[3] * il1 + ch[3] * ih1;
        } else {
          ink2q_mma32(c, a0, a1, a2, a3, b0, b1);
        }
        acc[n][0] = fmaf(ink2q_i2f<PER16>(c[0]) * s0, x.x, acc[n][0]);
        acc[n][1] = fmaf(ink2q_i2f<PER16>(c[1]) * s0, x.y, acc[n][1]);
        acc[n][2] = fmaf(ink2q_i2f<PER16>(c[2]) * s1, x.x, acc[n][2]);
        acc[n][3] = fmaf(ink2q_i2f<PER16>(c[3]) * s1, x.y, acc[n][3]);
      }
    }
  }
  const float es = escale ? escale[e] : 1.f;
#pragma unroll
  for (int n = 0; n < NT8; ++n) {
    if (n >= nt) break;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int row = row0 + warp * 16 + g + (c >= 2 ? 8 : 0), col = n * 8 + 2 * t + (c & 1);
      if (col < np && row < rows) {
        const float v = acc[n][c];
        Y[(size_t)pairs[col] * rows + row] = escale ? __fmul_rn(v, es) : v;
      }
    }
  }
}

// Bytes of caller scratch fcuda_ink2p_expert_iq_v2_ws needs for maxTiles tiles of K = in.
extern "C" size_t fcuda_ink2p_expert_iq_v2_scratch_bytes(int maxTiles, int in) {
  const size_t prow = (size_t)maxTiles * INK2P_NV_PAIRS;
  return prow * (size_t)in + prow * (size_t)(in / 32) * sizeof(float) + 16;
}

// fcuda_ink2p_expert_iq with caller-owned scratch of fcuda_ink2p_expert_iq_v2_scratch_bytes.
// The 16-wide-scale formats already issue two MMAs per fragment and keep 32-pair blocks;
// the others take two tiles of an expert per block so its weights are decoded once.
extern "C" int fcuda_ink2p_expert_iq_v2_ws(int fmt, const uint8_t *W, const int *offsets, const int *order,
                                           const int *tiles, const int *ntiles, int maxTiles, const float *X, int xDiv,
                                           int xStride, float *Y, int rows, int in, const float *escale,
                                           void *scratch) {
  if ((in & 255) != 0 || maxTiles < 1 || !scratch) return (int)cudaErrorInvalidValue;
  int8_t *Xq = (int8_t *)scratch;
  float *Xd = (float *)(((uintptr_t)(Xq + (size_t)maxTiles * INK2P_NV_PAIRS * in) + 15) & ~(uintptr_t)15);
  FCUDA_TS("k_ink2q_quant", maxTiles, k_ink2q_quant<<<maxTiles, INK2Q_THREADS, 0, g_stream>>>(
      offsets, order, tiles, ntiles, X, xDiv, xStride, Xq, Xd, in));
  const dim3 grid((rows + 127) / 128, maxTiles);
#define INK2Q_LAUNCH(F, NB) FCUDA_TS("k_ink2p_expert_iq_v2", grid, k_ink2p_expert_iq_v2<F, NB, 2><<<grid, INK2Q_THREADS, 0, g_stream>>>(W, offsets, tiles, ntiles, order, Xq, Xd, Y, rows, in, escale))
  switch (fmt) {
  case INK2_IQ2XS: INK2Q_LAUNCH(INK2_IQ2XS, 32); break;
  case INK2_IQ2S: INK2Q_LAUNCH(INK2_IQ2S, 32); break;
  case INK2_Q3K: INK2Q_LAUNCH(INK2_Q3K, 32); break;
  case INK2_IQ3XXS: INK2Q_LAUNCH(INK2_IQ3XXS, 64); break;
  case INK2_IQ3S: INK2Q_LAUNCH(INK2_IQ3S, 64); break;
  case INK2_IQ4NL: INK2Q_LAUNCH(INK2_IQ4NL, 64); break;
  default: return (int)cudaErrorInvalidValue;
  }
#undef INK2Q_LAUNCH
  return ink2_last_error();
}

// Drop-in for fcuda_ink2p_expert_iq (same signature and bucketing contract): the
// quantized activations live in a lazily grown device buffer, so the first call at a
// given size runs cudaMalloc and must not be inside a stream capture.
extern "C" int fcuda_ink2p_expert_iq_v2(int fmt, const uint8_t *W, const int *offsets, const int *order,
                                        const int *tiles, const int *ntiles, int maxTiles, const float *X, int xDiv,
                                        int xStride, float *Y, int rows, int in, const float *escale) {
  static void *buf = nullptr;
  static size_t cap = 0;
  const size_t need = fcuda_ink2p_expert_iq_v2_scratch_bytes(maxTiles, in);
  if (need > cap) {
    if (buf) cudaFree(buf);
    buf = nullptr;
    cap = 0;
    const cudaError_t err = fcuda_alloc_status(cudaMalloc(&buf, need));
    if (err != cudaSuccess) return (int)err;
    cap = need;
  }
  return fcuda_ink2p_expert_iq_v2_ws(fmt, W, offsets, order, tiles, ntiles, maxTiles, X, xDiv, xStride, Y, rows, in,
                                     escale, buf);
}
