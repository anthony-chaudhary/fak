// inkling2_prefill_attn.cuh: tiled tensor-core attention for the inkling2 prefill chunk.
//
// Same contract as k_ink2p_attend: the query at pos = pos0 + t sees ring keys (pos-n, pos],
// n = min(pos+1, window if > 0, cap); score = q.k*scale - sink + (dist < extent ? r.relBias[dist] : 0);
// softmax carries the implicit zero logit (running max starts at 0, denominator adds exp(-m)).
// One block per (kv head, query tile): the G query heads of the GQA group share each K/V tile
// staged in shared memory as fp16, QK^T and PV run on m16n8k16 fp16 MMA with fp32 accumulate,
// softmax is online in fp32. Included after inkling2_prefill.cuh (falls back to k_ink2p_attend
// for shapes it does not cover).

namespace ink2pa {

constexpr int kHD = 128, kRL = 16, kG = 3, kKS = kHD + 8; // padded smem row: conflict-free ldmatrix

__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const __half *p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

__device__ __forceinline__ void ldsm4t(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const __half *p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

__device__ __forceinline__ uint32_t h2(float lo, float hi) {
  __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t *>(&v);
}

__device__ __forceinline__ int nkeys(int pos, int window, int cap) {
  int n = pos + 1;
  if (window > 0 && n > window) n = window;
  if (n > cap) n = cap;
  return n;
}

__device__ __forceinline__ float reldot(const float *r, float4 a, float4 b, float4 c, float4 d) {
  float s = 0.f;
  s = fmaf(r[0], a.x, s); s = fmaf(r[1], a.y, s); s = fmaf(r[2], a.z, s); s = fmaf(r[3], a.w, s);
  s = fmaf(r[4], b.x, s); s = fmaf(r[5], b.y, s); s = fmaf(r[6], b.z, s); s = fmaf(r[7], b.w, s);
  s = fmaf(r[8], c.x, s); s = fmaf(r[9], c.y, s); s = fmaf(r[10], c.z, s); s = fmaf(r[11], c.w, s);
  s = fmaf(r[12], d.x, s); s = fmaf(r[13], d.y, s); s = fmaf(r[14], d.z, s); s = fmaf(r[15], d.w, s);
  return s;
}

// QW 16-query slabs per head, KT keys per tile; warps = kG * QW, warp w serves head kvh*kG + w/QW.
// REL 1 computes the relative term per tile as a 16 x (KT+16) MMA over the tile's distance band
// and gathers it from shared memory; REL 0 does the 16-dim dots in fp32 per score.
// PIPE 1 streams the next raw K/V tile with cp.async while the current one computes.
// E is the cache element: float tiles narrow to fp16 on the smem copy, __half tiles copy
// through unchanged, so an f16 cache stored with round-to-nearest gives the same tiles.
template <int QW, int KT, int REL, int PIPE, class E = float>
struct AttnSmem {
  static constexpr int kRS = KT + 16; // distance band row: NT + 2 n-tiles of 8
  static constexpr size_t rsFloats = REL ? (size_t)kG * QW * 16 * kRS : 0;
  static constexpr size_t bytes = (size_t)2 * KT * kKS * sizeof(__half) + rsFloats * 4 +
                                  (PIPE ? (size_t)2 * KT * kHD * sizeof(E) : 0);
};

__device__ __forceinline__ void cp16(void *dst, const void *src) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(src));
}

// Stores one 16-byte chunk of a K or V row (4 floats, 8 halves, or 16 E4M3FN bytes) into an fp16 smem tile.
__device__ __forceinline__ void tile16(__half *dst, float4 x) {
  *reinterpret_cast<uint2 *>(dst) = make_uint2(h2(x.x, x.y), h2(x.z, x.w));
}
__device__ __forceinline__ void tile16(__half *dst, uint4 x) { *reinterpret_cast<uint4 *>(dst) = x; }

__device__ __forceinline__ uint2 fp8x4_to_h2x2(uint32_t w, float sc) {
  const uint32_t lo = ink2_e4m3fn_decode2_scaled_h2((uint16_t)(w & 0xffffu), sc);
  const uint32_t hi = ink2_e4m3fn_decode2_scaled_h2((uint16_t)(w >> 16), sc);
  return make_uint2(lo, hi);
}
__device__ __forceinline__ uint2 int8x4_to_h2x2(uint32_t w, float sc) {
  const __half2 lo = __floats2half2_rn((float)(int8_t)(w & 0xffu) * sc, (float)(int8_t)((w >> 8) & 0xffu) * sc);
  const __half2 hi = __floats2half2_rn((float)(int8_t)((w >> 16) & 0xffu) * sc, (float)(int8_t)(w >> 24) * sc);
  return make_uint2(*reinterpret_cast<const uint32_t *>(&lo), *reinterpret_cast<const uint32_t *>(&hi));
}
__device__ __forceinline__ void tile16_int8(__half *dst, uint4 x, float sc) {
  uint2 *d = reinterpret_cast<uint2 *>(dst);
  d[0] = int8x4_to_h2x2(x.x, sc);
  d[1] = int8x4_to_h2x2(x.y, sc);
  d[2] = int8x4_to_h2x2(x.z, sc);
  d[3] = int8x4_to_h2x2(x.w, sc);
}
__device__ __forceinline__ void tile16_fp8(__half *dst, uint4 x, float sc) {
  uint2 *d = reinterpret_cast<uint2 *>(dst);
  d[0] = fp8x4_to_h2x2(x.x, sc);
  d[1] = fp8x4_to_h2x2(x.y, sc);
  d[2] = fp8x4_to_h2x2(x.z, sc);
  d[3] = fp8x4_to_h2x2(x.w, sc);
}

// Decode 4 packed E2M1 bytes (8 elements of one block) scaled by sc into 4 __half2s (8 __halfs).
__device__ __forceinline__ uint4 nvfp4x8_to_h2x4(uint32_t w, float sc) {
  return make_uint4(ink2_e2m1x2_decode_scaled_h2((uint8_t)(w & 0xffu), sc),
                    ink2_e2m1x2_decode_scaled_h2((uint8_t)((w >> 8) & 0xffu), sc),
                    ink2_e2m1x2_decode_scaled_h2((uint8_t)((w >> 16) & 0xffu), sc),
                    ink2_e2m1x2_decode_scaled_h2((uint8_t)(w >> 24), sc));
}

// Unpack 16 packed E2M1 bytes (32 elements = 2 consecutive blocks of 16) into 32 __halfs.
__device__ __forceinline__ void tile16_nvfp4(__half *dst, uint4 x, float sc0, float sc1) {
  uint4 *d = reinterpret_cast<uint4 *>(dst);
  d[0] = nvfp4x8_to_h2x4(x.x, sc0);
  d[1] = nvfp4x8_to_h2x4(x.y, sc0);
  d[2] = nvfp4x8_to_h2x4(x.z, sc1);
  d[3] = nvfp4x8_to_h2x4(x.w, sc1);
}

template <class E> struct Chunk16;
template <> struct Chunk16<float> { using V = float4; };
template <> struct Chunk16<__half> { using V = uint4; };
template <> struct Chunk16<uint8_t> { using V = uint4; };

template <int QW, int KT, int REL, int PIPE, class E = float, bool NVFP4 = false, bool AsymK8V4 = false>
__global__ void __launch_bounds__(32 * kG * QW) k_ink2p_attend_v2(const float *__restrict__ q,
                                                                  const float *__restrict__ r,
                                                                  const float *__restrict__ sinks,
                                                                  const float *__restrict__ relBias,
                                                                  const E *__restrict__ kcache,
                                                                  const E *__restrict__ vcache,
                                                                  float *__restrict__ out, int heads, int kvHeads,
                                                                  int extent, int window, int cap, float scale,
                                                                  int pos0, int T,
                                                                  const void *__restrict__ kscale = nullptr,
                                                                  const void *__restrict__ vscale = nullptr) {
  constexpr int NW = kG * QW, NT = KT / 8, QT = 16 * QW;
  using V16 = typename Chunk16<E>::V;
  constexpr bool kHasBlk16Scales = NVFP4 || AsymK8V4;
  constexpr int kKRowElem = (NVFP4 && !AsymK8V4) ? (kHD / 2) : kHD;
  constexpr int kVRowElem = kHasBlk16Scales ? (kHD / 2) : kHD;
  constexpr int CW = 16 / (int)sizeof(E);
  constexpr int kKCPR = kKRowElem / CW, kVCPR = kVRowElem / CW;
  constexpr int kBlksPerHead = kHD / 16; // 8
  extern __shared__ __align__(16) unsigned char smraw[];
  __half *sK = reinterpret_cast<__half *>(smraw);
  __half *sV = sK + KT * kKS;
  constexpr int RS = AttnSmem<QW, KT, REL, PIPE, E>::kRS;
  float *sRS = reinterpret_cast<float *>(sV + KT * kKS) + (threadIdx.x >> 5) * 16 * RS;
  E *stg = reinterpret_cast<E *>(reinterpret_cast<float *>(sV + KT * kKS) + AttnSmem<QW, KT, REL, PIPE, E>::rsFloats);
  __shared__ float sKS[KT], sVS[KT];
  __shared__ float sKS4[kHasBlk16Scales ? KT : 1][kHasBlk16Scales ? kBlksPerHead : 1];
  __shared__ float sVS4[kHasBlk16Scales ? KT : 1][kHasBlk16Scales ? kBlksPerHead : 1];
  const int kvh = blockIdx.x;
  const int tile = gridDim.y - 1 - blockIdx.y; // heaviest (latest) query tiles launch first
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, qd = lane & 3;
  const int h = kvh * kG + warp / QW;
  const int t0 = tile * QT + (warp % QW) * 16;
  const int kKVDim = kvHeads * kKRowElem;
  const int vKVDim = kvHeads * kVRowElem;
  const int blo = pos0 + tile * QT, bhi = pos0 + min(tile * QT + QT, T) - 1;
  const int kstart = blo + 1 - nkeys(blo, window, cap), kend = bhi;
  const bool wact = t0 < T;
  const int wlo = pos0 + t0, whi = pos0 + min(t0 + 16, T) - 1;
  const int wfirst = wlo + 1 - nkeys(wlo, window, cap);

  const int ta = t0 + g, tb = t0 + g + 8;
  const bool va = ta < T, vb = tb < T;
  const int posA = pos0 + ta, posB = pos0 + tb;
  const int firstA = posA + 1 - nkeys(posA, window, cap), firstB = posB + 1 - nkeys(posB, window, cap);

  // Q fragments, pre-scaled so the score accumulates straight onto the bias.
  uint32_t qa[8][4];
  {
    const float *qA = q + ((size_t)ta * heads + h) * kHD;
    const float *qB = q + ((size_t)tb * heads + h) * kHD;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int c = ks * 16 + 2 * qd;
      float2 a0 = va ? *reinterpret_cast<const float2 *>(qA + c) : make_float2(0.f, 0.f);
      float2 a1 = vb ? *reinterpret_cast<const float2 *>(qB + c) : make_float2(0.f, 0.f);
      float2 a2 = va ? *reinterpret_cast<const float2 *>(qA + c + 8) : make_float2(0.f, 0.f);
      float2 a3 = vb ? *reinterpret_cast<const float2 *>(qB + c + 8) : make_float2(0.f, 0.f);
      qa[ks][0] = h2(a0.x * scale, a0.y * scale);
      qa[ks][1] = h2(a1.x * scale, a1.y * scale);
      qa[ks][2] = h2(a2.x * scale, a2.y * scale);
      qa[ks][3] = h2(a3.x * scale, a3.y * scale);
    }
  }
  float rA[REL ? 1 : kRL], rB[REL ? 1 : kRL];
  uint32_t ra4[4];
  if constexpr (REL) {
    const float *pA = r + ((size_t)ta * heads + h) * kRL + 2 * qd;
    const float *pB = r + ((size_t)tb * heads + h) * kRL + 2 * qd;
    const float2 z = make_float2(0.f, 0.f);
    const float2 a0 = va ? *reinterpret_cast<const float2 *>(pA) : z, a2 = va ? *reinterpret_cast<const float2 *>(pA + 8) : z;
    const float2 a1 = vb ? *reinterpret_cast<const float2 *>(pB) : z, a3 = vb ? *reinterpret_cast<const float2 *>(pB + 8) : z;
    ra4[0] = h2(a0.x, a0.y);
    ra4[1] = h2(a1.x, a1.y);
    ra4[2] = h2(a2.x, a2.y);
    ra4[3] = h2(a3.x, a3.y);
  } else {
#pragma unroll
    for (int i = 0; i < kRL; ++i) {
      rA[i] = va ? r[((size_t)ta * heads + h) * kRL + i] : 0.f;
      rB[i] = vb ? r[((size_t)tb * heads + h) * kRL + i] : 0.f;
    }
  }
  const float nsA = va ? -sinks[(size_t)ta * heads + h] : 0.f;
  const float nsB = vb ? -sinks[(size_t)tb * heads + h] : 0.f;

  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float mA = 0.f, mB = 0.f, lA = 0.f, lB = 0.f; // running max starts at the implicit zero logit

  auto issue = [&](int kb) {
    for (int idx = threadIdx.x; idx < KT * kKCPR; idx += 32 * NW) {
      const int row = idx / kKCPR, c4 = (idx % kKCPR) * CW;
      const int p = kb + row;
      E *dk = stg + row * kKRowElem + c4;
      if (p <= kend) {
        const size_t off = (size_t)(p % cap) * kKVDim + (size_t)kvh * kKRowElem + c4;
        cp16(dk, kcache + off);
      } else {
        *reinterpret_cast<uint4 *>(dk) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    for (int idx = threadIdx.x; idx < KT * kVCPR; idx += 32 * NW) {
      const int row = idx / kVCPR, c4 = (idx % kVCPR) * CW;
      const int p = kb + row;
      E *dv = stg + KT * kKRowElem + row * kVRowElem + c4;
      if (p <= kend) {
        const size_t off = (size_t)(p % cap) * vKVDim + (size_t)kvh * kVRowElem + c4;
        cp16(dv, vcache + off);
      } else {
        *reinterpret_cast<uint4 *>(dv) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    asm volatile("cp.async.commit_group;" ::);
  };
  if constexpr (PIPE) issue(kstart);

  for (int kb = kstart; kb <= kend; kb += KT) {
    if constexpr (kHasBlk16Scales) {
      if (threadIdx.x < KT * kBlksPerHead) {
        const int row = threadIdx.x / kBlksPerHead;
        const int blk = threadIdx.x % kBlksPerHead;
        const int p = kb + row;
        if (p <= kend) {
          const size_t soff = (size_t)(p % cap) * (size_t)(kvHeads * kBlksPerHead) + (size_t)kvh * kBlksPerHead + blk;
          sKS4[row][blk] = AsymK8V4 ? __fmul_rn(ink2_e4m3fn_decode(__ldg(reinterpret_cast<const uint8_t *>(kscale) + soff)), 1.0f / 127.0f) : ink2_e4m3fn_decode(__ldg(reinterpret_cast<const uint8_t *>(kscale) + soff));
          sVS4[row][blk] = ink2_e4m3fn_decode(__ldg(reinterpret_cast<const uint8_t *>(vscale) + soff));
        } else {
          sKS4[row][blk] = 0.f;
          sVS4[row][blk] = 0.f;
        }
      }
    } else if constexpr (sizeof(E) == 1) {
      if (threadIdx.x < KT) {
        const int p = kb + threadIdx.x;
        if (p <= kend) {
          const size_t soff = (size_t)(p % cap) * (size_t)kvHeads + (size_t)kvh;
          sKS[threadIdx.x] = __ldg(reinterpret_cast<const float *>(kscale) + soff);
          sVS[threadIdx.x] = __ldg(reinterpret_cast<const float *>(vscale) + soff);
        } else {
          sKS[threadIdx.x] = 0.f;
          sVS[threadIdx.x] = 0.f;
        }
      }
    }
    if constexpr (PIPE) {
      asm volatile("cp.async.wait_all;" ::);
      __syncthreads();
      for (int idx = threadIdx.x; idx < KT * kKCPR; idx += 32 * NW) {
        const int row = idx / kKCPR, chunk = idx % kKCPR, c4 = chunk * CW;
        if constexpr (AsymK8V4) {
          tile16_int8(sK + row * kKS + c4, *reinterpret_cast<const uint4 *>(stg + row * kKRowElem + c4), sKS4[row][chunk]);
        } else if constexpr (NVFP4) {
          const int blk0 = chunk * 2;
          tile16_nvfp4(sK + row * kKS + chunk * 32, *reinterpret_cast<const uint4 *>(stg + row * kKRowElem + c4),
                       sKS4[row][blk0], sKS4[row][blk0 + 1]);
        } else if constexpr (sizeof(E) == 1) {
          tile16_fp8(sK + row * kKS + c4, *reinterpret_cast<const uint4 *>(stg + row * kKRowElem + c4), sKS[row]);
        } else {
          tile16(sK + row * kKS + c4, *reinterpret_cast<const V16 *>(stg + row * kKRowElem + c4));
        }
      }
      for (int idx = threadIdx.x; idx < KT * kVCPR; idx += 32 * NW) {
        const int row = idx / kVCPR, chunk = idx % kVCPR, c4 = chunk * CW;
        if constexpr (kHasBlk16Scales) {
          const int blk0 = chunk * 2;
          tile16_nvfp4(sV + row * kKS + chunk * 32,
                       *reinterpret_cast<const uint4 *>(stg + KT * kKRowElem + row * kVRowElem + c4),
                       sVS4[row][blk0], sVS4[row][blk0 + 1]);
        } else if constexpr (sizeof(E) == 1) {
          tile16_fp8(sV + row * kKS + c4, *reinterpret_cast<const uint4 *>(stg + KT * kKRowElem + row * kVRowElem + c4),
                     sVS[row]);
        } else {
          tile16(sV + row * kKS + c4, *reinterpret_cast<const V16 *>(stg + KT * kKRowElem + row * kVRowElem + c4));
        }
      }
      __syncthreads();
      if (kb + KT <= kend) issue(kb + KT);
    } else {
    __syncthreads();
    for (int idx = threadIdx.x; idx < KT * kKCPR; idx += 32 * NW) {
      const int row = idx / kKCPR, chunk = idx % kKCPR, c4 = chunk * CW;
      const int p = kb + row;
      V16 kv = {};
      if (p <= kend) {
        const size_t off = (size_t)(p % cap) * kKVDim + (size_t)kvh * kKRowElem + c4;
        kv = __ldg(reinterpret_cast<const V16 *>(kcache + off));
      }
      if constexpr (AsymK8V4) {
        tile16_int8(sK + row * kKS + c4, kv, sKS4[row][chunk]);
      } else if constexpr (NVFP4) {
        const int blk0 = chunk * 2;
        tile16_nvfp4(sK + row * kKS + chunk * 32, kv, sKS4[row][blk0], sKS4[row][blk0 + 1]);
      } else if constexpr (sizeof(E) == 1) {
        tile16_fp8(sK + row * kKS + c4, kv, sKS[row]);
      } else {
        tile16(sK + row * kKS + c4, kv);
      }
    }
    for (int idx = threadIdx.x; idx < KT * kVCPR; idx += 32 * NW) {
      const int row = idx / kVCPR, chunk = idx % kVCPR, c4 = chunk * CW;
      const int p = kb + row;
      V16 vv = {};
      if (p <= kend) {
        const size_t off = (size_t)(p % cap) * vKVDim + (size_t)kvh * kVRowElem + c4;
        vv = __ldg(reinterpret_cast<const V16 *>(vcache + off));
      }
      if constexpr (kHasBlk16Scales) {
        const int blk0 = chunk * 2;
        tile16_nvfp4(sV + row * kKS + chunk * 32, vv, sVS4[row][blk0], sVS4[row][blk0 + 1]);
      } else if constexpr (sizeof(E) == 1) {
        tile16_fp8(sV + row * kKS + c4, vv, sVS[row]);
      } else {
        tile16(sV + row * kKS + c4, vv);
      }
    }
    __syncthreads();
    }
    if (!wact || kb > whi || kb + KT - 1 < wfirst) continue;

    // bias first (sink + relative term), then the MMA accumulates the scaled q.k onto it.
    float s[NT][4];
    if constexpr (REL) {
      // band u = dist - (wlo - kb - (KT - 1)); row g, column c reads u = g - c + KT - 1
      const int dbase = wlo - kb - (KT - 1);
#pragma unroll
      for (int nt = 0; nt < NT + 2; ++nt) {
        const int dist = dbase + nt * 8 + g;
        uint32_t b0 = 0, b1 = 0;
        if (dist >= 0 && dist < extent) {
          const float *rp = relBias + (size_t)dist * kRL + 2 * qd;
          const float2 x0 = __ldg(reinterpret_cast<const float2 *>(rp)), x1 = __ldg(reinterpret_cast<const float2 *>(rp + 8));
          b0 = h2(x0.x, x0.y);
          b1 = h2(x1.x, x1.y);
        }
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        mma16816(c, ra4, b0, b1);
        *reinterpret_cast<float2 *>(sRS + g * RS + nt * 8 + 2 * qd) = make_float2(c[0], c[1]);
        *reinterpret_cast<float2 *>(sRS + (g + 8) * RS + nt * 8 + 2 * qd) = make_float2(c[2], c[3]);
      }
      __syncwarp();
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int u = g - (nt * 8 + 2 * qd + e) + KT - 1;
          s[nt][e] = nsA + sRS[g * RS + u];
          s[nt][2 + e] = nsB + sRS[(g + 8) * RS + u + 8];
        }
      }
    } else {
      const int D = posA - kb - 2 * qd; // row-A distance at n-tile 0, column parity 0
#pragma unroll
      for (int jj = 0; jj <= NT; ++jj) {
        const int j = jj - 1; // row A uses n-tile j, row B (8 positions later) n-tile j + 1
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int dist = D - j * 8 - e;
          float ra = 0.f, rb = 0.f;
          if (dist >= 0 && dist < extent) {
            const float4 *rp = reinterpret_cast<const float4 *>(relBias + (size_t)dist * kRL);
            const float4 b0 = __ldg(rp), b1 = __ldg(rp + 1), b2 = __ldg(rp + 2), b3 = __ldg(rp + 3);
            if (j >= 0) ra = reldot(rA, b0, b1, b2, b3);
            if (j + 1 < NT) rb = reldot(rB, b0, b1, b2, b3);
          }
          if (j >= 0) s[j][e] = nsA + ra;
          if (j + 1 < NT) s[j + 1][2 + e] = nsB + rb;
        }
      }
    }
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
      for (int np = 0; np < NT / 2; ++np) {
        uint32_t b0, b1, b2, b3;
        ldsm4(b0, b1, b2, b3, sK + (np * 16 + (lane >> 4) * 8 + (lane & 7)) * kKS + ks * 16 + ((lane >> 3) & 1) * 8);
        mma16816(s[2 * np], qa[ks], b0, b1);
        mma16816(s[2 * np + 1], qa[ks], b2, b3);
      }
    }
    // causal + window mask only where the tile straddles a row's key range
    const bool full = t0 + 16 <= T && kb >= pos0 + min(t0 + 16, T) - nkeys(whi, window, cap) && kb + KT - 1 <= wlo;
    if (!full) {
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int p = kb + nt * 8 + 2 * qd + e;
          if (!(va && p >= firstA && p <= posA)) s[nt][e] = -INFINITY;
          if (!(vb && p >= firstB && p <= posB)) s[nt][2 + e] = -INFINITY;
        }
      }
    }
    float xA = -INFINITY, xB = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      xA = fmaxf(xA, fmaxf(s[nt][0], s[nt][1]));
      xB = fmaxf(xB, fmaxf(s[nt][2], s[nt][3]));
    }
    xA = fmaxf(xA, __shfl_xor_sync(0xffffffff, xA, 1));
    xA = fmaxf(xA, __shfl_xor_sync(0xffffffff, xA, 2));
    xB = fmaxf(xB, __shfl_xor_sync(0xffffffff, xB, 1));
    xB = fmaxf(xB, __shfl_xor_sync(0xffffffff, xB, 2));
    const float nA = fmaxf(mA, xA), nB = fmaxf(mB, xB);
    // the running max rarely moves after the first tiles: skip the exact-1 rescale warp-wide
    if (__any_sync(0xffffffff, nA != mA || nB != mB)) {
      const float cA = __expf(mA - nA), cB = __expf(mB - nB);
      lA *= cA;
      lB *= cB;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        o[i][0] *= cA;
        o[i][1] *= cA;
        o[i][2] *= cB;
        o[i][3] *= cB;
      }
    }
    mA = nA;
    mB = nB;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      s[nt][0] = __expf(s[nt][0] - nA);
      s[nt][1] = __expf(s[nt][1] - nA);
      s[nt][2] = __expf(s[nt][2] - nB);
      s[nt][3] = __expf(s[nt][3] - nB);
      lA += s[nt][0] + s[nt][1];
      lB += s[nt][2] + s[nt][3];
    }
#pragma unroll
    for (int kk = 0; kk < NT / 2; ++kk) {
      const uint32_t pa[4] = {h2(s[2 * kk][0], s[2 * kk][1]), h2(s[2 * kk][2], s[2 * kk][3]),
                              h2(s[2 * kk + 1][0], s[2 * kk + 1][1]), h2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
      for (int dp = 0; dp < 8; ++dp) {
        uint32_t b0, b1, b2, b3;
        ldsm4t(b0, b1, b2, b3, sV + (kk * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * kKS + dp * 16 + (lane >> 4) * 8);
        mma16816(o[2 * dp], pa, b0, b1);
        mma16816(o[2 * dp + 1], pa, b2, b3);
      }
    }
  }
  if (!wact) return;
  lA += __shfl_xor_sync(0xffffffff, lA, 1);
  lA += __shfl_xor_sync(0xffffffff, lA, 2);
  lB += __shfl_xor_sync(0xffffffff, lB, 1);
  lB += __shfl_xor_sync(0xffffffff, lB, 2);
  const float iA = 1.f / (lA + __expf(-mA)), iB = 1.f / (lB + __expf(-mB));
  float *oA = out + ((size_t)ta * heads + h) * kHD + 2 * qd;
  float *oB = out + ((size_t)tb * heads + h) * kHD + 2 * qd;
  if constexpr (AsymK8V4) {
    auto fwht16_pf = [&](float &a0, float &a1, float &b0, float &b1) {
      float s0 = a0 + a1, s1 = a0 - a1;
      float t0 = b0 + b1, t1 = b0 - b1;
      auto shf = [&](float v, int m) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, m);
        return (qd & m) ? (ov - v) : (v + ov);
      };
      s0 = shf(s0, 1); s1 = shf(s1, 1); t0 = shf(t0, 1); t1 = shf(t1, 1);
      s0 = shf(s0, 2); s1 = shf(s1, 2); t0 = shf(t0, 2); t1 = shf(t1, 2);
      a0 = (s0 + t0) * 0.25f;
      a1 = (s1 + t1) * 0.25f;
      b0 = (s0 - t0) * 0.25f;
      b1 = (s1 - t1) * 0.25f;
    };
#pragma unroll
    for (int dp = 0; dp < 8; ++dp) {
      fwht16_pf(o[2 * dp][0], o[2 * dp][1], o[2 * dp + 1][0], o[2 * dp + 1][1]);
      fwht16_pf(o[2 * dp][2], o[2 * dp][3], o[2 * dp + 1][2], o[2 * dp + 1][3]);
    }
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    if (va) *reinterpret_cast<float2 *>(oA + i * 8) = make_float2(o[i][0] * iA, o[i][1] * iA);
    if (vb) *reinterpret_cast<float2 *>(oB + i * 8) = make_float2(o[i][2] * iB, o[i][3] * iB);
  }
}

template <int QW, int KT, int REL, int PIPE, class E = float, bool NVFP4 = false, bool AsymK8V4 = false>
static int ink2p_attend_v2_launch(const float *q, const float *r, const float *sinks, const float *relBias,
                                  const E *kcache, const E *vcache, float *out, int heads, int kvHeads,
                                  int extent, int window, int cap, float scale, int pos0, int T,
                                  const void *kscale = nullptr, const void *vscale = nullptr) {
  const dim3 grid(kvHeads, (T + 16 * QW - 1) / (16 * QW));
  constexpr size_t smem = AttnSmem<QW, KT, REL, PIPE, E>::bytes;
  if (smem > 48 * 1024) {
    const cudaError_t e = cudaFuncSetAttribute((const void *)k_ink2p_attend_v2<QW, KT, REL, PIPE, E, NVFP4, AsymK8V4>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e) return (int)e;
  }
  FCUDA_TS("k_ink2p_attend_v2", grid,
           (k_ink2p_attend_v2<QW, KT, REL, PIPE, E, NVFP4, AsymK8V4><<<grid, 32 * kG * QW, smem, g_stream>>>(
               q, r, sinks, relBias, kcache, vcache, out, heads, kvHeads, extent, window, cap, scale, pos0, T, kscale,
               vscale)));
  return ink2_last_error();
}

} // namespace ink2pa

extern "C" int fcuda_ink2p_attend_v2(const float *q, const float *r, const float *sinks, const float *relBias,
                                     const float *kcache, const float *vcache, float *out, int heads, int kvHeads,
                                     int headDim, int relLen, int extent, int window, int cap, float scale, int pos0,
                                     int T) {
  if (headDim != ink2pa::kHD || relLen != ink2pa::kRL || kvHeads < 1 || heads != ink2pa::kG * kvHeads || T < 1)
    return fcuda_ink2p_attend(q, r, sinks, relBias, kcache, vcache, out, heads, kvHeads, headDim, relLen, extent,
                              window, cap, scale, pos0, T);
  return ink2pa::ink2p_attend_v2_launch<2, 16, 1, 1>(q, r, sinks, relBias, kcache, vcache, out, heads, kvHeads, extent,
                                               window, cap, scale, pos0, T);
}

// fcuda_ink2p_attend_v2 over f16 caches (cap * kvDim halves per row). The float scalar
// fallback cannot read half caches, so a shape off the tensor-core tile is refused.
extern "C" int fcuda_ink2p_attend_v2_h(const float *q, const float *r, const float *sinks, const float *relBias,
                                       const void *kcache, const void *vcache, float *out, int heads, int kvHeads,
                                       int headDim, int relLen, int extent, int window, int cap, float scale, int pos0,
                                       int T) {
  if (headDim != ink2pa::kHD || relLen != ink2pa::kRL || kvHeads < 1 || heads != ink2pa::kG * kvHeads || T < 1)
    return (int)cudaErrorInvalidValue;
  return ink2pa::ink2p_attend_v2_launch<2, 16, 1, 1, __half>(q, r, sinks, relBias, (const __half *)kcache,
                                                            (const __half *)vcache, out, heads, kvHeads, extent,
                                                            window, cap, scale, pos0, T);
}

// fcuda_ink2p_attend_v2 over OCP E4M3FN caches with per-(token, kvHead) F32 scales.
// Dequantizes tiles into __half smem for mma16816 with FP32 accumulate.
extern "C" int fcuda_ink2p_attend_v2_fp8(const float *q, const float *r, const float *sinks, const float *relBias,
                                         const uint8_t *kcache, const uint8_t *vcache, const float *kscale,
                                         const float *vscale, float *out, int heads, int kvHeads, int headDim,
                                         int relLen, int extent, int window, int cap, float scale, int pos0, int T) {
  if (headDim != ink2pa::kHD || relLen != ink2pa::kRL || kvHeads < 1 || heads != ink2pa::kG * kvHeads || T < 1 ||
      cap < 1)
    return (int)cudaErrorInvalidValue;
  return ink2pa::ink2p_attend_v2_launch<2, 16, 1, 1, uint8_t, false>(q, r, sinks, relBias, kcache, vcache, out, heads,
                                                                     kvHeads, extent, window, cap, scale, pos0, T,
                                                                     kscale, vscale);
}

// fcuda_ink2p_attend_v2 over block-16 packed E2M1 caches with E4M3FN block scales.
// Dequantizes tiles into __half smem for mma16816 with FP32 accumulate.
extern "C" int fcuda_ink2p_attend_v2_nvfp4(const float *q, const float *r, const float *sinks, const float *relBias,
                                           const uint8_t *kcache, const uint8_t *vcache, const uint8_t *kscale,
                                           const uint8_t *vscale, float *out, int heads, int kvHeads, int headDim,
                                           int relLen, int extent, int window, int cap, float scale, int pos0, int T) {
  if (headDim != ink2pa::kHD || relLen != ink2pa::kRL || kvHeads < 1 || heads != ink2pa::kG * kvHeads || T < 1 ||
      cap < 1)
    return (int)cudaErrorInvalidValue;
  return ink2pa::ink2p_attend_v2_launch<2, 16, 1, 1, uint8_t, true>(q, r, sinks, relBias, kcache, vcache, out, heads,
                                                                    kvHeads, extent, window, cap, scale, pos0, T,
                                                                    kscale, vscale);
}

// fcuda_ink2p_attend_v2_asym attends over block-16 FP8 K + block-16 NVFP4 V with E4M3FN block scales.
extern "C" int fcuda_ink2p_attend_v2_asym(const float *q, const float *r, const float *sinks, const float *relBias,
                                          const uint8_t *kcache, const uint8_t *vcache, const uint8_t *kscale,
                                          const uint8_t *vscale, float *out, int heads, int kvHeads, int headDim,
                                          int relLen, int extent, int window, int cap, float scale, int pos0, int T) {
  if (headDim != ink2pa::kHD || relLen != ink2pa::kRL || kvHeads < 1 || heads != ink2pa::kG * kvHeads || T < 1 ||
      cap < 1)
    return (int)cudaErrorInvalidValue;
  return ink2pa::ink2p_attend_v2_launch<2, 16, 1, 1, uint8_t, false, true>(q, r, sinks, relBias, kcache, vcache, out,
                                                                           heads, kvHeads, extent, window, cap, scale,
                                                                           pos0, T, kscale, vscale);
}
