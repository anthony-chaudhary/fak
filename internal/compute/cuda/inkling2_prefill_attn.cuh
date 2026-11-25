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
// PIPE 1 streams the next raw fp32 K/V tile with cp.async while the current one computes.
template <int QW, int KT, int REL, int PIPE>
struct AttnSmem {
  static constexpr int kRS = KT + 16; // distance band row: NT + 2 n-tiles of 8
  static constexpr size_t rsFloats = REL ? (size_t)kG * QW * 16 * kRS : 0;
  static constexpr size_t bytes = (size_t)2 * KT * kKS * sizeof(__half) + rsFloats * 4 +
                                  (PIPE ? (size_t)2 * KT * kHD * 4 : 0);
};

__device__ __forceinline__ void cp16(void *dst, const void *src) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(src));
}

template <int QW, int KT, int REL, int PIPE>
__global__ void __launch_bounds__(32 * kG * QW) k_ink2p_attend_v2(const float *__restrict__ q,
                                                                  const float *__restrict__ r,
                                                                  const float *__restrict__ sinks,
                                                                  const float *__restrict__ relBias,
                                                                  const float *__restrict__ kcache,
                                                                  const float *__restrict__ vcache,
                                                                  float *__restrict__ out, int heads, int kvHeads,
                                                                  int extent, int window, int cap, float scale,
                                                                  int pos0, int T) {
  constexpr int NW = kG * QW, NT = KT / 8, QT = 16 * QW;
  extern __shared__ __align__(16) unsigned char smraw[];
  __half *sK = reinterpret_cast<__half *>(smraw);
  __half *sV = sK + KT * kKS;
  constexpr int RS = AttnSmem<QW, KT, REL, PIPE>::kRS;
  float *sRS = reinterpret_cast<float *>(sV + KT * kKS) + (threadIdx.x >> 5) * 16 * RS;
  float *stg = reinterpret_cast<float *>(sV + KT * kKS) + AttnSmem<QW, KT, REL, PIPE>::rsFloats;
  const int kvh = blockIdx.x;
  const int tile = gridDim.y - 1 - blockIdx.y; // heaviest (latest) query tiles launch first
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, qd = lane & 3;
  const int h = kvh * kG + warp / QW;
  const int t0 = tile * QT + (warp % QW) * 16;
  const int kvDim = kvHeads * kHD;
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
    for (int idx = threadIdx.x; idx < KT * 32; idx += 32 * NW) {
      const int row = idx >> 5, c4 = (idx & 31) * 4;
      const int p = kb + row;
      float *dk = stg + row * kHD + c4, *dv = dk + KT * kHD;
      if (p <= kend) {
        const size_t off = (size_t)(p % cap) * kvDim + (size_t)kvh * kHD + c4;
        cp16(dk, kcache + off);
        cp16(dv, vcache + off);
      } else {
        *reinterpret_cast<float4 *>(dk) = make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4 *>(dv) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
    asm volatile("cp.async.commit_group;" ::);
  };
  if constexpr (PIPE) issue(kstart);

  for (int kb = kstart; kb <= kend; kb += KT) {
    if constexpr (PIPE) {
      asm volatile("cp.async.wait_all;" ::);
      __syncthreads();
      for (int idx = threadIdx.x; idx < KT * 32; idx += 32 * NW) {
        const int row = idx >> 5, c4 = (idx & 31) * 4;
        const float4 kv = *reinterpret_cast<const float4 *>(stg + row * kHD + c4);
        const float4 vv = *reinterpret_cast<const float4 *>(stg + KT * kHD + row * kHD + c4);
        *reinterpret_cast<uint2 *>(sK + row * kKS + c4) = make_uint2(h2(kv.x, kv.y), h2(kv.z, kv.w));
        *reinterpret_cast<uint2 *>(sV + row * kKS + c4) = make_uint2(h2(vv.x, vv.y), h2(vv.z, vv.w));
      }
      __syncthreads();
      if (kb + KT <= kend) issue(kb + KT);
    } else {
    __syncthreads();
    for (int idx = threadIdx.x; idx < KT * 32; idx += 32 * NW) {
      const int row = idx >> 5, c4 = (idx & 31) * 4;
      const int p = kb + row;
      float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
      if (p <= kend) {
        const size_t off = (size_t)(p % cap) * kvDim + (size_t)kvh * kHD + c4;
        kv = __ldg(reinterpret_cast<const float4 *>(kcache + off));
        vv = __ldg(reinterpret_cast<const float4 *>(vcache + off));
      }
      *reinterpret_cast<uint2 *>(sK + row * kKS + c4) = make_uint2(h2(kv.x, kv.y), h2(kv.z, kv.w));
      *reinterpret_cast<uint2 *>(sV + row * kKS + c4) = make_uint2(h2(vv.x, vv.y), h2(vv.z, vv.w));
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
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    if (va) *reinterpret_cast<float2 *>(oA + i * 8) = make_float2(o[i][0] * iA, o[i][1] * iA);
    if (vb) *reinterpret_cast<float2 *>(oB + i * 8) = make_float2(o[i][2] * iB, o[i][3] * iB);
  }
}

template <int QW, int KT, int REL, int PIPE>
static int ink2p_attend_v2_launch(const float *q, const float *r, const float *sinks, const float *relBias,
                                  const float *kcache, const float *vcache, float *out, int heads, int kvHeads,
                                  int extent, int window, int cap, float scale, int pos0, int T) {
  const dim3 grid(kvHeads, (T + 16 * QW - 1) / (16 * QW));
  constexpr size_t smem = AttnSmem<QW, KT, REL, PIPE>::bytes;
  if (smem > 48 * 1024) {
    const cudaError_t e = cudaFuncSetAttribute((const void *)k_ink2p_attend_v2<QW, KT, REL, PIPE>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e) return (int)e;
  }
  FCUDA_TS("k_ink2p_attend_v2", grid, k_ink2p_attend_v2<QW, KT, REL, PIPE><<<grid, 32 * kG * QW, smem, g_stream>>>(
      q, r, sinks, relBias, kcache, vcache, out, heads, kvHeads, extent, window, cap, scale, pos0, T));
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
