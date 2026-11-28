// inkling2_prefill.cuh: batched prompt prefill for the resident inkling2_loop forward.
//
// Each kernel is the token-axis batch of one inkling2_loop.cuh decode kernel: token t
// of a T-token chunk sits at position pos0 + t, activations are [T][width] rows, and
// the per-token arithmetic keeps the decode kernel's order so the prompt KV, conv
// history and last-token logits match the per-token path up to GEMM reduction order.
// Included after inkling2_loop.cuh by cuda_kernels.cu.

// Depth-mix of one token per block; out is query-major [nq][T][n].
template <int NS, int NQ>
__device__ void ink2p_mix_body(Ink2MixPlainShared &sm, const Ink2States &st, size_t toff, const float *queries,
                               float *out, size_t qstride, int n) {
  constexpr int NV = NS * (1 + NQ);
  auto &part = sm.part;
  auto &tot = sm.tot;
  auto &scale = sm.scale;
  auto &wts = sm.wts;
  float acc[NV];
#pragma unroll
  for (int v = 0; v < NV; ++v) acc[v] = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float qv[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) qv[q] = queries[(size_t)q * n + i];
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      const float x = st.p[s][toff + i];
      acc[s] = fmaf(x, x, acc[s]);
#pragma unroll
      for (int q = 0; q < NQ; ++q) acc[NS + s * NQ + q] = fmaf(x, qv[q], acc[NS + s * NQ + q]);
    }
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    float x = acc[v];
    for (int d = 16; d > 0; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
    if (lane == 0) part[warp][v] = x;
  }
  __syncthreads();
  if (threadIdx.x < NV) {
    double t = 0;
    for (int w = 0; w < (int)(blockDim.x / 32); ++w) t += (double)part[w][threadIdx.x];
    tot[threadIdx.x] = t;
  }
  __syncthreads();
  if (threadIdx.x < NS) scale[threadIdx.x] = (float)(1.0 / sqrt(tot[threadIdx.x] / (double)n + 1e-6));
  __syncthreads();
  if (threadIdx.x < NQ) {
    const int q = threadIdx.x;
    const double inv = 1.0 / sqrt((double)n);
    double sc[NS], mx = -INFINITY;
    for (int s = 0; s < NS; ++s) {
      sc[s] = (double)(float)((double)scale[s] * tot[NS + s * NQ + q]) * inv;
      mx = fmax(mx, sc[s]);
    }
    double sum = 0;
    for (int s = 0; s < NS; ++s) {
      sc[s] = exp(sc[s] - mx);
      sum += sc[s];
    }
    for (int s = 0; s < NS; ++s) wts[q][s] = (float)(sc[s] / sum);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float xs[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s) xs[s] = __fmul_rn(st.p[s][toff + i], scale[s]);
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
      float a = 0.f;
#pragma unroll
      for (int s = 0; s < NS; ++s) a = __fadd_rn(a, __fmul_rn(wts[q][s], xs[s]));
      out[(size_t)q * qstride + toff + i] = a;
    }
  }
}

template <int NQ>
__global__ void __launch_bounds__(INK2_MIX_THREADS) k_ink2p_mix(Ink2States st, int ns, const float *queries, float *out,
                                                               int n, int T) {
  const size_t toff = (size_t)blockIdx.x * n, qs = (size_t)T * n;
  __shared__ Ink2MixPlainShared sm;
#define INK2P_MIX_CALL(NS) ink2p_mix_body<NS, NQ>(sm, st, toff, queries, out, qs, n)
  INK2_NS_DISPATCH(ns, INK2P_MIX_CALL)
#undef INK2P_MIX_CALL
}

// Y[t][r] = W[r,:] . X[t,:] for the small dense F32 weights; one warp per (row, token).
#define INK2P_GEMM_TOK 8
__global__ void k_ink2p_gemm_f32(const float *W, const float *X, float *Y, int rows, int cols, int T) {
  const int r = blockIdx.x;
  const int t = blockIdx.y * INK2P_GEMM_TOK + (threadIdx.x >> 5);
  if (t >= T) return;
  const int lane = threadIdx.x & 31;
  const float *wr = W + (size_t)r * cols, *x = X + (size_t)t * cols;
  float s = 0.f;
  for (int i = lane; i < cols; i += 32) s = fmaf(wr[i], x[i], s);
  s = ink2_warp_sum(s);
  if (lane == 0) Y[(size_t)t * rows + r] = s;
}

// Up to four controllers of one layer visit; blockIdx.z selects the controller, so one
// launch serves them all with each controller's arithmetic unchanged.
struct Ink2PCtrl {
  const float *cur[4], *msg[4], *W[4], *gain[4], *forget[4];
  float *g[4], *out[4];
};

__global__ void k_ink2p_ctrl_down(Ink2PCtrl cs, int n, int rank) {
  __shared__ float part[4][32];
  const int j = blockIdx.z;
  const float *cur = cs.cur[j], *msg = cs.msg[j], *W = cs.W[j];
  float *g = cs.g[j];
  const int r = blockIdx.x;
  const size_t toff = (size_t)blockIdx.y * n;
  cur += toff;
  msg += toff;
  const float *wc = W + (size_t)r * 2 * n, *wm = wc + n;
  float ssc = 0.f, ssm = 0.f, dc = 0.f, dm = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float c = cur[i], m = msg[i];
    ssc = fmaf(c, c, ssc);
    ssm = fmaf(m, m, ssm);
    dc = fmaf(wc[i], c, dc);
    dm = fmaf(wm[i], m, dm);
  }
  for (int d = 16; d > 0; d >>= 1) {
    ssc += __shfl_down_sync(0xffffffff, ssc, d);
    ssm += __shfl_down_sync(0xffffffff, ssm, d);
    dc += __shfl_down_sync(0xffffffff, dc, d);
    dm += __shfl_down_sync(0xffffffff, dm, d);
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) {
    part[0][warp] = ssc;
    part[1][warp] = ssm;
    part[2][warp] = dc;
    part[3][warp] = dm;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double t[4] = {0, 0, 0, 0};
    for (int w = 0; w < (int)(blockDim.x / 32); ++w)
      for (int v = 0; v < 4; ++v) t[v] += (double)part[v][w];
    const float sc = (float)(1.0 / sqrt(t[0] / (double)n + 1e-6));
    const float sm = (float)(1.0 / sqrt(t[1] / (double)n + 1e-6));
    g[(size_t)blockIdx.y * rank + r] = (float)((double)sc * t[2] + (double)sm * t[3]);
  }
}

// Controller tail over a tile of 32 outputs x INK2P_CTRL_TOK tokens: the gain/forget
// rows of the tile are staged once (transposed, so a warp reads 32 banks) instead of
// re-read per token.
#define INK2P_CTRL_TOK 8
__global__ void __launch_bounds__(256) k_ink2p_ctrl_apply(Ink2PCtrl cs, int n, int rank, int T) {
  const int j = blockIdx.z;
  const float *cur = cs.cur[j], *msg = cs.msg[j], *g = cs.g[j], *gainW = cs.gain[j], *forgetW = cs.forget[j];
  float *out = cs.out[j];
  extern __shared__ float csh[];
  float *gws = csh, *fws = csh + (size_t)rank * 33, *gs = csh + (size_t)rank * 66;
  const int i0 = blockIdx.x * 32, t0 = blockIdx.y * INK2P_CTRL_TOK;
  for (int idx = threadIdx.x; idx < 32 * rank; idx += blockDim.x) {
    const int il = idx / rank, r = idx % rank;
    const bool ok = i0 + il < n;
    gws[r * 33 + il] = ok ? gainW[(size_t)(i0 + il) * rank + r] : 0.f;
    fws[r * 33 + il] = ok ? forgetW[(size_t)(i0 + il) * rank + r] : 0.f;
  }
  for (int idx = threadIdx.x; idx < INK2P_CTRL_TOK * rank; idx += blockDim.x) {
    const int tl = idx / rank;
    gs[idx] = t0 + tl < T ? g[(size_t)(t0 + tl) * rank + idx % rank] : 0.f;
  }
  __syncthreads();
  const int il = threadIdx.x & 31, tl = threadIdx.x >> 5;
  const int i = i0 + il, tok = t0 + tl;
  if (i >= n || tok >= T) return;
  const float *gt = gs + (size_t)tl * rank;
  float gn = 0.f, fg = 0.f;
  for (int r = 0; r < rank; ++r) {
    const float gv = gt[r];
    gn = fmaf(gws[r * 33 + il], gv, gn);
    fg = fmaf(fws[r * 33 + il], gv, fg);
  }
  const float f = (float)tanh((double)fg), th = (float)tanh((double)gn);
  const size_t o = (size_t)tok * n + i;
  const float c = cur[o];
  out[o] = __fadd_rn(__fsub_rn(c, __fmul_rn(f, c)), __fmul_rn(th, msg[o]));
}

// Causal short conv walked along the token axis: one thread per channel carries the
// history in registers and writes the final history once.
#define INK2P_MAX_KN 8
__device__ __forceinline__ void ink2p_conv_body(const float *x, const float *w, const float *dyn, float *hist,
                                                float *out, const float *resid, int n, int heads, int kn, float norm,
                                                int T, float *log, const float *moeDown, const float *moeWts,
                                                int moeK) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  const int past = kn - 1;
  float *hc = hist + (size_t)c * past;
  const float *wc = w + (size_t)c * kn;
  float h[INK2P_MAX_KN], wv[INK2P_MAX_KN];
  for (int t = 0; t < past; ++t) h[t] = hc[t];
  for (int t = 0; t < kn; ++t) wv[t] = wc[t];
  // a verify chunk logs the entry history and every input so a later commit can keep
  // only the accepted prefix
  float *lc = log ? log + (size_t)c * (past + T) : nullptr;
  if (lc)
    for (int t = 0; t < past; ++t) lc[t] = h[t];
  const int hd = dyn ? c / (n / heads) : 0;
  const int dstride = heads * kn;
  for (int tok = 0; tok < T; ++tok) {
    float xc;
    if (moeDown) {
      // the sum of k_ink2p_combine, in routing order
      xc = 0.f;
      const size_t pair0 = (size_t)tok * moeK;
      for (int sl = 0; sl < moeK; ++sl) xc = __fadd_rn(xc, __fmul_rn(moeWts[pair0 + sl], moeDown[(pair0 + sl) * n + c]));
    } else {
      xc = x[(size_t)tok * n + c];
    }
    if (lc) lc[past + tok] = xc;
    float acc = 0.f;
    for (int t = 0; t < past; ++t) acc = __fadd_rn(acc, __fmul_rn(h[t], wv[t]));
    acc = __fadd_rn(acc, __fmul_rn(xc, wv[past]));
    if (dyn) {
      const float *dr = dyn + (size_t)tok * dstride + hd * kn;
      for (int lag = 0; lag < kn; ++lag) {
        const float xv = lag > 0 ? h[past - lag] : xc;
        acc = __fadd_rn(acc, __fmul_rn(dr[lag], xv));
      }
    }
    float o = __fmul_rn(__fadd_rn(xc, acc), norm);
    for (int t = 0; t + 1 < past; ++t) h[t] = h[t + 1];
    if (past > 0) h[past - 1] = xc;
    if (resid) o = __fadd_rn(resid[(size_t)tok * n + c], o);
    out[(size_t)tok * n + c] = o;
  }
  for (int t = 0; t < past; ++t) hc[t] = h[t];
}

// moeDown (nullable) makes input row t the MoE combine of the t-th moeK routed down rows
// by moeWts, so the combine needs no launch of its own.
__global__ void k_ink2p_conv(const float *x, const float *w, const float *dyn, float *hist, float *out,
                             const float *resid, int n, int heads, int kn, float norm, int T, float *log,
                             const float *moeDown, const float *moeWts, int moeK) {
  ink2p_conv_body(x, w, dyn, hist, out, resid, n, heads, kn, norm, T, log, moeDown, moeWts, moeK);
}

// Two logged convs of one verify visit (the k and v short convs) in one launch:
// blockIdx.y selects the conv; each runs k_ink2p_conv's arithmetic.
struct Ink2PConvPair {
  const float *x[2], *w[2], *dyn[2];
  float *hist[2], *out[2], *log[2];
  float norm[2];
};

__global__ void k_ink2p_conv_pair(Ink2PConvPair cp, int n, int heads, int kn, int T) {
  const int j = blockIdx.y;
  ink2p_conv_body(cp.x[j], cp.w[j], cp.dyn[j], cp.hist[j], cp.out[j], nullptr, n, heads, kn, cp.norm[j], T, cp.log[j],
                  nullptr, nullptr, 0);
}

extern "C" int fcuda_ink2p_conv_pair_log(const float *const *x, const float *const *w, const float *const *dyn,
                                         float *const *hist, float *const *log, float *const *out, const float *norm,
                                         int n, int heads, int kn, int T) {
  if (kn < 1 || kn > INK2P_MAX_KN) return (int)cudaErrorInvalidValue;
  Ink2PConvPair cp = {};
  for (int j = 0; j < 2; ++j) {
    cp.x[j] = x[j];
    cp.w[j] = w[j];
    cp.dyn[j] = dyn[j];
    cp.hist[j] = hist[j];
    cp.out[j] = out[j];
    cp.log[j] = log[j];
    cp.norm[j] = norm[j];
  }
  const dim3 grid((n + 31) / 32, 2);
  FCUDA_TS("k_ink2p_conv", grid, k_ink2p_conv_pair<<<grid, 32, 0, g_stream>>>(cp, n, heads, kn, T));
  return ink2_last_error();
}

// k_ink2_qkv_post over a chunk: blockIdx.y is the token, at position pos0 + t. T is the
// cache element (float, __half, or uint8_t E4M3FN/NVFP4); f16/fp8/nvfp4 appends run the
// fp32 ops of the float append in the same order and quantize only the stored value.
template <class T, bool NVFP4 = false, bool AsymK8V4 = false>
__global__ void k_ink2p_qkv_post(float *q, float *r, const float *k, const float *v, const float *qclip,
                                 const float *rclip, T *kcache, T *vcache, int heads, int kvHeads,
                                 int headDim, int relLen, float qkClip, float relClip, float eps, int global,
                                 float alpha, float floorTok, int cap, int pos0, const int *posDev,
                                 void *kscale = nullptr, void *vscale = nullptr, int *overflow = nullptr) {
  const int lane = threadIdx.x & 31;
  const int b = blockIdx.x, tok = blockIdx.y;
  const int pos = (posDev ? *posDev : pos0) + tok;
  const int idx = pos % cap;
  const int kvDim = kvHeads * headDim;
  if (b < heads) {
    float *qh = q + (size_t)tok * heads * headDim + (size_t)b * headDim;
    float *rh = r + (size_t)tok * heads * relLen + (size_t)b * relLen;
    ink2_clip_warp(qh, qclip + (size_t)b * headDim, qkClip, headDim);
    ink2_clip_warp(rh, rclip + (size_t)b * relLen, relClip, relLen);
    if (global) {
      const float tau = (float)(1.0 + (double)alpha * log(fmax(1.0, (double)(pos + 1) / (double)floorTok)));
      __syncwarp();
      for (int i = lane; i < headDim; i += 32) qh[i] = __fmul_rn(qh[i], tau);
      for (int i = lane; i < relLen; i += 32) rh[i] = __fmul_rn(rh[i], tau);
    }
    return;
  }
  if (b < heads + kvHeads) {
    const int h = b - heads;
    if constexpr (AsymK8V4) {
      const Ink2KVConv off = {};
      const int hBlks = headDim >> 4;
      ink2_put_k_fp8_blk16(kcache + (size_t)idx * kvDim + (size_t)h * headDim,
                           (uint8_t *)kscale + (size_t)idx * (kvHeads * hBlks) + (size_t)h * hBlks, overflow,
                           k + (size_t)tok * kvDim, off, h, headDim, kvHeads, qkClip);
    } else if constexpr (NVFP4) {
      const Ink2KVConv off = {};
      const int hBytes = headDim >> 1, hBlks = headDim >> 4;
      ink2_put_k_nvfp4(kcache + (size_t)idx * (kvHeads * hBytes) + (size_t)h * hBytes,
                       (uint8_t *)kscale + (size_t)idx * (kvHeads * hBlks) + (size_t)h * hBlks, overflow,
                       k + (size_t)tok * kvDim, off, h, headDim, kvHeads, qkClip);
    } else if constexpr (sizeof(T) == 1) {
      const Ink2KVConv off = {};
      ink2_put_k_fp8(kcache + (size_t)idx * kvDim + (size_t)h * headDim, (float *)kscale + (size_t)idx * kvHeads + h,
                     overflow, k + (size_t)tok * kvDim, off, h, headDim, kvHeads, qkClip);
    } else if constexpr (sizeof(T) == 2) {
      const Ink2KVConv off = {};
      ink2_put_k(kcache + (size_t)idx * kvDim + (size_t)h * headDim, k + (size_t)tok * kvDim, off, h, headDim, kvHeads,
                 qkClip);
    } else {
      float *dst = kcache + (size_t)idx * kvDim + (size_t)h * headDim;
      const float *src = k + (size_t)tok * kvDim + (size_t)h * headDim;
      for (int i = lane; i < headDim; i += 32) dst[i] = src[i];
      __syncwarp();
      ink2_clip_warp(dst, nullptr, qkClip, headDim);
    }
    return;
  }
  const int h = b - heads - kvHeads;
  if constexpr (NVFP4 || AsymK8V4) {
    const Ink2KVConv off = {};
    const int hBytes = headDim >> 1, hBlks = headDim >> 4;
    ink2_put_v_nvfp4<AsymK8V4>(vcache + (size_t)idx * (kvHeads * hBytes) + (size_t)h * hBytes,
                     (uint8_t *)vscale + (size_t)idx * (kvHeads * hBlks) + (size_t)h * hBlks, overflow,
                     v + (size_t)tok * kvDim, off, h, headDim, kvHeads, eps);
  } else if constexpr (sizeof(T) == 1) {
    const Ink2KVConv off = {};
    ink2_put_v_fp8(vcache + (size_t)idx * kvDim + (size_t)h * headDim, (float *)vscale + (size_t)idx * kvHeads + h,
                   overflow, v + (size_t)tok * kvDim, off, h, headDim, kvHeads, eps);
  } else if constexpr (sizeof(T) == 2) {
    const Ink2KVConv off = {};
    ink2_put_v(vcache + (size_t)idx * kvDim + (size_t)h * headDim, v + (size_t)tok * kvDim, off, h, headDim, kvHeads,
               eps);
  } else {
    const float *vh = v + (size_t)tok * kvDim + (size_t)h * headDim;
    double ss = 0;
    for (int i = lane; i < headDim; i += 32) ss += (double)vh[i] * (double)vh[i];
    for (int d = 16; d > 0; d >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, d);
    const float s = (float)(1.0 / sqrt(ss / (double)headDim + (double)eps));
    float *dst = vcache + (size_t)idx * kvDim + (size_t)h * headDim;
    for (int i = lane; i < headDim; i += 32) dst[i] = __fmul_rn(vh[i], s);
  }
}

// k_ink2_attend for every query of a chunk: block (head, token). The query at pos0+t
// sees keys (pos-n, pos], n bounded by the sliding window; relative bias only within
// extent (zero row past it). GLOBAL=false keeps the score row in dynamic shared
// memory; GLOBAL=true runs the identical arithmetic on a per-block row of gscore
// (span floats per block, token tile starting at tok0) for spans past the opt-in.
template <bool GLOBAL>
__global__ void __launch_bounds__(INK2_ATT_THREADS) k_ink2p_attend(const float *q, const float *r, const float *sinks,
                                                                  const float *relBias, const float *kcache,
                                                                  const float *vcache, float *out, int heads,
                                                                  int kvHeads, int headDim, int relLen, int extent,
                                                                  int window, int cap, float scale, int pos0,
                                                                  const int *posDev, float *gscore, int span,
                                                                  int tok0) {
  extern __shared__ float ink2p_att_dyn[];
  float *sc = GLOBAL ? gscore + ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * (size_t)span : ink2p_att_dyn;
  const int h = blockIdx.x, tok = tok0 + blockIdx.y;
  const int kvh = h / (heads / kvHeads);
  const int kvDim = kvHeads * headDim;
  const int pos = (posDev ? *posDev : pos0) + tok;
  int n = pos + 1;
  if (window > 0 && n > window) n = window;
  if (n > cap) n = cap;
  const int first = pos + 1 - n;
  const float *qh = q + (size_t)tok * heads * headDim + (size_t)h * headDim;
  const float *rh = r + (size_t)tok * heads * relLen + (size_t)h * relLen;
  const float sink = sinks[(size_t)tok * heads + h];
  __shared__ float shm[33];
  __shared__ double dsh[33];
  __shared__ float smax;
  float mx = -INFINITY;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
  for (int j = warp; j < n; j += nwarp) {
    const int p = first + j;
    const int dist = pos - p;
    const float *kr = kcache + (size_t)(p % cap) * kvDim + (size_t)kvh * headDim;
    float d = 0.f;
    for (int i = lane; i < headDim; i += 32) d = fmaf(qh[i], kr[i], d);
    float rd = 0.f;
    if (dist < extent) {
      const float *rb = relBias + (size_t)dist * relLen;
      for (int i = lane; i < relLen; i += 32) rd = fmaf(rb[i], rh[i], rd);
    }
    for (int o = 16; o > 0; o >>= 1) {
      d += __shfl_xor_sync(0xffffffff, d, o);
      rd += __shfl_xor_sync(0xffffffff, rd, o);
    }
    float bias = -sink;
    if (dist < extent) bias = __fadd_rn(bias, rd);
    const float s = __fadd_rn(__fmul_rn(d, scale), bias);
    if (lane == 0) sc[j] = s;
    mx = fmaxf(mx, s);
  }
  for (int d = 16; d > 0; d >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, d));
  if (lane == 0) shm[warp] = mx;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int w = 0; w < nwarp; ++w) m = fmaxf(m, shm[w]);
    smax = m < 0 ? 0 : m;
  }
  __syncthreads();
  // float exp and a float64 tree denominator: the decode kernel's float64 exp and
  // serial key-order sum cost more than the whole score pass at prefill widths
  const float m = smax;
  double part = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    const float ev = expf(sc[j] - m);
    sc[j] = ev;
    part += (double)ev;
  }
  const double denom = ink2_block_sum_d(part, dsh) + (double)expf(-m);
  for (int j = threadIdx.x; j < n; j += blockDim.x) sc[j] = (float)((double)sc[j] / denom);
  __syncthreads();
  __shared__ float gacc[INK2_ATT_GROUPS][256];
  const int groups = blockDim.x / headDim;
  const int dd = threadIdx.x % headDim, g = threadIdx.x / headDim;
  if (g < groups) {
    float acc = 0.f;
    for (int j = g; j < n; j += groups) {
      const float w = sc[j];
      const float *vr = vcache + (size_t)((first + j) % cap) * kvDim + (size_t)kvh * headDim;
      acc = fmaf(w, vr[dd], acc);
    }
    gacc[g][dd] = acc;
  }
  __syncthreads();
  if (threadIdx.x < headDim) {
    float acc = 0.f;
    for (int gg = 0; gg < groups; ++gg) acc += gacc[gg][threadIdx.x];
    out[(size_t)tok * heads * headDim + (size_t)h * headDim + threadIdx.x] = acc;
  }
}

// k_ink2_route per token: block t routes logits row t into ids/wts [t][k].
__global__ void k_ink2p_route(const float *logits, const float *bias, int *ids, float *wts, int nexp, int k,
                              float wscale) {
  __shared__ float probs[INK2_MAX_EXPERTS], sel[INK2_MAX_EXPERTS];
  const int lane = threadIdx.x;
  logits += (size_t)blockIdx.x * nexp;
  ids += (size_t)blockIdx.x * k;
  wts += (size_t)blockIdx.x * k;
  for (int i = lane; i < nexp; i += 32) {
    probs[i] = ink2_sigmoid(logits[i]);
    sel[i] = bias ? __fadd_rn(probs[i], bias[i]) : probs[i];
  }
  __syncwarp();
  float w[INK2_MAX_TOPK];
  float sum = 0.f;
  for (int t = 0; t < k; ++t) {
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = lane; i < nexp; i += 32)
      if (sel[i] > bv || (sel[i] == bv && i < bi)) {
        bv = sel[i];
        bi = i;
      }
    for (int d = 16; d > 0; d >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffff, bv, d);
      const int oi = __shfl_xor_sync(0xffffffff, bi, d);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    w[t] = probs[bi];
    sum = __fadd_rn(sum, w[t]);
    __syncwarp();
    if (lane == 0) {
      ids[t] = bi;
      sel[bi] = -INFINITY;
    }
    __syncwarp();
  }
  for (int t = 0; t < k; ++t) {
    if (t != lane) continue;
    float v = __fdiv_rn(w[t], __fadd_rn(sum, 1e-20f));
    if (wscale != 0.f) v = __fmul_rn(v, wscale);
    wts[t] = v;
  }
}

__global__ void k_ink2p_combine(const float *down, const float *wts, float *out, int k, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  const size_t t = blockIdx.y;
  float acc = 0.f;
  for (int s = 0; s < k; ++s) acc = __fadd_rn(acc, __fmul_rn(wts[t * k + s], down[(t * k + s) * n + j]));
  out[t * n + j] = acc;
}

// Pair order for the expert GEMV fallback: sorted[i] takes pair order[i]'s expert id and
// input row, so consecutive routed slots share an expert and its rows stay in L2.
__global__ void k_ink2p_gather_pairs(const int *ids, const int *order, const float *X, int xDiv, int xStride,
                                     int *sortedIds, float *sortedX, int width) {
  const int i = blockIdx.y, p = order[i];
  if (blockIdx.x == 0 && threadIdx.x == 0) sortedIds[i] = ids[p];
  const float *x = X + (size_t)(p / xDiv) * xStride;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < width; j += gridDim.x * blockDim.x)
    sortedX[(size_t)i * width + j] = x[j];
}

__global__ void k_ink2p_scatter_pairs(const int *order, const float *sortedY, float *Y, int rows) {
  const int i = blockIdx.y;
  const size_t dst = (size_t)order[i] * rows, src = (size_t)i * rows;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < rows; j += gridDim.x * blockDim.x) Y[dst + j] = sortedY[src + j];
}

// Expert-grouped NVFP4 GEMM for a chunk: the routed (token, slot) pairs are bucketed by
// expert on the device, then block (row tile, pair tile of one expert) decodes each
// weight block once and multiplies it against INK2P_NV_PAIRS staged activations.
#define INK2P_NV_PAIRS 32
#define INK2P_KCHUNK 256

__global__ void k_ink2p_bucket_count(const int *ids, int *counts, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&counts[ids[i]], 1);
}

// One block: exclusive scan of per-expert counts into offsets and the flat tile list
// (expert, first pair) of INK2P_NV_PAIRS-pair tiles; ntiles[0] counts it.
__global__ void k_ink2p_bucket_scan(const int *counts, int *offsets, int *tiles, int *ntiles, int nexp) {
  if (threadIdx.x != 0) return;
  int acc = 0, nt = 0;
  for (int e = 0; e < nexp; ++e) {
    offsets[e] = acc;
    for (int s = 0; s < counts[e]; s += INK2P_NV_PAIRS) {
      tiles[2 * nt] = e;
      tiles[2 * nt + 1] = acc + s;
      ++nt;
    }
    acc += counts[e];
  }
  offsets[nexp] = acc;
  ntiles[0] = nt;
}

// Stable fill, one block per expert: pairs of expert blockIdx.x land in ascending pair
// order, so the grouped output is deterministic.
__global__ void k_ink2p_bucket_fill(const int *ids, const int *offsets, int *order, int n) {
  __shared__ int warpBase[33];
  const int e = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  int base = offsets[e];
  for (int c = 0; c < n; c += blockDim.x) {
    const int i = c + threadIdx.x;
    const bool hit = i < n && ids[i] == e;
    const unsigned m = __ballot_sync(0xffffffff, hit);
    if (lane == 0) warpBase[warp] = __popc(m);
    __syncthreads();
    if (threadIdx.x == 0) {
      int acc = base;
      for (int w = 0; w < nw; ++w) {
        const int cnt = warpBase[w];
        warpBase[w] = acc;
        acc += cnt;
      }
      warpBase[32] = acc;
    }
    __syncthreads();
    if (hit) order[warpBase[warp] + __popc(m & ((1u << lane) - 1u))] = i;
    base = warpBase[32];
    __syncthreads();
  }
}

// E2M1 codes times an E4M3 block scale carry at most six significant bits and the
// activation is BF16-rounded already, so every product is exact in BF16 and the
// tensor-core MMA differs from the decode GEMV only in summation order. A warp owns
// INK2P_NV_MT m16 row tiles against every n8 tile of the routed pairs; activations are
// staged INK2P_KCHUNK columns at a time.
#define INK2P_NV_MT 2
#define INK2P_NV_XPAD 8

static __device__ __forceinline__ uint32_t ink2p_bf16x2(float lo, float hi) {
  return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);
}

static __device__ __forceinline__ uint32_t ink2p_nv_pair(const float *lut, uint32_t byte, float sc) {
  return ink2p_bf16x2(__fmul_rn(lut[byte & 0x0fu], sc), __fmul_rn(lut[byte >> 4], sc));
}

__global__ void __launch_bounds__(256) k_ink2p_expert_nvfp4(const uint8_t *Packed, const uint8_t *Scales,
                                                           const int *offsets, const int *order, const int *tiles,
                                                           const int *ntiles, const float *X,
                                                           int xDiv, int xStride, float *Y, int rows, int in,
                                                           const float *escale) {
  constexpr int NT = INK2P_NV_PAIRS / 8, S = INK2P_KCHUNK + INK2P_NV_XPAD;
  __shared__ unsigned short xsb[INK2P_NV_PAIRS * S];
  __shared__ float lut[16];
  __shared__ int pairs[INK2P_NV_PAIRS];
  if ((int)blockIdx.y >= ntiles[0]) return;
  const int e = tiles[2 * blockIdx.y], start = tiles[2 * blockIdx.y + 1];
  const int np = min(INK2P_NV_PAIRS, offsets[e + 1] - start);
  if (threadIdx.x < INK2P_NV_PAIRS) pairs[threadIdx.x] = threadIdx.x < np ? order[start + threadIdx.x] : -1;
  if (threadIdx.x < 16) lut[threadIdx.x] = nv4_e2m1(threadIdx.x);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int base = (blockIdx.x * (blockDim.x >> 5) + warp) * (16 * INK2P_NV_MT);
  const size_t ebase = (size_t)e * rows;
  const int nb = in / 16;
  const size_t rowBytes = (size_t)(in / 2);
  float acc[INK2P_NV_MT][NT][4];
#pragma unroll
  for (int m = 0; m < INK2P_NV_MT; ++m)
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;
  for (int k0 = 0; k0 < in; k0 += INK2P_KCHUNK) {
    const int kw = min(INK2P_KCHUNK, in - k0);
    __syncthreads();
    for (int idx = threadIdx.x; idx < INK2P_NV_PAIRS * kw; idx += blockDim.x) {
      const int p = idx / kw, i = idx % kw;
      float v = 0.f;
      if (p < np) v = nv4_bf16_rne(X[(size_t)(pairs[p] / xDiv) * xStride + k0 + i]);
      xsb[p * S + i] = (unsigned short)(__float_as_uint(v) >> 16);
    }
    __syncthreads();
    if (base >= rows) continue;
    for (int kl = 0; kl < kw / 16; ++kl) {
      const int kb = k0 / 16 + kl;
      uint32_t b0[NT], b1[NT];
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const unsigned short *xr = xsb + (n * 8 + g) * S + kl * 16 + 2 * t;
        b0[n] = *(const uint32_t *)xr;
        b1[n] = *(const uint32_t *)(xr + 8);
      }
#pragma unroll
      for (int m = 0; m < INK2P_NV_MT; ++m) {
        if (base + m * 16 >= rows) break;
        const size_t w0 = ebase + base + m * 16 + g, w1 = w0 + 8;
        const uint2 p0 = *(const uint2 *)(Packed + w0 * rowBytes + (size_t)kb * 8);
        const uint2 p1 = *(const uint2 *)(Packed + w1 * rowBytes + (size_t)kb * 8);
        const float s0 = nv4_e4m3fn(Scales[w0 * (size_t)nb + kb]);
        const float s1 = nv4_e4m3fn(Scales[w1 * (size_t)nb + kb]);
        const uint32_t a0 = ink2p_nv_pair(lut, (p0.x >> (8 * t)) & 0xffu, s0);
        const uint32_t a1 = ink2p_nv_pair(lut, (p1.x >> (8 * t)) & 0xffu, s1);
        const uint32_t a2 = ink2p_nv_pair(lut, (p0.y >> (8 * t)) & 0xffu, s0);
        const uint32_t a3 = ink2p_nv_pair(lut, (p1.y >> (8 * t)) & 0xffu, s1);
#pragma unroll
        for (int n = 0; n < NT; ++n)
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                       "{%0,%1,%2,%3};"
                       : "+f"(acc[m][n][0]), "+f"(acc[m][n][1]), "+f"(acc[m][n][2]), "+f"(acc[m][n][3])
                       : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[n]), "r"(b1[n]));
      }
    }
  }
  if (base >= rows) return;
  const float es = escale ? escale[e] : 1.f;
#pragma unroll
  for (int m = 0; m < INK2P_NV_MT; ++m) {
    if (base + m * 16 >= rows) break;
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int row = base + m * 16 + g + (c >= 2 ? 8 : 0), col = n * 8 + 2 * t + (c & 1);
        if (col < np) {
          const float v = acc[m][n][c];
          Y[(size_t)pairs[col] * rows + row] = escale ? __fmul_rn(v, es) : v;
        }
      }
  }
}

// Grouped expert GEMM for the IQ and Q3_K stacks: per super-block every lane decodes
// its 8 weights once (the decode kernel lane map) and dots them against the staged
// pairs of the tile. Each warp covers INK2P_IQ_RPW rows.
#define INK2P_IQ_RPW 4
#define INK2P_IQ_HALF 16
#define INK2P_IQ_THREADS 512

// ink2p_iq_decode: the 8 weights of one lane in one super-block; w[0..3] multiply
// x[o0+j] and w[4..7] x[o1+j] (o1 = o0+4 except the split nibble halves of IQ4_NL).
template <int FMT>
static __device__ __forceinline__ void ink2p_iq_decode(const unsigned char *blk, int lane, float *w, int &o0, int &o1) {
  if (FMT == INK2_Q3K) {
    const int e0 = lane * 8, half = e0 >> 7, j = (e0 & 127) >> 5, sub = (e0 & 31) >> 4, l0 = e0 & 15;
    const int is = half * 8 + j * 2 + sub;
    const unsigned char mbit = (unsigned char)(1u << (half * 4 + j));
    const unsigned char *hm = blk, *qs = blk + 32, *sc = blk + 96;
    const float dall = __half2float(*(const __half *)(blk + 108));
    int scv;
    if (is < 8) scv = (sc[is] & 0x0f) | (((sc[8 + (is & 3)] >> (2 * (is >> 2))) & 3) << 4);
    else scv = (sc[is - 8] >> 4) | (((sc[8 + (is & 3)] >> (2 * (is >> 2))) & 3) << 4);
    const float dl = dall * (float)(scv - 32);
#pragma unroll
    for (int tt = 0; tt < 8; ++tt) {
      const int li = sub * 16 + l0 + tt;
      w[tt] = dl * (float)((int)((qs[half * 32 + li] >> (2 * j)) & 3) - ((hm[li] & mbit) ? 0 : 4));
    }
    o0 = e0;
    o1 = e0 + 4;
    return;
  }
  if (FMT == INK2_IQ4NL) {
    const int ib = lane >> 2, il = lane & 3;
    const unsigned char *b = blk + 18 * ib;
    const unsigned char *q4 = b + 2 + 4 * il;
    const float d = __half2float(*(const __half *)b);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      w[j] = d * (float)fcuda_kvalues_iq4nl[q4[j] & 0xf];
      w[4 + j] = d * (float)fcuda_kvalues_iq4nl[q4[j] >> 4];
    }
    o0 = 32 * ib + 4 * il;
    o1 = o0 + 16;
    return;
  }
  const int ib32 = lane >> 2, l = lane & 3;
  const float d = __half2float(*(const __half *)blk);
  unsigned long long grid;
  unsigned char signs;
  float db;
  if (FMT == INK2_IQ2XS) {
    const unsigned short qv = ((const unsigned short *)(blk + 2))[4 * ib32 + l];
    const unsigned char sc = blk[66 + ib32];
    db = d * (0.5f + (float)(l < 2 ? (sc & 0x0f) : (sc >> 4))) * 0.25f;
    grid = fcuda_iq2xs_grid[qv & 511];
    signs = fcuda_ksigns_iq2xs[qv >> 9];
  } else if (FMT == INK2_IQ2S) {
    const unsigned int idx = blk[2 + 4 * ib32 + l] | (((unsigned int)blk[66 + ib32] << (8 - 2 * l)) & 0x300);
    const unsigned char sc = blk[74 + ib32];
    db = d * (0.5f + (float)(l < 2 ? (sc & 0x0f) : (sc >> 4))) * 0.25f;
    grid = fcuda_iq2s_grid[idx];
    signs = blk[34 + 4 * ib32 + l];
  } else if (FMT == INK2_IQ3XXS) {
    const unsigned int aux32 = fcuda_iq_u32_at(blk + 66 + 4 * ib32);
    db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    signs = fcuda_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
    grid = (unsigned long long)fcuda_iq3xxs_grid[blk[2 + 8 * ib32 + 2 * l]] |
           ((unsigned long long)fcuda_iq3xxs_grid[blk[2 + 8 * ib32 + 2 * l + 1]] << 32);
  } else {
    const unsigned int high = blk[66 + ib32];
    const unsigned int idx1 = blk[2 + 8 * ib32 + 2 * l] | ((high << (8 - 2 * l)) & 0x100);
    const unsigned int idx2 = blk[2 + 8 * ib32 + 2 * l + 1] | ((high << (7 - 2 * l)) & 0x100);
    signs = blk[74 + 4 * ib32 + l];
    const unsigned int nibble = (blk[106 + ib32 / 2] >> (4 * (ib32 & 1))) & 0x0f;
    db = d * (float)(1 + 2 * (int)nibble);
    grid = (unsigned long long)fcuda_iq3s_grid[idx1] | ((unsigned long long)fcuda_iq3s_grid[idx2] << 32);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float v = (float)((grid >> (8 * j)) & 0xff);
    w[j] = db * ((signs & (1u << j)) ? -v : v);
  }
  o0 = lane * 8;
  o1 = o0 + 4;
}

template <int FMT>
__global__ void __launch_bounds__(INK2P_IQ_THREADS) k_ink2p_expert_iq(const unsigned char *W, const int *offsets,
                                                                     const int *order, const int *tiles,
                                                                     const int *ntiles, const float *X, int xDiv,
                                                                     int xStride, float *Y, int rows, int in,
                                                                     const float *escale) {
  constexpr int SB = FMT == INK2_IQ2XS ? 74 : FMT == INK2_IQ2S ? 82 : FMT == INK2_IQ3XXS ? 98 : (FMT == INK2_IQ3S || FMT == INK2_Q3K) ? 110 : 144;
  __shared__ __align__(16) float xs[INK2P_IQ_HALF][256];
  __shared__ int pairs[INK2P_NV_PAIRS];
  if ((int)blockIdx.y >= ntiles[0]) return;
  const int e = tiles[2 * blockIdx.y], start = tiles[2 * blockIdx.y + 1];
  const int np = min(INK2P_NV_PAIRS, offsets[e + 1] - start);
  if (threadIdx.x < INK2P_NV_PAIRS) pairs[threadIdx.x] = threadIdx.x < np ? order[start + threadIdx.x] : -1;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int o0 = (blockIdx.x * (blockDim.x >> 5) + warp) * INK2P_IQ_RPW;
  const size_t ebase = (size_t)e * rows;
  const int nsb = in / 256;
  const float es = escale ? escale[e] : 1.f;
  // the tile runs in halves so a block of INK2P_IQ_THREADS threads keeps its
  // accumulators in registers while staging each activation once per row block
  for (int h0 = 0; h0 < np; h0 += INK2P_IQ_HALF) {
    const int nh = min(INK2P_IQ_HALF, np - h0);
    float sum[INK2P_IQ_RPW][INK2P_IQ_HALF];
#pragma unroll
    for (int r = 0; r < INK2P_IQ_RPW; ++r)
#pragma unroll
      for (int p = 0; p < INK2P_IQ_HALF; ++p) sum[r][p] = 0.f;
    for (int sb = 0; sb < nsb; ++sb) {
      __syncthreads();
      for (int idx = threadIdx.x; idx < nh * 256; idx += blockDim.x) {
        const int p = idx >> 8, i = idx & 255;
        xs[p][i] = X[(size_t)(pairs[h0 + p] / xDiv) * xStride + sb * 256 + i];
      }
      __syncthreads();
      // every row of the warp decodes first so each staged activation is read once
      float w[INK2P_IQ_RPW][8];
      int x0 = 0, x1 = 0;
#pragma unroll
      for (int r = 0; r < INK2P_IQ_RPW; ++r) {
        const int o = min(o0 + r, rows - 1);
        ink2p_iq_decode<FMT>(W + ((ebase + o) * nsb + sb) * SB, lane, w[r], x0, x1);
      }
#pragma unroll
      for (int p = 0; p < INK2P_IQ_HALF; ++p) {
        if (p >= nh) break;
        const float4 a = *(const float4 *)&xs[p][x0];
        const float4 b = *(const float4 *)&xs[p][x1];
#pragma unroll
        for (int r = 0; r < INK2P_IQ_RPW; ++r) {
          float part = sum[r][p];
          part = fmaf(w[r][0], a.x, part);
          part = fmaf(w[r][1], a.y, part);
          part = fmaf(w[r][2], a.z, part);
          part = fmaf(w[r][3], a.w, part);
          part = fmaf(w[r][4], b.x, part);
          part = fmaf(w[r][5], b.y, part);
          part = fmaf(w[r][6], b.z, part);
          part = fmaf(w[r][7], b.w, part);
          sum[r][p] = part;
        }
      }
    }
#pragma unroll
    for (int r = 0; r < INK2P_IQ_RPW; ++r) {
#pragma unroll
      for (int p = 0; p < INK2P_IQ_HALF; ++p) {
        if (p >= nh) break;
        float v = sum[r][p];
        for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
        const int o = o0 + r;
        if (lane == 0 && o < rows) Y[(size_t)pairs[h0 + p] * rows + o] = escale ? __fmul_rn(v, es) : v;
      }
    }
  }
}

static int ink2p_smem_optin(const void *fn, size_t bytes) {
  if (bytes + INK2_SMEM_STATIC_ALLOWANCE <= 48 * 1024) return 0;
  return (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
}

extern "C" int fcuda_ink2p_mix(const float *const *states, int ns, const float *queries, int nq, float *out, int n,
                               int T) {
  if (ns < 1 || ns > INK2_MAX_STATES || nq < 1 || nq > INK2_MAX_QUERIES || T < 1) return (int)cudaErrorInvalidValue;
  Ink2States st;
  for (int i = 0; i < INK2_MAX_STATES; ++i) st.p[i] = i < ns ? states[i] : nullptr;
  switch (nq) {
  case 1: FCUDA_TS("k_ink2p_mix", T, k_ink2p_mix<1><<<T, INK2_MIX_THREADS, 0, g_stream>>>(st, ns, queries, out, n, T)); break;
  case 2: FCUDA_TS("k_ink2p_mix", T, k_ink2p_mix<2><<<T, INK2_MIX_THREADS, 0, g_stream>>>(st, ns, queries, out, n, T)); break;
  case 3: FCUDA_TS("k_ink2p_mix", T, k_ink2p_mix<3><<<T, INK2_MIX_THREADS, 0, g_stream>>>(st, ns, queries, out, n, T)); break;
  default: FCUDA_TS("k_ink2p_mix", T, k_ink2p_mix<4><<<T, INK2_MIX_THREADS, 0, g_stream>>>(st, ns, queries, out, n, T)); break;
  }
  return ink2_last_error();
}

extern "C" int fcuda_ink2p_gemm_f32(const float *W, const float *X, float *Y, int rows, int cols, int T) {
  const dim3 grid(rows, (T + INK2P_GEMM_TOK - 1) / INK2P_GEMM_TOK);
  FCUDA_TS("k_ink2p_gemm_f32", grid, k_ink2p_gemm_f32<<<grid, 32 * INK2P_GEMM_TOK, 0, g_stream>>>(W, X, Y, rows, cols, T));
  return ink2_last_error();
}

extern "C" int fcuda_ink2p_ctrl_down(const float *cur, const float *msg, const float *W, float *g, int n, int rank,
                                     int T) {
  Ink2PCtrl cs = {};
  cs.cur[0] = cur;
  cs.msg[0] = msg;
  cs.W[0] = W;
  cs.g[0] = g;
  FCUDA_TS("k_ink2p_ctrl_down", dim3(rank, T), k_ink2p_ctrl_down<<<dim3(rank, T), 256, 0, g_stream>>>(cs, n, rank));
  return ink2_last_error();
}

extern "C" int fcuda_ink2p_ctrl_apply(const float *cur, const float *msg, const float *g, const float *gainW,
                                      const float *forgetW, float *out, int n, int rank, int T) {
  const dim3 grid((n + 31) / 32, (T + INK2P_CTRL_TOK - 1) / INK2P_CTRL_TOK);
  const size_t smem = ((size_t)rank * 66 + (size_t)INK2P_CTRL_TOK * rank) * sizeof(float);
  const int oe = ink2p_smem_optin((const void *)k_ink2p_ctrl_apply, smem);
  if (oe) return oe;
  Ink2PCtrl cs = {};
  cs.cur[0] = cur;
  cs.msg[0] = msg;
  cs.g[0] = (float *)g;
  cs.gain[0] = gainW;
  cs.forget[0] = forgetW;
  cs.out[0] = out;
  FCUDA_TS("k_ink2p_ctrl_apply", grid, k_ink2p_ctrl_apply<<<grid, 256, smem, g_stream>>>(cs, n, rank, T));
  return ink2_last_error();
}

// fcuda_ink2p_ctrl_down then fcuda_ink2p_ctrl_apply for J <= 4 controllers in one launch
// each; arrays are indexed by controller.
extern "C" int fcuda_ink2p_ctrl_multi(const float *const *cur, const float *const *msg, const float *const *W,
                                      const float *const *gainW, const float *const *forgetW, float *const *g,
                                      float *const *out, int J, int n, int rank, int T) {
  if (J < 1 || J > 4) return (int)cudaErrorInvalidValue;
  Ink2PCtrl cs = {};
  for (int j = 0; j < J; ++j) {
    cs.cur[j] = cur[j];
    cs.msg[j] = msg[j];
    cs.W[j] = W[j];
    cs.gain[j] = gainW[j];
    cs.forget[j] = forgetW[j];
    cs.g[j] = g[j];
    cs.out[j] = out[j];
  }
  FCUDA_TS("k_ink2p_ctrl_down", dim3(rank, T, J), k_ink2p_ctrl_down<<<dim3(rank, T, J), 256, 0, g_stream>>>(cs, n, rank));
  const dim3 grid((n + 31) / 32, (T + INK2P_CTRL_TOK - 1) / INK2P_CTRL_TOK, J);
  const size_t smem = ((size_t)rank * 66 + (size_t)INK2P_CTRL_TOK * rank) * sizeof(float);
  const int oe = ink2p_smem_optin((const void *)k_ink2p_ctrl_apply, smem);
  if (oe) return oe;
  FCUDA_TS("k_ink2p_ctrl_apply", grid, k_ink2p_ctrl_apply<<<grid, 256, smem, g_stream>>>(cs, n, rank, T));
  return ink2_last_error();
}

extern "C" int fcuda_ink2p_conv(const float *x, const float *w, const float *dyn, float *hist, float *out,
                                const float *resid, int n, int heads, int kn, float norm, int T) {
  if (kn < 1 || kn > INK2P_MAX_KN) return (int)cudaErrorInvalidValue;
  // one thread walks a channel along the whole chunk, so small blocks spread it over every SM
  FCUDA_TS("k_ink2p_conv", (n + 31) / 32, k_ink2p_conv<<<(n + 31) / 32, 32, 0, g_stream>>>(
      x, w, dyn, hist, out, resid, n, heads, kn, norm, T, nullptr, nullptr, nullptr, 0));
  return ink2_last_error();
}

// fcuda_ink2p_conv over a verify chunk: log is [n][kn-1+T] for this conv's channels.
extern "C" int fcuda_ink2p_conv_log(const float *x, const float *w, const float *dyn, float *hist, float *log,
                                    float *out, const float *resid, int n, int heads, int kn, float norm, int T) {
  if (kn < 1 || kn > INK2P_MAX_KN) return (int)cudaErrorInvalidValue;
  FCUDA_TS("k_ink2p_conv", (n + 31) / 32, k_ink2p_conv<<<(n + 31) / 32, 32, 0, g_stream>>>(
      x, w, dyn, hist, out, resid, n, heads, kn, norm, T, log, nullptr, nullptr, 0));
  return ink2_last_error();
}

// fcuda_ink2p_conv_log (no dynamic taps) on the MoE combine of down ([T*k][n]) by wts.
extern "C" int fcuda_ink2p_combine_conv_log(const float *down, const float *wts, int k, const float *w, float *hist,
                                            float *log, float *out, const float *resid, int n, int kn, float norm,
                                            int T) {
  if (kn < 1 || kn > INK2P_MAX_KN) return (int)cudaErrorInvalidValue;
  FCUDA_TS("k_ink2p_conv", (n + 31) / 32, k_ink2p_conv<<<(n + 31) / 32, 32, 0, g_stream>>>(
      nullptr, w, nullptr, hist, out, resid, n, 1, kn, norm, T, log, down, wts, k));
  return ink2_last_error();
}

// The history after the first keep tokens of a logged chunk: log[c][keep .. keep+past).
__global__ void k_ink2p_conv_commit(float *hist, const float *log, int channels, int past, int L, int keep,
                                    const int *keepDev) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= channels * past) return;
  if (keepDev) keep = *keepDev;
  const int c = i / past, j = i % past;
  hist[i] = log[(size_t)c * L + keep + j];
}

extern "C" int fcuda_ink2p_conv_commit(float *hist, const float *log, int channels, int past, int L, int keep) {
  if (past < 1 || keep < 0 || keep + past > L) return (int)cudaErrorInvalidValue;
  const int n = channels * past;
  FCUDA_TS("k_ink2p_conv_commit", (n + 255) / 256,
           k_ink2p_conv_commit<<<(n + 255) / 256, 256, 0, g_stream>>>(hist, log, channels, past, L, keep, nullptr));
  return ink2_last_error();
}

// fcuda_ink2p_conv_commit with keep read from device memory (0 <= keep <= L-past).
extern "C" int fcuda_ink2p_conv_commit_dev(float *hist, const float *log, int channels, int past, int L,
                                           const int *keepDev) {
  if (past < 1) return (int)cudaErrorInvalidValue;
  const int n = channels * past;
  FCUDA_TS("k_ink2p_conv_commit", (n + 255) / 256,
           k_ink2p_conv_commit<<<(n + 255) / 256, 256, 0, g_stream>>>(hist, log, channels, past, L, 0, keepDev));
  return ink2_last_error();
}

extern "C" int fcuda_ink2p_qkv_post(float *q, float *r, const float *k, const float *v, const float *qclip,
                                    const float *rclip, float *kcache, float *vcache, int heads, int kvHeads,
                                    int headDim, int relLen, float qkClip, float relClip, float eps, int global,
                                    float alpha, float floorTok, int cap, int pos0, int T) {
  if (headDim > 256) return (int)cudaErrorInvalidValue;
  const dim3 grid(heads + 2 * kvHeads, T);
  FCUDA_TS("k_ink2p_qkv_post", grid, k_ink2p_qkv_post<float><<<grid, 32, 0, g_stream>>>(
      q, r, k, v, qclip, rclip, kcache, vcache, heads, kvHeads, headDim, relLen, qkClip, relClip, eps, global, alpha,
      floorTok, cap, pos0, nullptr));
  return ink2_last_error();
}

// fcuda_ink2p_qkv_post into f16 caches (cap * kvDim halves per row).
extern "C" int fcuda_ink2p_qkv_post_h(float *q, float *r, const float *k, const float *v, const float *qclip,
                                      const float *rclip, void *kcache, void *vcache, int heads, int kvHeads,
                                      int headDim, int relLen, float qkClip, float relClip, float eps, int global,
                                      float alpha, float floorTok, int cap, int pos0, int T) {
  if (headDim > 32 * INK2_KV_REGS) return (int)cudaErrorInvalidValue;
  const dim3 grid(heads + 2 * kvHeads, T);
  FCUDA_TS("k_ink2p_qkv_post", grid, k_ink2p_qkv_post<__half><<<grid, 32, 0, g_stream>>>(
      q, r, k, v, qclip, rclip, (__half *)kcache, (__half *)vcache, heads, kvHeads, headDim, relLen, qkClip, relClip,
      eps, global, alpha, floorTok, cap, pos0, nullptr));
  return ink2_last_error();
}

// fcuda_ink2p_qkv_post into OCP E4M3FN caches with per-(token, kvHead) F32 scales.
extern "C" int fcuda_ink2p_qkv_post_fp8(float *q, float *r, const float *k, const float *v, const float *qclip,
                                        const float *rclip, uint8_t *kcache, uint8_t *vcache, float *kscale,
                                        float *vscale, int *overflow, int heads, int kvHeads, int headDim, int relLen,
                                        float qkClip, float relClip, float eps, int global, float alpha, float floorTok,
                                        int cap, int pos0, int T) {
  if (headDim < 1 || headDim > 32 * INK2_KV_REGS || kvHeads < 1 || heads < 1 || cap < 1 || T < 1)
    return (int)cudaErrorInvalidValue;
  const dim3 grid(heads + 2 * kvHeads, T);
  FCUDA_TS("k_ink2p_qkv_post", grid, k_ink2p_qkv_post<uint8_t><<<grid, 32, 0, g_stream>>>(
      q, r, k, v, qclip, rclip, kcache, vcache, heads, kvHeads, headDim, relLen, qkClip, relClip, eps, global, alpha,
      floorTok, cap, pos0, nullptr, kscale, vscale, overflow));
  return ink2_last_error();
}

// fcuda_ink2p_qkv_post into block-16 packed E2M1 caches with E4M3FN block scales.
extern "C" int fcuda_ink2p_qkv_post_nvfp4(float *q, float *r, const float *k, const float *v, const float *qclip,
                                          const float *rclip, uint8_t *kcache, uint8_t *vcache, uint8_t *kscale,
                                          uint8_t *vscale, int *overflow, int heads, int kvHeads, int headDim,
                                          int relLen, float qkClip, float relClip, float eps, int global, float alpha,
                                          float floorTok, int cap, int pos0, int T) {
  if (headDim < 16 || (headDim & 15) != 0 || headDim > 32 * INK2_KV_REGS || kvHeads < 1 || heads < 1 || cap < 1 ||
      T < 1)
    return (int)cudaErrorInvalidValue;
  const dim3 grid(heads + 2 * kvHeads, T);
  FCUDA_TS("k_ink2p_qkv_post", grid, (k_ink2p_qkv_post<uint8_t, true><<<grid, 32, 0, g_stream>>>(
      q, r, k, v, qclip, rclip, kcache, vcache, heads, kvHeads, headDim, relLen, qkClip, relClip, eps, global, alpha,
      floorTok, cap, pos0, nullptr, kscale, vscale, overflow)));
  return ink2_last_error();
}

// fcuda_ink2p_qkv_post_asym writes block-16 FP8 K + block-16 NVFP4 V with E4M3FN block scales.
extern "C" int fcuda_ink2p_qkv_post_asym(float *q, float *r, const float *k, const float *v, const float *qclip,
                                         const float *rclip, uint8_t *kcache, uint8_t *vcache, uint8_t *kscale,
                                         uint8_t *vscale, int *overflow, int heads, int kvHeads, int headDim,
                                         int relLen, float qkClip, float relClip, float eps, int global, float alpha,
                                         float floorTok, int cap, int pos0, int T) {
  if (headDim < 16 || (headDim & 15) != 0 || headDim > 32 * INK2_KV_REGS || kvHeads < 1 || heads < 1 || cap < 1 ||
      T < 1)
    return (int)cudaErrorInvalidValue;
  const dim3 grid(heads + 2 * kvHeads, T);
  FCUDA_TS("k_ink2p_qkv_post", grid, (k_ink2p_qkv_post<uint8_t, false, true><<<grid, 32, 0, g_stream>>>(
      q, r, k, v, qclip, rclip, kcache, vcache, heads, kvHeads, headDim, relLen, qkClip, relClip, eps, global, alpha,
      floorTok, cap, pos0, nullptr, kscale, vscale, overflow)));
  return ink2_last_error();
}

// fcuda_ink2p_qkv_post with the chunk position read from device memory, so a captured
// verify graph replays at any position.
extern "C" int fcuda_ink2p_qkv_post_dev(float *q, float *r, const float *k, const float *v, const float *qclip,
                                        const float *rclip, float *kcache, float *vcache, int heads, int kvHeads,
                                        int headDim, int relLen, float qkClip, float relClip, float eps, int global,
                                        float alpha, float floorTok, int cap, const int *posDev, int T) {
  if (headDim > 256) return (int)cudaErrorInvalidValue;
  const dim3 grid(heads + 2 * kvHeads, T);
  FCUDA_TS("k_ink2p_qkv_post", grid, k_ink2p_qkv_post<float><<<grid, 32, 0, g_stream>>>(
      q, r, k, v, qclip, rclip, kcache, vcache, heads, kvHeads, headDim, relLen, qkClip, relClip, eps, global, alpha,
      floorTok, cap, 0, posDev));
  return ink2_last_error();
}

// fcuda_ink2p_qkv_post_h_dev is fcuda_ink2p_qkv_post_dev into f16 caches.
extern "C" int fcuda_ink2p_qkv_post_h_dev(float *q, float *r, const float *k, const float *v, const float *qclip,
                                          const float *rclip, void *kcache, void *vcache, int heads, int kvHeads,
                                          int headDim, int relLen, float qkClip, float relClip, float eps, int global,
                                          float alpha, float floorTok, int cap, const int *posDev, int T) {
  if (headDim > 32 * INK2_KV_REGS) return (int)cudaErrorInvalidValue;
  const dim3 grid(heads + 2 * kvHeads, T);
  FCUDA_TS("k_ink2p_qkv_post", grid, k_ink2p_qkv_post<__half><<<grid, 32, 0, g_stream>>>(
      q, r, k, v, qclip, rclip, (__half *)kcache, (__half *)vcache, heads, kvHeads, headDim, relLen, qkClip, relClip,
      eps, global, alpha, floorTok, cap, 0, posDev));
  return ink2_last_error();
}

// fcuda_ink2p_qkv_post_fp8_dev is fcuda_ink2p_qkv_post_dev into OCP E4M3FN caches with per-(token, kvHead) F32 scales.
extern "C" int fcuda_ink2p_qkv_post_fp8_dev(float *q, float *r, const float *k, const float *v, const float *qclip,
                                            const float *rclip, uint8_t *kcache, uint8_t *vcache, float *kscale,
                                            float *vscale, int *overflow, int heads, int kvHeads, int headDim,
                                            int relLen, float qkClip, float relClip, float eps, int global,
                                            float alpha, float floorTok, int cap, const int *posDev, int T) {
  if (headDim < 1 || headDim > 32 * INK2_KV_REGS || kvHeads < 1 || heads < 1 || cap < 1 || T < 1)
    return (int)cudaErrorInvalidValue;
  const dim3 grid(heads + 2 * kvHeads, T);
  FCUDA_TS("k_ink2p_qkv_post", grid, k_ink2p_qkv_post<uint8_t><<<grid, 32, 0, g_stream>>>(
      q, r, k, v, qclip, rclip, kcache, vcache, heads, kvHeads, headDim, relLen, qkClip, relClip, eps, global, alpha,
      floorTok, cap, 0, posDev, kscale, vscale, overflow));
  return ink2_last_error();
}

// fcuda_ink2p_qkv_post_nvfp4_dev is fcuda_ink2p_qkv_post_dev into block-16 packed E2M1 caches with E4M3FN block scales.
extern "C" int fcuda_ink2p_qkv_post_nvfp4_dev(float *q, float *r, const float *k, const float *v, const float *qclip,
                                              const float *rclip, uint8_t *kcache, uint8_t *vcache, uint8_t *kscale,
                                              uint8_t *vscale, int *overflow, int heads, int kvHeads, int headDim,
                                              int relLen, float qkClip, float relClip, float eps, int global,
                                              float alpha, float floorTok, int cap, const int *posDev, int T) {
  if (headDim < 16 || (headDim & 15) != 0 || headDim > 32 * INK2_KV_REGS || kvHeads < 1 || heads < 1 || cap < 1 ||
      T < 1)
    return (int)cudaErrorInvalidValue;
  const dim3 grid(heads + 2 * kvHeads, T);
  FCUDA_TS("k_ink2p_qkv_post", grid, (k_ink2p_qkv_post<uint8_t, true><<<grid, 32, 0, g_stream>>>(
      q, r, k, v, qclip, rclip, kcache, vcache, heads, kvHeads, headDim, relLen, qkClip, relClip, eps, global, alpha,
      floorTok, cap, 0, posDev, kscale, vscale, overflow)));
  return ink2_last_error();
}

static float *g_ink2p_att_scratch = nullptr;
static size_t g_ink2p_att_scratch_cap = 0;            // floats
static size_t g_ink2p_att_budget = (size_t)64 << 20;  // bytes per global-path token tile
static int g_ink2p_att_global_mode = -1;
static long long g_ink2p_att_smem_limit = -1; // opt-in bytes minus the kernel's static smem

static bool ink2p_att_force_global(void) {
  if (g_ink2p_att_global_mode < 0) {
    const char *v = getenv("FAK_INKLING2_PREFILL_ATTEND_GLOBAL");
    g_ink2p_att_global_mode = (v && v[0] == '1' && v[1] == 0) ? 1 : 0;
  }
  return g_ink2p_att_global_mode == 1;
}

// Test hook: 1 forces the global score-row path, 0 picks by span; returns the previous mode.
extern "C" int fcuda_ink2p_attend_global_select(int mode) {
  const int prev = ink2p_att_force_global() ? 1 : 0;
  g_ink2p_att_global_mode = mode ? 1 : 0;
  return prev;
}

// Test hook: the global path's per-tile scratch budget in bytes (0 keeps it); returns the previous.
extern "C" unsigned long long fcuda_ink2p_attend_global_budget(unsigned long long bytes) {
  const unsigned long long prev = g_ink2p_att_budget;
  if (bytes) g_ink2p_att_budget = (size_t)bytes;
  return prev;
}

static int ink2p_att_smem_fits(size_t bytes, bool *fits) {
  if (g_ink2p_att_smem_limit < 0) {
    int dev = 0, optin = 0;
    cudaFuncAttributes fa;
    cudaError_t e = cudaGetDevice(&dev);
    if (e == cudaSuccess) e = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (e == cudaSuccess) e = cudaFuncGetAttributes(&fa, (const void *)k_ink2p_attend<false>);
    if (e != cudaSuccess) {
      (void)cudaGetLastError();
      return (int)e;
    }
    const long long lim = (long long)optin - (long long)fa.sharedSizeBytes;
    g_ink2p_att_smem_limit = lim > 0 ? lim : 0;
  }
  *fits = (long long)bytes <= g_ink2p_att_smem_limit;
  return 0;
}

// Growth is refused while a capture is open: a captured graph bakes the pointer.
static int ink2p_att_reserve(size_t need) {
  if (need <= g_ink2p_att_scratch_cap) return 0;
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  cudaError_t e = cudaStreamIsCapturing(g_stream, &capture);
  if (e != cudaSuccess) return (int)e;
  if (g_capture_open || capture != cudaStreamCaptureStatusNone) return (int)cudaErrorStreamCaptureUnsupported;
  const size_t floor = g_ink2p_att_budget / sizeof(float);
  const size_t want = need > floor ? need : floor;
  if (g_ink2p_att_scratch) {
    float *retiring = g_ink2p_att_scratch;
    g_ink2p_att_scratch = nullptr;
    g_ink2p_att_scratch_cap = 0;
    fak_alloc_untrack(retiring);
    e = cudaFree(retiring);
    if (e != cudaSuccess) return (int)e;
  }
  float *p = nullptr;
  e = fcuda_alloc_status(cudaMalloc(&p, want * sizeof(float)));
  if (e != cudaSuccess || !p) {
    (void)cudaGetLastError();
    return (int)(e != cudaSuccess ? e : cudaErrorMemoryAllocation);
  }
  fak_alloc_track(p, FAK_ALLOC_ATTENTION_SCRATCH, FAK_ALLOC_DEVICE_LOCAL, FAK_ALLOC_SITE("inkling2-prefill-attend"),
                  want * sizeof(float));
  g_ink2p_att_scratch = p;
  g_ink2p_att_scratch_cap = want;
  return 0;
}

// Shared-memory scores when span fits the opt-in (unless forced global), else global
// score rows launched over token tiles bounded by the scratch budget.
static int ink2p_attend_launch(const float *q, const float *r, const float *sinks, const float *relBias,
                               const float *kcache, const float *vcache, float *out, int heads, int kvHeads,
                               int headDim, int relLen, int extent, int window, int cap, float scale, int pos0,
                               const int *posDev, int T, int span) {
  if (headDim > 256 || INK2_ATT_THREADS / headDim > INK2_ATT_GROUPS) return (int)cudaErrorInvalidValue;
  if (span < 1) span = 1;
  const size_t smem = (size_t)span * sizeof(float);
  bool fits = false;
  if (!ink2p_att_force_global()) {
    const int fe = ink2p_att_smem_fits(smem, &fits);
    if (fe) return fe;
  }
  if (fits) {
    const int oe = ink2p_smem_optin((const void *)k_ink2p_attend<false>, smem);
    if (oe) {
      (void)cudaGetLastError();
      return oe;
    }
    const dim3 grid(heads, T);
    FCUDA_TS("k_ink2p_attend", grid, k_ink2p_attend<false><<<grid, INK2_ATT_THREADS, smem, g_stream>>>(
        q, r, sinks, relBias, kcache, vcache, out, heads, kvHeads, headDim, relLen, extent, window, cap, scale, pos0,
        posDev, nullptr, span, 0));
    return ink2_last_error();
  }
  const size_t row = (size_t)heads * (size_t)span;
  size_t tile = g_ink2p_att_budget / (row * sizeof(float));
  if (tile < 1) tile = 1;
  if (tile > (size_t)T) tile = (size_t)T;
  const int re = ink2p_att_reserve(row * tile);
  if (re) return re;
  for (int t0 = 0; t0 < T; t0 += (int)tile) {
    const int nt = T - t0 < (int)tile ? T - t0 : (int)tile;
    const dim3 grid(heads, nt);
    FCUDA_TS("k_ink2p_attend_global", grid, k_ink2p_attend<true><<<grid, INK2_ATT_THREADS, 0, g_stream>>>(
        q, r, sinks, relBias, kcache, vcache, out, heads, kvHeads, headDim, relLen, extent, window, cap, scale, pos0,
        posDev, g_ink2p_att_scratch, span, t0));
    const int le = ink2_last_error();
    if (le) return le;
  }
  return 0;
}

extern "C" int fcuda_ink2p_attend(const float *q, const float *r, const float *sinks, const float *relBias,
                                  const float *kcache, const float *vcache, float *out, int heads, int kvHeads,
                                  int headDim, int relLen, int extent, int window, int cap, float scale, int pos0,
                                  int T) {
  int span = pos0 + T;
  if (window > 0 && span > window) span = window;
  if (span > cap) span = cap;
  return ink2p_attend_launch(q, r, sinks, relBias, kcache, vcache, out, heads, kvHeads, headDim, relLen, extent,
                             window, cap, scale, pos0, nullptr, T, span);
}

// fcuda_ink2p_attend with the chunk position read from device memory; scores are
// sized for the longest span any position can see.
extern "C" int fcuda_ink2p_attend_dev(const float *q, const float *r, const float *sinks, const float *relBias,
                                      const float *kcache, const float *vcache, float *out, int heads, int kvHeads,
                                      int headDim, int relLen, int extent, int window, int cap, float scale,
                                      const int *posDev, int T) {
  int span = cap;
  if (window > 0 && span > window) span = window;
  return ink2p_attend_launch(q, r, sinks, relBias, kcache, vcache, out, heads, kvHeads, headDim, relLen, extent,
                             window, cap, scale, 0, posDev, T, span);
}

extern "C" int fcuda_ink2p_route(const float *logits, const float *bias, int *ids, float *wts, int nexp, int k,
                                 float wscale, int T) {
  if (nexp > INK2_MAX_EXPERTS || k > INK2_MAX_TOPK || k > nexp) return (int)cudaErrorInvalidValue;
  FCUDA_TS("k_ink2p_route", T, k_ink2p_route<<<T, 32, 0, g_stream>>>(logits, bias, ids, wts, nexp, k, wscale));
  return ink2_last_error();
}

extern "C" int fcuda_ink2p_combine(const float *down, const float *wts, float *out, int k, int n, int T) {
  const dim3 grid((n + 255) / 256, T);
  FCUDA_TS("k_ink2p_combine", grid, k_ink2p_combine<<<grid, 256, 0, g_stream>>>(down, wts, out, k, n));
  return ink2_last_error();
}

// Bucket the npairs routed expert ids by expert. Scratch: counts [nexp], offsets
// [nexp+1], order [npairs], tiles [2*(npairs/INK2P_NV_PAIRS + nexp)], ntiles [1].
extern "C" int fcuda_ink2p_bucket(const int *ids, int npairs, int nexp, int *counts, int *offsets, int *order,
                                  int *tiles, int *ntiles) {
  if (nexp > INK2_MAX_EXPERTS) return (int)cudaErrorInvalidValue;
  cudaError_t e = cudaSuccess;
  FCUDA_TS("fill", dim3(0, 0), e = cudaMemsetAsync(counts, 0, (size_t)nexp * sizeof(int), g_stream));
  if (e != cudaSuccess) return (int)e;
  FCUDA_TS("k_ink2p_bucket_count", (npairs + 255) / 256, k_ink2p_bucket_count<<<(npairs + 255) / 256, 256, 0, g_stream>>>(ids, counts, npairs));
  FCUDA_TS("k_ink2p_bucket_scan", 1, k_ink2p_bucket_scan<<<1, 32, 0, g_stream>>>(counts, offsets, tiles, ntiles, nexp));
  FCUDA_TS("k_ink2p_bucket_fill", nexp, k_ink2p_bucket_fill<<<nexp, 256, 0, g_stream>>>(ids, offsets, order, npairs));
  return ink2_last_error();
}

extern "C" int fcuda_ink2p_gather_pairs(const int *ids, const int *order, const float *X, int xDiv, int xStride,
                                        int *sortedIds, float *sortedX, int width, int npairs) {
  const dim3 grid((width + 255) / 256 < 8 ? (width + 255) / 256 : 8, npairs);
  FCUDA_TS("k_ink2p_gather_pairs", grid, k_ink2p_gather_pairs<<<grid, 256, 0, g_stream>>>(ids, order, X, xDiv, xStride, sortedIds, sortedX, width));
  return ink2_last_error();
}

extern "C" int fcuda_ink2p_scatter_pairs(const int *order, const float *sortedY, float *Y, int rows, int npairs) {
  const dim3 grid((rows + 255) / 256 < 8 ? (rows + 255) / 256 : 8, npairs);
  FCUDA_TS("k_ink2p_scatter_pairs", grid, k_ink2p_scatter_pairs<<<grid, 256, 0, g_stream>>>(order, sortedY, Y, rows));
  return ink2_last_error();
}

extern "C" int fcuda_ink2p_max_tiles(int npairs, int nexp) { return npairs / INK2P_NV_PAIRS + nexp; }

// Grouped NVFP4 expert GEMM over the bucketed pairs: pair i reads X row
// (i / xDiv) * xStride and writes Y row i.
extern "C" int fcuda_ink2p_expert_nvfp4(const uint8_t *W, const uint8_t *scales, const int *offsets, const int *order,
                                        const int *tiles, const int *ntiles, int maxTiles, const float *X, int xDiv,
                                        int xStride, float *Y, int rows, int in, const float *escale) {
  if (!scales || (in & 15) != 0 || (rows & 15) != 0) return (int)cudaErrorInvalidValue;
  const int rowsPerBlock = 8 * 16 * INK2P_NV_MT;
  const dim3 grid((rows + rowsPerBlock - 1) / rowsPerBlock, maxTiles);
  FCUDA_TS("k_ink2p_expert_nvfp4", grid, k_ink2p_expert_nvfp4<<<grid, 256, 0, g_stream>>>(
      W, scales, offsets, order, tiles, ntiles, X, xDiv, xStride, Y, rows, in, escale));
  return ink2_last_error();
}

// Grouped IQ/Q3_K expert GEMM over bucketed pairs; fmt as fcuda_ink2_expert_gemv
// (1..6). Returns cudaErrorInvalidValue for NVFP4 or an unknown format.
extern "C" int fcuda_ink2p_expert_iq(int fmt, const uint8_t *W, const int *offsets, const int *order, const int *tiles,
                                     const int *ntiles, int maxTiles, const float *X, int xDiv, int xStride, float *Y,
                                     int rows, int in, const float *escale) {
  if ((in & 255) != 0) return (int)cudaErrorInvalidValue;
  const int rowsPerBlock = (INK2P_IQ_THREADS / 32) * INK2P_IQ_RPW;
  const dim3 grid((rows + rowsPerBlock - 1) / rowsPerBlock, maxTiles);
#define INK2P_IQ_LAUNCH(F) FCUDA_TS("k_ink2p_expert_iq", grid, k_ink2p_expert_iq<F><<<grid, INK2P_IQ_THREADS, 0, g_stream>>>(W, offsets, order, tiles, ntiles, X, xDiv, xStride, Y, rows, in, escale))
  switch (fmt) {
  case INK2_IQ2XS: INK2P_IQ_LAUNCH(INK2_IQ2XS); break;
  case INK2_IQ2S: INK2P_IQ_LAUNCH(INK2_IQ2S); break;
  case INK2_IQ3XXS: INK2P_IQ_LAUNCH(INK2_IQ3XXS); break;
  case INK2_IQ3S: INK2P_IQ_LAUNCH(INK2_IQ3S); break;
  case INK2_IQ4NL: INK2P_IQ_LAUNCH(INK2_IQ4NL); break;
  case INK2_Q3K: INK2P_IQ_LAUNCH(INK2_Q3K); break;
  default: return (int)cudaErrorInvalidValue;
  }
#undef INK2P_IQ_LAUNCH
  return ink2_last_error();
}
