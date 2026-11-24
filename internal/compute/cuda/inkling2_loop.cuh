// inkling2_loop.cuh: device ops for the resident inkling2_loop decode forward.
//
// Each kernel mirrors one host step of internal/model/inkling2loop_forward.go. Where
// the host reduction is a sequential float32 (or float64) loop, the kernel keeps that
// order with explicit _rn intrinsics so the device token path stays as close to the
// host reference as the GEMV reduction order allows. Every kernel reads the token
// position from the device input row (ink2_pos), so a captured graph replays across
// tokens. Included last by cuda_kernels.cu: uses the IQ grids, fcuda_iq_dot8 and the
// NVFP4 decoders defined by earlier headers of the same translation unit.

#include <cooperative_groups.h>

#define INK2_MAX_STATES 8
#define INK2_MAX_QUERIES 4
#define INK2_MAX_TOPK 16

struct Ink2States {
  const float *p[INK2_MAX_STATES];
};

static __device__ __forceinline__ int ink2_pos(const float *meta) { return __float_as_int(meta[0]); }

static __device__ __forceinline__ float ink2_sigmoid(float x) {
  return (float)(1.0 / (1.0 + exp((double)(-x))));
}

static __device__ __forceinline__ double ink2_block_sum_d(double v, double *sh) {
  for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    double s = 0;
    for (int w = 0; w < (int)((blockDim.x + 31) / 32); ++w) s += sh[w];
    sh[32] = s;
  }
  __syncthreads();
  return sh[32];
}

// inkling2RMS over `rows` rows of width n: y = x * s (* w); float partials per thread,
// float64 across warps.
__global__ void __launch_bounds__(1024) k_ink2_rms_rows(const float *x, const float *w, float *y, int n, float eps) {
  __shared__ float part[32];
  __shared__ float sfac;
  const float *xr = x + (size_t)blockIdx.x * n;
  float *yr = y + (size_t)blockIdx.x * n;
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) ss = fmaf(xr[i], xr[i], ss);
  for (int d = 16; d > 0; d >>= 1) ss += __shfl_down_sync(0xffffffff, ss, d);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    double t = 0;
    for (int w = 0; w < (int)(blockDim.x / 32); ++w) t += (double)part[w];
    sfac = (float)(1.0 / sqrt(t / (double)n + (double)eps));
  }
  __syncthreads();
  const float s = sfac;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = __fmul_rn(xr[i], s);
    if (w) v = __fmul_rn(v, w[i]);
    yr[i] = v;
  }
}

// Two-input RMS (eps 1e-6, no weight) into the halves of a controller pair row.
__global__ void k_ink2_ctrl_pair(const float *cur, const float *msg, float *pair, int n) {
  __shared__ double sh[33];
  const float *x = blockIdx.x == 0 ? cur : msg;
  float *y = pair + (size_t)blockIdx.x * n;
  double ss = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) ss += (double)x[i] * (double)x[i];
  ss = ink2_block_sum_d(ss, sh);
  const float s = (float)(1.0 / sqrt(ss / (double)n + 1e-6));
  for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = __fmul_rn(x[i], s);
}

// inkling2Mix for nq queries over one state list: every state RMS-normalized (eps 1e-6),
// scored against each query, softmax in float64, float32 weighted sum in state order.
// One pass gathers every sum of squares and raw query dot (float per thread, float64
// across warps); the score is the normalized dot (scale * raw dot).
#define INK2_MIX_THREADS 1024
template <int NS, int NQ>
__device__ void ink2_mix_body(const Ink2States &st, const float *queries, float *out, int n) {
  constexpr int NV = NS * (1 + NQ);
  __shared__ float part[INK2_MIX_THREADS / 32][INK2_MAX_STATES * (1 + INK2_MAX_QUERIES)];
  __shared__ double tot[INK2_MAX_STATES * (1 + INK2_MAX_QUERIES)];
  __shared__ float scale[INK2_MAX_STATES];
  __shared__ float wts[INK2_MAX_QUERIES][INK2_MAX_STATES];
  float acc[NV];
#pragma unroll
  for (int v = 0; v < NV; ++v) acc[v] = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float qv[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) qv[q] = queries[(size_t)q * n + i];
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      const float x = st.p[s][i];
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
    for (int s = 0; s < NS; ++s) xs[s] = __fmul_rn(st.p[s][i], scale[s]);
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
      float a = 0.f;
#pragma unroll
      for (int s = 0; s < NS; ++s) a = __fadd_rn(a, __fmul_rn(wts[q][s], xs[s]));
      out[(size_t)q * n + i] = a;
    }
  }
}

template <int NQ>
__global__ void __launch_bounds__(INK2_MIX_THREADS) k_ink2_mix(Ink2States st, int ns, const float *queries, float *out, int n) {
  switch (ns) {
  case 1: ink2_mix_body<1, NQ>(st, queries, out, n); break;
  case 2: ink2_mix_body<2, NQ>(st, queries, out, n); break;
  case 3: ink2_mix_body<3, NQ>(st, queries, out, n); break;
  case 4: ink2_mix_body<4, NQ>(st, queries, out, n); break;
  case 5: ink2_mix_body<5, NQ>(st, queries, out, n); break;
  case 6: ink2_mix_body<6, NQ>(st, queries, out, n); break;
  case 7: ink2_mix_body<7, NQ>(st, queries, out, n); break;
  default: ink2_mix_body<8, NQ>(st, queries, out, n); break;
  }
}

// Cluster form of the depth mix: INK2_MIX_CTAS blocks each own one slice of the hidden
// dimension (one element per thread) and meet over distributed shared memory, so the
// reductions use eight SMs where k_ink2_mix uses one. Scores and weights are computed
// one (query, state) pair per lane, which keeps the float64 softmax off a single lane.
// With norms set, each query row is RMS-normalized (weight norms[q]) before it is
// written, which removes the RMS launch that follows a mix with no controller between.
// Adapted from the single-pass MIX of realugbun/llama.cpp ggml/src/ggml-cuda/inkling2.cu:931-1073@2eb2530ec4 (MIT).
#define INK2_MIX_CTAS 8
struct Ink2Norms {
  const float *w[INK2_MAX_QUERIES];
};

struct Ink2MixShared {
  float part[32][INK2_MAX_STATES * (1 + INK2_MAX_QUERIES)];
  double blk[INK2_MAX_STATES * (1 + INK2_MAX_QUERIES)];
  double tot[INK2_MAX_STATES * (1 + INK2_MAX_QUERIES)];
  float scale[INK2_MAX_STATES];
  double esc[INK2_MAX_QUERIES][INK2_MAX_STATES];
  float wts[INK2_MAX_QUERIES][INK2_MAX_STATES];
  double nblk[INK2_MAX_QUERIES];
  float nfac[INK2_MAX_QUERIES];
};

template <int NS, int NQ>
__device__ void ink2_mixc_body(Ink2MixShared &sm, const Ink2States &st, const float *queries, const Ink2Norms &nw,
                               int norm, float eps, float *out, int n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  constexpr int NV = NS * (1 + NQ);
  auto &part = sm.part;
  auto &blk = sm.blk;
  auto &tot = sm.tot;
  auto &scale = sm.scale;
  auto &esc = sm.esc;
  auto &wts = sm.wts;
  auto &nblk = sm.nblk;
  auto &nfac = sm.nfac;
  const unsigned rank = cluster.block_rank();
  const int i = (int)rank * blockDim.x + threadIdx.x;
  const bool live = i < n;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = (blockDim.x + 31) >> 5;
  float xs[NS], acc[NV];
#pragma unroll
  for (int s = 0; s < NS; ++s) xs[s] = live ? st.p[s][i] : 0.f;
  {
    float qv[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) qv[q] = live ? queries[(size_t)q * n + i] : 0.f;
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      acc[s] = xs[s] * xs[s];
#pragma unroll
      for (int q = 0; q < NQ; ++q) acc[NS + s * NQ + q] = xs[s] * qv[q];
    }
  }
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    float x = acc[v];
    for (int d = 16; d > 0; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
    if (lane == 0) part[warp][v] = x;
  }
  __syncthreads();
  if (threadIdx.x < NV) {
    double t = 0;
    for (int w = 0; w < nwarp; ++w) t += (double)part[w][threadIdx.x];
    blk[threadIdx.x] = t;
  }
  cluster.sync();
  if (threadIdx.x < NV) {
    double t = 0;
    for (unsigned r = 0; r < cluster.num_blocks(); ++r) t += cluster.map_shared_rank(blk, r)[threadIdx.x];
    tot[threadIdx.x] = t;
  }
  __syncthreads();
  if (threadIdx.x < NS) scale[threadIdx.x] = (float)(1.0 / sqrt(tot[threadIdx.x] / (double)n + 1e-6));
  __syncthreads();
  if (threadIdx.x < 32) {
    const int q = threadIdx.x / NS, s = threadIdx.x % NS;
    const bool on = threadIdx.x < NS * NQ;
    const double inv = 1.0 / sqrt((double)n);
    const double sc = on ? (double)(float)((double)scale[s] * tot[NS + s * NQ + q]) * inv : 0.0;
    if (on) esc[q][s] = sc;
    __syncwarp();
    double mx = -INFINITY;
    if (on)
      for (int t = 0; t < NS; ++t) mx = fmax(mx, esc[q][t]);
    __syncwarp();
    if (on) esc[q][s] = exp(sc - mx);
    __syncwarp();
    if (on) {
      double sum = 0;
      for (int t = 0; t < NS; ++t) sum += esc[q][t];
      wts[q][s] = (float)(esc[q][s] / sum);
    }
  }
  __syncthreads();
  float o[NQ];
#pragma unroll
  for (int s = 0; s < NS; ++s) xs[s] = __fmul_rn(xs[s], scale[s]);
#pragma unroll
  for (int q = 0; q < NQ; ++q) {
    float a = 0.f;
#pragma unroll
    for (int s = 0; s < NS; ++s) a = __fadd_rn(a, __fmul_rn(wts[q][s], xs[s]));
    o[q] = a;
  }
  if (norm) {
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
      float x = o[q] * o[q];
      for (int d = 16; d > 0; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
      if (lane == 0) part[warp][q] = x;
    }
    __syncthreads();
    if (threadIdx.x < NQ) {
      double t = 0;
      for (int w = 0; w < nwarp; ++w) t += (double)part[w][threadIdx.x];
      nblk[threadIdx.x] = t;
    }
    cluster.sync();
    if (threadIdx.x < NQ) {
      double t = 0;
      for (unsigned r = 0; r < cluster.num_blocks(); ++r) t += cluster.map_shared_rank(nblk, r)[threadIdx.x];
      nfac[threadIdx.x] = (float)(1.0 / sqrt(t / (double)n + (double)eps));
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
      float v = __fmul_rn(o[q], nfac[q]);
      if (nw.w[q] && live) v = __fmul_rn(v, nw.w[q][i]);
      o[q] = v;
    }
  }
  if (live)
#pragma unroll
    for (int q = 0; q < NQ; ++q) out[(size_t)q * n + i] = o[q];
  // a block's shared memory must outlive every remote read of it
  cluster.sync();
#endif
}

template <int NQ>
__global__ void __launch_bounds__(1024)
    k_ink2_mixc(Ink2States st, int ns, const float *queries, Ink2Norms nw, int norm, float eps, float *out, int n) {
  __shared__ Ink2MixShared sm;
  switch (ns) {
  case 1: ink2_mixc_body<1, NQ>(sm, st, queries, nw, norm, eps, out, n); break;
  case 2: ink2_mixc_body<2, NQ>(sm, st, queries, nw, norm, eps, out, n); break;
  case 3: ink2_mixc_body<3, NQ>(sm, st, queries, nw, norm, eps, out, n); break;
  case 4: ink2_mixc_body<4, NQ>(sm, st, queries, nw, norm, eps, out, n); break;
  case 5: ink2_mixc_body<5, NQ>(sm, st, queries, nw, norm, eps, out, n); break;
  case 6: ink2_mixc_body<6, NQ>(sm, st, queries, nw, norm, eps, out, n); break;
  case 7: ink2_mixc_body<7, NQ>(sm, st, queries, nw, norm, eps, out, n); break;
  default: ink2_mixc_body<8, NQ>(sm, st, queries, nw, norm, eps, out, n); break;
  }
}

// Block-per-row F32 GEMV: y[r] = W[r,:] . x for the small dense matrices (router,
// controllers, sink and conv projections), which have few rows and long rows.
__global__ void k_ink2_gemv_f32(const float *W, const float *x, float *y, int rows, int cols) {
  __shared__ float part[8];
  const int r = blockIdx.x;
  const float *wr = W + (size_t)r * cols;
  float s = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) s = fmaf(wr[i], x[i], s);
  for (int d = 16; d > 0; d >>= 1) s += __shfl_down_sync(0xffffffff, s, d);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int w = 0; w < (int)(blockDim.x / 32); ++w) t += part[w];
    y[r] = t;
  }
}

// inkling2Control head: g[r] = Down[r,:] . [rms(cur), rms(msg)] with both RMS scales
// (eps 1e-6) folded out of the dot; one block per rank row.
__global__ void k_ink2_ctrl_down(const float *cur, const float *msg, const float *W, float *g, int n) {
  __shared__ float part[4][32];
  const int r = blockIdx.x;
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
    g[r] = (float)((double)sc * t[2] + (double)sm * t[3]);
  }
}

// inkling2Control tail: gain/forget rank-`rank` GEMVs (warp per output), then
// out = cur - tanh(f)*cur + tanh(g)*msg.
__global__ void k_ink2_ctrl_apply(const float *cur, const float *msg, const float *g, const float *gainW,
                                  const float *forgetW, float *out, int n, int rank) {
  const int lane = threadIdx.x & 31;
  const int i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (i >= n) return;
  const float *gw = gainW + (size_t)i * rank, *fw = forgetW + (size_t)i * rank;
  float gn = 0.f, fg = 0.f;
  for (int r = lane; r < rank; r += 32) {
    const float gv = g[r];
    gn = fmaf(gw[r], gv, gn);
    fg = fmaf(fw[r], gv, fg);
  }
  for (int d = 16; d > 0; d >>= 1) {
    gn += __shfl_down_sync(0xffffffff, gn, d);
    fg += __shfl_down_sync(0xffffffff, fg, d);
  }
  if (lane == 0) {
    const float f = (float)tanh((double)fg), t = (float)tanh((double)gn);
    out[i] = __fadd_rn(__fsub_rn(cur[i], __fmul_rn(f, cur[i])), __fmul_rn(t, msg[i]));
  }
}

// convolve: causal short conv with residual over one channel range of a slot's history.
// dyn (nullable) adds per-head input-dependent taps laid out [head][tap]; resid
// (nullable) is added to the result.
__global__ void k_ink2_conv(const float *x, const float *w, const float *dyn, float *hist, float *out,
                            const float *resid, int n, int heads, int kn, float norm) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  const int past = kn - 1;
  float *hc = hist + (size_t)c * past;
  const float *wc = w + (size_t)c * kn;
  const float xc = x[c];
  float acc = 0.f;
  for (int t = 0; t < past; ++t) acc = __fadd_rn(acc, __fmul_rn(hc[t], wc[t]));
  acc = __fadd_rn(acc, __fmul_rn(xc, wc[past]));
  if (dyn) {
    const int hd = c / (n / heads);
    for (int lag = 0; lag < kn; ++lag) {
      const float xv = lag > 0 ? hc[past - lag] : xc;
      acc = __fadd_rn(acc, __fmul_rn(dyn[hd * kn + lag], xv));
    }
  }
  float o = __fmul_rn(__fadd_rn(xc, acc), norm);
  for (int t = 0; t + 1 < past; ++t) hc[t] = hc[t + 1];
  hc[past - 1] = xc;
  if (resid) o = __fadd_rn(resid[c], o);
  out[c] = o;
}

static __device__ __forceinline__ float ink2_warp_sum(float v) {
  for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xffffffff, v, d);
  return v;
}

// inkling2Clip on one warp: x (scaled first when scale != null) shrunk to L2 <= thr.
static __device__ void ink2_clip_warp(float *x, const float *scale, float thr, int n) {
  const int lane = threadIdx.x & 31;
  float ss = 0.f;
  for (int i = lane; i < n; i += 32) {
    float v = x[i];
    if (scale) v = __fmul_rn(v, scale[i]);
    x[i] = v;
    ss = fmaf(v, v, ss);
  }
  ss = ink2_warp_sum(ss);
  const float norm = __fadd_rn((float)sqrt((double)ss), 1e-6f);
  float f = __fdiv_rn(thr, norm);
  if (f > 1.f) f = 1.f;
  if (f < 0.f) f = 0.f;
  __syncwarp();
  for (int i = lane; i < n; i += 32) x[i] = __fmul_rn(x[i], f);
}

// Post-projection attention prep for one layer visit: per-head q/r clips, k clip, v RMS,
// the global log-scale tau on q and r, and the append of k/v into the slot cache at the
// current position (ring index for local slots). Blocks: [0,heads) q+r heads, then
// kvHeads k heads, then kvHeads v heads; one warp per head.
__global__ void k_ink2_qkv_post(float *q, float *r, const float *k, const float *v, const float *qclip,
                                const float *rclip, float *kcache, float *vcache, const float *meta,
                                int heads, int kvHeads, int headDim, int relLen, float qkClip, float relClip,
                                float eps, int global, float alpha, float floorTok, int cap) {
  const int lane = threadIdx.x & 31;
  const int b = blockIdx.x;
  const int pos = ink2_pos(meta);
  const int idx = pos % cap;
  if (b < heads) {
    float *qh = q + (size_t)b * headDim;
    float *rh = r + (size_t)b * relLen;
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
  const int kvDim = kvHeads * headDim;
  if (b < heads + kvHeads) {
    const int h = b - heads;
    float *dst = kcache + (size_t)idx * kvDim + (size_t)h * headDim;
    for (int i = lane; i < headDim; i += 32) dst[i] = k[h * headDim + i];
    __syncwarp();
    ink2_clip_warp(dst, nullptr, qkClip, headDim);
    return;
  }
  const int h = b - heads - kvHeads;
  const float *vh = v + (size_t)h * headDim;
  double ss = 0;
  for (int i = lane; i < headDim; i += 32) ss += (double)vh[i] * (double)vh[i];
  for (int d = 16; d > 0; d >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, d);
  const float s = (float)(1.0 / sqrt(ss / (double)headDim + (double)eps));
  float *dst = vcache + (size_t)idx * kvDim + (size_t)h * headDim;
  for (int i = lane; i < headDim; i += 32) dst[i] = __fmul_rn(vh[i], s);
}

// attend: causal attention of the newest token over one slot, relative-position bias
// within extent, per-head sink logit, a zero sink in the softmax denominator, GQA.
// One block per head; scores are computed thread-per-key in host order, the softmax
// denominator serially in key order, the output thread-per-dim in key order.
#define INK2_ATT_THREADS 512
#define INK2_ATT_GROUPS 4
__global__ void __launch_bounds__(INK2_ATT_THREADS) k_ink2_attend(const float *q, const float *r, const float *sinks, const float *relBias,
                              const float *kcache, const float *vcache, double *scores, float *out,
                              const float *meta, int heads, int kvHeads, int headDim, int relLen,
                              int extent, int window, int cap, float scale) {
  const int h = blockIdx.x;
  const int kvh = h / (heads / kvHeads);
  const int kvDim = kvHeads * headDim;
  const int pos = ink2_pos(meta);
  int n = pos + 1;
  if (window > 0 && n > window) n = window;
  if (n > cap) n = cap;
  const int first = pos + 1 - n;
  const float *qh = q + (size_t)h * headDim;
  const float *rh = r + (size_t)h * relLen;
  double *sc = scores + (size_t)h * cap;
  __shared__ double shm[33];
  __shared__ double sdenom;
  double mx = -INFINITY;
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
    float bias = -sinks[h];
    if (dist < extent) bias = __fadd_rn(bias, rd);
    const double s = (double)__fadd_rn(__fmul_rn(d, scale), bias);
    if (lane == 0) sc[j] = s;
    mx = fmax(mx, s);
  }
  for (int d = 16; d > 0; d >>= 1) mx = fmax(mx, __shfl_down_sync(0xffffffff, mx, d));
  if ((threadIdx.x & 31) == 0) shm[threadIdx.x >> 5] = mx;
  __syncthreads();
  __shared__ double smax;
  if (threadIdx.x == 0) {
    double m = -INFINITY;
    for (int w = 0; w < (int)((blockDim.x + 31) / 32); ++w) m = fmax(m, shm[w]);
    smax = m < 0 ? 0 : m;
  }
  __syncthreads();
  const double m = smax;
  for (int j = threadIdx.x; j < n; j += blockDim.x) sc[j] = exp(sc[j] - m);
  __syncthreads();
  if (threadIdx.x == 0) {
    // the zero sink logit joins the denominator; keys summed in order
    double denom = exp(-m);
    for (int j = 0; j < n; ++j) denom += sc[j];
    sdenom = denom;
  }
  __syncthreads();
  const double denom = sdenom;
  extern __shared__ float wsh[];
  for (int j = threadIdx.x; j < n; j += blockDim.x) wsh[j] = (float)(sc[j] / denom);
  __syncthreads();
  // INK2_ATT_GROUPS key groups per output dim, combined in group order
  __shared__ float gacc[INK2_ATT_GROUPS][256];
  const int groups = blockDim.x / headDim;
  const int dd = threadIdx.x % headDim, g = threadIdx.x / headDim;
  if (g < groups) {
    float acc = 0.f;
    for (int j = g; j < n; j += groups) {
      const float w = wsh[j];
      const float *vr = vcache + (size_t)((first + j) % cap) * kvDim + (size_t)kvh * headDim;
      acc = fmaf(w, vr[dd], acc);
    }
    gacc[g][dd] = acc;
  }
  __syncthreads();
  if (threadIdx.x < headDim) {
    float acc = 0.f;
    for (int gg = 0; gg < groups; ++gg) acc += gacc[gg][threadIdx.x];
    out[(size_t)h * headDim + threadIdx.x] = acc;
  }
}

__global__ void k_ink2_sigmoid_mul(float *x, const float *gate, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __fmul_rn(x[i], ink2_sigmoid(gate[i]));
}

// MoE routing: sigmoid probabilities, top-k of probs+bias (ties to the lower index),
// weights renormalized over the selected probabilities and scaled. One warp.
__global__ void k_ink2_route(const float *logits, const float *bias, int *ids, float *wts, int nexp, int k,
                             float wscale) {
  __shared__ float probs[256], sel[256];
  const int lane = threadIdx.x;
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

// Multiply each routed slot's rows by its expert's per-tensor scale.
__global__ void k_ink2_expert_scale(float *y, const int *ids, const float *scale, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows) y[(size_t)blockIdx.y * rows + i] = __fmul_rn(y[(size_t)blockIdx.y * rows + i], scale[ids[blockIdx.y]]);
}

// SITU activation of one routed slot: act = beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb).
__global__ void k_ink2_situ(const float *gu, float *act, int width, int actStride, double beta, double lb) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= width) return;
  const float *row = gu + (size_t)blockIdx.y * 2 * width;
  const double g = (double)row[j];
  double u = (double)row[width + j];
  const double a = beta * tanh(g / beta) * (1.0 / (1.0 + exp(-g)));
  if (lb > 0) u = lb * tanh(u / lb);
  act[(size_t)blockIdx.y * actStride + j] = (float)(a * u);
}

// out[j] = sum over routed slots, in routing order, of weight * down[slot][j].
__global__ void k_ink2_moe_combine(const float *down, const float *wts, float *out, int k, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  float acc = 0.f;
  for (int t = 0; t < k; ++t) acc = __fadd_rn(acc, __fmul_rn(wts[t], down[(size_t)t * n + j]));
  out[j] = acc;
}

__global__ void k_ink2_add(const float *a, const float *b, float *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __fadd_rn(a[i], b[i]);
}

// Final softcap (when cap != 0) then argmax over n with ties to the lower index: every
// block folds its best (value, index) into one 64-bit key with atomicMax.
static __device__ __forceinline__ unsigned long long ink2_argkey(float v, int i) {
  const unsigned int b = __float_as_uint(v);
  const unsigned int ord = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return ((unsigned long long)ord << 32) | (unsigned long long)(0xffffffffu - (unsigned int)i);
}

__global__ void k_ink2_softcap_argmax(float *logits, int n, float cap, unsigned long long *key) {
  __shared__ unsigned long long best[32];
  unsigned long long mine = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    float v = logits[i];
    if (cap != 0.f) {
      v = __fmul_rn(cap, (float)tanh((double)__fdiv_rn(v, cap)));
      logits[i] = v;
    }
    const unsigned long long kv = ink2_argkey(v, i);
    if (kv > mine) mine = kv;
  }
  for (int d = 16; d > 0; d >>= 1) {
    const unsigned long long o = __shfl_xor_sync(0xffffffff, mine, d);
    if (o > mine) mine = o;
  }
  if ((threadIdx.x & 31) == 0) best[threadIdx.x >> 5] = mine;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < (int)(blockDim.x / 32); ++w)
      if (best[w] > mine) mine = best[w];
    atomicMax(key, mine);
  }
}

__global__ void k_ink2_argkey_index(const unsigned long long *key, int *outIdx) {
  *outIdx = (int)(0xffffffffu - (unsigned int)(*key & 0xffffffffull));
}

// Expert-indexed GEMVs: blockIdx.y is the routed slot, ids[slot] selects the expert's
// rows in one stacked weight tensor, x advances by xStride per slot (0 shares one
// input), y is [slot][rows]. Each format's per-row arithmetic is the resident
// single-expert kernel's, so routing on the device changes no expert output.
#define INK2_IQ_ROWS_PER_WARP 4

// SITU activation of one gate/up pair, in float64 like the host forward.
static __device__ __forceinline__ float ink2_situ_val(float gf, float uf, double beta, double lb) {
  const double g = (double)gf;
  double u = (double)uf;
  const double a = beta * tanh(g / beta) * (1.0 / (1.0 + exp(-g)));
  if (lb > 0) u = lb * tanh(u / lb);
  return (float)(a * u);
}

// Activation element i of one routed slot: x itself, or (situW > 0) the SITU of the
// slot's [gate | up] row of width situW, zero-padded up to the weight's K.
static __device__ __forceinline__ float ink2_stage(const float *x, int i, int situW, double beta, double lb) {
  if (situW == 0) return x[i];
  return i < situW ? ink2_situ_val(x[i], x[situW + i], beta, lb) : 0.f;
}
enum { INK2_NVFP4 = 0, INK2_IQ2XS, INK2_IQ2S, INK2_IQ3XXS, INK2_IQ3S, INK2_IQ4NL, INK2_Q3K };

template <int FMT>
__global__ void k_ink2_expert_iq(const unsigned char *W, const int *ids, const float *X, int xStride, float *Y,
                                 int rows, int in, const float *escale, int situW, double beta, double lb) {
  extern __shared__ float xsh[];
  const int lane = threadIdx.x & 31;
  const int slot = blockIdx.y;
  const float *xg = X + (size_t)slot * xStride;
  for (int i = threadIdx.x; i < in; i += blockDim.x) xsh[i] = ink2_stage(xg, i, situW, beta, lb);
  __syncthreads();
  const float *x = xsh;
  const int nsb = in / 256;
  constexpr int SB = FMT == INK2_IQ2XS ? 74 : FMT == INK2_IQ2S ? 82 : FMT == INK2_IQ3XXS ? 98 : (FMT == INK2_IQ3S || FMT == INK2_Q3K) ? 110 : 144;
  const size_t ebase = (size_t)ids[slot] * rows;
  for (int rr = 0; rr < INK2_IQ_ROWS_PER_WARP; ++rr) {
  const int o = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * INK2_IQ_ROWS_PER_WARP + rr;
  if (o >= rows) break;
  const unsigned char *wrow = W + (ebase + o) * nsb * SB;
  float sum = 0.f;
  if (FMT == INK2_Q3K) {
    // ggml block_q3_K: hmask[32], qs[64], scales[12] (6-bit, offset 32), f16 d.
    // Lane l covers values 8l..8l+7 of each super-block: one (half, shift, sub) group.
    const int e0 = lane * 8, half = e0 >> 7, j = (e0 & 127) >> 5, sub = (e0 & 31) >> 4, l0 = e0 & 15;
    const int is = half * 8 + j * 2 + sub;
    const unsigned char mbit = (unsigned char)(1u << (half * 4 + j));
    const float *xrow = x + e0;
    for (int sb = 0; sb < nsb; ++sb) {
      const unsigned char *blk = wrow + (size_t)sb * 110;
      const unsigned char *hm = blk, *qs = blk + 32, *sc = blk + 96;
      const float dall = __half2float(*(const __half *)(blk + 108));
      int scv;
      if (is < 8) scv = (sc[is] & 0x0f) | (((sc[8 + (is & 3)] >> (2 * (is >> 2))) & 3) << 4);
      else scv = (sc[is - 8] >> 4) | (((sc[8 + (is & 3)] >> (2 * (is >> 2))) & 3) << 4);
      const float dl = dall * (float)(scv - 32);
      const float *xs = xrow + (size_t)sb * 256;
      float part = 0.f;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const int li = sub * 16 + l0 + t;
        const int qv = (int)((qs[half * 32 + li] >> (2 * j)) & 3) - ((hm[li] & mbit) ? 0 : 4);
        part = fmaf((float)qv, xs[t], part);
      }
      sum = fmaf(dl, part, sum);
    }
  } else if (FMT == INK2_IQ4NL) {
    const int ib = lane >> 2, il = lane & 3;
    const float *xrow = x + 32 * ib + 4 * il;
    for (int sb = 0; sb < nsb; ++sb) {
      const unsigned char *blk = wrow + (size_t)sb * 144 + 18 * ib;
      const unsigned char *q4 = blk + 2 + 4 * il;
      const float d = __half2float(*(const __half *)blk);
      const float *xs = xrow + (size_t)sb * 256;
      const float4 lo = *(const float4 *)xs;
      const float4 hi = *(const float4 *)(xs + 16);
      const float xl[4] = {lo.x, lo.y, lo.z, lo.w};
      const float xh[4] = {hi.x, hi.y, hi.z, hi.w};
      float part = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        part = fmaf((float)fcuda_kvalues_iq4nl[q4[j] & 0xf], xl[j], part);
        part = fmaf((float)fcuda_kvalues_iq4nl[q4[j] >> 4], xh[j], part);
      }
      sum = fmaf(d, part, sum);
    }
  } else {
    const int ib32 = lane >> 2, l = lane & 3;
    const float *xrow = x + lane * 8;
    for (int sb = 0; sb < nsb; ++sb) {
      const unsigned char *blk = wrow + (size_t)sb * SB;
      const float d = __half2float(*(const __half *)blk);
      if (FMT == INK2_IQ2XS) {
        const unsigned short qv = ((const unsigned short *)(blk + 2))[4 * ib32 + l];
        const unsigned char sc = blk[66 + ib32];
        const float db = d * (0.5f + (float)(l < 2 ? (sc & 0x0f) : (sc >> 4))) * 0.25f;
        sum = fmaf(db, fcuda_iq_dot8(fcuda_iq2xs_grid[qv & 511], fcuda_ksigns_iq2xs[qv >> 9], xrow + (size_t)sb * 256), sum);
      } else if (FMT == INK2_IQ2S) {
        const unsigned int idx = blk[2 + 4 * ib32 + l] | (((unsigned int)blk[66 + ib32] << (8 - 2 * l)) & 0x300);
        const unsigned char sign = blk[34 + 4 * ib32 + l];
        const unsigned char sc = blk[74 + ib32];
        const float db = d * (0.5f + (float)(l < 2 ? (sc & 0x0f) : (sc >> 4))) * 0.25f;
        sum = fmaf(db, fcuda_iq_dot8(fcuda_iq2s_grid[idx], sign, xrow + (size_t)sb * 256), sum);
      } else if (FMT == INK2_IQ3XXS) {
        const unsigned int aux32 = fcuda_iq_u32_at(blk + 66 + 4 * ib32);
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        const unsigned char signs = fcuda_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        const unsigned long long grid = (unsigned long long)fcuda_iq3xxs_grid[blk[2 + 8 * ib32 + 2 * l]] |
                                        ((unsigned long long)fcuda_iq3xxs_grid[blk[2 + 8 * ib32 + 2 * l + 1]] << 32);
        sum = fmaf(db, fcuda_iq_dot8(grid, signs, xrow + (size_t)sb * 256), sum);
      } else {
        const unsigned int high = blk[66 + ib32];
        const unsigned int idx1 = blk[2 + 8 * ib32 + 2 * l] | ((high << (8 - 2 * l)) & 0x100);
        const unsigned int idx2 = blk[2 + 8 * ib32 + 2 * l + 1] | ((high << (7 - 2 * l)) & 0x100);
        const unsigned char sign = blk[74 + 4 * ib32 + l];
        const unsigned int nibble = (blk[106 + ib32 / 2] >> (4 * (ib32 & 1))) & 0x0f;
        const float db = d * (float)(1 + 2 * (int)nibble);
        const unsigned long long grid = (unsigned long long)fcuda_iq3s_grid[idx1] |
                                        ((unsigned long long)fcuda_iq3s_grid[idx2] << 32);
        sum = fmaf(db, fcuda_iq_dot8(grid, sign, xrow + (size_t)sb * 256), sum);
      }
    }
  }
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) sum += __shfl_down_sync(0xffffffff, sum, delta);
  if (lane == 0) Y[(size_t)slot * rows + o] = escale ? __fmul_rn(sum, escale[ids[slot]]) : sum;
  }
}

// NVFP4 (E2M1 codes, one E4M3 scale per 16 values, global scale 1): a block of 8 warps
// covers 32 rows of one routed slot, each warp 4 rows; the BF16-rounded activation is
// staged once per block in shared memory.
#define INK2_NV_ROWS_PER_WARP 4
__global__ void k_ink2_expert_nvfp4(const uint8_t *Packed, const uint8_t *Scales, const int *ids, const float *X,
                                    int xStride, float *Y, int rows, int in, const float *escale, int situW,
                                    double beta, double lb) {
  extern __shared__ float xs[];
  __shared__ float lut[16];
  const int slot = blockIdx.y;
  const float *x = X + (size_t)slot * xStride;
  for (int i = threadIdx.x; i < in; i += blockDim.x) xs[i] = nv4_bf16_rne(ink2_stage(x, i, situW, beta, lb));
  if (threadIdx.x < 16) lut[threadIdx.x] = nv4_e2m1(threadIdx.x);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int o0 = (blockIdx.x * (blockDim.x >> 5) + warp) * INK2_NV_ROWS_PER_WARP;
  const size_t ebase = (size_t)ids[slot] * rows;
  const int nb = in / 16;
  float sum[INK2_NV_ROWS_PER_WARP];
#pragma unroll
  for (int r = 0; r < INK2_NV_ROWS_PER_WARP; ++r) sum[r] = 0.f;
  for (int b = lane; b < nb; b += 32) {
    const float *xb = xs + b * 16;
#pragma unroll
    for (int r = 0; r < INK2_NV_ROWS_PER_WARP; ++r) {
      const int o = o0 + r;
      if (o < rows) {
        const size_t row = ebase + o;
        const uint2 pk = *(const uint2 *)(Packed + row * (size_t)(in / 2) + (size_t)b * 8);
        const float sc = nv4_e4m3fn(Scales[row * (size_t)nb + b]);
        float part = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint32_t word = j < 4 ? pk.x : pk.y;
          const uint32_t byte = (word >> (8 * (j & 3))) & 0xffu;
          part = fmaf(lut[byte & 0x0fu], xb[2 * j], part);
          part = fmaf(lut[byte >> 4], xb[2 * j + 1], part);
        }
        sum[r] = fmaf(part, sc, sum[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < INK2_NV_ROWS_PER_WARP; ++r) {
    float v = sum[r];
    for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    const int o = o0 + r;
    if (lane == 0 && o < rows) Y[(size_t)slot * rows + o] = escale ? __fmul_rn(v, escale[ids[slot]]) : v;
  }
}

static inline int ink2_last_error(void) { return (int)cudaGetLastError(); }

extern "C" int fcuda_ink2_zero(void *p, size_t bytes) {
  cudaError_t e = cudaSuccess;
  FCUDA_TS("fill", dim3(0, 0), e = cudaMemsetAsync(p, 0, bytes, g_stream));
  return (int)e;
}

extern "C" int fcuda_ink2_rms_rows(const float *x, const float *w, float *y, int rows, int n, float eps) {
  FCUDA_TS("k_ink2_rms_rows", rows, k_ink2_rms_rows<<<rows, 1024, 0, g_stream>>>(x, w, y, n, eps));
  return ink2_last_error();
}

extern "C" int fcuda_ink2_ctrl_pair(const float *cur, const float *msg, float *pair, int n) {
  FCUDA_TS("k_ink2_ctrl_pair", 2, k_ink2_ctrl_pair<<<2, 256, 0, g_stream>>>(cur, msg, pair, n));
  return ink2_last_error();
}

static int fcuda_ink2_mix_plain(const Ink2States &st, const float *queries, int nq, float *out, int n);

// -1 unknown, 0 no cluster launch on this device, 1 cluster launch available.
static int g_ink2_cluster = -1;

static bool ink2_cluster_ok(void) {
  if (g_ink2_cluster < 0) {
    int dev = 0, attr = 0;
    g_ink2_cluster = cudaGetDevice(&dev) == cudaSuccess &&
                             cudaDeviceGetAttribute(&attr, cudaDevAttrClusterLaunch, dev) == cudaSuccess && attr
                         ? 1
                         : 0;
    cudaGetLastError();
  }
  return g_ink2_cluster == 1;
}

template <int NQ>
static cudaError_t ink2_mixc_launch(const Ink2States &st, int ns, const float *queries, const Ink2Norms &nw, int norm,
                                    float eps, float *out, int n, int threads) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(INK2_MIX_CTAS);
  cfg.blockDim = dim3(threads);
  cfg.stream = g_stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = INK2_MIX_CTAS;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, k_ink2_mixc<NQ>, st, ns, queries, nw, norm, eps, out, n);
}

// fcuda_ink2_mix: out[q] = mix of the states for query q, then (norms != null) RMS with
// weight norms[q] (nullable entries skip the weight). Devices without cluster launch run
// the single-block mix and one k_ink2_rms_rows per query.
extern "C" int fcuda_ink2_mix(const float *const *states, int ns, const float *queries, int nq,
                              const float *const *norms, float eps, float *out, int n) {
  if (ns < 1 || ns > INK2_MAX_STATES || nq < 1 || nq > INK2_MAX_QUERIES) return (int)cudaErrorInvalidValue;
  Ink2States st;
  for (int i = 0; i < INK2_MAX_STATES; ++i) st.p[i] = i < ns ? states[i] : nullptr;
  Ink2Norms nw;
  for (int q = 0; q < INK2_MAX_QUERIES; ++q) nw.w[q] = norms && q < nq ? norms[q] : nullptr;
  const int threads = ((n + INK2_MIX_CTAS - 1) / INK2_MIX_CTAS + 31) & ~31;
  if (threads <= 1024 && ink2_cluster_ok()) {
    cudaError_t e = cudaSuccess;
    const int norm = norms ? 1 : 0;
    switch (nq) {
    case 1: FCUDA_TS("k_ink2_mixc", INK2_MIX_CTAS, e = ink2_mixc_launch<1>(st, ns, queries, nw, norm, eps, out, n, threads)); break;
    case 2: FCUDA_TS("k_ink2_mixc", INK2_MIX_CTAS, e = ink2_mixc_launch<2>(st, ns, queries, nw, norm, eps, out, n, threads)); break;
    case 3: FCUDA_TS("k_ink2_mixc", INK2_MIX_CTAS, e = ink2_mixc_launch<3>(st, ns, queries, nw, norm, eps, out, n, threads)); break;
    default: FCUDA_TS("k_ink2_mixc", INK2_MIX_CTAS, e = ink2_mixc_launch<4>(st, ns, queries, nw, norm, eps, out, n, threads)); break;
    }
    return (int)e;
  }
  const int rc = fcuda_ink2_mix_plain(st, queries, nq, out, n);
  if (rc != 0 || !norms) return rc;
  for (int q = 0; q < nq; ++q) {
    float *row = out + (size_t)q * n;
    FCUDA_TS("k_ink2_rms_rows", 1, k_ink2_rms_rows<<<1, 1024, 0, g_stream>>>(row, norms[q], row, n, eps));
  }
  return ink2_last_error();
}

static int fcuda_ink2_mix_plain(const Ink2States &st, const float *queries, int nq, float *out, int n) {
  int ns = 0;
  while (ns < INK2_MAX_STATES && st.p[ns]) ++ns;
  switch (nq) {
  case 1: FCUDA_TS("k_ink2_mix", 1, k_ink2_mix<1><<<1, INK2_MIX_THREADS, 0, g_stream>>>(st, ns, queries, out, n)); break;
  case 2: FCUDA_TS("k_ink2_mix", 1, k_ink2_mix<2><<<1, INK2_MIX_THREADS, 0, g_stream>>>(st, ns, queries, out, n)); break;
  case 3: FCUDA_TS("k_ink2_mix", 1, k_ink2_mix<3><<<1, INK2_MIX_THREADS, 0, g_stream>>>(st, ns, queries, out, n)); break;
  default: FCUDA_TS("k_ink2_mix", 1, k_ink2_mix<4><<<1, INK2_MIX_THREADS, 0, g_stream>>>(st, ns, queries, out, n)); break;
  }
  return ink2_last_error();
}

extern "C" int fcuda_ink2_gemv_f32(const float *W, const float *x, float *y, int rows, int cols) {
  FCUDA_TS("k_ink2_gemv_f32", rows, k_ink2_gemv_f32<<<rows, 256, 0, g_stream>>>(W, x, y, rows, cols));
  return ink2_last_error();
}

extern "C" int fcuda_ink2_ctrl_down(const float *cur, const float *msg, const float *W, float *g, int n, int rank) {
  FCUDA_TS("k_ink2_ctrl_down", rank, k_ink2_ctrl_down<<<rank, 256, 0, g_stream>>>(cur, msg, W, g, n));
  return ink2_last_error();
}

extern "C" int fcuda_ink2_ctrl_apply(const float *cur, const float *msg, const float *g, const float *gainW,
                                     const float *forgetW, float *out, int n, int rank) {
  FCUDA_TS("k_ink2_ctrl_apply", (n + 7) / 8, k_ink2_ctrl_apply<<<(n + 7) / 8, 256, 0, g_stream>>>(
      cur, msg, g, gainW, forgetW, out, n, rank));
  return ink2_last_error();
}

extern "C" int fcuda_ink2_conv(const float *x, const float *w, const float *dyn, float *hist, float *out,
                               const float *resid, int n, int heads, int kn, float norm) {
  FCUDA_TS("k_ink2_conv", (n + 255) / 256, k_ink2_conv<<<(n + 255) / 256, 256, 0, g_stream>>>(
      x, w, dyn, hist, out, resid, n, heads, kn, norm));
  return ink2_last_error();
}

extern "C" int fcuda_ink2_qkv_post(float *q, float *r, const float *k, const float *v, const float *qclip,
                                   const float *rclip, float *kcache, float *vcache, const float *meta, int heads,
                                   int kvHeads, int headDim, int relLen, float qkClip, float relClip, float eps,
                                   int global, float alpha, float floorTok, int cap) {
  if (headDim > 256) return (int)cudaErrorInvalidValue;
  const int blocks = heads + 2 * kvHeads;
  FCUDA_TS("k_ink2_qkv_post", blocks, k_ink2_qkv_post<<<blocks, 32, 0, g_stream>>>(
      q, r, k, v, qclip, rclip, kcache, vcache, meta, heads, kvHeads, headDim, relLen, qkClip, relClip, eps, global,
      alpha, floorTok, cap));
  return ink2_last_error();
}

// headDim <= 256 and INK2_ATT_THREADS / headDim <= INK2_ATT_GROUPS (the per-group partials).
extern "C" int fcuda_ink2_attend(const float *q, const float *r, const float *sinks, const float *relBias,
                                 const float *kcache, const float *vcache, double *scores, float *out,
                                 const float *meta, int heads, int kvHeads, int headDim, int relLen, int extent,
                                 int window, int cap, float scale) {
  if (headDim > 256 || INK2_ATT_THREADS / headDim > INK2_ATT_GROUPS) return (int)cudaErrorInvalidValue;
  FCUDA_TS("k_ink2_attend", heads, k_ink2_attend<<<heads, INK2_ATT_THREADS, (size_t)cap * sizeof(float), g_stream>>>(
      q, r, sinks, relBias, kcache, vcache, scores, out, meta, heads, kvHeads, headDim, relLen, extent, window, cap,
      scale));
  return ink2_last_error();
}

extern "C" int fcuda_ink2_sigmoid_mul(float *x, const float *gate, int n) {
  FCUDA_TS("k_ink2_sigmoid_mul", (n + 255) / 256, k_ink2_sigmoid_mul<<<(n + 255) / 256, 256, 0, g_stream>>>(x, gate, n));
  return ink2_last_error();
}

extern "C" int fcuda_ink2_route(const float *logits, const float *bias, int *ids, float *wts, int nexp, int k,
                                float wscale) {
  if (nexp > 256 || k > INK2_MAX_TOPK || k > nexp) return (int)cudaErrorInvalidValue;
  FCUDA_TS("k_ink2_route", 1, k_ink2_route<<<1, 32, 0, g_stream>>>(logits, bias, ids, wts, nexp, k, wscale));
  return ink2_last_error();
}

extern "C" int fcuda_ink2_expert_scale(float *y, const int *ids, const float *scale, int k, int rows) {
  FCUDA_TS("k_ink2_expert_scale", dim3((rows + 255) / 256, k), k_ink2_expert_scale<<<dim3((rows + 255) / 256, k), 256, 0, g_stream>>>(
      y, ids, scale, rows));
  return ink2_last_error();
}

extern "C" int fcuda_ink2_situ(const float *gu, float *act, int k, int width, int actStride, double beta, double lb) {
  FCUDA_TS("k_ink2_situ", dim3((width + 255) / 256, k), k_ink2_situ<<<dim3((width + 255) / 256, k), 256, 0, g_stream>>>(
      gu, act, width, actStride, beta, lb));
  return ink2_last_error();
}

extern "C" int fcuda_ink2_moe_combine(const float *down, const float *wts, float *out, int k, int n) {
  FCUDA_TS("k_ink2_moe_combine", (n + 255) / 256, k_ink2_moe_combine<<<(n + 255) / 256, 256, 0, g_stream>>>(down, wts, out, k, n));
  return ink2_last_error();
}

extern "C" int fcuda_ink2_add(const float *a, const float *b, float *out, int n) {
  FCUDA_TS("k_ink2_add", (n + 255) / 256, k_ink2_add<<<(n + 255) / 256, 256, 0, g_stream>>>(a, b, out, n));
  return ink2_last_error();
}

// outIdx[0] receives the index; outIdx[2..3] (8-byte aligned) hold the reduction key.
extern "C" int fcuda_ink2_softcap_argmax(float *logits, int n, float cap, int *outIdx) {
  unsigned long long *key = (unsigned long long *)(outIdx + 2);
  cudaError_t e = cudaSuccess;
  FCUDA_TS("fill", dim3(0, 0), e = cudaMemsetAsync(key, 0, sizeof(*key), g_stream));
  if (e != cudaSuccess) return (int)e;
  FCUDA_TS("k_ink2_softcap_argmax", 128, k_ink2_softcap_argmax<<<128, 256, 0, g_stream>>>(logits, n, cap, key));
  FCUDA_TS("k_ink2_argkey_index", 1, k_ink2_argkey_index<<<1, 1, 0, g_stream>>>(key, outIdx));
  return ink2_last_error();
}

// fmt: 0 NVFP4 (scales = the stacked E4M3 scale plane), 1 IQ2_XS, 2 IQ2_S, 3 IQ3_XXS,
// 4 IQ3_S, 5 IQ4_NL (in = padded K), 6 Q3_K. Returns cudaErrorInvalidValue for an unknown format.
extern "C" int fcuda_ink2_expert_gemv(int fmt, const uint8_t *W, const uint8_t *scales, const int *ids,
                                      const float *X, int xStride, float *Y, int k, int rows, int in,
                                      const float *escale, int situW, double beta, double lb) {
  const dim3 iqgrid((rows + 8 * INK2_IQ_ROWS_PER_WARP - 1) / (8 * INK2_IQ_ROWS_PER_WARP), k);
  switch (fmt) {
  case INK2_NVFP4:
    if (!scales || (in & 15) != 0) return (int)cudaErrorInvalidValue;
    {
      const dim3 nvgrid((rows + 31) / 32, k);
      FCUDA_TS("k_ink2_expert_nvfp4", nvgrid, k_ink2_expert_nvfp4<<<nvgrid, 256, in * sizeof(float), g_stream>>>(
          W, scales, ids, X, xStride, Y, rows, in, escale, situW, beta, lb));
    }
    break;
  case INK2_IQ2XS:
    FCUDA_TS("k_ink2_expert_iq", iqgrid, k_ink2_expert_iq<INK2_IQ2XS><<<iqgrid, 256, in * sizeof(float), g_stream>>>(W, ids, X, xStride, Y, rows, in, escale, situW, beta, lb));
    break;
  case INK2_IQ2S:
    FCUDA_TS("k_ink2_expert_iq", iqgrid, k_ink2_expert_iq<INK2_IQ2S><<<iqgrid, 256, in * sizeof(float), g_stream>>>(W, ids, X, xStride, Y, rows, in, escale, situW, beta, lb));
    break;
  case INK2_IQ3XXS:
    FCUDA_TS("k_ink2_expert_iq", iqgrid, k_ink2_expert_iq<INK2_IQ3XXS><<<iqgrid, 256, in * sizeof(float), g_stream>>>(W, ids, X, xStride, Y, rows, in, escale, situW, beta, lb));
    break;
  case INK2_IQ3S:
    FCUDA_TS("k_ink2_expert_iq", iqgrid, k_ink2_expert_iq<INK2_IQ3S><<<iqgrid, 256, in * sizeof(float), g_stream>>>(W, ids, X, xStride, Y, rows, in, escale, situW, beta, lb));
    break;
  case INK2_Q3K:
    FCUDA_TS("k_ink2_expert_iq", iqgrid, k_ink2_expert_iq<INK2_Q3K><<<iqgrid, 256, in * sizeof(float), g_stream>>>(W, ids, X, xStride, Y, rows, in, escale, situW, beta, lb));
    break;
  case INK2_IQ4NL:
    FCUDA_TS("k_ink2_expert_iq", iqgrid, k_ink2_expert_iq<INK2_IQ4NL><<<iqgrid, 256, in * sizeof(float), g_stream>>>(W, ids, X, xStride, Y, rows, in, escale, situW, beta, lb));
    break;
  default:
    return (int)cudaErrorInvalidValue;
  }
  return ink2_last_error();
}

// Capture-once per-token graph for the resident forward. Begin opens a capture on
// g_stream; End instantiates and keeps the exec; Launch replays it and fences.
static cudaGraphExec_t g_ink2_exec = nullptr;

extern "C" int fcuda_ink2_graph_begin(void) {
  if (g_capture_open) return (int)cudaErrorInvalidValue;
  const cudaError_t e = cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal);
  if (e == cudaSuccess) g_capture_open = true;
  return (int)e;
}

extern "C" int fcuda_ink2_graph_end(void) {
  cudaGraph_t graph = nullptr;
  const cudaError_t e = cudaStreamEndCapture(g_stream, &graph);
  g_capture_open = false;
  if (e != cudaSuccess) {
    if (graph) cudaGraphDestroy(graph);
    return (int)e;
  }
  if (g_ink2_exec) {
    cudaGraphExecDestroy(g_ink2_exec);
    g_ink2_exec = nullptr;
  }
  const cudaError_t ie = cudaGraphInstantiate(&g_ink2_exec, graph, 0);
  cudaGraphDestroy(graph);
  return (int)ie;
}

extern "C" int fcuda_ink2_graph_launch(void) {
  if (!g_ink2_exec) return (int)cudaErrorInvalidValue;
  return (int)cudaGraphLaunch(g_ink2_exec, g_stream);
}

extern "C" void fcuda_ink2_graph_reset(void) {
  if (g_ink2_exec) {
    cudaGraphExecDestroy(g_ink2_exec);
    g_ink2_exec = nullptr;
  }
}
