// inkling2_attend_split.cuh: k_ink2_attend spread over more blocks, bit for bit.
// The reference runs one 512-thread block per head (24 blocks on an 84-SM card), so
// its key scores, double exps and value sums all sit on a quarter of the SMs. Here
// the same arithmetic runs in three launches:
//   scores  (head, chunk) blocks, one warp per key with the reference dot and shuffle
//           order; each block leaves its chunk max in the head's output row, which is
//           free until the last launch writes it;
//   exp     (head, chunk) blocks take the head max from those chunk maxes (fmax is
//           exact in any order), exponentiate their chunk in place and park the max
//           in the head's rel-query row, which only the scores launch reads;
//   output  (head, 32-dim slice) blocks redo the serial key-order denominator, then
//           the reference's 4 key groups per dim, combined in group order.
// FAK_INKLING2_ATTEND_SPLIT=0 keeps the reference kernel.

#ifndef INK2_ATS_CHUNKS
#define INK2_ATS_CHUNKS 8
#endif
#define INK2_ATS_WARPS 8
#ifndef INK2_ATS_UNROLL
#define INK2_ATS_UNROLL 8
#endif

static __device__ __forceinline__ int ink2_ats_keys(const float *meta, int window, int cap, int *first) {
  const int pos = ink2_pos(meta);
  int n = pos + 1;
  if (window > 0 && n > window) n = window;
  if (n > cap) n = cap;
  *first = pos + 1 - n;
  return n;
}

__global__ void __launch_bounds__(INK2_ATS_WARPS * 32)
    k_ink2_attend_scores(const float *q, const float *r, const float *sinks, const float *relBias,
                         const float *kcache, double *scores, float *out, const float *meta, int heads, int kvHeads,
                         int headDim, int relLen, int extent, int window, int cap, float scale, Ink2Seq sq) {
  ink2_pdl_enter();
  const int h = blockIdx.x, c = blockIdx.y, seq = blockIdx.z;
  q = ink2_row(q, sq.q, seq);
  r = ink2_row(r, sq.r, seq);
  sinks = ink2_row(sinks, sq.sinks, seq);
  kcache = ink2_row(kcache, sq.kc, seq);
  scores = ink2_row(scores, sq.scores, seq);
  out = ink2_row(out, sq.out, seq);
  meta = ink2_row(meta, sq.meta, seq);
  const int kvh = h / (heads / kvHeads);
  const int kvDim = kvHeads * headDim;
  int first;
  const int n = ink2_ats_keys(meta, window, cap, &first);
  const int pos = first + n - 1;
  const int per = (n + gridDim.y - 1) / gridDim.y;
  const int j0 = c * per, j1 = min(n, j0 + per);
  const float *qh = q + (size_t)h * headDim;
  const float *rh = r + (size_t)h * relLen;
  double *sc = scores + (size_t)h * cap;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  double mx = -INFINITY;
  for (int j = j0 + warp; j < j1; j += INK2_ATS_WARPS) {
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
  __shared__ double wmax[INK2_ATS_WARPS];
  if (lane == 0) wmax[warp] = mx;
  __syncthreads();
  if (threadIdx.x == 0) {
    double m = -INFINITY;
    for (int w = 0; w < INK2_ATS_WARPS; ++w) m = fmax(m, wmax[w]);
    ((double *)(out + (size_t)h * headDim))[c] = m;
  }
}

__global__ void __launch_bounds__(256)
    k_ink2_attend_exp(double *scores, const float *out, float *r, const float *meta, int headDim, int relLen,
                      int window, int cap, int chunks, Ink2Seq sq) {
  ink2_pdl_enter();
  const int h = blockIdx.x, c = blockIdx.y, seq = blockIdx.z;
  scores = ink2_row(scores, sq.scores, seq);
  out = ink2_row(out, sq.out, seq);
  r = ink2_row(r, sq.r, seq);
  meta = ink2_row(meta, sq.meta, seq);
  int first;
  const int n = ink2_ats_keys(meta, window, cap, &first);
  const double *cm = (const double *)(out + (size_t)h * headDim);
  double m = -INFINITY;
  for (int i = 0; i < chunks; ++i) m = fmax(m, cm[i]);
  if (m < 0) m = 0;
  if (c == 0 && threadIdx.x == 0) *(double *)(r + (size_t)h * relLen) = m;
  const int per = (n + gridDim.y - 1) / gridDim.y;
  const int j0 = c * per, j1 = min(n, j0 + per);
  double *sc = scores + (size_t)h * cap;
  for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) sc[j] = exp(sc[j] - m);
}

// Block: 32 dims x INK2_ATT_THREADS / headDim key groups (the reference's grouping).
// STAGED keeps the exps and weights in shared memory; past the opt-in limit the
// unstaged form reads them from global and divides in place, giving the same bits.
template <bool STAGED>
__global__ void __launch_bounds__(INK2_ATT_THREADS)
    k_ink2_attend_out(const double *scores, const float *vcache, const float *r, float *out, const float *meta,
                      int heads, int kvHeads, int headDim, int relLen, int window, int cap, Ink2Seq sq) {
  ink2_pdl_enter();
  const int h = blockIdx.x, seq = blockIdx.z;
  scores = ink2_row(scores, sq.scores, seq);
  vcache = ink2_row(vcache, sq.vc, seq);
  r = ink2_row(r, sq.r, seq);
  out = ink2_row(out, sq.out, seq);
  meta = ink2_row(meta, sq.meta, seq);
  const int kvh = h / (heads / kvHeads);
  const int kvDim = kvHeads * headDim;
  int first;
  const int n = ink2_ats_keys(meta, window, cap, &first);
  const double *sc = scores + (size_t)h * cap;
  // the exps are staged in shared memory so the serial denominator waits on adds, not loads
  extern __shared__ double esh[];
  float *wsh = (float *)(esh + cap);
  __shared__ double sdenom;
  if (STAGED) {
    for (int j = threadIdx.x; j < n; j += blockDim.x) esh[j] = sc[j];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double m = *(const double *)(r + (size_t)h * relLen);
    double denom = exp(-m);
    for (int j = 0; j < n; ++j) denom += STAGED ? esh[j] : sc[j];
    sdenom = denom;
  }
  __syncthreads();
  const double denom = sdenom;
  if (STAGED) {
    for (int j = threadIdx.x; j < n; j += blockDim.x) wsh[j] = (float)(esh[j] / denom);
    __syncthreads();
  }
#define INK2_ATS_W(j) (STAGED ? wsh[(j)] : (float)(sc[(j)] / denom))
  const int groups = INK2_ATT_THREADS / headDim;
  const int dd = blockIdx.y * 32 + (threadIdx.x & 31), g = threadIdx.x >> 5;
  __shared__ float gacc[INK2_ATT_GROUPS][32];
  if (g < groups && dd < headDim) {
    // values are fetched INK2_ATS_UNROLL keys ahead of the serial fmaf chain
    const float *vbase = vcache + (size_t)kvh * headDim + dd;
    float acc = 0.f;
    int j = g;
    for (; j + (INK2_ATS_UNROLL - 1) * groups < n; j += INK2_ATS_UNROLL * groups) {
      float vv[INK2_ATS_UNROLL];
#pragma unroll
      for (int u = 0; u < INK2_ATS_UNROLL; ++u) vv[u] = vbase[(size_t)((first + j + u * groups) % cap) * kvDim];
#pragma unroll
      for (int u = 0; u < INK2_ATS_UNROLL; ++u) acc = fmaf(INK2_ATS_W(j + u * groups), vv[u], acc);
    }
    for (; j < n; j += groups) acc = fmaf(INK2_ATS_W(j), vbase[(size_t)((first + j) % cap) * kvDim], acc);
    gacc[g][threadIdx.x & 31] = acc;
  }
  __syncthreads();
  if (threadIdx.x < 32 && dd < headDim) {
    float acc = 0.f;
    for (int gg = 0; gg < groups; ++gg) acc += gacc[gg][threadIdx.x];
    out[(size_t)h * headDim + dd] = acc;
  }
#undef INK2_ATS_W
}

static int g_ink2_ats_mode = -1;

static bool ink2_ats_enabled(void) {
  if (g_ink2_ats_mode < 0) {
    const char *v = getenv("FAK_INKLING2_ATTEND_SPLIT");
    g_ink2_ats_mode = (v && v[0] == '0' && v[1] == 0) ? 0 : 1;
  }
  return g_ink2_ats_mode == 1;
}

// Test hook: 1 selects the split attend, 0 the reference; returns the previous mode.
extern "C" int fcuda_ink2_attend_split_select(int mode) {
  const int prev = ink2_ats_enabled() ? 1 : 0;
  g_ink2_ats_mode = mode ? 1 : 0;
  return prev;
}

// Launches the split attend; false (nothing launched) when the shape keeps the
// reference: the chunk maxes need headDim / 2 >= INK2_ATS_CHUNKS doubles of out and
// the parked max needs an even relLen >= 2 (8-byte aligned rows). blockIdx.z is the
// sequence of sq; every sequence's r and out rows must keep that alignment.
static bool ink2_attend_split(const float *q, const float *r, const float *sinks, const float *relBias,
                              const float *kcache, const float *vcache, double *scores, float *out, const float *meta,
                              int heads, int kvHeads, int headDim, int relLen, int extent, int window, int cap,
                              float scale, const Ink2Seq &sq, int C) {
  if (!ink2_ats_enabled() || headDim / 2 < INK2_ATS_CHUNKS || (headDim & 31) != 0 || relLen < 2 || (relLen & 1) != 0 ||
      ((uintptr_t)r & 7) != 0 || ((uintptr_t)out & 7) != 0 || (sq.r & 1) != 0 || (sq.out & 1) != 0)
    return false;
  size_t osmem = (size_t)cap * (sizeof(double) + sizeof(float));
  static size_t optin = 0;
  static size_t refused = 0;
  bool staged = true;
  if (osmem > optin) {
    if (refused != 0 && osmem >= refused) {
      staged = false;
    } else if (cudaFuncSetAttribute(k_ink2_attend_out<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)osmem) !=
               cudaSuccess) {
      cudaGetLastError();
      refused = osmem;
      staged = false;
    } else {
      optin = osmem;
    }
  }
  if (!staged) osmem = 0;
  const dim3 g1(heads, INK2_ATS_CHUNKS, C), g3(heads, headDim / 32, C);
  FCUDA_TS("k_ink2_attend_scores", g1, k_ink2_attend_scores<<<g1, INK2_ATS_WARPS * 32, 0, g_stream>>>(
                                           q, r, sinks, relBias, kcache, scores, out, meta, heads, kvHeads, headDim,
                                           relLen, extent, window, cap, scale, sq));
  FCUDA_TS("k_ink2_attend_exp", g1, k_ink2_attend_exp<<<g1, 256, 0, g_stream>>>(scores, out, (float *)r, meta, headDim,
                                                                                  relLen, window, cap, INK2_ATS_CHUNKS, sq));
  if (staged) {
    FCUDA_TS("k_ink2_attend_out", g3, k_ink2_attend_out<true><<<g3, (INK2_ATT_THREADS / headDim) * 32, osmem, g_stream>>>(
                                          scores, vcache, r, out, meta, heads, kvHeads, headDim, relLen, window,
                                          cap, sq));
  } else {
    FCUDA_TS("k_ink2_attend_out", g3, k_ink2_attend_out<false><<<g3, (INK2_ATT_THREADS / headDim) * 32, 0, g_stream>>>(
                                          scores, vcache, r, out, meta, heads, kvHeads, headDim, relLen, window,
                                          cap, sq));
  }
  return true;
}
