// inkling2_attend_split.cuh: k_ink2_attend spread over more blocks.
// The reference runs one 512-thread block per head (24 blocks on an 84-SM card), so
// its key scores, double exps and value sums all sit on a quarter of the SMs. Here
// the same scores and exps run in three launches:
//   scores  (head, chunk) blocks, one warp per key with the reference dot and shuffle
//           order; each block leaves its chunk max in the head's output row, which is
//           free until the last launch writes it;
//   exp     (head, chunk) blocks take the head max from those chunk maxes (fmax is
//           exact in any order), exponentiate their chunk in place and park the max
//           in the head's rel-query row, which only the scores launch reads;
//   output  (head, 32-dim slice) blocks sum the denominator, then the value groups
//           per dim, combined in group order (exact mode: the reference's serial
//           key-order denominator and 4 key groups, bit for bit).
// FAK_INKLING2_ATTEND_SPLIT=0 keeps the reference kernel.
//
// The serial denominator costs about 40 ns per key of dependent DADD latency, which
// dominates decode at depth. FAK_INKLING2_ATTEND_TREE=1 sums it as a fixed-shape tree
// instead and runs 16 key groups per dim: deterministic, but not bit-identical to the
// reference (the order of the double adds and of the float value sums changes), so it
// stays opt-in until a logit-KL receipt clears it against the batch-size noise floor.
// The default keeps the bit-identical serial output launch.

#ifndef INK2_ATS_CHUNKS
#define INK2_ATS_CHUNKS 32
#endif
#define INK2_ATS_TREE_GROUPS 16
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

template <class T, bool NVFP4 = false>
__global__ void __launch_bounds__(INK2_ATS_WARPS * 32)
    k_ink2_attend_scores(const float *q, const float *r, const float *sinks, const float *relBias,
                         const T *kcache, double *scores, float *out, const float *meta, int heads, int kvHeads,
                         int headDim, int relLen, int extent, int window, int cap, float scale, Ink2Seq sq,
                         const void *kscale = nullptr) {
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
    const size_t slotIdx = (size_t)(p % cap);
    float d = 0.f;
    if constexpr (NVFP4) {
      const int hBytes = headDim >> 1, hBlks = headDim >> 4;
      const uint8_t *kr = (const uint8_t *)kcache + slotIdx * (size_t)(kvHeads * hBytes) + (size_t)kvh * hBytes;
      const uint8_t *ks = (const uint8_t *)kscale + slotIdx * (size_t)(kvHeads * hBlks) + (size_t)kvh * hBlks;
      for (int bi = lane; bi < hBytes; bi += 32) {
        const float2 kv = ink2_e2m1x2_decode_f2(kr[bi]);
        const float bsc = ink2_e4m3fn_decode(ks[bi >> 3]);
        const int d0 = bi << 1;
        d = fmaf(qh[d0], __fmul_rn(kv.x, bsc), d);
        d = fmaf(qh[d0 + 1], __fmul_rn(kv.y, bsc), d);
      }
    } else {
      const T *kr = kcache + slotIdx * kvDim + (size_t)kvh * headDim;
      for (int i = lane; i < headDim; i += 32) d = fmaf(qh[i], ink2_kv_f(kr[i]), d);
    }
    float rd = 0.f;
    if (dist < extent) {
      const float *rb = relBias + (size_t)dist * relLen;
      for (int i = lane; i < relLen; i += 32) rd = fmaf(rb[i], rh[i], rd);
    }
    for (int o = 16; o > 0; o >>= 1) {
      d += __shfl_xor_sync(0xffffffff, d, o);
      rd += __shfl_xor_sync(0xffffffff, rd, o);
    }
    if constexpr (!NVFP4 && sizeof(T) == 1)
      d = __fmul_rn(d, ((const float *)kscale)[slotIdx * (size_t)kvHeads + (size_t)kvh]);
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
template <class T, bool NVFP4>
static __device__ __forceinline__ float ink2_ats_vval(const T *vcache, const void *vscale, size_t slotIdx, int kvHeads,
                                                      int headDim, int kvh, int dd) {
  if constexpr (NVFP4) {
    const int hBytes = headDim >> 1, hBlks = headDim >> 4;
    const uint8_t packed = ((const uint8_t *)vcache)[slotIdx * (size_t)(kvHeads * hBytes) + (size_t)kvh * hBytes + (dd >> 1)];
    const uint8_t scb = ((const uint8_t *)vscale)[slotIdx * (size_t)(kvHeads * hBlks) + (size_t)kvh * hBlks + (dd >> 4)];
    const float2 vv = ink2_e2m1x2_decode_f2(packed);
    return __fmul_rn((dd & 1) ? vv.y : vv.x, ink2_e4m3fn_decode(scb));
  } else {
    const T *vbase = vcache + (size_t)kvh * headDim + dd;
    return ink2_kv_val(vbase[slotIdx * (size_t)(kvHeads * headDim)], (const float *)vscale, slotIdx, kvHeads, kvh);
  }
}

template <bool STAGED, class T, bool NVFP4 = false>
__global__ void __launch_bounds__(INK2_ATT_THREADS)
    k_ink2_attend_out(const double *scores, const T *vcache, const float *r, float *out, const float *meta,
                      int heads, int kvHeads, int headDim, int relLen, int window, int cap, Ink2Seq sq,
                      const void *vscale = nullptr) {
  ink2_pdl_enter();
  const int h = blockIdx.x, seq = blockIdx.z;
  scores = ink2_row(scores, sq.scores, seq);
  vcache = ink2_row(vcache, sq.vc, seq);
  r = ink2_row(r, sq.r, seq);
  out = ink2_row(out, sq.out, seq);
  meta = ink2_row(meta, sq.meta, seq);
  const int kvh = h / (heads / kvHeads);
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
    float acc = 0.f;
    int j = g;
    for (; j + (INK2_ATS_UNROLL - 1) * groups < n; j += INK2_ATS_UNROLL * groups) {
      float vv[INK2_ATS_UNROLL];
#pragma unroll
      for (int u = 0; u < INK2_ATS_UNROLL; ++u) {
        const size_t slotIdx = (size_t)((first + j + u * groups) % cap);
        vv[u] = ink2_ats_vval<T, NVFP4>(vcache, vscale, slotIdx, kvHeads, headDim, kvh, dd);
      }
#pragma unroll
      for (int u = 0; u < INK2_ATS_UNROLL; ++u) acc = fmaf(INK2_ATS_W(j + u * groups), vv[u], acc);
    }
    for (; j < n; j += groups) {
      const size_t slotIdx = (size_t)((first + j) % cap);
      acc = fmaf(INK2_ATS_W(j), ink2_ats_vval<T, NVFP4>(vcache, vscale, slotIdx, kvHeads, headDim, kvh, dd), acc);
    }
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

// The tree output launch: block of 32 dims x INK2_ATS_TREE_GROUPS key groups. Each
// thread sums a key-strided slice of the exps, warps reduce by a fixed xor pattern and
// thread 0 adds the sink term and the warp partials in warp order. STAGED keeps the
// weights in shared memory; unstaged recomputes them from global, giving the same bits.
template <bool STAGED, class T, bool NVFP4 = false>
__global__ void __launch_bounds__(INK2_ATS_TREE_GROUPS * 32)
    k_ink2_attend_out_tree(const double *scores, const T *vcache, const float *r, float *out, const float *meta,
                           int heads, int kvHeads, int headDim, int relLen, int window, int cap, Ink2Seq sq,
                           const void *vscale = nullptr) {
  ink2_pdl_enter();
  const int h = blockIdx.x, seq = blockIdx.z;
  scores = ink2_row(scores, sq.scores, seq);
  vcache = ink2_row(vcache, sq.vc, seq);
  r = ink2_row(r, sq.r, seq);
  out = ink2_row(out, sq.out, seq);
  meta = ink2_row(meta, sq.meta, seq);
  const int kvh = h / (heads / kvHeads);
  int first;
  const int n = ink2_ats_keys(meta, window, cap, &first);
  const double *sc = scores + (size_t)h * cap;
  extern __shared__ float wsh[];
  __shared__ double wpart[INK2_ATS_TREE_GROUPS];
  __shared__ double sdenom;
  const int lane = threadIdx.x & 31, g = threadIdx.x >> 5;
  double part = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) part += sc[j];
  for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
  if (lane == 0) wpart[g] = part;
  __syncthreads();
  if (threadIdx.x == 0) {
    double denom = exp(-*(const double *)(r + (size_t)h * relLen));
    for (int w = 0; w < INK2_ATS_TREE_GROUPS; ++w) denom += wpart[w];
    sdenom = denom;
  }
  __syncthreads();
  const double denom = sdenom;
  if (STAGED) {
    for (int j = threadIdx.x; j < n; j += blockDim.x) wsh[j] = (float)(sc[j] / denom);
    __syncthreads();
  }
#define INK2_ATS_TW(j) (STAGED ? wsh[(j)] : (float)(sc[(j)] / denom))
  const int dd = blockIdx.y * 32 + lane;
  __shared__ float gacc[INK2_ATS_TREE_GROUPS][32];
  float acc = 0.f;
  if (dd < headDim) {
    const int groups = INK2_ATS_TREE_GROUPS;
    int j = g;
    for (; j + (INK2_ATS_UNROLL - 1) * groups < n; j += INK2_ATS_UNROLL * groups) {
      float vv[INK2_ATS_UNROLL];
#pragma unroll
      for (int u = 0; u < INK2_ATS_UNROLL; ++u) {
        const size_t slotIdx = (size_t)((first + j + u * groups) % cap);
        vv[u] = ink2_ats_vval<T, NVFP4>(vcache, vscale, slotIdx, kvHeads, headDim, kvh, dd);
      }
#pragma unroll
      for (int u = 0; u < INK2_ATS_UNROLL; ++u) acc = fmaf(INK2_ATS_TW(j + u * groups), vv[u], acc);
    }
    for (; j < n; j += groups) {
      const size_t slotIdx = (size_t)((first + j) % cap);
      acc = fmaf(INK2_ATS_TW(j), ink2_ats_vval<T, NVFP4>(vcache, vscale, slotIdx, kvHeads, headDim, kvh, dd), acc);
    }
  }
#undef INK2_ATS_TW
  gacc[g][lane] = acc;
  __syncthreads();
  if (threadIdx.x < 32 && dd < headDim) {
    float a = 0.f;
    for (int gg = 0; gg < INK2_ATS_TREE_GROUPS; ++gg) a += gacc[gg][threadIdx.x];
    out[(size_t)h * headDim + dd] = a;
  }
}

// Keys per staged weight tile: a multiple of every groups * INK2_ATS_UNROLL, so a
// thread keeps the keys congruent to its group across tiles and accumulates them in
// the same order as the staged form. Shared memory is fixed at this tile size, so the
// cost follows the live key count, not cap.
#define INK2_ATS_TILE 2048

// Past the staging limit in the exact mode: the denominator is a block-parallel sum over
// the live keys, one fp32 reciprocal per head, and the weights are staged a tile at a
// time. It matches the staged form to rounding, not bit for bit, and scales strictly
// with live key count n rather than cap.
template <class T, bool NVFP4 = false>
__global__ void __launch_bounds__(INK2_ATT_THREADS)
    k_ink2_attend_out_tiled(const double *scores, const T *vcache, const float *r, float *out, const float *meta,
                            int heads, int kvHeads, int headDim, int relLen, int window, int cap, Ink2Seq sq,
                            const void *vscale = nullptr) {
  ink2_pdl_enter();
  const int h = blockIdx.x, seq = blockIdx.z;
  scores = ink2_row(scores, sq.scores, seq);
  vcache = ink2_row(vcache, sq.vc, seq);
  r = ink2_row(r, sq.r, seq);
  out = ink2_row(out, sq.out, seq);
  meta = ink2_row(meta, sq.meta, seq);
  const int kvh = h / (heads / kvHeads);
  int first;
  const int n = ink2_ats_keys(meta, window, cap, &first);
  const double *sc = scores + (size_t)h * cap;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __shared__ double wpart[INK2_ATT_THREADS / 32];
  __shared__ float sinv;
  double part = 0.0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) part += sc[j];
  for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
  if (lane == 0) wpart[warp] = part;
  __syncthreads();
  if (threadIdx.x == 0) {
    double denom = exp(-*(const double *)(r + (size_t)h * relLen));
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) denom += wpart[w];
    sinv = (float)(1.0 / denom);
  }
  __syncthreads();
  const float inv = sinv;
  const int groups = INK2_ATT_THREADS / headDim;
  const int dd = blockIdx.y * 32 + lane, g = warp;
  const bool active = g < groups && dd < headDim;
  __shared__ float wsh[INK2_ATS_TILE];
  __shared__ float gacc[INK2_ATT_GROUPS][32];
  float acc = 0.f;
  for (int t0 = 0; t0 < n; t0 += INK2_ATS_TILE) {
    const int tn = min(INK2_ATS_TILE, n - t0);
    __syncthreads();
    for (int j = threadIdx.x; j < tn; j += blockDim.x) wsh[j] = __fmul_rn((float)sc[t0 + j], inv);
    __syncthreads();
    if (active) {
      const int kt = first + t0;
      int j = g;
      for (; j + (INK2_ATS_UNROLL - 1) * groups < tn; j += INK2_ATS_UNROLL * groups) {
        float vv[INK2_ATS_UNROLL];
#pragma unroll
        for (int u = 0; u < INK2_ATS_UNROLL; ++u) {
          const size_t slotIdx = (size_t)((kt + j + u * groups) % cap);
          vv[u] = ink2_ats_vval<T, NVFP4>(vcache, vscale, slotIdx, kvHeads, headDim, kvh, dd);
        }
#pragma unroll
        for (int u = 0; u < INK2_ATS_UNROLL; ++u) acc = fmaf(wsh[j + u * groups], vv[u], acc);
      }
      for (; j < tn; j += groups) {
        const size_t slotIdx = (size_t)((kt + j) % cap);
        acc = fmaf(wsh[j], ink2_ats_vval<T, NVFP4>(vcache, vscale, slotIdx, kvHeads, headDim, kvh, dd), acc);
      }
    }
  }
  if (active) gacc[g][lane] = acc;
  __syncthreads();
  if (threadIdx.x < 32 && dd < headDim) {
    float a = 0.f;
    for (int gg = 0; gg < groups; ++gg) a += gacc[gg][threadIdx.x];
    out[(size_t)h * headDim + dd] = a;
  }
}

// Past the staging limit in the tree mode (>24576 cap): same fixed INK2_ATS_TILE
// shared-memory weight staging and single fp32 reciprocal as k_ink2_attend_out_tiled,
// keeping the tree mode's 16 key groups and avoiding unstaged global fp64 divides.
template <class T, bool NVFP4 = false>
__global__ void __launch_bounds__(INK2_ATS_TREE_GROUPS * 32)
    k_ink2_attend_out_tree_tiled(const double *scores, const T *vcache, const float *r, float *out, const float *meta,
                                 int heads, int kvHeads, int headDim, int relLen, int window, int cap, Ink2Seq sq,
                                 const void *vscale = nullptr) {
  ink2_pdl_enter();
  const int h = blockIdx.x, seq = blockIdx.z;
  scores = ink2_row(scores, sq.scores, seq);
  vcache = ink2_row(vcache, sq.vc, seq);
  r = ink2_row(r, sq.r, seq);
  out = ink2_row(out, sq.out, seq);
  meta = ink2_row(meta, sq.meta, seq);
  const int kvh = h / (heads / kvHeads);
  int first;
  const int n = ink2_ats_keys(meta, window, cap, &first);
  const double *sc = scores + (size_t)h * cap;
  __shared__ double wpart[INK2_ATS_TREE_GROUPS];
  __shared__ float sinv;
  const int lane = threadIdx.x & 31, g = threadIdx.x >> 5;
  double part = 0.0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) part += sc[j];
  for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
  if (lane == 0) wpart[g] = part;
  __syncthreads();
  if (threadIdx.x == 0) {
    double denom = exp(-*(const double *)(r + (size_t)h * relLen));
    for (int w = 0; w < INK2_ATS_TREE_GROUPS; ++w) denom += wpart[w];
    sinv = (float)(1.0 / denom);
  }
  __syncthreads();
  const float inv = sinv;
  const int dd = blockIdx.y * 32 + lane;
  __shared__ float wsh[INK2_ATS_TILE];
  __shared__ float gacc[INK2_ATS_TREE_GROUPS][32];
  float acc = 0.f;
  const int groups = INK2_ATS_TREE_GROUPS;
  for (int t0 = 0; t0 < n; t0 += INK2_ATS_TILE) {
    const int tn = min(INK2_ATS_TILE, n - t0);
    __syncthreads();
    for (int j = threadIdx.x; j < tn; j += blockDim.x) wsh[j] = __fmul_rn((float)sc[t0 + j], inv);
    __syncthreads();
    if (dd < headDim) {
      const int kt = first + t0;
      int j = g;
      for (; j + (INK2_ATS_UNROLL - 1) * groups < tn; j += INK2_ATS_UNROLL * groups) {
        float vv[INK2_ATS_UNROLL];
#pragma unroll
        for (int u = 0; u < INK2_ATS_UNROLL; ++u) {
          const size_t slotIdx = (size_t)((kt + j + u * groups) % cap);
          vv[u] = ink2_ats_vval<T, NVFP4>(vcache, vscale, slotIdx, kvHeads, headDim, kvh, dd);
        }
#pragma unroll
        for (int u = 0; u < INK2_ATS_UNROLL; ++u) acc = fmaf(wsh[j + u * groups], vv[u], acc);
      }
      for (; j < tn; j += groups) {
        const size_t slotIdx = (size_t)((kt + j) % cap);
        acc = fmaf(wsh[j], ink2_ats_vval<T, NVFP4>(vcache, vscale, slotIdx, kvHeads, headDim, kvh, dd), acc);
      }
    }
  }
  gacc[g][lane] = acc;
  __syncthreads();
  if (threadIdx.x < 32 && dd < headDim) {
    float a = 0.f;
    for (int gg = 0; gg < INK2_ATS_TREE_GROUPS; ++gg) a += gacc[gg][threadIdx.x];
    out[(size_t)h * headDim + dd] = a;
  }
}

static int g_ink2_ats_exact = -1;

static bool ink2_ats_exact(void) {
  if (g_ink2_ats_exact < 0) {
    const char *v = getenv("FAK_INKLING2_ATTEND_TREE");
    g_ink2_ats_exact = (v && v[0] == '1' && v[1] == 0) ? 0 : 1;
  }
  return g_ink2_ats_exact == 1;
}

// Test hook: 1 selects the bit-identical serial output launch, 0 the tree; returns the
// previous mode.
extern "C" int fcuda_ink2_attend_exact_select(int mode) {
  const int prev = ink2_ats_exact() ? 1 : 0;
  g_ink2_ats_exact = mode ? 1 : 0;
  return prev;
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
// sequence of sq; every sequence's r and out rows must keep that alignment. T is the
// cache element; each instantiation keeps its own shared-memory opt-in state, as the
// attribute is per kernel.
template <class T, bool NVFP4 = false>
static bool ink2_attend_split(const float *q, const float *r, const float *sinks, const float *relBias,
                              const T *kcache, const T *vcache, double *scores, float *out, const float *meta,
                              int heads, int kvHeads, int headDim, int relLen, int extent, int window, int cap,
                              float scale, const Ink2Seq &sq, int C, const void *kscale = nullptr,
                              const void *vscale = nullptr) {
  // Size chunk count to the layer's maximum live key span: local sliding-window
  // layers (window=128) and small/medium global caches (cap<=4096) keep 8 chunks
  // (192 blocks at 24 heads) so short/windowed decode does not pay 4x block launch
  // overhead across 47 visits; deep global caches (cap>4096) widen to 16 chunks.
  const int maxKeys = (window > 0 && window < cap) ? window : cap;
  int chunks = maxKeys > 4096 ? 16 : 8;
  if (chunks > INK2_ATS_CHUNKS) chunks = INK2_ATS_CHUNKS;
  if (!ink2_ats_enabled() || headDim / 2 < chunks || chunks < 1 || (headDim & 31) != 0 || relLen < 2 ||
      (relLen & 1) != 0 || ((uintptr_t)r & 7) != 0 || ((uintptr_t)out & 7) != 0 || (sq.r & 1) != 0 ||
      (sq.out & 1) != 0)
    return false;
  // the opt-in is per kernel and per staging footprint: exact stages exps and weights,
  // the tree only weights
  const bool exact = ink2_ats_exact();
  size_t osmem = (size_t)cap * (exact ? sizeof(double) + sizeof(float) : sizeof(float));
  static size_t optin[2] = {0, 0}, refused[2] = {0, 0};
  const int k = exact ? 0 : 1;
  bool staged = true;
  if (osmem > optin[k]) {
    const cudaError_t e =
        refused[k] != 0 && osmem >= refused[k] ? cudaErrorInvalidValue
        : exact ? cudaFuncSetAttribute(k_ink2_attend_out<true, T, NVFP4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)osmem)
                : cudaFuncSetAttribute(k_ink2_attend_out_tree<true, T, NVFP4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)osmem);
    if (e != cudaSuccess) {
      cudaGetLastError();
      if (refused[k] == 0 || osmem < refused[k]) refused[k] = osmem;
      staged = false;
    } else {
      optin[k] = osmem;
    }
  }
  if (!staged) osmem = 0;
  const dim3 g1(heads, chunks, C), g3(heads, headDim / 32, C);
  FCUDA_TS("k_ink2_attend_scores", g1, (k_ink2_attend_scores<T, NVFP4><<<g1, INK2_ATS_WARPS * 32, 0, g_stream>>>(
                                           q, r, sinks, relBias, kcache, scores, out, meta, heads, kvHeads, headDim,
                                           relLen, extent, window, cap, scale, sq, kscale)));
  FCUDA_TS("k_ink2_attend_exp", g1, k_ink2_attend_exp<<<g1, 256, 0, g_stream>>>(scores, out, (float *)r, meta, headDim,
                                                                                  relLen, window, cap, chunks, sq));
  const int othreads = (INK2_ATT_THREADS / headDim) * 32;
  if (!exact) {
    if (staged) {
      FCUDA_TS("k_ink2_attend_out_tree", g3, (k_ink2_attend_out_tree<true, T, NVFP4><<<g3, INK2_ATS_TREE_GROUPS * 32, osmem, g_stream>>>(
                                                 scores, vcache, r, out, meta, heads, kvHeads, headDim, relLen, window, cap, sq, vscale)));
    } else {
      FCUDA_TS("k_ink2_attend_out_tree_tiled", g3,
               (k_ink2_attend_out_tree_tiled<T, NVFP4><<<g3, INK2_ATS_TREE_GROUPS * 32, 0, g_stream>>>(
                   scores, vcache, r, out, meta, heads, kvHeads, headDim, relLen, window, cap, sq, vscale)));
    }
  } else if (staged) {
    FCUDA_TS("k_ink2_attend_out", g3, (k_ink2_attend_out<true, T, NVFP4><<<g3, othreads, osmem, g_stream>>>(
                                          scores, vcache, r, out, meta, heads, kvHeads, headDim, relLen, window, cap, sq, vscale)));
  } else {
    FCUDA_TS("k_ink2_attend_out_tiled", g3, (k_ink2_attend_out_tiled<T, NVFP4><<<g3, othreads, 0, g_stream>>>(
                                                scores, vcache, r, out, meta, heads, kvHeads, headDim, relLen, window,
                                                cap, sq, vscale)));
  }
  return true;
}
