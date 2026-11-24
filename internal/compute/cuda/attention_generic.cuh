// Generic-head-dim f32 attention for decode and the dense prompt panel.
//
// One block owns WARPS*NS query rows of a single KV head group (row = token*grp + g),
// stages 32-key K/V tiles in shared memory once and reuses them for every row.
// Scores: lane j computes the full dot for key j of the tile, so the softmax max and
// sum are two warp-shuffle reductions per tile instead of a block reduction per key.
// Values: lane owns dims d = lane + 32*i.
//
// A row's keys are cut into attn_generic_chunk(n) chunks; each chunk runs its own
// online softmax and the chunk states are folded in key order by attn_generic_fold.
// Decode spreads the chunks over grid.z and folds them in k_attention_generic_merge;
// the prompt panel folds them in registers. Both apply the same operations in the
// same order, so a panel row is bitwise the decode step over the same keys.
//
// Semantics match k_flash_attention / k_prompt_causal_attention_panel: row t of head h
// reads KV head h / (nH/nKV) over keys [0, prefix+t], score = dot(q, k) * scale,
// output acc / l with l == 0 mapped to zero. Only f32 reduction order differs.

#include "attention_generic_plan.h"

// Folds chunk state (mC, lC, xC) into the running total (mT, lT, xT). Explicit
// rounding intrinsics pin the operation sequence across the two fold sites.
struct attn_generic_fold_coef {
  float m, a, b;
};
static __device__ __forceinline__ attn_generic_fold_coef attn_generic_fold(float mT, float mC) {
  attn_generic_fold_coef c;
  c.m = fmaxf(mT, mC);
  c.a = expf(mT - c.m);
  c.b = expf(mC - c.m);
  return c;
}
static __device__ __forceinline__ float attn_generic_fold_value(float xT, float xC, attn_generic_fold_coef c) {
  return __fmaf_rn(xC, c.b, __fmul_rn(xT, c.a));
}

// nsplits > 1: decode partial mode. Every row sees the same n keys, block z owns
// chunk z and writes its unfolded (m, l, acc) to Part. nsplits == 1: the block
// walks every key its rows can see and folds each row's chunks itself.
template <int NS>
__global__ void __launch_bounds__(ATTN_GENERIC_WARPS * 32)
k_attention_generic_tiled(const float *__restrict__ Q, const float *__restrict__ K,
                          const float *__restrict__ V, float *__restrict__ Out,
                          float *__restrict__ Part, int tokens, int prefix, int nH, int nKV,
                          int hd, float scale, int nsplits) {
  constexpr int ROWS = ATTN_GENERIC_WARPS * NS;
  const int grp = nH / nKV;
  const int kvh = blockIdx.y;
  const int split = blockIdx.z;
  const int rowBase = blockIdx.x * ROWS;
  const int groupRows = tokens * grp;
  const int w = nKV * hd;
  const int kstride = hd | 1;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  extern __shared__ float smem[];
  float *qs = smem;                             // [ROWS][hd]
  float *ks = qs + ROWS * hd;                   // [TILE][kstride]
  float *vs = ks + ATTN_GENERIC_TILE * kstride; // [TILE][hd]
  float *ps = vs + ATTN_GENERIC_TILE * hd;      // [ROWS][TILE]

  for (int i = threadIdx.x; i < ROWS * hd; i += blockDim.x) {
    int r = i / hd, d = i - r * hd;
    int row = rowBase + r;
    float q = 0.f;
    if (row < groupRows) {
      int token = row / grp, g = row - token * grp;
      q = Q[(size_t)token * nH * hd + (size_t)(kvh * grp + g) * hd + d];
    }
    qs[i] = q;
  }

  // Block row r = s*WARPS + warp; limit is the last visible key, -1 for padding rows.
  int limit[NS], chunk[NS];
#pragma unroll
  for (int s = 0; s < NS; s++) {
    int row = rowBase + s * ATTN_GENERIC_WARPS + warp;
    limit[s] = row < groupRows ? prefix + row / grp : -1;
    chunk[s] = attn_generic_chunk(limit[s] + 1);
  }
  const int lastRow = min(rowBase + ROWS, groupRows) - 1;
  int keyBegin = 0;
  int keyEnd = prefix + lastRow / grp + 1;
  if (nsplits > 1) {
    const int c = attn_generic_chunk(keyEnd);
    keyBegin = split * c;
    keyEnd = min(keyEnd, keyBegin + c);
  }

  float mRun[NS], lRun[NS], acc[NS][ATTN_GENERIC_ACC];
  float mTot[NS], lTot[NS], accTot[NS][ATTN_GENERIC_ACC];
#pragma unroll
  for (int s = 0; s < NS; s++) {
    mRun[s] = mTot[s] = -INFINITY;
    lRun[s] = lTot[s] = 0.f;
#pragma unroll
    for (int i = 0; i < ATTN_GENERIC_ACC; i++) acc[s][i] = accTot[s][i] = 0.f;
  }

  for (int t0 = keyBegin; t0 < keyEnd; t0 += ATTN_GENERIC_TILE) {
    const int tn = min(ATTN_GENERIC_TILE, keyEnd - t0);
    __syncthreads(); // query rows visible; previous tile fully consumed
    for (int i = threadIdx.x; i < tn * hd; i += blockDim.x) {
      int j = i / hd, d = i - j * hd;
      size_t off = (size_t)(t0 + j) * w + (size_t)kvh * hd + d;
      ks[j * kstride + d] = K[off];
      vs[j * hd + d] = V[off];
    }
    __syncthreads();

    float dot[NS];
#pragma unroll
    for (int s = 0; s < NS; s++) dot[s] = 0.f;
    if (lane < tn) {
      const float *kr = ks + lane * kstride;
      for (int d = 0; d < hd; d++) {
        float kd = kr[d];
#pragma unroll
        for (int s = 0; s < NS; s++) dot[s] += qs[(s * ATTN_GENERIC_WARPS + warp) * hd + d] * kd;
      }
    }

#pragma unroll
    for (int s = 0; s < NS; s++) {
      const int lim = limit[s];
      if (lim < t0) continue; // warp-uniform: padding row, or tile entirely past its window
      if (nsplits == 1 && t0 > 0 && t0 % chunk[s] == 0) {
        attn_generic_fold_coef c = attn_generic_fold(mTot[s], mRun[s]);
        lTot[s] = attn_generic_fold_value(lTot[s], lRun[s], c);
#pragma unroll
        for (int i = 0; i < ATTN_GENERIC_ACC; i++) {
          accTot[s][i] = attn_generic_fold_value(accTot[s][i], acc[s][i], c);
          acc[s][i] = 0.f;
        }
        mTot[s] = c.m;
        mRun[s] = -INFINITY;
        lRun[s] = 0.f;
      }
      const int r = s * ATTN_GENERIC_WARPS + warp;
      const bool valid = lane < tn && t0 + lane <= lim;
      const float sc = dot[s] * scale;
      float mx = valid ? sc : -INFINITY;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
      const float mNew = fmaxf(mRun[s], mx);
      const float corr = expf(mRun[s] - mNew);
      const float p = valid ? expf(sc - mNew) : 0.f;
      float sum = p;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
      lRun[s] = lRun[s] * corr + sum;
      mRun[s] = mNew;
      ps[r * ATTN_GENERIC_TILE + lane] = p;
      __syncwarp();
#pragma unroll
      for (int i = 0; i < ATTN_GENERIC_ACC; i++) acc[s][i] *= corr;
      const int jn = min(tn, lim - t0 + 1);
      const float *pr = ps + r * ATTN_GENERIC_TILE;
      for (int j = 0; j < jn; j++) {
        const float pj = pr[j];
        const float *vr = vs + j * hd;
#pragma unroll
        for (int i = 0; i < ATTN_GENERIC_ACC; i++) {
          int d = lane + 32 * i;
          if (d < hd) acc[s][i] += pj * vr[d];
        }
      }
    }
  }

#pragma unroll
  for (int s = 0; s < NS; s++) {
    int row = rowBase + s * ATTN_GENERIC_WARPS + warp;
    if (row >= groupRows) continue;
    int token = row / grp, h = kvh * grp + (row - token * grp);
    size_t outRow = (size_t)token * nH + h;
    if (nsplits == 1) {
      attn_generic_fold_coef c = attn_generic_fold(mTot[s], mRun[s]);
      float l = attn_generic_fold_value(lTot[s], lRun[s], c);
      float invL = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
      for (int i = 0; i < ATTN_GENERIC_ACC; i++) {
        int d = lane + 32 * i;
        if (d < hd) Out[outRow * hd + d] = attn_generic_fold_value(accTot[s][i], acc[s][i], c) * invL;
      }
    } else {
      size_t slot = outRow * nsplits + split;
#pragma unroll
      for (int i = 0; i < ATTN_GENERIC_ACC; i++) {
        int d = lane + 32 * i;
        if (d < hd) Part[slot * hd + d] = acc[s][i];
      }
      if (lane == 0) {
        float *ml = Part + (size_t)tokens * nH * nsplits * hd;
        ml[2 * slot] = mRun[s];
        ml[2 * slot + 1] = lRun[s];
      }
    }
  }
}

// Folds the decode chunk partials of one (token, head) row in chunk order.
__global__ void k_attention_generic_merge(const float *__restrict__ Part, float *__restrict__ Out,
                                          int rows, int hd, int nsplits) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float *ml = Part + (size_t)rows * nsplits * hd + (size_t)row * nsplits * 2;
  for (int d = threadIdx.x; d < hd; d += blockDim.x) {
    float mT = -INFINITY, lT = 0.f, xT = 0.f;
    for (int s = 0; s < nsplits; s++) {
      attn_generic_fold_coef c = attn_generic_fold(mT, ml[2 * s]);
      lT = attn_generic_fold_value(lT, ml[2 * s + 1], c);
      xT = attn_generic_fold_value(xT, Part[((size_t)row * nsplits + s) * hd + d], c);
      mT = c.m;
    }
    float invL = lT > 0.f ? 1.f / lT : 0.f;
    Out[(size_t)row * hd + d] = xT * invL;
  }
}

static unsigned long long g_attn_generic_launches[2] = {0, 0}; // [decode, panel]
static float *g_attn_generic_part = nullptr;
static size_t g_attn_generic_part_cap = 0; // floats

// Decode partials need nH*splits*(hd+2) floats. The first reservation takes the
// geometry's worst case (MAX_SPLITS chunks), so a model's decode never reallocates
// a pointer an already captured graph baked in. Growth is refused while a capture
// is open, and the caller then keeps the legacy kernel.
static bool attn_generic_reserve_part(size_t need, size_t worst) {
  if (need <= g_attn_generic_part_cap) return true;
  if (g_capture_open) return false;
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(g_stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone) {
    return false;
  }
  size_t cap = worst > need ? worst : need;
  if (g_attn_generic_part) {
    float *retiring = g_attn_generic_part;
    g_attn_generic_part = nullptr;
    g_attn_generic_part_cap = 0;
    fak_alloc_untrack(retiring);
    if (cudaFree(retiring) != cudaSuccess) return false;
  }
  float *p = nullptr;
  cudaError_t e = fcuda_alloc_status(cudaMalloc(&p, cap * sizeof(float)));
  if (e != cudaSuccess || !p) return false;
  fak_alloc_track(p, FAK_ALLOC_ATTENTION_SCRATCH, FAK_ALLOC_DEVICE_LOCAL,
                  FAK_ALLOC_SITE("attention-generic-splitkv"), cap * sizeof(float));
  g_attn_generic_part = p;
  g_attn_generic_part_cap = cap;
  return true;
}

template <int NS>
static bool attn_generic_smem_admit(size_t bytes) {
  static int admitted = -1; // max dynamic shared bytes this instantiation accepts
  if (bytes <= ATTN_GENERIC_DEFAULT_SMEM) return true;
  if (admitted < 0) {
    int want = (int)attn_generic_smem_bytes(ATTN_GENERIC_MAX_HD, ATTN_GENERIC_WARPS * NS);
    if (cudaFuncSetAttribute(k_attention_generic_tiled<NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, want) ==
        cudaSuccess) {
      admitted = want;
    } else {
      (void)cudaGetLastError(); // a refused opt-in is not a stream fault; the legacy kernel runs
      admitted = ATTN_GENERIC_DEFAULT_SMEM;
    }
  }
  return bytes <= (size_t)admitted;
}

template <int NS>
static void attn_generic_launch_tiled(dim3 grid, size_t smem, const float *dQ, const float *dK,
                                      const float *dV, float *dOut, float *part, int tokens,
                                      int prefix, int nH, int nKV, int hd, float scale, int nsplits) {
  FCUDA_TS("k_attention_generic_tiled", grid,
           k_attention_generic_tiled<NS><<<grid, ATTN_GENERIC_WARPS * 32, smem, g_stream>>>(
               dQ, dK, dV, dOut, part, tokens, prefix, nH, nKV, hd, scale, nsplits));
}

// Returns -2 when the generic path is off or does not admit the geometry (nothing was
// enqueued, the caller runs its legacy kernel), otherwise 0. Launch status is left for
// the caller's own status read so both callers keep their existing error contracts.
// which: 0 decode (tokens == 1, chunks spread over grid.z), 1 prompt panel.
static int attn_generic_launch(const float *dQ, const float *dK, const float *dV, float *dOut,
                               int tokens, int prefix, int nH, int nKV, int hd, float scale,
                               int which) {
  if (attn_generic_mode_legacy(getenv("FAK_CUDA_ATTN_GENERIC"))) return -2;
  if (!attn_generic_admits(tokens, prefix, nH, nKV, hd)) return -2;
  if (which == 0 && tokens != 1) return -2;
  const int grp = nH / nKV;
  const int groupRows = tokens * grp;
  const int ns = attn_generic_slots(groupRows);
  const int rows = ATTN_GENERIC_WARPS * ns;
  const size_t smem = attn_generic_smem_bytes(hd, rows);
  bool smemOK = ns == 1 ? attn_generic_smem_admit<1>(smem)
                : ns == 2 ? attn_generic_smem_admit<2>(smem)
                          : attn_generic_smem_admit<4>(smem);
  if (!smemOK) return -2;
  const int rowBlocks = (groupRows + rows - 1) / rows;
  int nsplits = which == 0 ? attn_generic_splits(prefix + 1) : 1;
  float *part = nullptr;
  if (nsplits > 1) {
    size_t slotWidth = (size_t)hd + 2;
    size_t need = (size_t)nH * nsplits * slotWidth;
    size_t worst = (size_t)nH * ATTN_GENERIC_MAX_SPLITS * slotWidth;
    if (!attn_generic_reserve_part(need, worst)) return -2;
    part = g_attn_generic_part;
  }
  dim3 grid(rowBlocks, nKV, nsplits);
  if (ns == 1) {
    attn_generic_launch_tiled<1>(grid, smem, dQ, dK, dV, dOut, part, tokens, prefix, nH, nKV, hd, scale, nsplits);
  } else if (ns == 2) {
    attn_generic_launch_tiled<2>(grid, smem, dQ, dK, dV, dOut, part, tokens, prefix, nH, nKV, hd, scale, nsplits);
  } else {
    attn_generic_launch_tiled<4>(grid, smem, dQ, dK, dV, dOut, part, tokens, prefix, nH, nKV, hd, scale, nsplits);
  }
  if (nsplits > 1) {
    FCUDA_TS("k_attention_generic_merge", nH,
             k_attention_generic_merge<<<nH, 128, 0, g_stream>>>(part, dOut, nH, hd, nsplits));
  }
  g_attn_generic_launches[which]++;
  return 0;
}

extern "C" unsigned long long fak_attn_generic_launch_count(int which) {
  return which == 0 || which == 1 ? g_attn_generic_launches[which] : 0ULL;
}
