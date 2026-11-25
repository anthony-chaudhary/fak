// inkling2_dspark_markov.cuh: the DSpark rnn_hidden Markov head on the device, with a
// base-logit shortlist. Drafted row i picks the argmax over the M best base logits
// lm_head(h_i) of lm_head(h_i) + lm_head(z_i), where z_i = hidden_out(latent_i) + bias
// and latent_i comes from a GRU-style state fed the previous drafted token's
// markov_w1 row. Included after inkling2_dspark.cuh by cuda_kernels.cu.

#define INK2D_TOPM_MAX 64
#define INK2D_TOPM_SLICES 64

// Selection order of the shortlist: larger value first, ties to the lower index; an
// index below zero is no candidate.
static __device__ __forceinline__ bool ink2d_topm_beats(float bv, int bi, float av, int ai) {
  return bi >= 0 && (ai < 0 || bv > av || (bv == av && bi < ai));
}

// A block's best (value, index): shuffle-reduced per warp, then across the warps through
// buf, which the caller alternates between rounds so one barrier per round is enough.
static __device__ __forceinline__ void ink2d_topm_block_best(float &v, int &i, float *bufV, int *bufI) {
  for (int d = 16; d > 0; d >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffff, v, d);
    const int oi = __shfl_xor_sync(0xffffffff, i, d);
    if (ink2d_topm_beats(ov, oi, v, i)) {
      v = ov;
      i = oi;
    }
  }
  if ((threadIdx.x & 31) == 0) {
    bufV[threadIdx.x >> 5] = v;
    bufI[threadIdx.x >> 5] = i;
  }
  __syncthreads();
  v = bufV[0];
  i = bufI[0];
  for (int w = 1; w < (int)(blockDim.x >> 5); ++w)
    if (ink2d_topm_beats(bufV[w], bufI[w], v, i)) {
      v = bufV[w];
      i = bufI[w];
    }
}

// Stage 1: each block keeps the M best of one vocab slice of one row, in selection
// order. A thread owns every 256th logit of the slice and caches the best it still
// holds; only the thread whose logit was taken rescans. The slice is at most 64 * 256
// logits (taken is one bit per owned logit).
__global__ void __launch_bounds__(256) k_ink2d_topm_slice(const float *logits, int rowStride, int vocab, int M,
                                                          float *candV, int *candI) {
  ink2_pdl_enter();
  __shared__ float bufV[2][8];
  __shared__ int bufI[2][8];
  const int row = blockIdx.y, slice = blockIdx.x;
  const int per = (vocab + INK2D_TOPM_SLICES - 1) / INK2D_TOPM_SLICES;
  const int lo = slice * per, hi = min(vocab, lo + per);
  const float *l = logits + (size_t)row * rowStride;
  unsigned long long taken = 0;
  float mine = -INFINITY;
  int minei = -1;
  for (int m = 0; m < M; ++m) {
    if (minei < 0) {
      int j = 0;
      for (int t = lo + threadIdx.x; t < hi; t += 256, ++j) {
        const float v = l[t];
        if (!((taken >> j) & 1ull) && (minei < 0 || v > mine)) {
          mine = v;
          minei = t;
        }
      }
    }
    float best = mine;
    int besti = minei;
    ink2d_topm_block_best(best, besti, bufV[m & 1], bufI[m & 1]);
    if (threadIdx.x == 0) {
      const size_t o = ((size_t)row * INK2D_TOPM_SLICES + slice) * M + m;
      candV[o] = besti >= 0 ? best : -INFINITY;
      candI[o] = besti;
    }
    if (besti >= 0 && besti == minei) {
      taken |= 1ull << ((minei - lo) >> 8);
      mine = -INFINITY;
      minei = -1;
    }
  }
}

// Stage 2: the M best of one row's slice candidates, written to out[row][M] in
// selection order (ties to the lower token). A thread owns every 256th candidate and
// caches the best it still holds.
__global__ void __launch_bounds__(256) k_ink2d_topm_merge(const float *candV, const int *candI, int M, int *out) {
  ink2_pdl_enter();
  __shared__ float bufV[2][8];
  __shared__ int bufI[2][8];
  const int row = blockIdx.x;
  const int n = INK2D_TOPM_SLICES * M;
  const float *v = candV + (size_t)row * n;
  const int *id = candI + (size_t)row * n;
  unsigned int taken = 0;
  float mine = -INFINITY;
  int minei = -1, minej = 0;
  for (int m = 0; m < M; ++m) {
    if (minei < 0) {
      int j = 0;
      for (int i = threadIdx.x; i < n; i += 256, ++j) {
        const float cv = v[i];
        const int ci = id[i];
        if (!((taken >> j) & 1u) && ink2d_topm_beats(cv, ci, mine, minei)) {
          mine = cv;
          minei = ci;
          minej = j;
        }
      }
    }
    float best = mine;
    int besti = minei;
    ink2d_topm_block_best(best, besti, bufV[m & 1], bufI[m & 1]);
    if (threadIdx.x == 0) out[(size_t)row * M + m] = besti >= 0 ? besti : 0;
    if (besti >= 0 && besti == minei) {
      taken |= 1u << minej;
      mine = -INFINITY;
      minei = -1;
    }
  }
}

// in = [state | markov_w1[prev] | h | fRNN]: prev is the anchor at step 0, else the token
// drafted at step-1 (idx[4*(step-1)]); fRNN is the ring row of position *posDev-1; the
// state starts at zero on step 0.
__global__ void k_ink2d_markov_in(const float *state, const unsigned short *w1, const int *anchor, const int *idx,
                                  int step, const float *h, const float *frnnRing, const int *posDev, int cap,
                                  int rank, int hidden, float *in) {
  ink2_pdl_enter();
  const int prev = step == 0 ? *anchor : idx[4 * (step - 1)];
  const float *fr = frnnRing + (size_t)((*posDev - 1 + cap) % cap) * hidden;
  const int n = 2 * rank + 2 * hidden;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    float v;
    if (i < rank)
      v = step == 0 ? 0.f : state[i];
    else if (i < 2 * rank)
      v = __uint_as_float((unsigned)w1[(size_t)prev * rank + (i - rank)] << 16);
    else if (i < 2 * rank + hidden)
      v = h[i - 2 * rank];
    else
      v = fr[i - 2 * rank - hidden];
    in[i] = v;
  }
}

// GRU-style update of state from raw + bias, and latent = tanh(raw[2r:] + bias).
__global__ void k_ink2d_markov_gate(const float *raw, const float *bias, float *state, float *latent, int rank,
                                    int step) {
  ink2_pdl_enter();
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= rank) return;
  const float g = 1.f / (1.f + expf(-(raw[c] + bias[c])));
  const float cand = tanhf(raw[rank + c] + bias[rank + c]);
  const float prev = step == 0 ? 0.f : state[c];
  state[c] = g * prev + (1.f - g) * cand;
  latent[c] = tanhf(raw[2 * rank + c] + bias[2 * rank + c]);
}

// One step's draft token: argmax over the shortlist of base[c] + lm_head[c].(z + zb),
// lm_head a Q5_K [vocab][in] matrix decoded as k_q5k_gemm does. One warp per candidate.
// idx[4*step + 1] receives the float bits of the winner's lead over the runner-up, the
// draft's confidence in the step.
__global__ void k_ink2d_markov_pick(const unsigned char *W, int in, const float *base, const int *cand, int M,
                                    const float *z, const float *zb, int *idx, int step) {
  ink2_pdl_enter();
  __shared__ float zs[4096];
  __shared__ float sv[32];
  __shared__ float sv2[32];
  __shared__ int si[32];
  for (int i = threadIdx.x; i < in; i += blockDim.x) zs[i] = z[i] + zb[i];
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
  float best = -INFINITY, second = -INFINITY;
  int besti = 0x7fffffff;
  const int nsb = in / 256;
  for (int m = warp; m < M; m += nwarp) {
    const int o = cand[m];
    const unsigned char *wr = W + (size_t)o * nsb * 176;
    float sum = 0.f;
    for (int sb = 0; sb < nsb; ++sb) {
      const unsigned char *b = wr + (size_t)sb * 176;
      const float d = __half2float(*(const __half *)b);
      const float dm = __half2float(*(const __half *)(b + 2));
      const unsigned char *sc = b + 4, *qh = b + 16, *ql = b + 48;
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        unsigned char scale, mn;
        getScaleMinK4_dev(g, sc, &scale, &mn);
        const int pair = g >> 1;
        int q = (g & 1) ? (ql[pair * 32 + lane] >> 4) : (ql[pair * 32 + lane] & 15);
        if (qh[lane] & (1u << g)) q += 16;
        sum += (d * (float)scale * (float)q - dm * (float)mn) * zs[sb * 256 + g * 32 + lane];
      }
    }
    for (int s = 16; s > 0; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
    const float v = base[o] + sum;
    if (v > best || (v == best && o < besti)) {
      second = best;
      best = v;
      besti = o;
    } else if (v > second) {
      second = v;
    }
  }
  if (lane == 0) {
    sv[warp] = best;
    sv2[warp] = second;
    si[warp] = besti;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float runner = sv2[0];
    for (int w = 1; w < nwarp; ++w) {
      if (sv2[w] > runner) runner = sv2[w];
      if (sv[w] > sv[0] || (sv[w] == sv[0] && si[w] < si[0])) {
        if (sv[0] > runner) runner = sv[0];
        sv[0] = sv[w];
        si[0] = si[w];
      } else if (sv[w] > runner) {
        runner = sv[w];
      }
    }
    idx[4 * step] = si[0];
    idx[4 * step + 1] = (int)__float_as_uint(sv[0] - runner);
  }
}

extern "C" int fcuda_ink2d_topm(const float *logits, int rows, int rowStride, int vocab, int M, float *candV,
                                int *candI, int *out) {
  if (M < 1 || M > INK2D_TOPM_MAX || vocab > INK2D_TOPM_SLICES * 64 * 256) return (int)cudaErrorInvalidValue;
  const dim3 g1(INK2D_TOPM_SLICES, rows);
  FCUDA_TS("k_ink2d_topm_slice", g1,
           k_ink2d_topm_slice<<<g1, 256, 0, g_stream>>>(logits, rowStride, vocab, M, candV, candI));
  FCUDA_TS("k_ink2d_topm_merge", rows, k_ink2d_topm_merge<<<rows, 256, 0, g_stream>>>(candV, candI, M, out));
  return ink2_last_error();
}

extern "C" int fcuda_ink2d_markov_in(const float *state, const unsigned short *w1, const int *anchor, const int *idx,
                                     int step, const float *h, const float *frnnRing, const int *posDev, int cap,
                                     int rank, int hidden, float *in) {
  const int n = 2 * rank + 2 * hidden;
  FCUDA_TS("k_ink2d_markov_in", (n + 255) / 256, k_ink2d_markov_in<<<(n + 255) / 256, 256, 0, g_stream>>>(
      state, w1, anchor, idx, step, h, frnnRing, posDev, cap, rank, hidden, in));
  return ink2_last_error();
}

extern "C" int fcuda_ink2d_markov_gate(const float *raw, const float *bias, float *state, float *latent, int rank,
                                       int step) {
  FCUDA_TS("k_ink2d_markov_gate", (rank + 255) / 256,
           k_ink2d_markov_gate<<<(rank + 255) / 256, 256, 0, g_stream>>>(raw, bias, state, latent, rank, step));
  return ink2_last_error();
}

extern "C" int fcuda_ink2d_markov_pick_q5k(const unsigned char *W, int in, const float *base, const int *cand, int M,
                                           const float *z, const float *zb, int *idx, int step) {
  if (in % 256 != 0 || in > 4096) return (int)cudaErrorInvalidValue;
  FCUDA_TS("k_ink2d_markov_pick", 1,
           k_ink2d_markov_pick<<<1, 512, 0, g_stream>>>(W, in, base, cand, M, z, zb, idx, step));
  return ink2_last_error();
}
