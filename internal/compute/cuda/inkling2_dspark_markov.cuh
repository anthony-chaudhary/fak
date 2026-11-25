// inkling2_dspark_markov.cuh: the DSpark rnn_hidden Markov head on the device, with a
// base-logit shortlist. Drafted row i picks the argmax over the M best base logits
// lm_head(h_i) of lm_head(h_i) + lm_head(z_i), where z_i = hidden_out(latent_i) + bias
// and latent_i comes from a GRU-style state fed the previous drafted token's
// markov_w1 row. Included after inkling2_dspark.cuh by cuda_kernels.cu.

#define INK2D_TOPM_MAX 64
#define INK2D_TOPM_SLICES 64

// Stage 1: each block keeps the M best of one vocab slice of one row.
__global__ void k_ink2d_topm_slice(const float *logits, int rowStride, int vocab, int M, float *candV, int *candI) {
  ink2_pdl_enter();
  __shared__ float bv[256];
  __shared__ int bi[256];
  __shared__ int chosen[INK2D_TOPM_MAX];
  const int row = blockIdx.y, slice = blockIdx.x;
  const int per = (vocab + INK2D_TOPM_SLICES - 1) / INK2D_TOPM_SLICES;
  const int lo = slice * per, hi = min(vocab, lo + per);
  const float *l = logits + (size_t)row * rowStride;
  for (int m = 0; m < M; ++m) {
    float best = -INFINITY;
    int besti = -1;
    for (int t = lo + threadIdx.x; t < hi; t += blockDim.x) {
      bool used = false;
      for (int c = 0; c < m; ++c) used |= chosen[c] == t;
      const float v = l[t];
      if (!used && (besti < 0 || v > best)) {
        best = v;
        besti = t;
      }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = besti;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        const float ov = bv[threadIdx.x + s];
        const int oi = bi[threadIdx.x + s];
        const int mi = bi[threadIdx.x];
        if (oi >= 0 && (mi < 0 || ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < mi))) {
          bv[threadIdx.x] = ov;
          bi[threadIdx.x] = oi;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      chosen[m] = bi[0];
      const size_t o = ((size_t)row * INK2D_TOPM_SLICES + slice) * M + m;
      candV[o] = bi[0] >= 0 ? bv[0] : -INFINITY;
      candI[o] = bi[0];
    }
    __syncthreads();
  }
}

// Stage 2: the M best of one row's slice candidates, written to out[row][M].
__global__ void k_ink2d_topm_merge(const float *candV, const int *candI, int M, int *out) {
  ink2_pdl_enter();
  __shared__ float v[INK2D_TOPM_SLICES * INK2D_TOPM_MAX];
  __shared__ int id[INK2D_TOPM_SLICES * INK2D_TOPM_MAX];
  __shared__ float bv[256];
  __shared__ int bp[256];
  const int row = blockIdx.x;
  const int n = INK2D_TOPM_SLICES * M;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    v[i] = candV[(size_t)row * n + i];
    id[i] = candI[(size_t)row * n + i];
  }
  __syncthreads();
  for (int m = 0; m < M; ++m) {
    float best = -INFINITY;
    int bestp = -1;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
      if (id[i] >= 0 && (bestp < 0 || v[i] > best || (v[i] == best && id[i] < id[bestp]))) {
        best = v[i];
        bestp = i;
      }
    bv[threadIdx.x] = best;
    bp[threadIdx.x] = bestp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        const int op = bp[threadIdx.x + s], mp = bp[threadIdx.x];
        if (op >= 0 && (mp < 0 || bv[threadIdx.x + s] > bv[threadIdx.x] ||
                        (bv[threadIdx.x + s] == bv[threadIdx.x] && id[op] < id[mp]))) {
          bv[threadIdx.x] = bv[threadIdx.x + s];
          bp[threadIdx.x] = op;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const int p = bp[0];
      out[(size_t)row * M + m] = p >= 0 ? id[p] : 0;
      if (p >= 0) id[p] = -1;
    }
    __syncthreads();
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
__global__ void k_ink2d_markov_pick(const unsigned char *W, int in, const float *base, const int *cand, int M,
                                    const float *z, const float *zb, int *idx, int step) {
  ink2_pdl_enter();
  __shared__ float zs[4096];
  __shared__ float sv[32];
  __shared__ int si[32];
  for (int i = threadIdx.x; i < in; i += blockDim.x) zs[i] = z[i] + zb[i];
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
  float best = -INFINITY;
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
      best = v;
      besti = o;
    }
  }
  if (lane == 0) {
    sv[warp] = best;
    si[warp] = besti;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < nwarp; ++w)
      if (sv[w] > sv[0] || (sv[w] == sv[0] && si[w] < si[0])) {
        sv[0] = sv[w];
        si[0] = si[w];
      }
    idx[4 * step] = si[0];
  }
}

extern "C" int fcuda_ink2d_topm(const float *logits, int rows, int rowStride, int vocab, int M, float *candV,
                                int *candI, int *out) {
  if (M < 1 || M > INK2D_TOPM_MAX) return (int)cudaErrorInvalidValue;
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
