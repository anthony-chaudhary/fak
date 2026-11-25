// inkling2_verify_rows.cuh: few-row variants of decode kernels for the speculative
// verify and the DSpark draft, where T (rows) is at most 9. Included after
// inkling2_dspark_markov.cuh by cuda_kernels.cu.

#define INK2V_MAX_ROWS 9
// output rows per block: the staged X is reused across them
#define INK2V_BLOCK_ROWS 128

// Scale and min of group g from a Q5_K super-block's 12 packed scale bytes held as three
// words (getScaleMinK4_dev on registers).
static __device__ __forceinline__ unsigned int ink2v_k4_byte(const uint4 &h, int i) {
  const unsigned int w = i < 4 ? h.y : i < 8 ? h.z : h.w;
  return (w >> (8 * (i & 3))) & 0xffu;
}

static __device__ __forceinline__ void ink2v_k4_scale_min(const uint4 &h, int g, unsigned char *sc, unsigned char *mn) {
  if (g < 4) {
    *sc = (unsigned char)(ink2v_k4_byte(h, g) & 63);
    *mn = (unsigned char)(ink2v_k4_byte(h, g + 4) & 63);
  } else {
    *sc = (unsigned char)((ink2v_k4_byte(h, g + 4) & 0x0f) | ((ink2v_k4_byte(h, g - 4) >> 6) << 4));
    *mn = (unsigned char)((ink2v_k4_byte(h, g + 4) >> 4) | ((ink2v_k4_byte(h, g) >> 6) << 4));
  }
}

// Y[t][o] = W[o] . X[t] for a Q5_K W and up to INK2V_MAX_ROWS rows of X, which are
// staged once in shared memory; each weight is decoded once per block and applied to
// every row. Every row keeps k_q5k_gemm<1>'s expression, K order and shuffle-down
// reduction, so a row's result equals the single-row GEMV's bit for bit. HDR16 reads a
// super-block's d, dmin and scale bytes as one 16-byte load (W must be 16-byte aligned)
// instead of a byte load per group; it pays from three rows up and costs at one.
template <int R, bool HDR16>
__global__ void k_ink2v_q5k_rows(const unsigned char *W, const float *X, float *Y, int out, int in, int blockRows) {
  ink2_pdl_enter();
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < R * in; i += blockDim.x) xs[i] = X[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int nsb = in / 256;
  const int end = min(out, (blockIdx.x + 1) * blockRows);
  // each warp takes two output rows at a time, so one staged-X read feeds both
  for (int o = blockIdx.x * blockRows + 2 * (threadIdx.x >> 5); o < end; o += 2 * (blockDim.x >> 5)) {
    const bool two = o + 1 < end;
    float s0[R] = {}, s1[R] = {};
    const unsigned char *w0 = W + (size_t)o * nsb * 176;
    const unsigned char *w1 = two ? w0 + (size_t)nsb * 176 : w0;
    for (int sb = 0; sb < nsb; ++sb) {
      const unsigned char *b0 = w0 + (size_t)sb * 176, *b1 = w1 + (size_t)sb * 176;
      uint4 h0 = make_uint4(0u, 0u, 0u, 0u), h1 = h0;
      float d0, dm0, d1, dm1;
      if (HDR16) {
        h0 = *(const uint4 *)b0;
        h1 = *(const uint4 *)b1;
        d0 = __half2float(__ushort_as_half((unsigned short)(h0.x & 0xffffu)));
        dm0 = __half2float(__ushort_as_half((unsigned short)(h0.x >> 16)));
        d1 = __half2float(__ushort_as_half((unsigned short)(h1.x & 0xffffu)));
        dm1 = __half2float(__ushort_as_half((unsigned short)(h1.x >> 16)));
      } else {
        d0 = __half2float(*(const __half *)b0), dm0 = __half2float(*(const __half *)(b0 + 2));
        d1 = __half2float(*(const __half *)b1), dm1 = __half2float(*(const __half *)(b1 + 2));
      }
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        unsigned char sc0, mn0, sc1, mn1;
        if (HDR16) {
          ink2v_k4_scale_min(h0, g, &sc0, &mn0);
          ink2v_k4_scale_min(h1, g, &sc1, &mn1);
        } else {
          getScaleMinK4_dev(g, b0 + 4, &sc0, &mn0);
          getScaleMinK4_dev(g, b1 + 4, &sc1, &mn1);
        }
        const int pair = g >> 1;
        int q0 = (g & 1) ? (b0[48 + pair * 32 + lane] >> 4) : (b0[48 + pair * 32 + lane] & 15);
        int q1 = (g & 1) ? (b1[48 + pair * 32 + lane] >> 4) : (b1[48 + pair * 32 + lane] & 15);
        if (b0[16 + lane] & (1u << g)) q0 += 16;
        if (b1[16 + lane] & (1u << g)) q1 += 16;
        const float wv0 = d0 * (float)sc0 * (float)q0 - dm0 * (float)mn0;
        const float wv1 = d1 * (float)sc1 * (float)q1 - dm1 * (float)mn1;
        const int k = sb * 256 + g * 32 + lane;
#pragma unroll
        for (int t = 0; t < R; ++t) {
          const float x = xs[t * in + k];
          s0[t] += wv0 * x;
          s1[t] += wv1 * x;
        }
      }
    }
#pragma unroll
    for (int t = 0; t < R; ++t) {
#pragma unroll
      for (int z = 16; z; z >>= 1) {
        s0[t] += __shfl_down_sync(0xffffffff, s0[t], z);
        s1[t] += __shfl_down_sync(0xffffffff, s1[t], z);
      }
      if (lane == 0) {
        Y[(size_t)t * out + o] = s0[t];
        if (two) Y[(size_t)t * out + o + 1] = s1[t];
      }
    }
  }
}

// Threads per block of the staged-X row kernels. From four rows the staged X leaves
// room for fewer blocks on an SM, so each block carries twice the warps.
static int ink2v_rows_threads(int rows) { return rows >= 4 ? 512 : 256; }

template <int R>
static int ink2v_q5k_rows_launch(const unsigned char *W, const float *X, float *Y, int out, int in) {
  const size_t smem = (size_t)R * in * sizeof(float);
  const bool hdr16 = R >= 3 && ((uintptr_t)W & 15) == 0;
  const int oe = hdr16 ? ink2p_smem_optin((const void *)k_ink2v_q5k_rows<R, true>, smem)
                       : ink2p_smem_optin((const void *)k_ink2v_q5k_rows<R, false>, smem);
  if (oe) return oe;
  // enough blocks to fill the SMs on a small matrix, else INK2V_BLOCK_ROWS rows per
  // block so the staged X is re-read from L2 as few times as possible
  int blockRows = INK2V_BLOCK_ROWS;
  if (out < INK2V_BLOCK_ROWS * 256) blockRows = max(8, ((out / 256) + 7) & ~7);
  const int blocks = (out + blockRows - 1) / blockRows;
  const int threads = ink2v_rows_threads(R);
  if (hdr16) {
    FCUDA_TS("k_ink2v_q5k_rows", blocks, k_ink2v_q5k_rows<R, true><<<blocks, threads, smem, g_stream>>>(W, X, Y, out, in, blockRows));
  } else {
    FCUDA_TS("k_ink2v_q5k_rows", blocks, k_ink2v_q5k_rows<R, false><<<blocks, threads, smem, g_stream>>>(W, X, Y, out, in, blockRows));
  }
  return ink2_last_error();
}

extern "C" int fcuda_ink2v_q5k_rows(const unsigned char *W, const float *X, float *Y, int out, int in, int rows) {
  if (in % 256 != 0) return (int)cudaErrorInvalidValue;
  switch (rows) {
  case 1: return ink2v_q5k_rows_launch<1>(W, X, Y, out, in);
  case 2: return ink2v_q5k_rows_launch<2>(W, X, Y, out, in);
  case 3: return ink2v_q5k_rows_launch<3>(W, X, Y, out, in);
  case 4: return ink2v_q5k_rows_launch<4>(W, X, Y, out, in);
  case 5: return ink2v_q5k_rows_launch<5>(W, X, Y, out, in);
  case 6: return ink2v_q5k_rows_launch<6>(W, X, Y, out, in);
  case 7: return ink2v_q5k_rows_launch<7>(W, X, Y, out, in);
  case 8: return ink2v_q5k_rows_launch<8>(W, X, Y, out, in);
  case 9: return ink2v_q5k_rows_launch<9>(W, X, Y, out, in);
  }
  return (int)cudaErrorInvalidValue;
}

// out[t*k + j] = x[t] for every row t and routed slot j: each (token, slot) pair gets
// its token's input row, so the decode expert GEMV can run over T*k pairs.
__global__ void k_ink2v_repeat_rows(const float *x, float *out, int n, int k) {
  ink2_pdl_enter();
  const int pair = blockIdx.y;
  const float *src = x + (size_t)(pair / k) * n;
  float *dst = out + (size_t)pair * n;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

extern "C" int fcuda_ink2v_repeat_rows(const float *x, float *out, int n, int rows, int k) {
  const dim3 grid((n + 255) / 256, rows * k);
  FCUDA_TS("k_ink2v_repeat_rows", grid, k_ink2v_repeat_rows<<<grid, 256, 0, g_stream>>>(x, out, n, k));
  return ink2_last_error();
}

// k_ink2_q8_gemv over R activation rows: each warp loads its weight row's blocks once
// and runs every row's DP4A chain on them. Per row the block term, accumulation
// order and shuffle tree are k_ink2_q8_gemv's, so each row equals the single-row y.
template <int MAXB, int R>
__global__ void __launch_bounds__(INK2_Q8_WARPS * 32)
    k_ink2v_q8_rows(const signed char *__restrict__ W, const float *__restrict__ Wscale,
                    const signed char *__restrict__ qX, const float *__restrict__ xScale, float *__restrict__ Y, int out,
                    int in) {
  ink2_pdl_enter();
  const int lane = threadIdx.x & 31;
  const int o = blockIdx.x * INK2_Q8_WARPS + (threadIdx.x >> 5);
  if (o >= out) return;
  const int nblk = in >> 5;
  const int4 *wrow = (const int4 *)(W + (size_t)o * in);
  const float *wsc = Wscale + (size_t)o * nblk;
  int4 w0[MAXB], w1[MAXB];
  float ws[MAXB];
#pragma unroll
  for (int j = 0; j < MAXB; ++j) {
    const int b = lane + 32 * j;
    if (b < nblk) {
      w0[j] = __ldcs(wrow + 2 * b);
      w1[j] = __ldcs(wrow + 2 * b + 1);
      ws[j] = __ldcs(wsc + b);
    }
  }
#pragma unroll
  for (int t = 0; t < R; ++t) {
    const int4 *xrow = (const int4 *)(qX + (size_t)t * in);
    const float *xsc = xScale + (size_t)t * nblk;
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < MAXB; ++j) {
      const int b = lane + 32 * j;
      if (b < nblk) {
        const int4 x0 = __ldg(xrow + 2 * b), x1 = __ldg(xrow + 2 * b + 1);
        int acc = __dp4a(w0[j].x, x0.x, 0);
        acc = __dp4a(w0[j].y, x0.y, acc);
        acc = __dp4a(w0[j].z, x0.z, acc);
        acc = __dp4a(w0[j].w, x0.w, acc);
        acc = __dp4a(w1[j].x, x1.x, acc);
        acc = __dp4a(w1[j].y, x1.y, acc);
        acc = __dp4a(w1[j].z, x1.z, acc);
        acc = __dp4a(w1[j].w, x1.w, acc);
        sum += (float)acc * ws[j] * __ldg(xsc + b);
      }
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) Y[(size_t)t * out + o] = sum;
  }
}

template <int MAXB>
static void ink2v_q8_rows_launch(const signed char *W, const float *Wscale, const signed char *codes,
                                 const float *scales, float *Y, int out, int in, int rows) {
  const int grid = (out + INK2_Q8_WARPS - 1) / INK2_Q8_WARPS;
#define INK2V_Q8_CASE(R)                                                                                              \
  case R:                                                                                                             \
    FCUDA_TS("k_ink2v_q8_rows", grid,                                                                                 \
             k_ink2v_q8_rows<MAXB, R><<<grid, INK2_Q8_WARPS * 32, 0, g_stream>>>(W, Wscale, codes, scales, Y, out, in)); \
    break;
  switch (rows) {
    INK2V_Q8_CASE(1) INK2V_Q8_CASE(2) INK2V_Q8_CASE(3) INK2V_Q8_CASE(4) INK2V_Q8_CASE(5)
    INK2V_Q8_CASE(6) INK2V_Q8_CASE(7) INK2V_Q8_CASE(8) INK2V_Q8_CASE(9)
  }
#undef INK2V_Q8_CASE
}

// Y[t] = W.X[t] for a resident Q8_0 (block 32) W and rows <= 9 rows of X; q holds the
// rows' codes ([rows][in] bytes) then their block scales; quantize = 0 reuses the codes an
// earlier call on the same X left in q. gate (nullable, needs quantize) quantizes
// X * sigmoid(gate) instead, leaving X unscaled. Returns -1 (nothing launched) outside
// the kernel's contract.
extern "C" int fcuda_ink2v_q8_rows(const int8_t *W, const float *Wscale, const float *X, const float *gate, void *q,
                                   int quantize, float *Y, int out, int in, int rows, int block) {
  const int nblk = in / 32;
  const bool aligned = (((uintptr_t)W | (uintptr_t)q) & 15) == 0 && (in & 15) == 0;
  if (block != 32 || in % 32 != 0 || !aligned || nblk > 288 || rows < 1 || rows > INK2V_MAX_ROWS ||
      (gate && !quantize))
    return -1;
  signed char *codes = (signed char *)q;
  float *scales = (float *)(codes + (size_t)rows * in);
  // rows are contiguous in X, the codes and the scales, so the one-row gated kernel
  // covers them as rows * nblk blocks
  if (gate)
    FCUDA_TS("k_ink2_q8_quant_gated", rows * nblk,
             k_ink2_q8_quant_gated<<<rows * nblk, 32, 0, g_stream>>>(X, gate, codes, scales));
  else if (quantize)
    FCUDA_TS("k_q8_quant_act", dim3(nblk, rows),
             k_q8_quant_act<<<dim3(nblk, rows), 32, 0, g_stream>>>(X, codes, scales, rows, in, 32));
  const signed char *w8 = (const signed char *)W;
  if (nblk <= 96)
    ink2v_q8_rows_launch<3>(w8, Wscale, codes, scales, Y, out, in, rows);
  else if (nblk <= 128)
    ink2v_q8_rows_launch<4>(w8, Wscale, codes, scales, Y, out, in, rows);
  else
    ink2v_q8_rows_launch<9>(w8, Wscale, codes, scales, Y, out, in, rows);
  return ink2_last_error();
}

// Up to INK2V_Q8_MAX_SEGS Q8_0 products of one layer visit in a single launch: segment
// i owns blocks [blk0, blk0 + ceil(out / INK2_Q8_WARPS)) and runs k_ink2v_q8_rows on its
// own weight, codes and output, so every row equals that kernel's.
#define INK2V_Q8_MAX_SEGS 8
struct Ink2VQ8Seg {
  const signed char *W;
  const float *Wscale;
  const signed char *qX;
  const float *xScale;
  float *Y;
  int out, blk0;
};
struct Ink2VQ8Segs {
  Ink2VQ8Seg s[INK2V_Q8_MAX_SEGS];
};

template <int MAXB, int R>
__global__ void __launch_bounds__(INK2_Q8_WARPS * 32) k_ink2v_q8_segs(Ink2VQ8Segs segs, int nseg, int in) {
  ink2_pdl_enter();
  int si = 0;
  for (int i = 1; i < nseg; ++i)
    if ((int)blockIdx.x >= segs.s[i].blk0) si = i;
  const signed char *__restrict__ W = segs.s[si].W;
  const float *__restrict__ Wscale = segs.s[si].Wscale;
  const signed char *__restrict__ qX = segs.s[si].qX;
  const float *__restrict__ xScale = segs.s[si].xScale;
  float *__restrict__ Y = segs.s[si].Y;
  const int out = segs.s[si].out;
  const int lane = threadIdx.x & 31;
  const int o = ((int)blockIdx.x - segs.s[si].blk0) * INK2_Q8_WARPS + (threadIdx.x >> 5);
  if (o >= out) return;
  const int nblk = in >> 5;
  const int4 *wrow = (const int4 *)(W + (size_t)o * in);
  const float *wsc = Wscale + (size_t)o * nblk;
  int4 w0[MAXB], w1[MAXB];
  float ws[MAXB];
#pragma unroll
  for (int j = 0; j < MAXB; ++j) {
    const int b = lane + 32 * j;
    if (b < nblk) {
      w0[j] = __ldcs(wrow + 2 * b);
      w1[j] = __ldcs(wrow + 2 * b + 1);
      ws[j] = __ldcs(wsc + b);
    }
  }
#pragma unroll
  for (int t = 0; t < R; ++t) {
    const int4 *xrow = (const int4 *)(qX + (size_t)t * in);
    const float *xsc = xScale + (size_t)t * nblk;
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < MAXB; ++j) {
      const int b = lane + 32 * j;
      if (b < nblk) {
        const int4 x0 = __ldg(xrow + 2 * b), x1 = __ldg(xrow + 2 * b + 1);
        int acc = __dp4a(w0[j].x, x0.x, 0);
        acc = __dp4a(w0[j].y, x0.y, acc);
        acc = __dp4a(w0[j].z, x0.z, acc);
        acc = __dp4a(w0[j].w, x0.w, acc);
        acc = __dp4a(w1[j].x, x1.x, acc);
        acc = __dp4a(w1[j].y, x1.y, acc);
        acc = __dp4a(w1[j].z, x1.z, acc);
        acc = __dp4a(w1[j].w, x1.w, acc);
        sum += (float)acc * ws[j] * __ldg(xsc + b);
      }
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) Y[(size_t)t * out + o] = sum;
  }
}

template <int MAXB>
static void ink2v_q8_segs_launch(const Ink2VQ8Segs &segs, int nseg, int grid, int in, int rows) {
#define INK2V_Q8_SEG_CASE(R)                                                                                          \
  case R:                                                                                                             \
    FCUDA_TS("k_ink2v_q8_segs", grid,                                                                                 \
             k_ink2v_q8_segs<MAXB, R><<<grid, INK2_Q8_WARPS * 32, 0, g_stream>>>(segs, nseg, in));                    \
    break;
  switch (rows) {
    INK2V_Q8_SEG_CASE(1) INK2V_Q8_SEG_CASE(2) INK2V_Q8_SEG_CASE(3) INK2V_Q8_SEG_CASE(4) INK2V_Q8_SEG_CASE(5)
    INK2V_Q8_SEG_CASE(6) INK2V_Q8_SEG_CASE(7) INK2V_Q8_SEG_CASE(8) INK2V_Q8_SEG_CASE(9)
  }
#undef INK2V_Q8_SEG_CASE
}

// Y[i][t] = W[i].X[i][t] for nseg <= INK2V_Q8_MAX_SEGS resident Q8_0 (block 32) weights
// of one input width and rows <= 9 rows each, in one GEMV launch. q holds the codes of
// every distinct X (in first-use order, [rows][in] bytes each) and then their block
// scales; distinct inputs that sit back to back are quantized by one launch. Returns
// -1 (nothing launched) outside the kernel's contract.
extern "C" int fcuda_ink2v_q8_segs(int nseg, const int8_t *const *W, const float *const *Wscale,
                                   const float *const *X, void *q, float *const *Y, const int *out, int in,
                                   int rows) {
  const int nblk = in / 32;
  if (nseg < 1 || nseg > INK2V_Q8_MAX_SEGS || in % 32 != 0 || (in & 15) != 0 || nblk > 288 || rows < 1 ||
      rows > INK2V_MAX_ROWS || ((uintptr_t)q & 15) != 0)
    return -1;
  const float *xs[INK2V_Q8_MAX_SEGS];
  int xi[INK2V_Q8_MAX_SEGS], nx = 0;
  for (int i = 0; i < nseg; ++i) {
    if (((uintptr_t)W[i] & 15) != 0 || out[i] < 1) return -1;
    int d = 0;
    while (d < nx && xs[d] != X[i]) ++d;
    if (d == nx) xs[nx++] = X[i];
    xi[i] = d;
  }
  signed char *codes = (signed char *)q;
  float *scales = (float *)(codes + (size_t)nx * rows * in);
  bool packed = true;
  for (int d = 1; d < nx; ++d) packed = packed && xs[d] == xs[d - 1] + (size_t)rows * in;
  if (packed) {
    FCUDA_TS("k_q8_quant_act", dim3(nblk, nx * rows),
             k_q8_quant_act<<<dim3(nblk, nx * rows), 32, 0, g_stream>>>(xs[0], codes, scales, nx * rows, in, 32));
  } else {
    for (int d = 0; d < nx; ++d)
      FCUDA_TS("k_q8_quant_act", dim3(nblk, rows),
               k_q8_quant_act<<<dim3(nblk, rows), 32, 0, g_stream>>>(xs[d], codes + (size_t)d * rows * in,
                                                                     scales + (size_t)d * rows * nblk, rows, in, 32));
  }
  Ink2VQ8Segs segs;
  int grid = 0;
  for (int i = 0; i < INK2V_Q8_MAX_SEGS; ++i) {
    Ink2VQ8Seg &g = segs.s[i];
    if (i >= nseg) {
      g = segs.s[0];
      g.out = 0;
      g.blk0 = 0x7fffffff;
      continue;
    }
    g.W = (const signed char *)W[i];
    g.Wscale = Wscale[i];
    g.qX = codes + (size_t)xi[i] * rows * in;
    g.xScale = scales + (size_t)xi[i] * rows * nblk;
    g.Y = Y[i];
    g.out = out[i];
    g.blk0 = grid;
    grid += (out[i] + INK2_Q8_WARPS - 1) / INK2_Q8_WARPS;
  }
  if (nblk <= 96)
    ink2v_q8_segs_launch<3>(segs, nseg, grid, in, rows);
  else if (nblk <= 128)
    ink2v_q8_segs_launch<4>(segs, nseg, grid, in, rows);
  else
    ink2v_q8_segs_launch<9>(segs, nseg, grid, in, rows);
  return ink2_last_error();
}

// The cluster depth mix (k_ink2_mixc) for each row of a verify chunk: cluster y mixes
// token y, whose states sit at row y of each [T][n] state, and writes query q of it at
// out[q][y] ([nq][T][n], the batched mix layout), with the optional RMS norm folded in.
template <int NQ>
__global__ void __launch_bounds__(1024)
    k_ink2v_mixc(Ink2States st, int ns, const float *queries, Ink2Norms nw, int norm, float eps, float *out, int n,
                 int T) {
  ink2_pdl_enter();
  __shared__ Ink2MixShared sm;
  const size_t toff = (size_t)blockIdx.y * n;
  Ink2States sr;
  for (int s = 0; s < INK2_MAX_STATES; ++s) sr.p[s] = st.p[s] ? st.p[s] + toff : nullptr;
  float *o = out + toff;
  const size_t qs = (size_t)T * n;
#define INK2V_MIXC_CALL(NS) ink2_mixc_body<NS, NQ>(sm, sr, queries, nw, norm, eps, o, n, qs)
  INK2_NS_DISPATCH(ns, INK2V_MIXC_CALL)
#undef INK2V_MIXC_CALL
}

template <int NQ>
static cudaError_t ink2v_mixc_launch(const Ink2States &st, int ns, const float *queries, const Ink2Norms &nw, int norm,
                                     float eps, float *out, int n, int T, int threads) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(INK2_MIX_CTAS, T);
  cfg.blockDim = dim3(threads);
  cfg.stream = g_stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = INK2_MIX_CTAS;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, k_ink2v_mixc<NQ>, st, ns, queries, nw, norm, eps, out, n, T);
}

// Row-batched fcuda_ink2_mix: out is [nq][T][n]; norms (nullable) folds an RMS per query.
// Returns -1 (nothing launched) without cluster launch, so the caller keeps its own path.
extern "C" int fcuda_ink2v_mix(const float *const *states, int ns, const float *queries, int nq,
                               const float *const *norms, float eps, float *out, int n, int T) {
  if (ns < 1 || ns > INK2_MAX_STATES || nq < 1 || nq > INK2_MAX_QUERIES || T < 1) return (int)cudaErrorInvalidValue;
  const int threads = ((n + INK2_MIX_CTAS - 1) / INK2_MIX_CTAS + 31) & ~31;
  if (threads > 1024 || threads < ns * (1 + nq) || !ink2_cluster_ok()) return -1;
  Ink2States st;
  for (int i = 0; i < INK2_MAX_STATES; ++i) st.p[i] = i < ns ? states[i] : nullptr;
  Ink2Norms nw;
  for (int q = 0; q < INK2_MAX_QUERIES; ++q) nw.w[q] = norms && q < nq ? norms[q] : nullptr;
  const int norm = norms ? 1 : 0;
  cudaError_t e = cudaSuccess;
  const dim3 grid(INK2_MIX_CTAS, T);
  switch (nq) {
  case 1: FCUDA_TS("k_ink2v_mixc", grid, e = ink2v_mixc_launch<1>(st, ns, queries, nw, norm, eps, out, n, T, threads)); break;
  case 2: FCUDA_TS("k_ink2v_mixc", grid, e = ink2v_mixc_launch<2>(st, ns, queries, nw, norm, eps, out, n, T, threads)); break;
  case 3: FCUDA_TS("k_ink2v_mixc", grid, e = ink2v_mixc_launch<3>(st, ns, queries, nw, norm, eps, out, n, T, threads)); break;
  default: FCUDA_TS("k_ink2v_mixc", grid, e = ink2v_mixc_launch<4>(st, ns, queries, nw, norm, eps, out, n, T, threads)); break;
  }
  return (int)e;
}

// k_ink2v_q5k_rows for a Q6_K W: every row keeps k_q6k_gemm<1>'s expression and order.
template <int R>
__global__ void k_ink2v_q6k_rows(const unsigned char *W, const float *X, float *Y, int out, int in, int blockRows) {
  ink2_pdl_enter();
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < R * in; i += blockDim.x) xs[i] = X[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int nsb = in / 256;
  for (int o = blockIdx.x * blockRows + (threadIdx.x >> 5); o < min(out, (blockIdx.x + 1) * blockRows);
       o += blockDim.x >> 5) {
    float sums[R] = {};
    const unsigned char *wr = W + (size_t)o * nsb * 210;
    for (int sb = 0; sb < nsb; ++sb) {
      const unsigned char *b = wr + (size_t)sb * 210;
      const unsigned char *ql = b, *qh = b + 128;
      const signed char *sc = (const signed char *)(b + 192);
      const float d = __half2float(*(const __half *)(b + 208));
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int qo = h * 64, ho = h * 32, so = h * 8, base = h * 128, is = lane >> 4;
        const int q1 = (ql[qo + lane] & 15) | (((qh[ho + lane] >> 0) & 3) << 4);
        const int q2 = (ql[qo + lane + 32] & 15) | (((qh[ho + lane] >> 2) & 3) << 4);
        const int q3 = (ql[qo + lane] >> 4) | (((qh[ho + lane] >> 4) & 3) << 4);
        const int q4 = (ql[qo + lane + 32] >> 4) | (((qh[ho + lane] >> 6) & 3) << 4);
#pragma unroll
        for (int t = 0; t < R; ++t) {
          const float *xb = xs + (size_t)t * in + (size_t)sb * 256;
          sums[t] += d * (float)sc[so + is] * (q1 - 32) * xb[base + lane];
          sums[t] += d * (float)sc[so + is + 2] * (q2 - 32) * xb[base + lane + 32];
          sums[t] += d * (float)sc[so + is + 4] * (q3 - 32) * xb[base + lane + 64];
          sums[t] += d * (float)sc[so + is + 6] * (q4 - 32) * xb[base + lane + 96];
        }
      }
    }
#pragma unroll
    for (int t = 0; t < R; ++t) {
#pragma unroll
      for (int z = 16; z; z >>= 1) sums[t] += __shfl_down_sync(0xffffffff, sums[t], z);
      if (lane == 0) Y[(size_t)t * out + o] = sums[t];
    }
  }
}

template <int R>
static int ink2v_q6k_rows_launch(const unsigned char *W, const float *X, float *Y, int out, int in) {
  const size_t smem = (size_t)R * in * sizeof(float);
  const int oe = ink2p_smem_optin((const void *)k_ink2v_q6k_rows<R>, smem);
  if (oe) return oe;
  int blockRows = INK2V_BLOCK_ROWS;
  if (out < INK2V_BLOCK_ROWS * 256) blockRows = max(8, ((out / 256) + 7) & ~7);
  const int blocks = (out + blockRows - 1) / blockRows;
  FCUDA_TS("k_ink2v_q6k_rows", blocks,
           k_ink2v_q6k_rows<R><<<blocks, ink2v_rows_threads(R), smem, g_stream>>>(W, X, Y, out, in, blockRows));
  return ink2_last_error();
}

extern "C" int fcuda_ink2v_q6k_rows(const unsigned char *W, const float *X, float *Y, int out, int in, int rows) {
  if (in % 256 != 0) return (int)cudaErrorInvalidValue;
  switch (rows) {
  case 1: return ink2v_q6k_rows_launch<1>(W, X, Y, out, in);
  case 2: return ink2v_q6k_rows_launch<2>(W, X, Y, out, in);
  case 3: return ink2v_q6k_rows_launch<3>(W, X, Y, out, in);
  case 4: return ink2v_q6k_rows_launch<4>(W, X, Y, out, in);
  case 5: return ink2v_q6k_rows_launch<5>(W, X, Y, out, in);
  case 6: return ink2v_q6k_rows_launch<6>(W, X, Y, out, in);
  case 7: return ink2v_q6k_rows_launch<7>(W, X, Y, out, in);
  case 8: return ink2v_q6k_rows_launch<8>(W, X, Y, out, in);
  case 9: return ink2v_q6k_rows_launch<9>(W, X, Y, out, in);
  }
  return (int)cudaErrorInvalidValue;
}
