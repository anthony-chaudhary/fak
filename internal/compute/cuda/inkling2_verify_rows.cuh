// inkling2_verify_rows.cuh: few-row variants of decode kernels for the speculative
// verify and the DSpark draft, where T (rows) is at most 9. Included after
// inkling2_dspark_markov.cuh by cuda_kernels.cu.

#define INK2V_MAX_ROWS 9
// output rows per block: the staged X is reused across them
#define INK2V_BLOCK_ROWS 128

// Y[t][o] = W[o] . X[t] for a Q5_K W and up to INK2V_MAX_ROWS rows of X, which are
// staged once in shared memory; each weight is decoded once per block and applied to
// every row. Every row keeps k_q5k_gemm<1>'s expression, K order and shuffle-down
// reduction, so a row's result equals the single-row GEMV's bit for bit.
template <int R>
__global__ void k_ink2v_q5k_rows(const unsigned char *W, const float *X, float *Y, int out, int in) {
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < R * in; i += blockDim.x) xs[i] = X[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int nsb = in / 256;
  for (int o = blockIdx.x * INK2V_BLOCK_ROWS + (threadIdx.x >> 5); o < min(out, (blockIdx.x + 1) * INK2V_BLOCK_ROWS);
       o += blockDim.x >> 5) {
  float sums[R] = {};
  const unsigned char *wr = W + (size_t)o * nsb * 176;
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
      const float wv = d * (float)scale * (float)q - dm * (float)mn;
      const int k = sb * 256 + g * 32 + lane;
#pragma unroll
      for (int t = 0; t < R; ++t) sums[t] += wv * xs[t * in + k];
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
static int ink2v_q5k_rows_launch(const unsigned char *W, const float *X, float *Y, int out, int in) {
  const size_t smem = (size_t)R * in * sizeof(float);
  const int oe = ink2p_smem_optin((const void *)k_ink2v_q5k_rows<R>, smem);
  if (oe) return oe;
  const int blocks = (out + INK2V_BLOCK_ROWS - 1) / INK2V_BLOCK_ROWS;
  FCUDA_TS("k_ink2v_q5k_rows", blocks, k_ink2v_q5k_rows<R><<<blocks, 256, smem, g_stream>>>(W, X, Y, out, in));
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
// rows' codes ([rows][in] bytes) then their block scales. Returns cudaErrorNotSupported
// (nothing launched) outside the kernel's contract.
extern "C" int fcuda_ink2v_q8_rows(const int8_t *W, const float *Wscale, const float *X, void *q, float *Y, int out,
                                   int in, int rows, int block) {
  const int nblk = in / 32;
  const bool aligned = (((uintptr_t)W | (uintptr_t)q) & 15) == 0 && (in & 15) == 0;
  if (block != 32 || in % 32 != 0 || !aligned || nblk > 288 || rows < 1 || rows > INK2V_MAX_ROWS)
    return -1;
  signed char *codes = (signed char *)q;
  float *scales = (float *)(codes + (size_t)rows * in);
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
