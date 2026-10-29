// ---- Qwen3.5/3.6 whole-operation Gated-DeltaNet decode (#4725) -----------------

__device__ __forceinline__ float qwen35_gdn_silu(float x) {
  return x / (1.0f + (float)exp((double)-x));
}

__device__ __forceinline__ float qwen35_gdn_softplus(float x) {
  if (x > 20.0f) return x;
  return (float)log1p(exp((double)x));
}

// One block computes one output row. Unlike four separate Backend.MatMul calls, the
// grid spans qkv, z, beta, and decay projections in ONE launch; every block selects
// its source weight/result band while sharing the resident normalized input.

__device__ __forceinline__ float qwen35_gdn_weight(
    const void *weight, const float *scale, int q8, size_t index, int rowWidth) {
  if (!q8) return ((const float *)weight)[index];
  int row = (int)(index / (size_t)rowWidth);
  int col = (int)(index - (size_t)row * rowWidth);
  int nblk = rowWidth / 32;
  return (float)((const signed char *)weight)[index] * scale[(size_t)row * nblk + col / 32];
}

__global__ void k_fp16_split(const float *src, __half *hi, __half *lo, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  __half h = __float2half_rn(src[i]);
  hi[i] = h;
  lo[i] = __float2half_rn(src[i] - __half2float(h));
}

// A dequant + fp16 GEMM returns 0, or one of these without launching when a scratch
// allocation misses, so a caller fails closed instead of a kernel writing through a null
// pointer: the f32 dequant (or per-row activation-code) scratch, or the fp16 split halves.
#define FCUDA_SCRATCH_ALLOC_FAILED 80001
#define FCUDA_SPLIT_SCRATCH_ALLOC_FAILED 80002
// g_scratch_alloc_test_fault makes the Nth next scratch allocation of a dequant + fp16 GEMM
// report a miss (fcuda_scratch_alloc_test_fault(N); 0 disarms), so tests can prove each
// fail-closed site: 1 is the f32 dequant copy, 2..5 the fp16 split halves.
static int g_scratch_alloc_test_fault = 0;
extern "C" void fcuda_scratch_alloc_test_fault(int nth) { g_scratch_alloc_test_fault = nth > 0 ? nth : 0; }
static void *scratch_malloc(size_t bytes) {
  if (g_scratch_alloc_test_fault > 0 && --g_scratch_alloc_test_fault == 0) return nullptr;
  return fcuda_malloc(bytes);
}
// Same three ordered GEMMs for both already-split weight routes.
static void fp16_compensated_gemm(const __half *wHi, const __half *wLo,
    const __half *xHi, const __half *xLo, float *Y, int out, int in, int tokens) {
  const float one = 1.0f, zero = 0.0f;
  FCUDA_TS("cublas_gemm_ex", dim3(out, tokens), cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, out, tokens, in, &one,
      wHi, CUDA_R_16F, in, xHi, CUDA_R_16F, in, &zero, Y, CUDA_R_32F, out,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  FCUDA_TS("cublas_gemm_ex", dim3(out, tokens), cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, out, tokens, in, &one,
      wLo, CUDA_R_16F, in, xHi, CUDA_R_16F, in, &one, Y, CUDA_R_32F, out,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  FCUDA_TS("cublas_gemm_ex", dim3(out, tokens), cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, out, tokens, in, &one,
      wHi, CUDA_R_16F, in, xLo, CUDA_R_16F, in, &one, Y, CUDA_R_32F, out,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
static int fp16_compensated_matmul(const float *W, const float *X, float *Y,
                                   int out, int in, int tokens) {
  size_t wn = (size_t)out * in, xn = (size_t)tokens * in;
  __half *wHi = (__half *)scratch_malloc(wn * sizeof(__half));
  __half *wLo = (__half *)scratch_malloc(wn * sizeof(__half));
  __half *xHi = (__half *)scratch_malloc(xn * sizeof(__half));
  __half *xLo = (__half *)scratch_malloc(xn * sizeof(__half));
  if (!wHi || !wLo || !xHi || !xLo) {
    fcuda_free(wHi); fcuda_free(wLo); fcuda_free(xHi); fcuda_free(xLo);
    return FCUDA_SPLIT_SCRATCH_ALLOC_FAILED;
  }
  FCUDA_TS("k_fp16_split", (wn + 255) / 256, k_fp16_split<<<(wn + 255) / 256, 256, 0, g_stream>>>(W, wHi, wLo, wn));
  FCUDA_TS("k_fp16_split", (xn + 255) / 256, k_fp16_split<<<(xn + 255) / 256, 256, 0, g_stream>>>(X, xHi, xLo, xn));
  fp16_compensated_gemm(wHi, wLo, xHi, xLo, Y, out, in, tokens);
  fcuda_free(wHi); fcuda_free(wLo); fcuda_free(xHi); fcuda_free(xLo);
  return 0;
}

__global__ void k_q8_dequant_transient(const signed char *codes, const float *scales,
                                         float *W, int rows, int cols, int block) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (i >= total) return;
  int row = i / cols, col = i % cols;
  W[i] = (float)codes[i] * scales[(size_t)row * (cols / block) + col / block];
}

// Q8 sequence-prefill weights need only the two fp16 halves, not an
// additional full F32 weight image (#14097). Explicit rounding preserves the
// old F32 store followed by k_fp16_split; no dequant/subtract contraction.
__global__ void k_q8_dequant_split(const signed char *codes, const float *scales,
    __half *hi, __half *lo, size_t n, int cols, int block) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  size_t row = i / (size_t)cols, col = i % (size_t)cols;
  float value = __fmul_rn((float)codes[i], scales[row * (cols / block) + col / block]);
  __half h = __float2half_rn(value);
  hi[i] = h;
  lo[i] = __float2half_rn(__fsub_rn(value, __half2float(h)));
}

static int q8_dequant_sgemm(const void *codes, const float *scales,
                            const float *X, float *Y, int rows, int cols, int tokens, int block) {
  size_t wn = (size_t)rows * cols, xn = (size_t)tokens * cols;
  __half *wHi = (__half *)scratch_malloc(wn * sizeof(__half));
  __half *wLo = (__half *)scratch_malloc(wn * sizeof(__half));
  __half *xHi = (__half *)scratch_malloc(xn * sizeof(__half));
  __half *xLo = (__half *)scratch_malloc(xn * sizeof(__half));
  if (!wHi || !wLo || !xHi || !xLo) {
    fcuda_free(wHi); fcuda_free(wLo); fcuda_free(xHi); fcuda_free(xLo);
    return FCUDA_SPLIT_SCRATCH_ALLOC_FAILED;
  }
  FCUDA_TS("k_q8_dequant_split", (wn + 255) / 256, k_q8_dequant_split<<<(wn + 255) / 256, 256, 0, g_stream>>>(
      (const signed char *)codes, scales, wHi, wLo, wn, cols, block));
  FCUDA_TS("k_fp16_split", (xn + 255) / 256, k_fp16_split<<<(xn + 255) / 256, 256, 0, g_stream>>>(X, xHi, xLo, xn));
  fp16_compensated_gemm(wHi, wLo, xHi, xLo, Y, rows, cols, tokens);
  fcuda_free(wHi); fcuda_free(wLo); fcuda_free(xHi); fcuda_free(xLo);
  return 0;
}

__global__ void k_qwen35_gdn_fused_in_proj(
    const float *x,
    const void *wQKV, const float *sQKV, int qQKV,
    const void *wZ, const float *sZ, int qZ,
    const void *wB, const float *sB, int qB,
    const void *wA, const float *sA, int qA,
    float *mixed, float *z, float *b, float *a,
    int hidden, int convDim, int valueDim, int nV) {
  int token = blockIdx.y;
  x += (size_t)token * hidden;
  mixed += (size_t)token * convDim;
  z += (size_t)token * valueDim;
  b += (size_t)token * nV;
  a += (size_t)token * nV;
  int globalRow = blockIdx.x;
  int row = globalRow;
  const void *w = wQKV;
  const float *scale = sQKV;
  int q8 = qQKV;
  float *dst = mixed;
  if (row >= convDim) {
    row -= convDim;
    w = wZ; scale = sZ; q8 = qZ; dst = z;
    if (row >= valueDim) {
      row -= valueDim;
      w = wB; scale = sB; q8 = qB; dst = b;
      if (row >= nV) {
        row -= nV;
        w = wA; scale = sA; q8 = qA; dst = a;
      }
    }
  }
  float sum = 0.0f;
  size_t base = (size_t)row * hidden;
  if (q8) {
    const signed char *codes = (const signed char *)w + base;
    int nblk = hidden / 32;
    const float *rowScale = scale + (size_t)row * nblk;
    for (int blk = threadIdx.x; blk < nblk; blk += blockDim.x) {
      const signed char *qb = codes + blk * 32;
      const float *xb = x + blk * 32;
      float blockSum = 0.0f;
#pragma unroll
      for (int i = 0; i < 32; i += 4) {
        char4 q = *(const char4 *)(qb + i);
        float4 xv = *(const float4 *)(xb + i);
        blockSum = fmaf((float)q.x, xv.x, blockSum);
        blockSum = fmaf((float)q.y, xv.y, blockSum);
        blockSum = fmaf((float)q.z, xv.z, blockSum);
        blockSum = fmaf((float)q.w, xv.w, blockSum);
      }
      sum = fmaf(blockSum, rowScale[blk], sum);
    }
  } else {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
      sum += ((const float *)w)[base + i] * x[i];
  }
  __shared__ float warpSums[8];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
  if (lane == 0) warpSums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = lane < 8 ? warpSums[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) dst[row] = sum;
  }
}

// Depthwise causal conv over [oldest ... newest, current], followed by SiLU.
// The same channel thread shifts its K-1 history in place, so no cross-thread state
// race exists and the returned conv state already carries the current mixed row.
__global__ void k_qwen35_gdn_conv_state(
    const float *mixed, const float *convW, float *convState, float *convOut,
    int convDim, int K) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= convDim) return;
  float acc = 0.0f;
  const float *cw = convW + (size_t)c * K;
  for (int j = 0; j < K - 1; j++) acc += cw[j] * convState[(size_t)j * convDim + c];
  acc += cw[K - 1] * mixed[c];
  convOut[c] = qwen35_gdn_silu(acc);
  // K==1 is a CPU-valid pointwise causal convolution with no history state.
  // Do not form K-2 or touch the zero-capacity convState allocation in that case.
  if (K > 1) {
    for (int j = 0; j < K - 2; j++)
      convState[(size_t)j * convDim + c] = convState[(size_t)(j + 1) * convDim + c];
    convState[(size_t)(K - 2) * convDim + c] = mixed[c];
  }
}

// Normalize each distinct q/k head once. q is additionally scaled by 1/sqrt(kHd),
// matching the existing CPU recurrence before value-head group expansion.
// Prompt-panel tiling, column split, and split out-norm adapted from mw00/project-maya
// src/prefill/kernels.cu (gdn_conv_tiled_kernel, gdn_rec_cols_kernel, gdn_out_norm_kernel) at
// 4615f346d9311dd1ce9f5738ab04446f023a6e42 (MIT; Copyright (c) 2026 Peasant Smith and the
// Project Maya contributors; Copyright (c) 2026 Niko1221 and the Strata contributors).
// Every element keeps the expression and order of the one-block-per-head panel it replaced,
// so prompt outputs and recurrent/conv state stay bitwise that panel's (#9359).

// Panel convolution computes every prompt row from the immutable incoming history and panel.
// Rows depend only on inputs, so grid.y tiles the prompt; k_qwen35_gdn_conv_panel_history
// commits the final K-1 inputs after every tile has read the incoming history.
constexpr int kQwen35GDNConvTile = 64;
__global__ void k_qwen35_gdn_conv_panel_tiled(
    const float *mixed, const float *convW, const float *convState, float *convOut,
    int tokens, int convDim, int K) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= convDim) return;
  int t0 = blockIdx.y * kQwen35GDNConvTile;
  int t1 = t0 + kQwen35GDNConvTile < tokens ? t0 + kQwen35GDNConvTile : tokens;
  const float *cw = convW + (size_t)c * K;
  for (int token = t0; token < t1; ++token) {
    float acc = 0.0f;
#pragma unroll 1
    for (int j = 0; j < K; ++j) {
      int panel = token + j - (K - 1);
      float value = panel < 0 ? convState[(size_t)(K - 1 + panel) * convDim + c]
                              : mixed[(size_t)panel * convDim + c];
      acc += cw[j] * value;
    }
    convOut[(size_t)token * convDim + c] = qwen35_gdn_silu(acc);
  }
}

__global__ void k_qwen35_gdn_conv_panel_history(
    const float *mixed, float *convState, int tokens, int convDim, int K) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= convDim) return;
  if (K > 1) {
    for (int j = 0; j < K - 1; ++j) {
      int panel = tokens - (K - 1) + j;
      float value = panel < 0 ? convState[(size_t)(K - 1 + panel) * convDim + c]
                              : mixed[(size_t)panel * convDim + c];
      convState[(size_t)j * convDim + c] = value;
    }
  }
}
__global__ void k_qwen35_gdn_qk_norm(
    const float *convOut, float *qNorm, float *kNorm, int nK, int kHd) {
  int h = blockIdx.x;
  int d = threadIdx.x;
  if (h >= nK) return;
  int keyDim = nK * kHd;
  extern __shared__ float smem[];
  float *qss = smem;
  float *kss = smem + blockDim.x;
  float qv = d < kHd ? convOut[h * kHd + d] : 0.0f;
  float kv = d < kHd ? convOut[keyDim + h * kHd + d] : 0.0f;
  qss[d] = qv * qv;
  kss[d] = kv * kv;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (d < off) {
      qss[d] += qss[d + off];
      kss[d] += kss[d + off];
    }
    __syncthreads();
  }
  if (d < kHd) {
    float qinv = (float)(1.0 / sqrt((double)qss[0] + 1e-6));
    float kinv = (float)(1.0 / sqrt((double)kss[0] + 1e-6));
    float scale = (float)(1.0 / sqrt((double)kHd));
    qNorm[h * kHd + d] = qv * qinv * scale;
    kNorm[h * kHd + d] = kv * kinv;
  }
}

__global__ void k_qwen35_gdn_qk_norm_panel(
    const float *convOut, float *qNorm, float *kNorm, int tokens, int convDim, int nK, int kHd) {
  int token = blockIdx.y;
  int h = blockIdx.x;
  int d = threadIdx.x;
  if (token >= tokens || h >= nK) return;
  int keyDim = nK * kHd;
  const float *row = convOut + (size_t)token * convDim;
  extern __shared__ float smem[];
  float *qss = smem;
  float *kss = smem + blockDim.x;
  float qv = d < kHd ? row[h * kHd + d] : 0.0f;
  float kv = d < kHd ? row[keyDim + h * kHd + d] : 0.0f;
  qss[d] = qv * qv;
  kss[d] = kv * kv;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (d < off) { qss[d] += qss[d + off]; kss[d] += kss[d + off]; }
    __syncthreads();
  }
  if (d < kHd) {
    float qinv = (float)(1.0 / sqrt((double)qss[0] + 1e-6));
    float kinv = (float)(1.0 / sqrt((double)kss[0] + 1e-6));
    float scale = (float)(1.0 / sqrt((double)kHd));
    size_t base = ((size_t)token * nK + h) * kHd + d;
    qNorm[base] = qv * qinv * scale;
    kNorm[base] = kv * kinv;
  }
}

// One value head per block; one value dimension per thread. A thread owns every
// recurrent_state[h,i,d] element for its d, so it can perform decay, k^T S,
// rank-1 update, and q^T S without atomics. The block then reduces the per-head
// readout norm and applies norm[d] * RMS(core[d]) * silu(z[h,d]) in the same launch.
// beta, decay and the inverse RMS are per-head scalars with FP64 transcendentals (1/64 rate
// on sm_89): thread 0 evaluates each once and the block reads it back, the same bits as
// every thread evaluating it (#9359).
// kRegisterState holds the thread's state column in registers, as the prompt panel does: one
// global read and one write per state element instead of two of each. Every element keeps
// the expression and order of the global-state loop, which stays the path for key heads
// wider than kQwen35GDNDecodeRegisterMaxKeyHeadDim or value heads wider than
// kQwen35GDNDecodeRegisterMaxThreads, so outputs and state are bitwise (#9359).
constexpr int kQwen35GDNDecodeRegisterMaxKeyHeadDim = 128;
constexpr int kQwen35GDNDecodeRegisterMaxThreads = 256;
template <bool kRegisterState>
__device__ __forceinline__ void qwen35_gdn_recurrent_gated_norm_body(
    const float *convOut, const float *qNorm, const float *kNorm,
    const float *z, const float *b, const float *a,
    const float *aLog, const float *dtBias, const float *norm,
    float *state, float *core,
    int nK, int nV, int kHd, int vHd, float eps) {
  int h = blockIdx.x;
  int d = threadIdx.x;
  if (h >= nV) return;
  int repeat = nV / nK;
  int kh = h / repeat;
  int keyDim = nK * kHd;
  __shared__ float headScalar[3];
  if (d == 0) {
    float headBeta = 1.0f / (1.0f + (float)exp((double)-b[h]));
    float aa = (float)exp((double)aLog[h]);
    float dt = qwen35_gdn_softplus(a[h] + dtBias[h]);
    float decayArg = -aa * dt;
    headScalar[0] = headBeta;
    headScalar[1] = (float)exp((double)decayArg);
  }
  __syncthreads();
  float beta = headScalar[0];
  float decay = headScalar[1];
  float readout = 0.0f;
  if (d < vHd) {
    float kvmem = 0.0f;
    if constexpr (kRegisterState) {
      constexpr int maxKHd = kQwen35GDNDecodeRegisterMaxKeyHeadDim;
      const float *kRow = kNorm + kh * kHd;
      const float *qRow = qNorm + kh * kHd;
      float column[maxKHd];
#pragma unroll
      for (int i = 0; i < maxKHd; ++i) {
        if (i < kHd) column[i] = state[((size_t)h * kHd + i) * vHd + d];
      }
#pragma unroll
      for (int i = 0; i < maxKHd; ++i) {
        if (i < kHd) {
          float sd = column[i] * decay;
          column[i] = sd;
          kvmem += sd * kRow[i];
        }
      }
      float v = convOut[2 * keyDim + h * vHd + d];
      float delta = (v - kvmem) * beta;
#pragma unroll
      for (int i = 0; i < maxKHd; ++i) {
        if (i < kHd) {
          float sd = column[i] + kRow[i] * delta;
          column[i] = sd;
          readout += sd * qRow[i];
        }
      }
#pragma unroll
      for (int i = 0; i < maxKHd; ++i) {
        if (i < kHd) state[((size_t)h * kHd + i) * vHd + d] = column[i];
      }
    } else {
      for (int i = 0; i < kHd; i++) {
        size_t si = ((size_t)h * kHd + i) * vHd + d;
        float sd = state[si] * decay;
        state[si] = sd;
        kvmem += sd * kNorm[kh * kHd + i];
      }
      float v = convOut[2 * keyDim + h * vHd + d];
      float delta = (v - kvmem) * beta;
      for (int i = 0; i < kHd; i++) {
        size_t si = ((size_t)h * kHd + i) * vHd + d;
        float sd = state[si] + kNorm[kh * kHd + i] * delta;
        state[si] = sd;
        readout += sd * qNorm[kh * kHd + i];
      }
    }
    core[h * vHd + d] = readout;
  }
  extern __shared__ float ss[];
  ss[d] = d < vHd ? readout * readout : 0.0f;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (d < off) ss[d] += ss[d + off];
    __syncthreads();
  }
  if (d == 0) headScalar[2] = (float)(1.0 / sqrt((double)ss[0] / (double)vHd + (double)eps));
  __syncthreads();
  if (d < vHd) {
    float inv = headScalar[2];
    int vd = h * vHd + d;
    core[vd] = norm[d] * (readout * inv) * qwen35_gdn_silu(z[vd]);
  }
}

__global__ void k_qwen35_gdn_recurrent_gated_norm(
    const float *convOut, const float *qNorm, const float *kNorm,
    const float *z, const float *b, const float *a,
    const float *aLog, const float *dtBias, const float *norm,
    float *state, float *core,
    int nK, int nV, int kHd, int vHd, float eps) {
  qwen35_gdn_recurrent_gated_norm_body<false>(
      convOut, qNorm, kNorm, z, b, a, aLog, dtBias, norm, state, core, nK, nV, kHd, vHd, eps);
}

__global__ void __launch_bounds__(kQwen35GDNDecodeRegisterMaxThreads) k_qwen35_gdn_recurrent_gated_norm_regstate(
    const float *convOut, const float *qNorm, const float *kNorm,
    const float *z, const float *b, const float *a,
    const float *aLog, const float *dtBias, const float *norm,
    float *state, float *core,
    int nK, int nV, int kHd, int vHd, float eps) {
  qwen35_gdn_recurrent_gated_norm_body<true>(
      convOut, qNorm, kNorm, z, b, a, aLog, dtBias, norm, state, core, nK, nV, kHd, vHd, eps);
}

// Prompt gate scalars per (token, value head), evaluated once rather than once per value column
// (their FP64 transcendentals dominated the old per-head panel). In place: b becomes beta and a
// becomes decay; both are sequence scratch read only by k_qwen35_gdn_recurrent_panel_cols.
__global__ void k_qwen35_gdn_gate_panel(
    float *b, float *a, const float *aLog, const float *dtBias, int tokens, int nV) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)tokens * nV) return;
  int h = (int)(i % (size_t)nV);
  float beta = 1.0f / (1.0f + (float)exp((double)-b[i]));
  float aa = (float)exp((double)aLog[h]);
  float dt = qwen35_gdn_softplus(a[i] + dtBias[h]);
  b[i] = beta;
  a[i] = (float)exp((double)(-aa * dt));
}

// One thread per (value head, value column) advances the whole prompt in token order, holding
// its column of recurrent_state[h,:,d] in registers; a block covers kQwen35GDNPanelCols columns
// of one head and stages the token's q/k rows in shared memory. It writes the raw readout; the
// head-coupled norm runs in k_qwen35_gdn_out_norm_panel.
constexpr int kQwen35GDNPanelCols = 32;
constexpr int kQwen35GDNPanelMaxKeyHeadDim = 128;
__global__ void __launch_bounds__(kQwen35GDNPanelCols) k_qwen35_gdn_recurrent_panel_cols(
    const float *convOut, const float *qNorm, const float *kNorm,
    const float *beta, const float *decay, float *state, float *readout,
    int tokens, int convDim, int nK, int nV, int kHd, int vHd) {
  constexpr int maxKHd = kQwen35GDNPanelMaxKeyHeadDim;
  int colBlocks = (vHd + kQwen35GDNPanelCols - 1) / kQwen35GDNPanelCols;
  int h = blockIdx.x / colBlocks;
  int d = (blockIdx.x - h * colBlocks) * kQwen35GDNPanelCols + threadIdx.x;
  if (h >= nV) return;
  bool live = d < vHd;
  int kh = h / (nV / nK);
  int keyDim = nK * kHd;
  __shared__ float sq[maxKHd], sk[maxKHd];
  float localState[maxKHd];
  if (live) {
#pragma unroll
    for (int i = 0; i < maxKHd; ++i) {
      if (i < kHd) localState[i] = state[((size_t)h * kHd + i) * vHd + d];
    }
  }
  for (int token = 0; token < tokens; ++token) {
    const float *qRow = qNorm + ((size_t)token * nK + kh) * kHd;
    const float *kRow = kNorm + ((size_t)token * nK + kh) * kHd;
    __syncthreads();
    for (int i = threadIdx.x; i < kHd; i += kQwen35GDNPanelCols) { sq[i] = qRow[i]; sk[i] = kRow[i]; }
    __syncthreads();
    if (!live) continue;
    float tokenBeta = beta[(size_t)token * nV + h];
    float tokenDecay = decay[(size_t)token * nV + h];
    float kvmem = 0.0f;
#pragma unroll
    for (int i = 0; i < maxKHd; ++i) {
      if (i < kHd) {
        float sd = localState[i] * tokenDecay;
        localState[i] = sd;
        kvmem += sd * sk[i];
      }
    }
    float v = convOut[(size_t)token * convDim + 2 * keyDim + h * vHd + d];
    float delta = (v - kvmem) * tokenBeta;
    float out = 0.0f;
#pragma unroll
    for (int i = 0; i < maxKHd; ++i) {
      if (i < kHd) {
        float sd = localState[i] + sk[i] * delta;
        localState[i] = sd;
        out += sd * sq[i];
      }
    }
    readout[(size_t)token * nV * vHd + (size_t)h * vHd + d] = out;
  }
  if (live) {
#pragma unroll
    for (int i = 0; i < maxKHd; ++i) {
      if (i < kHd) state[((size_t)h * kHd + i) * vHd + d] = localState[i];
    }
  }
}

// Original chunk expansion of the delta rule (#9359), not a borrowed kernel.
// With D(t,j)=product(decay[j+1:t]), S_before=decay[t]*S[t-1],
// u[t]=beta[t]*(v[t]-k[t]^T*S_before),
// S[t]=D(t,-1)*S0+sum(j<=t) D(t,j)*k[j]*u[j]. Substituting this into u gives
// a unit-lower-triangular solve of only 16 scalars per value column. Q/K dot
// products and readouts run across tokens; the state update runs across keys.
// Products are formed directly: dividing cumulative decays would produce NaNs
// when a gate underflows to zero. The serial FP32 evaluation remains the oracle.
constexpr int kQwen35GDNChunk = 16;
constexpr int kQwen35GDNChunkCols = 8;
__global__ void k_qwen35_gdn_recurrent_chunks(
    const float *convOut, const float *qNorm, const float *kNorm,
    const float *beta, const float *decay, float *state, float *readout,
    int tokens, int convDim, int nK, int nV, int kHd, int vHd) {
  constexpr int T = kQwen35GDNChunk, C = kQwen35GDNChunkCols;
  constexpr int K = kQwen35GDNPanelMaxKeyHeadDim;
  int colBlocks = (vHd + C - 1) / C;
  int h = blockIdx.x / colBlocks;
  int d0 = (blockIdx.x % colBlocks) * C;
  int kh = h / (nV / nK), keyDim = nK * kHd;
  int tid = threadIdx.x;
  __shared__ float s[K * C], q[T * K], k[T * K];
  __shared__ float kk[T * T], qk[T * T];
  __shared__ float ks[T * C], qs[T * C], u[T * C];
  __shared__ float cumulative[T], tail[T], gates[T];
  for (int x = tid; x < kHd * C; x += blockDim.x) {
    int d = d0 + x % C;
    s[x] = d < vHd ? state[((size_t)h * kHd + x / C) * vHd + d] : 0.0f;
  }
  __syncthreads();
  for (int base = 0; base < tokens; base += T) {
    int count = tokens - base < T ? tokens - base : T;
    for (int x = tid; x < count * kHd; x += blockDim.x) {
      int t = x / kHd, i = x % kHd;
      size_t src = ((size_t)(base + t) * nK + kh) * kHd + i;
      q[t * K + i] = qNorm[src]; k[t * K + i] = kNorm[src];
    }
    if (tid < count) {
      float c = 1.0f, last = 1.0f;
      for (int j = 0; j <= tid; ++j) c *= decay[(size_t)(base + j) * nV + h];
      for (int j = tid + 1; j < count; ++j) last *= decay[(size_t)(base + j) * nV + h];
      cumulative[tid] = c; tail[tid] = last;
      gates[tid] = beta[(size_t)(base + tid) * nV + h];
    }
    __syncthreads();
    for (int x = tid; x < count * count; x += blockDim.x) {
      int t = x / count, j = x % count;
      if (j > t) continue;
      float kd = 0.0f, qd = 0.0f, attenuation = 1.0f;
      for (int i = 0; i < kHd; ++i) {
        kd += k[t * K + i] * k[j * K + i];
        qd += q[t * K + i] * k[j * K + i];
      }
      for (int i = j + 1; i <= t; ++i) attenuation *= decay[(size_t)(base + i) * nV + h];
      kk[t * T + j] = kd * attenuation;
      qk[t * T + j] = qd * attenuation;
    }
    for (int x = tid; x < count * C; x += blockDim.x) {
      int t = x / C, d = x % C;
      float kd = 0.0f, qd = 0.0f;
      for (int i = 0; i < kHd; ++i) {
        kd += k[t * K + i] * s[i * C + d];
        qd += q[t * K + i] * s[i * C + d];
      }
      ks[x] = cumulative[t] * kd; qs[x] = cumulative[t] * qd;
    }
    __syncthreads();
    if (tid < C) {
      int d = d0 + tid;
      for (int t = 0; t < count; ++t) {
        float predicted = ks[t * C + tid];
        for (int j = 0; j < t; ++j) predicted += kk[t * T + j] * u[j * C + tid];
        float v = d < vHd ? convOut[(size_t)(base + t) * convDim + 2 * keyDim + h * vHd + d] : 0.0f;
        u[t * C + tid] = gates[t] * (v - predicted);
      }
    }
    __syncthreads();
    for (int x = tid; x < count * C; x += blockDim.x) {
      int t = x / C, d = x % C;
      float out = qs[x];
      for (int j = 0; j <= t; ++j) out += qk[t * T + j] * u[j * C + d];
      if (d0 + d < vHd) readout[((size_t)(base + t) * nV + h) * vHd + d0 + d] = out;
    }
    for (int x = tid; x < kHd * C; x += blockDim.x) {
      int i = x / C, d = x % C;
      float next = cumulative[count - 1] * s[x];
      for (int j = 0; j < count; ++j) next += tail[j] * k[j * K + i] * u[j * C + d];
      s[x] = next;
    }
    // No thread may replace this tile's q/k/u while another reads its update.
    __syncthreads();
  }
  for (int x = tid; x < kHd * C; x += blockDim.x) {
    int d = d0 + x % C;
    if (d < vHd) state[((size_t)h * kHd + x / C) * vHd + d] = s[x];
  }
}

// Gated out-norm, in place on core: one block per (token, value head) reduces readout^2 over the
// same power-of-two shared-memory tree as the decode kernel and applies
// norm[d] * readout * RMS^-1 * silu(z). The inverse RMS is evaluated once per block.
__global__ void k_qwen35_gdn_out_norm_panel(
    const float *z, const float *norm, float *core, int nV, int vHd, float eps) {
  int token = blockIdx.x;
  int h = blockIdx.y;
  int d = threadIdx.x;
  extern __shared__ float ss[];
  __shared__ float inv;
  size_t base = (size_t)token * nV * vHd + (size_t)h * vHd;
  float readout = d < vHd ? core[base + d] : 0.0f;
  ss[d] = d < vHd ? readout * readout : 0.0f;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (d < off) ss[d] += ss[d + off];
    __syncthreads();
  }
  if (d == 0) inv = (float)(1.0 / sqrt((double)ss[0] / (double)vHd + (double)eps));
  __syncthreads();
  if (d < vHd) core[base + d] = norm[d] * (readout * inv) * qwen35_gdn_silu(z[base + d]);
}
__global__ void k_qwen35_gdn_out_proj(
    const void *w, const float *scale, int q8,
    const float *x, float *out, int hidden, int valueDim) {
  int row = blockIdx.x;
  int token = blockIdx.y;
  x += (size_t)token * valueDim;
  out += (size_t)token * hidden;
  float sum = 0.0f;
  size_t base = (size_t)row * valueDim;
  if (q8) {
    const signed char *codes = (const signed char *)w + base;
    int nblk = valueDim / 32;
    const float *rowScale = scale + (size_t)row * nblk;
    for (int blk = threadIdx.x; blk < nblk; blk += blockDim.x) {
      const signed char *qb = codes + blk * 32;
      const float *xb = x + blk * 32;
      float blockSum = 0.0f;
#pragma unroll
      for (int i = 0; i < 32; i += 4) {
        char4 q = *(const char4 *)(qb + i);
        float4 xv = *(const float4 *)(xb + i);
        blockSum = fmaf((float)q.x, xv.x, blockSum);
        blockSum = fmaf((float)q.y, xv.y, blockSum);
        blockSum = fmaf((float)q.z, xv.z, blockSum);
        blockSum = fmaf((float)q.w, xv.w, blockSum);
      }
      sum = fmaf(blockSum, rowScale[blk], sum);
    }
  } else {
    for (int i = threadIdx.x; i < valueDim; i += blockDim.x)
      sum += ((const float *)w)[base + i] * x[i];
  }
  __shared__ float warpSums[8];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
  if (lane == 0) warpSums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = lane < 8 ? warpSums[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) out[row] = sum;
  }
}

static int qwen35_gdn_threads(int n) {
  int threads = 1;
  while (threads < n && threads < 1024) threads <<= 1;
  return threads;
}

static int qwen35_gdn_launch_status(int stage) {
  // cudaGetLastError (not Peek) consumes this launch's status so a stale sticky
  // value cannot be mistaken for a later stage. The deterministic test seam
  // substitutes a launch error without poisoning the CUDA context.
  cudaError_t e = cudaGetLastError();
  if (g_qwen35_gdn_test_fault_stage == stage) {
    g_qwen35_gdn_test_fault_stage = 0;
    e = cudaErrorInvalidConfiguration;
  }
  if (e == cudaSuccess) return 0;
  fprintf(stderr, "fak-cuda: Qwen3.5 GDN stage %d launch failed: %s\n", stage, cudaGetErrorString(e));
  return stage * 10000 + (int)e;
}

// qwen35_gdn_fence is the capture-legal stream fence. Inside an open graph capture no
// work has executed and a synchronize would invalidate the capture, so it returns success;
// the graph fence (fcuda_graph_end_launch) reports a replayed fault instead.
static cudaError_t qwen35_gdn_fence(void) {
  return g_capture_open ? cudaSuccess : cudaStreamSynchronize(g_stream);
}

static int qwen35_gdn_drain_after_error(int status) {
  // A later stage can fail to launch after earlier stages were enqueued. Drain
  // those stages before returning so Go can safely invalidate/free their buffers.
  cudaError_t e = qwen35_gdn_fence();
  if (e == cudaSuccess) return status;
  fprintf(stderr, "fak-cuda: Qwen3.5 GDN drain after launch failure also failed: %s\n", cudaGetErrorString(e));
  return 70000 + (int)e;
}

extern "C" void fcuda_qwen35_gdn_test_fault(int stage) {
  g_qwen35_gdn_test_fault_stage = stage;
}

// Test seam: force the global-state decode kernel so tests can hold both paths to the same
// bits; and count register-state launches so a test can prove which path ran.
static int g_qwen35_gdn_decode_global_state_test = 0;
static size_t g_qwen35_gdn_decode_register_state_launches = 0;
extern "C" void fcuda_qwen35_gdn_decode_force_global_state(int on) {
  g_qwen35_gdn_decode_global_state_test = on != 0;
}
extern "C" size_t fcuda_qwen35_gdn_decode_register_state_launches(void) {
  return g_qwen35_gdn_decode_register_state_launches;
}

extern "C" int fcuda_qwen35_gdn_decode_f32(
    const float *dX,
    const void *dInQKV, const float *dInQKVScale, int inQKVQ8,
    const void *dInZ, const float *dInZScale, int inZQ8,
    const void *dInB, const float *dInBScale, int inBQ8,
    const void *dInA, const float *dInAScale, int inAQ8,
    const float *dConvW, const float *dALog, const float *dDtBias,
    const float *dNorm,
    const void *dOutW, const float *dOutWScale, int outWQ8,
    float *dConvState, float *dRecurrentState, float *dOut,
    float *dMixed, float *dZ, float *dB, float *dA, float *dConvOut,
    float *dQNorm, float *dKNorm, float *dCore,
    int hidden, int nK, int nV, int kHd, int vHd, int convKernel, float rmsEps) {
  if (!dX || !dInQKV || !dInZ || !dInB || !dInA || !dConvW || !dALog ||
      !dDtBias || !dNorm || !dOutW || !dConvState || !dRecurrentState || !dOut ||
      !dMixed || !dZ || !dB || !dA || !dConvOut || !dQNorm || !dKNorm || !dCore ||
      hidden <= 0 || nK <= 0 || nV <= 0 || kHd <= 0 || vHd <= 0 || convKernel < 1 ||
      nV % nK != 0 || kHd > 1024 || vHd > 1024 || !(rmsEps > 0.0f) || !isfinite(rmsEps))
    return -1;
  long long keyDim64 = (long long)nK * kHd;
  long long valueDim64 = (long long)nV * vHd;
  long long convDim64 = 2 * keyDim64 + valueDim64;
  long long fusedRows64 = convDim64 + valueDim64 + 2LL * nV;
  if (keyDim64 > 2147483647LL || valueDim64 > 2147483647LL ||
      convDim64 > 2147483647LL || fusedRows64 > 2147483647LL) return -1;
  int valueDim = (int)valueDim64;
  int convDim = (int)convDim64;

  // Clear and surface stale launch state before adding any new work. If the
  // prior stream is still busy, synchronize it before returning the refusal.
  cudaError_t stale = cudaGetLastError();
  if (stale != cudaSuccess) {
    fprintf(stderr, "fak-cuda: Qwen3.5 GDN preflight found stale CUDA status: %s\n", cudaGetErrorString(stale));
    return qwen35_gdn_drain_after_error(10000 + (int)stale);
  }
  FCUDA_TS("k_qwen35_gdn_fused_in_proj", (int)fusedRows64, k_qwen35_gdn_fused_in_proj<<<(int)fusedRows64, 256, 0, g_stream>>>(
      dX,
      dInQKV, dInQKVScale, inQKVQ8,
      dInZ, dInZScale, inZQ8,
      dInB, dInBScale, inBQ8,
      dInA, dInAScale, inAQ8,
      dMixed, dZ, dB, dA, hidden, convDim, valueDim, nV));
  int status = qwen35_gdn_launch_status(2);
  if (status != 0) return qwen35_gdn_drain_after_error(status);

  unsigned int convBlocks = (unsigned int)(((unsigned long long)convDim + 255ULL) / 256ULL);
  FCUDA_TS("k_qwen35_gdn_conv_state", convBlocks, k_qwen35_gdn_conv_state<<<convBlocks, 256, 0, g_stream>>>(
      dMixed, dConvW, dConvState, dConvOut, convDim, convKernel));
  status = qwen35_gdn_launch_status(3);
  if (status != 0) return qwen35_gdn_drain_after_error(status);

  int qThreads = qwen35_gdn_threads(kHd);
  FCUDA_TS("k_qwen35_gdn_qk_norm", nK, k_qwen35_gdn_qk_norm<<<nK, qThreads, (size_t)2 * qThreads * sizeof(float), g_stream>>>(
      dConvOut, dQNorm, dKNorm, nK, kHd));
  status = qwen35_gdn_launch_status(4);
  if (status != 0) return qwen35_gdn_drain_after_error(status);

  int vThreads = qwen35_gdn_threads(vHd);
  bool registerState = !g_qwen35_gdn_decode_global_state_test &&
                       kHd <= kQwen35GDNDecodeRegisterMaxKeyHeadDim &&
                       vThreads <= kQwen35GDNDecodeRegisterMaxThreads;
  if (registerState) {
    FCUDA_TS("k_qwen35_gdn_recurrent_gated_norm_regstate", nV, k_qwen35_gdn_recurrent_gated_norm_regstate<<<nV, vThreads, (size_t)vThreads * sizeof(float), g_stream>>>(
        dConvOut, dQNorm, dKNorm, dZ, dB, dA, dALog, dDtBias, dNorm,
        dRecurrentState, dCore, nK, nV, kHd, vHd, rmsEps));
  } else {
    FCUDA_TS("k_qwen35_gdn_recurrent_gated_norm", nV, k_qwen35_gdn_recurrent_gated_norm<<<nV, vThreads, (size_t)vThreads * sizeof(float), g_stream>>>(
        dConvOut, dQNorm, dKNorm, dZ, dB, dA, dALog, dDtBias, dNorm,
        dRecurrentState, dCore, nK, nV, kHd, vHd, rmsEps));
  }
  status = qwen35_gdn_launch_status(5);
  if (status != 0) return qwen35_gdn_drain_after_error(status);
  if (registerState) g_qwen35_gdn_decode_register_state_launches++;

  FCUDA_TS("k_qwen35_gdn_out_proj", hidden, k_qwen35_gdn_out_proj<<<hidden, 256, 0, g_stream>>>(
      dOutW, dOutWScale, outWQ8, dCore, dOut, hidden, valueDim));
  status = qwen35_gdn_launch_status(6);
  if (status != 0) return qwen35_gdn_drain_after_error(status);

  cudaError_t completed = qwen35_gdn_fence();
  if (g_qwen35_gdn_test_fault_stage == 7) {
    g_qwen35_gdn_test_fault_stage = 0;
    if (g_capture_open) {
      g_graph_fence_test_fault = 1; // a captured fault surfaces at the graph fence
    } else {
      completed = cudaErrorLaunchFailure;
    }
  }
  if (completed != cudaSuccess) {
    fprintf(stderr, "fak-cuda: Qwen3.5 GDN asynchronous execution failed: %s\n", cudaGetErrorString(completed));
    return 70000 + (int)completed;
  }
  g_qwen35_gdn_operations++;
  return 0;
}

/* Sequence spine adapted from llama.cpp gated_delta_net.cu at
 * 0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd (MIT). Unlike calling the public
 * decode ABI from Go, this loop preserves every panel row and recurrent state on
 * g_stream and synchronizes once at the sequence boundary. */
extern "C" int fcuda_qwen35_gdn_sequence_f32(
    const float *dX, int tokens,
    const void *dInQKV, const float *dInQKVScale, int inQKVQ8,
    const void *dInZ, const float *dInZScale, int inZQ8,
    const void *dInB, const float *dInBScale, int inBQ8,
    const void *dInA, const float *dInAScale, int inAQ8,
    const float *dConvW, const float *dALog, const float *dDtBias,
    const float *dNorm,
    const void *dOutW, const float *dOutWScale, int outWQ8,
    float *dConvState, float *dRecurrentState, float *dOut,
    float *dMixed, float *dZ, float *dB, float *dA, float *dConvOut,
    float *dQNorm, float *dKNorm, float *dCore,
    int hidden, int nK, int nV, int kHd, int vHd, int convKernel, float rmsEps) {
  if (tokens <= 0 || !dX || !dOut) return -1;
  const int convDim = 2 * nK * kHd + nV * vHd;
  const int fusedRows = convDim + nV * vHd + 2 * nV;
  if (hidden <= 0 || nK <= 0 || nV <= 0 || kHd <= 0 || vHd <= 0 || convKernel <= 0 ||
      !dInQKV || !dInZ || !dInB || !dInA || !dConvW || !dALog || !dDtBias || !dNorm ||
      !dOutW || !dConvState || !dRecurrentState || !dMixed || !dZ || !dB || !dA ||
      !dConvOut || !dQNorm || !dKNorm || !dCore) return -1;
  // The panel keeps one key-dimension column of state in registers and the q/k rows in
  // fixed shared arrays; a wider key head is refused, never truncated.
  if (kHd > kQwen35GDNPanelMaxKeyHeadDim || nV % nK != 0) return -1;
  cudaGetLastError();
  if (tokens >= 128 && inQKVQ8 && inZQ8 && inBQ8 && inAQ8) {
    int dq = q8_dequant_sgemm(dInQKV, dInQKVScale, dX, dMixed, convDim, hidden, tokens, 32);
    if (dq == 0) dq = q8_dequant_sgemm(dInZ, dInZScale, dX, dZ, nV * vHd, hidden, tokens, 32);
    if (dq == 0) dq = q8_dequant_sgemm(dInB, dInBScale, dX, dB, nV, hidden, tokens, 32);
    if (dq == 0) dq = q8_dequant_sgemm(dInA, dInAScale, dX, dA, nV, hidden, tokens, 32);
    if (dq != 0) return qwen35_gdn_drain_after_error(dq);
  } else {
    FCUDA_TS("k_qwen35_gdn_fused_in_proj", dim3(fusedRows, tokens), k_qwen35_gdn_fused_in_proj<<<dim3(fusedRows, tokens), 256, 0, g_stream>>>(
      dX, dInQKV, dInQKVScale, inQKVQ8, dInZ, dInZScale, inZQ8,
      dInB, dInBScale, inBQ8, dInA, dInAScale, inAQ8,
      dMixed, dZ, dB, dA, hidden, convDim, nV * vHd, nV));
  }
  int status = qwen35_gdn_launch_status(2);
  if (status != 0) return qwen35_gdn_drain_after_error(status);
  int convBlocks = (convDim + 255) / 256;
  int convTiles = (tokens + kQwen35GDNConvTile - 1) / kQwen35GDNConvTile;
  FCUDA_TS("k_qwen35_gdn_conv_panel_tiled", dim3(convBlocks, convTiles), k_qwen35_gdn_conv_panel_tiled<<<dim3(convBlocks, convTiles), 256, 0, g_stream>>>(
      dMixed, dConvW, dConvState, dConvOut, tokens, convDim, convKernel));
  status = qwen35_gdn_launch_status(3);
  if (status != 0) return qwen35_gdn_drain_after_error(status);
  FCUDA_TS("k_qwen35_gdn_conv_panel_history", convBlocks, k_qwen35_gdn_conv_panel_history<<<convBlocks, 256, 0, g_stream>>>(
      dMixed, dConvState, tokens, convDim, convKernel));
  status = qwen35_gdn_launch_status(3);
  if (status != 0) return qwen35_gdn_drain_after_error(status);
  int qThreads = qwen35_gdn_threads(kHd);
  FCUDA_TS("k_qwen35_gdn_qk_norm_panel", dim3(nK, tokens), k_qwen35_gdn_qk_norm_panel<<<dim3(nK, tokens), qThreads, (size_t)2 * qThreads * sizeof(float), g_stream>>>(
      dConvOut, dQNorm, dKNorm, tokens, convDim, nK, kHd));
  status = qwen35_gdn_launch_status(4);
  if (status != 0) return qwen35_gdn_drain_after_error(status);
  size_t gates = (size_t)tokens * nV;
  FCUDA_TS("k_qwen35_gdn_gate_panel", (unsigned)((gates + 255) / 256), k_qwen35_gdn_gate_panel<<<(unsigned)((gates + 255) / 256), 256, 0, g_stream>>>(
      dB, dA, dALog, dDtBias, tokens, nV));
  status = qwen35_gdn_launch_status(5);
  if (status != 0) return qwen35_gdn_drain_after_error(status);
  int colBlocks = (vHd + kQwen35GDNPanelCols - 1) / kQwen35GDNPanelCols;
  // Explicit opt-in until physical NVIDIA state/logit parity and end-to-end
  // performance qualify the changed FP32 summation order. No allocation or ABI
  // change: static shared storage is below 24 KiB, within CUDA's 48 KiB floor.
  const char *recurrence = getenv("FAK_CUDA_GDN_RECURRENCE");
  if (tokens >= kQwen35GDNChunk && recurrence && strcmp(recurrence, "chunk16") == 0) {
    int chunkCols = (vHd + kQwen35GDNChunkCols - 1) / kQwen35GDNChunkCols;
    FCUDA_TS("k_qwen35_gdn_recurrent_chunks", nV * chunkCols, k_qwen35_gdn_recurrent_chunks<<<nV * chunkCols, 128, 0, g_stream>>>(
        dConvOut, dQNorm, dKNorm, dB, dA, dRecurrentState, dCore, tokens, convDim, nK, nV, kHd, vHd));
  } else {
    FCUDA_TS("k_qwen35_gdn_recurrent_panel_cols", nV * colBlocks, k_qwen35_gdn_recurrent_panel_cols<<<nV * colBlocks, kQwen35GDNPanelCols, 0, g_stream>>>(
        dConvOut, dQNorm, dKNorm, dB, dA, dRecurrentState, dCore, tokens, convDim, nK, nV, kHd, vHd));
  }
  status = qwen35_gdn_launch_status(5);
  if (status != 0) return qwen35_gdn_drain_after_error(status);
  int vThreads = qwen35_gdn_threads(vHd);
  FCUDA_TS("k_qwen35_gdn_out_norm_panel", dim3(tokens, nV), k_qwen35_gdn_out_norm_panel<<<dim3(tokens, nV), vThreads, (size_t)vThreads * sizeof(float), g_stream>>>(
      dZ, dNorm, dCore, nV, vHd, rmsEps));
  status = qwen35_gdn_launch_status(5);
  if (status != 0) return qwen35_gdn_drain_after_error(status);
  if (tokens >= 128 && outWQ8) {
    int dq = q8_dequant_sgemm(dOutW, dOutWScale, dCore, dOut, hidden, nV * vHd, tokens, 32);
    if (dq != 0) return qwen35_gdn_drain_after_error(dq);
  } else {
    FCUDA_TS("k_qwen35_gdn_out_proj", dim3(hidden, tokens), k_qwen35_gdn_out_proj<<<dim3(hidden, tokens), 256, 0, g_stream>>>(
        dOutW, dOutWScale, outWQ8, dCore, dOut, hidden, nV * vHd));
  }
  status = qwen35_gdn_launch_status(6);
  if (status != 0) return qwen35_gdn_drain_after_error(status);  cudaError_t sync = cudaStreamSynchronize(g_stream);
  if (sync != cudaSuccess) return 70000 + (int) sync;
  g_qwen35_gdn_operations += (size_t) tokens;
  return 0;
}

extern "C" size_t fcuda_qwen35_gdn_operations(void) { return g_qwen35_gdn_operations; }
extern "C" void fcuda_qwen35_gdn_operations_reset(void) { g_qwen35_gdn_operations = 0; }
