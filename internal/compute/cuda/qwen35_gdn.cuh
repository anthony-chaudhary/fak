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
  k_fp16_split<<<(wn + 255) / 256, 256, 0, g_stream>>>(W, wHi, wLo, wn);
  k_fp16_split<<<(xn + 255) / 256, 256, 0, g_stream>>>(X, xHi, xLo, xn);
  const float one = 1.0f, zero = 0.0f;
  cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, out, tokens, in, &one,
      wHi, CUDA_R_16F, in, xHi, CUDA_R_16F, in, &zero, Y, CUDA_R_32F, out,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, out, tokens, in, &one,
      wLo, CUDA_R_16F, in, xHi, CUDA_R_16F, in, &one, Y, CUDA_R_32F, out,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, out, tokens, in, &one,
      wHi, CUDA_R_16F, in, xLo, CUDA_R_16F, in, &one, Y, CUDA_R_32F, out,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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

static int q8_dequant_sgemm(const void *codes, const float *scales,
                            const float *X, float *Y, int rows, int cols, int tokens, int block) {
  float *wide = (float *)scratch_malloc((size_t)rows * cols * sizeof(float));
  if (!wide) return FCUDA_SCRATCH_ALLOC_FAILED;
  k_q8_dequant_transient<<<((size_t)rows * cols + 255) / 256, 256, 0, g_stream>>>(
      (const signed char *)codes, scales, wide, rows, cols, block);
  int status = fp16_compensated_matmul(wide, X, Y, rows, cols, tokens);
  fcuda_free(wide);
  return status;
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
// Panel convolution computes every prompt row from the immutable incoming history and panel,
// then commits only the final K-1 inputs as the next recurrent history. Channels are independent.
__global__ void k_qwen35_gdn_conv_panel(
    const float *mixed, const float *convW, float *convState, float *convOut,
    int tokens, int convDim, int K) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= convDim) return;
  const float *cw = convW + (size_t)c * K;
  for (int token = 0; token < tokens; ++token) {
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

// One value head per block; one value dimension per thread. A thread owns every
// recurrent_state[h,i,d] element for its d, so it can perform decay, k^T S,
// rank-1 update, and q^T S without atomics. The block then reduces the per-head
// readout norm and applies norm[d] * RMS(core[d]) * silu(z[h,d]) in the same launch.
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
__global__ void k_qwen35_gdn_recurrent_gated_norm(
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
  float beta = 1.0f / (1.0f + (float)exp((double)-b[h]));
  float aa = (float)exp((double)aLog[h]);
  float dt = qwen35_gdn_softplus(a[h] + dtBias[h]);
  float decayArg = -aa * dt;
  float decay = (float)exp((double)decayArg);
  float readout = 0.0f;
  if (d < vHd) {
    float kvmem = 0.0f;
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
    core[h * vHd + d] = readout;
  }
  extern __shared__ float ss[];
  ss[d] = d < vHd ? readout * readout : 0.0f;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (d < off) ss[d] += ss[d + off];
    __syncthreads();
  }
  if (d < vHd) {
    float inv = (float)(1.0 / sqrt((double)ss[0] / (double)vHd + (double)eps));
    int vd = h * vHd + d;
    core[vd] = norm[d] * (readout * inv) * qwen35_gdn_silu(z[vd]);
  }
}

// One persistent block per value head owns its recurrent state and advances the whole prompt
// in token order on device. Different heads are independent; no grid-wide synchronization is needed.
__global__ void k_qwen35_gdn_recurrent_panel(
    const float *convOut, const float *qNorm, const float *kNorm,
    const float *z, const float *b, const float *a,
    const float *aLog, const float *dtBias, const float *norm,
    float *state, float *core, int tokens, int convDim,
    int nK, int nV, int kHd, int vHd, float eps) {
  int h = blockIdx.x;
  int d = threadIdx.x;
  if (h >= nV) return;
  int repeat = nV / nK;
  int kh = h / repeat;
  int keyDim = nK * kHd;
  extern __shared__ float ss[];
  constexpr int maxKHd = 128;
  float localState[maxKHd];
  if (d < vHd) {
#pragma unroll
    for (int i = 0; i < maxKHd; ++i) {
      if (i < kHd) localState[i] = state[((size_t)h * kHd + i) * vHd + d];
    }
  }

  for (int token = 0; token < tokens; ++token) {
    const float *qRow = qNorm + ((size_t)token * nK + kh) * kHd;
    const float *kRow = kNorm + ((size_t)token * nK + kh) * kHd;
    const float *convRow = convOut + (size_t)token * convDim;
    const float *zRow = z + (size_t)token * nV * vHd;
    const float *bRow = b + (size_t)token * nV;
    const float *aRow = a + (size_t)token * nV;
    float beta = 1.0f / (1.0f + (float)exp((double)-bRow[h]));
    float aa = (float)exp((double)aLog[h]);
    float dt = qwen35_gdn_softplus(aRow[h] + dtBias[h]);
    float decay = (float)exp((double)(-aa * dt));
    float readout = 0.0f;
    if (d < vHd) {
      float kvmem = 0.0f;
#pragma unroll
      for (int i = 0; i < maxKHd; ++i) {
        if (i < kHd) {
          float sd = localState[i] * decay;
          localState[i] = sd;
          kvmem += sd * kRow[i];
        }
      }
      float v = convRow[2 * keyDim + h * vHd + d];
      float delta = (v - kvmem) * beta;
#pragma unroll
      for (int i = 0; i < maxKHd; ++i) {
        if (i < kHd) {
          float sd = localState[i] + kRow[i] * delta;
          localState[i] = sd;
          readout += sd * qRow[i];
        }
      }
    }
    ss[d] = d < vHd ? readout * readout : 0.0f;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
      if (d < off) ss[d] += ss[d + off];
      __syncthreads();
    }
    if (d < vHd) {
      float inv = (float)(1.0 / sqrt((double)ss[0] / (double)vHd + (double)eps));
      int vd = h * vHd + d;
      core[(size_t)token * nV * vHd + vd] = norm[d] * (readout * inv) * qwen35_gdn_silu(zRow[vd]);
    }
    __syncthreads();
  }
  if (d < vHd) {
#pragma unroll
    for (int i = 0; i < maxKHd; ++i) {
      if (i < kHd) state[((size_t)h * kHd + i) * vHd + d] = localState[i];
    }
  }
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

static int qwen35_gdn_drain_after_error(int status) {
  // A later stage can fail to launch after earlier stages were enqueued. Drain
  // those stages before returning so Go can safely invalidate/free their buffers.
  cudaError_t e = cudaStreamSynchronize(g_stream);
  if (e == cudaSuccess) return status;
  fprintf(stderr, "fak-cuda: Qwen3.5 GDN drain after launch failure also failed: %s\n", cudaGetErrorString(e));
  return 70000 + (int)e;
}

extern "C" void fcuda_qwen35_gdn_test_fault(int stage) {
  g_qwen35_gdn_test_fault_stage = stage;
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
  k_qwen35_gdn_fused_in_proj<<<(int)fusedRows64, 256, 0, g_stream>>>(
      dX,
      dInQKV, dInQKVScale, inQKVQ8,
      dInZ, dInZScale, inZQ8,
      dInB, dInBScale, inBQ8,
      dInA, dInAScale, inAQ8,
      dMixed, dZ, dB, dA, hidden, convDim, valueDim, nV);
  int status = qwen35_gdn_launch_status(2);
  if (status != 0) return qwen35_gdn_drain_after_error(status);

  unsigned int convBlocks = (unsigned int)(((unsigned long long)convDim + 255ULL) / 256ULL);
  k_qwen35_gdn_conv_state<<<convBlocks, 256, 0, g_stream>>>(
      dMixed, dConvW, dConvState, dConvOut, convDim, convKernel);
  status = qwen35_gdn_launch_status(3);
  if (status != 0) return qwen35_gdn_drain_after_error(status);

  int qThreads = qwen35_gdn_threads(kHd);
  k_qwen35_gdn_qk_norm<<<nK, qThreads, (size_t)2 * qThreads * sizeof(float), g_stream>>>(
      dConvOut, dQNorm, dKNorm, nK, kHd);
  status = qwen35_gdn_launch_status(4);
  if (status != 0) return qwen35_gdn_drain_after_error(status);

  int vThreads = qwen35_gdn_threads(vHd);
  k_qwen35_gdn_recurrent_gated_norm<<<nV, vThreads, (size_t)vThreads * sizeof(float), g_stream>>>(
      dConvOut, dQNorm, dKNorm, dZ, dB, dA, dALog, dDtBias, dNorm,
      dRecurrentState, dCore, nK, nV, kHd, vHd, rmsEps);
  status = qwen35_gdn_launch_status(5);
  if (status != 0) return qwen35_gdn_drain_after_error(status);

  k_qwen35_gdn_out_proj<<<hidden, 256, 0, g_stream>>>(
      dOutW, dOutWScale, outWQ8, dCore, dOut, hidden, valueDim);
  status = qwen35_gdn_launch_status(6);
  if (status != 0) return qwen35_gdn_drain_after_error(status);

  cudaError_t completed = cudaStreamSynchronize(g_stream);
  if (g_qwen35_gdn_test_fault_stage == 7) {
    g_qwen35_gdn_test_fault_stage = 0;
    completed = cudaErrorLaunchFailure;
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
  cudaGetLastError();
  if (tokens >= 128 && inQKVQ8 && inZQ8 && inBQ8 && inAQ8) {
    int dq = q8_dequant_sgemm(dInQKV, dInQKVScale, dX, dMixed, convDim, hidden, tokens, 32);
    if (dq == 0) dq = q8_dequant_sgemm(dInZ, dInZScale, dX, dZ, nV * vHd, hidden, tokens, 32);
    if (dq == 0) dq = q8_dequant_sgemm(dInB, dInBScale, dX, dB, nV, hidden, tokens, 32);
    if (dq == 0) dq = q8_dequant_sgemm(dInA, dInAScale, dX, dA, nV, hidden, tokens, 32);
    if (dq != 0) return qwen35_gdn_drain_after_error(dq);
  } else {
    k_qwen35_gdn_fused_in_proj<<<dim3(fusedRows, tokens), 256, 0, g_stream>>>(
      dX, dInQKV, dInQKVScale, inQKVQ8, dInZ, dInZScale, inZQ8,
      dInB, dInBScale, inBQ8, dInA, dInAScale, inAQ8,
      dMixed, dZ, dB, dA, hidden, convDim, nV * vHd, nV);
  }
  int status = qwen35_gdn_launch_status(2);
  if (status != 0) return qwen35_gdn_drain_after_error(status);
  int convBlocks = (convDim + 255) / 256;
  k_qwen35_gdn_conv_panel<<<convBlocks, 256, 0, g_stream>>>(
      dMixed, dConvW, dConvState, dConvOut, tokens, convDim, convKernel);
  status = qwen35_gdn_launch_status(3);
  if (status != 0) return qwen35_gdn_drain_after_error(status);
  int qThreads = qwen35_gdn_threads(kHd);
  k_qwen35_gdn_qk_norm_panel<<<dim3(nK, tokens), qThreads, (size_t)2 * qThreads * sizeof(float), g_stream>>>(
      dConvOut, dQNorm, dKNorm, tokens, convDim, nK, kHd);
  status = qwen35_gdn_launch_status(4);
  if (status != 0) return qwen35_gdn_drain_after_error(status);
  int vThreads = qwen35_gdn_threads(vHd);
  k_qwen35_gdn_recurrent_panel<<<nV, vThreads, (size_t)vThreads * sizeof(float), g_stream>>>(
      dConvOut, dQNorm, dKNorm, dZ, dB, dA, dALog, dDtBias, dNorm,
      dRecurrentState, dCore, tokens, convDim, nK, nV, kHd, vHd, rmsEps);
  status = qwen35_gdn_launch_status(5);
  if (status != 0) return qwen35_gdn_drain_after_error(status);
  if (tokens >= 128 && outWQ8) {
    int dq = q8_dequant_sgemm(dOutW, dOutWScale, dCore, dOut, hidden, nV * vHd, tokens, 32);
    if (dq != 0) return qwen35_gdn_drain_after_error(dq);
  } else {
    k_qwen35_gdn_out_proj<<<dim3(hidden, tokens), 256, 0, g_stream>>>(
        dOutW, dOutWScale, outWQ8, dCore, dOut, hidden, nV * vHd);
  }
  status = qwen35_gdn_launch_status(6);
  if (status != 0) return qwen35_gdn_drain_after_error(status);  cudaError_t sync = cudaStreamSynchronize(g_stream);
  if (sync != cudaSuccess) return 70000 + (int) sync;
  g_qwen35_gdn_operations += (size_t) tokens;
  return 0;
}

extern "C" size_t fcuda_qwen35_gdn_operations(void) { return g_qwen35_gdn_operations; }
extern "C" void fcuda_qwen35_gdn_operations_reset(void) { g_qwen35_gdn_operations = 0; }
