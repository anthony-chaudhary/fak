// y[P,out] = x[P,in] @ W[out,in]^T, all row-major. Column-major cuBLAS recipe:
// treat row-major W[out,in] as col-major [in,out] (op=T), row-major X[P,in] as col-major
// [in,P] (op=N); the col-major out[out,P] result IS row-major Y[P,out]. Verified by index.
extern "C" void fcuda_matmul_f32(const float *dW, const float *dX, float *dY, int out, int in, int P) {
  const float alpha = 1.0f, beta = 0.0f;
  // F32 SGEMM. The F16 HGEMM peer below checks its status; this path historically did NOT, so a
  // failed launch (e.g. a cuBLAS/launch-config change on a new arch+toolkit pair — #972 first
  // surfaced on sm_80 / CUDA 13.0) left dY stale and the device output looked structurally wrong
  // (a non-block-aligned out dim came back short) with no diagnostic. Check + name the dims so the
  // next witness run pinpoints the failure instead of reporting silent garbage.
  cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
  FCUDA_TS("cublas_sgemm", dim3(out, P), st = cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N,
              out, P, in,
              &alpha,
              dW, in,
              dX, in,
              &beta,
              dY, out));
  if (st != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "fak-cuda: cublasSgemm(out=%d,in=%d,P=%d) failed: %d\n", out, in, P, (int)st);
    // FAIL-SAFE (#972): a failed SGEMM leaves dY untouched, and dY is a RECYCLED pool buffer,
    // so a stale (smaller) prior allocation's content reads back as a plausible-but-wrong vector
    // (the "out=257 -> got 64" ghost the issue reports). Zero the full out*P result so a failed
    // launch yields an unambiguous all-zero vector the cosine/shape gate rejects deterministically,
    // never a short/garbage read that masquerades as a real result.
    FCUDA_TS("fill", dim3(0, 0), cudaMemsetAsync(dY, 0, (size_t)out * (size_t)P * sizeof(float), g_stream));
  }
}

// ---- fp16 compute path (#484): F16 weights + tensor-core HGEMM ------------------
// The first device backend ran F32 SGEMM only. fp16/tensor-cores is the precision axis toward
// llama.cpp throughput (bench_llamacpp.py measures F16). Weights are narrowed to __half at H2D
// (Caps.UploadDtype); the GEMM runs on tensor cores via cublasGemmEx with F32 accumulation, so
// the output stays f32 and the rest of the op chain is untouched. It is an Approx peer of the
// cpuref Reference (looser cosine gate than the Q8 lane — see cudaFP16CosineMin in cuda.go),
// NOT bit-identity.

// k_f32_to_f16 narrows a staged f32 buffer to F16, element-for-element (row-major upload).
__global__ void k_f32_to_f16(const float *src, __half *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}
extern "C" void fcuda_f32_to_f16(void *dDstHalf, const float *dSrc, int n) {
  FCUDA_TS("k_f32_to_f16", (n + 255) / 256, k_f32_to_f16<<<(n + 255) / 256, 256, 0, g_stream>>>(dSrc, (__half *)dDstHalf, n));
}

// k_f32_to_f16_T narrows AND transpose-repacks a row-major f32 weight [out,in] into a
// column-major F16 weight [out,in]: dst[o + i*out] = (half)src[o*in + i]. This is the `Layout`
// repack at H2D — the ColMajor weight is laid out once at upload so the HGEMM reads it with
// op_N instead of transposing per call. Indexed by the SOURCE element s = o*in + i.
__global__ void k_f32_to_f16_T(const float *src, __half *dst, int out, int in) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= out * in) return;
  int o = s / in, i = s % in;
  dst[o + (size_t)i * out] = __float2half(src[s]);
}
extern "C" void fcuda_f32_to_f16_T(void *dDstHalf, const float *dSrc, int out, int in) {
  int total = out * in;
  FCUDA_TS("k_f32_to_f16_T", (total + 255) / 256, k_f32_to_f16_T<<<(total + 255) / 256, 256, 0, g_stream>>>(dSrc, (__half *)dDstHalf, out, in));
}

// fcuda_matmul_f16: Y[P,out] = X[P,in] @ W[out,in]^T on tensor cores. W is resident __half;
// X (f32) is converted to __half in a pooled scratch; cublasGemmEx accumulates in F32 and
// writes f32 Y. The column-major recipe mirrors fcuda_matmul_f32:
//   colMajor==0 (row-major W [out,in]):  op_T on A, lda=in  (W treated as col-major [in,out]).
//   colMajor!=0 (W repacked col-major):  op_N on A, lda=out (W IS col-major [out,in]).
// Both yield C col-major [out,P] == row-major Y[P,out] (ldc=out), B = X col-major [in,P] (ldb=in).
extern "C" void fcuda_matmul_f16(const void *dWhalf, const float *dX, float *dY,
                                 int out, int in, int P, int colMajor) {
  const __half *A = (const __half *)dWhalf;
  __half *dXh = (__half *)fcuda_malloc((size_t)P * in * sizeof(__half));
  FCUDA_TS("k_f32_to_f16", (P * in + 255) / 256, k_f32_to_f16<<<(P * in + 255) / 256, 256, 0, g_stream>>>(dX, dXh, P * in));
  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st;
  if (colMajor) {
    FCUDA_TS("cublas_gemm_ex", dim3(out, P), st = cublasGemmEx(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, out, P, in,
                      &alpha, A, CUDA_R_16F, out, dXh, CUDA_R_16F, in,
                      &beta, dY, CUDA_R_32F, out,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    FCUDA_TS("cublas_gemm_ex", dim3(out, P), st = cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, out, P, in,
                      &alpha, A, CUDA_R_16F, in, dXh, CUDA_R_16F, in,
                      &beta, dY, CUDA_R_32F, out,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  if (st != CUBLAS_STATUS_SUCCESS) fprintf(stderr, "fak-cuda: cublasGemmEx(HGEMM) failed: %d\n", (int)st);
  fcuda_free(dXh);
}

// ---- native quantized device GEMM (#485): Q8_0 + Q4_K, no dequant-to-f32 --------
// The weight stays NARROW in VRAM (int8 codes / Q4_K super-block bytes) and the GEMM
// consumes it directly — no dequant-to-f32 round trip, so the VRAM/bandwidth win the
// quantized format buys is kept. Both are Approx peers of the cpuref Reference (per-dtype
// recorded cosine floors in cuda.go: cudaQ8CosineMin tighter than cudaQ4KCosineMin), NOT
// bit-identity. The activation arrives f32-resident; the kernels quantize (Q8_0) or
// dequant-fuse (Q4_K) on device, accumulate in F32, and write f32 Y so the rest of the op
// chain (RMSNorm/RoPE/SwiGLU/Attention) stays f32 and unchanged.

// q8round_dev reproduces cpuref q8round byte-for-byte: truncate toward zero, then round the
// fractional part half-away-from-zero, clamp to [-127,127]. The on-device activation quant
// must round the SAME way the cpuref reference does so the int8 lane stays tight to f32.
__device__ signed char q8round_dev(float x) {
  int t = (int)x; // C cast truncates toward zero, like Go int32(x)
  float f = x - (float)t;
  if (f >= 0.5f) t++;
  else if (f <= -0.5f) t--;
  if (t > 127) return 127;
  if (t < -127) return -127;
  return (signed char)t;
}

// k_q8_quant_act quantizes the f32 activation X[P,in] to Q8_0 ON DEVICE: per block of `block`
// elements, d = amax/127 and code = q8round(x/d) — exactly cpuref quantizeVecQ8. One block per
// (row t, block b); 64 power-of-two threads stride the (=32) block elements for the amax reduce.
__global__ void k_q8_quant_act(const float *X, signed char *qX, float *xScale,
                               int P, int in, int block) {
  int b = blockIdx.x, t = blockIdx.y;
  int nblk = in / block;
  if (t >= P || b >= nblk) return;
  const float *xb = X + (size_t)t * in + (size_t)b * block;
  __shared__ float red[64];
  float a = 0.f;
  for (int i = threadIdx.x; i < block; i += blockDim.x) a = fmaxf(a, fabsf(xb[i]));
  red[threadIdx.x] = a;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  float d = red[0] / 127.f;
  if (threadIdx.x == 0) xScale[(size_t)t * nblk + b] = d;
  float inv = d > 0.f ? 1.f / d : 0.f;
  signed char *qb = qX + (size_t)t * in + (size_t)b * block;
  for (int i = threadIdx.x; i < block; i += blockDim.x)
    qb[i] = d > 0.f ? q8round_dev(xb[i] * inv) : (signed char)0;
}

// k_q8_gemm: Y[t,o] = Σ_b (Σ_i qW[o,b,i]·qX[t,b,i]) · dW[o,b] · dX[t,b]. One warp owns
// each output row. Its lanes distribute quant blocks, use signed DP4A for the integer dot, then
// reduce through shuffle instructions. Eight independent rows per CUDA block keep the SM occupied
// without the former per-row shared-memory reduction and its eight block-wide barriers.
__global__ void k_q8_gemm(const signed char *W, const float *Wscale,
                          const signed char *qX, const float *xScale,
                          float *Y, int out, int in, int P, int block) {
  constexpr int warps = 8;
  int lane = threadIdx.x & 31;
  int o = blockIdx.x * warps + (threadIdx.x >> 5);
  int t = blockIdx.y;
  if (o >= out || t >= P) return;
  int nblk = in / block;
  const signed char *wrow = W + (size_t)o * in;
  const float *wsc = Wscale + (size_t)o * nblk;
  const signed char *xrow = qX + (size_t)t * in;
  const float *xsc = xScale + (size_t)t * nblk;
  float sum = 0.f;
  for (int b = lane; b < nblk; b += 32) {
    const int *wb = (const int *)(wrow + (size_t)b * block);
    const int *xb = (const int *)(xrow + (size_t)b * block);
    int acc = 0;
    for (int i = 0; i < block / 4; ++i) acc = __dp4a(wb[i], xb[i], acc);
    sum += (float)acc * wsc[b] * xsc[b];
  }
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, delta);
  if (lane == 0) Y[(size_t)t * out + o] = sum;
}

// k_q8_gemm_rowtile is the prompt-panel Q8_0 GEMM (block 32 only). Each warp owns
// kQ8TileOuts outputs and up to kQ8TileRows rows, so one weight-block load feeds every row of
// the tile. Every (t,o) keeps k_q8_gemm's arithmetic exactly: lane L takes blocks
// b = L, L+32, ...; acc is the exact integer dot; sum += (float)acc * wsc[b] * xsc[b] in that
// order. The 32 lane partials are then summed in __shfl_down's grouping (16, 8, 4, 2, 1),
// replayed in shared memory by one lane per (t,o). The output is bitwise k_q8_gemm's.
constexpr int kQ8TileRows = 8;
constexpr int kQ8TileOuts = 4;
constexpr int kQ8TileWarps = 8;
__global__ void __launch_bounds__(kQ8TileWarps * 32)
k_q8_gemm_rowtile(const signed char *__restrict__ W, const float *__restrict__ Wscale,
                  const signed char *__restrict__ qX, const float *__restrict__ xScale,
                  float *__restrict__ Y, int out, int in, int P, int rowsPerTile) {
  constexpr int R = kQ8TileRows, O = kQ8TileOuts, K = R * O;
  static_assert(K == 32, "one lane replays the reduction of one (row, output) partial");
  __shared__ float part[kQ8TileWarps][K][33];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int t0 = blockIdx.x * rowsPerTile;
  int o0 = (blockIdx.y * kQ8TileWarps + warp) * O;
  if (o0 >= out || t0 >= P) return;
  int rows = min(rowsPerTile, P - t0);
  int nblk = in >> 5;
  const signed char *wrow[O];
  const float *wsc[O];
#pragma unroll
  for (int j = 0; j < O; ++j) {
    int o = min(o0 + j, out - 1);
    wrow[j] = W + (size_t)o * in;
    wsc[j] = Wscale + (size_t)o * nblk;
  }
  const signed char *xrows = qX + (size_t)t0 * in;
  const float *xsc = xScale + (size_t)t0 * nblk;
  float sum[O][R];
#pragma unroll
  for (int j = 0; j < O; ++j)
#pragma unroll
    for (int r = 0; r < R; ++r) sum[j][r] = 0.f;
  for (int b = lane; b < nblk; b += 32) {
    int4 wlo[O], whi[O];
    float ws[O];
#pragma unroll
    for (int j = 0; j < O; ++j) {
      const int4 *wb = (const int4 *)(wrow[j] + ((size_t)b << 5));
      wlo[j] = wb[0];
      whi[j] = wb[1];
      ws[j] = wsc[j][b];
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (r < rows) {
        const int4 *xb = (const int4 *)(xrows + (size_t)r * in + ((size_t)b << 5));
        int4 xlo = xb[0], xhi = xb[1];
        float xs = xsc[(size_t)r * nblk + b];
#pragma unroll
        for (int j = 0; j < O; ++j) {
          int acc = 0;
          acc = __dp4a(wlo[j].x, xlo.x, acc);
          acc = __dp4a(wlo[j].y, xlo.y, acc);
          acc = __dp4a(wlo[j].z, xlo.z, acc);
          acc = __dp4a(wlo[j].w, xlo.w, acc);
          acc = __dp4a(whi[j].x, xhi.x, acc);
          acc = __dp4a(whi[j].y, xhi.y, acc);
          acc = __dp4a(whi[j].z, xhi.z, acc);
          acc = __dp4a(whi[j].w, xhi.w, acc);
          sum[j][r] += (float)acc * ws[j] * xs;
        }
      }
    }
  }
#pragma unroll
  for (int j = 0; j < O; ++j)
#pragma unroll
    for (int r = 0; r < R; ++r) part[warp][r * O + j][lane] = sum[j][r];
  __syncwarp();
  const float *v = part[warp][lane];
  float a[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) a[i] = v[i] + v[i + 16];
#pragma unroll
  for (int i = 0; i < 8; ++i) a[i] = a[i] + a[i + 8];
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = a[i] + a[i + 4];
  a[0] = a[0] + a[2];
  a[1] = a[1] + a[3];
  int r = lane / O, j = lane - r * O;
  if (r < rows && o0 + j < out) Y[(size_t)(t0 + r) * out + o0 + j] = a[0] + a[1];
}

// Persistent Q8 activation-quantization scratch (#969), grown ONCE and reused like g_attn_scratch.
// The per-call fcuda_malloc/fcuda_free pair tripped graph capture: the pooled allocator misses
// (-> cudaMalloc) the first time a given size class is requested, and Q8 GEMV is the decode hot
// path captured into the graph, so a fresh size class mid-capture crashed replay. Two static
// buffers sized to the largest (P*in)/(P*nblk) ever seen, reused across calls — safe because
// g_stream serializes calls and each quant-then-GEMM writes-then-reads its scratch within one call.
static signed char *g_q8_qX = nullptr;
static int g_q8_qX_cap = 0; // bytes
static float *g_q8_xScale = nullptr;
static int g_q8_xScale_cap = 0; // floats
// q8_quant_rows grows the activation scratch, then quantizes every row of X on its own. Returns 0
// or FCUDA_SCRATCH_ALLOC_FAILED before any launch.
static int q8_quant_rows(const float *dX, int in, int P, int block) {
  int nblk = in / block;
  int needQX = P * in;          // int8 activation codes
  int needScale = P * nblk;     // per-block act scales (floats)
  if (needQX > g_q8_qX_cap) {
    if (g_q8_qX) cudaFree(g_q8_qX);
    g_q8_qX = nullptr;
    g_q8_qX_cap = 0;
    if (cudaMalloc(&g_q8_qX, (size_t)needQX) != cudaSuccess || !g_q8_qX) {
      g_q8_qX = nullptr;
      return FCUDA_SCRATCH_ALLOC_FAILED;
    }
    g_q8_qX_cap = needQX;
  }
  if (needScale > g_q8_xScale_cap) {
    if (g_q8_xScale) cudaFree(g_q8_xScale);
    g_q8_xScale = nullptr;
    g_q8_xScale_cap = 0;
    if (cudaMalloc(&g_q8_xScale, (size_t)needScale * sizeof(float)) != cudaSuccess || !g_q8_xScale) {
      g_q8_xScale = nullptr;
      return FCUDA_SCRATCH_ALLOC_FAILED;
    }
    g_q8_xScale_cap = needScale;
  }
  FCUDA_TS("k_q8_quant_act", dim3(nblk, P), k_q8_quant_act<<<dim3(nblk, P), 64, 0, g_stream>>>(dX, g_q8_qX, g_q8_xScale, P, in, block));
  return 0;
}
// q8_rows_matmul is the per-row Q8_0 GEMM: each activation row is quantized on its own and
// each (row, output) dot runs the same k_q8_gemm warp sum whatever P is, so a row's result is
// bitwise the P=1 decode GEMV's. Returns 0 or FCUDA_SCRATCH_ALLOC_FAILED before any launch.
static int q8_rows_matmul(const int8_t *dCodes, const float *dScales, const float *dX,
                          float *dY, int out, int in, int P, int block) {
  int rc = q8_quant_rows(dX, in, P, block);
  if (rc != 0) return rc;
  FCUDA_TS("k_q8_gemm", dim3((out + 7) / 8, P), k_q8_gemm<<<dim3((out + 7) / 8, P), 256, 0, g_stream>>>((const signed char *)dCodes, dScales, g_q8_qX, g_q8_xScale,
                                                dY, out, in, P, block));
  return 0;
}
extern "C" int fcuda_q8_matmul_f32(const int8_t *dCodes, const float *dScales, const float *dX,
                                   float *dY, int out, int in, int P, int block) {
  if (P >= 128) return q8_dequant_sgemm(dCodes, dScales, dX, dY, out, in, P, block);
  return q8_rows_matmul(dCodes, dScales, dX, dY, out, in, P, block);
}
// fcuda_q8_matmul_rows_f32 is the BitExact prompt-panel GEMM: every row bitwise the P=1 decode
// GEMV for every P. Block-32 weights with 16-byte aligned codes run k_q8_gemm_rowtile over
// balanced tiles of at most kQ8TileRows rows; anything else runs k_q8_gemm itself.
extern "C" int fcuda_q8_matmul_rows_f32(const int8_t *dCodes, const float *dScales, const float *dX,
                                        float *dY, int out, int in, int P, int block) {
  if (!dCodes || !dScales || !dX || !dY || out <= 0 || in <= 0 || P <= 0 || P > 65535 ||
      block <= 0 || in % block != 0 || P > INT_MAX / in) return -1;
  int outGroups = (out + kQ8TileWarps * kQ8TileOuts - 1) / (kQ8TileWarps * kQ8TileOuts);
  if (block != 32 || ((uintptr_t)dCodes & 15) != 0 || outGroups > 65535)
    return q8_rows_matmul(dCodes, dScales, dX, dY, out, in, P, block);
  int rc = q8_quant_rows(dX, in, P, block);
  if (rc != 0) return rc;
  int tiles = (P + kQ8TileRows - 1) / kQ8TileRows;
  int rowsPerTile = (P + tiles - 1) / tiles;
  FCUDA_TS("k_q8_gemm_rowtile", dim3(tiles, outGroups), k_q8_gemm_rowtile<<<dim3(tiles, outGroups), kQ8TileWarps * 32, 0, g_stream>>>(
      (const signed char *)dCodes, dScales, g_q8_qX, g_q8_xScale, dY, out, in, P, rowsPerTile));
  return 0;
}

// getScaleMinK4_dev reproduces the GGUF loader's getScaleMinK4 (internal/ggufload) bit-for-bit:
// the 6-bit (scale,min) for the j-th 32-elem sub-block, unpacked from the 12 packed scale bytes.
__device__ void getScaleMinK4_dev(int j, const unsigned char *q, unsigned char *sc, unsigned char *mn) {
  if (j < 4) {
    *sc = q[j] & 63;
    *mn = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
    *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

__global__ void k_q4k_coeffs(const unsigned char *Q4K, float2 *coeff, int out, int in) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int nsb = in / 256;
  int total = out * nsb * 8;
  if (i >= total) return;
  int group = i & 7;
  int block = i >> 3;
  const unsigned char *blk = Q4K + (size_t)block * 144;
  unsigned char sc, mn;
  getScaleMinK4_dev(group, blk + 4, &sc, &mn);
  float d = __half2float(*(const __half *)blk);
  float dmin = __half2float(*(const __half *)(blk + 2));
  coeff[i] = make_float2(d * (float)sc, dmin * (float)mn);
}

__global__ void k_q4k_dequant_transient(const unsigned char *Q4K, float *W, int out, int in) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = out * in;
  if (i >= total) return;
  int o = i / in, k = i % in;
  int sb = k / 256, group = (k % 256) / 32, lane = k & 31;
  const unsigned char *blk = Q4K + ((size_t)o * (in / 256) + sb) * 144;
  float d = __half2float(*(const __half *)blk), dm = __half2float(*(const __half *)(blk + 2));
  unsigned char sc, mn; getScaleMinK4_dev(group, blk + 4, &sc, &mn);
  unsigned char packed = blk[16 + (group >> 1) * 32 + lane];
  int q = (group & 1) ? packed >> 4 : packed & 15;
  W[i] = d * (float)sc * (float)q - dm * (float)mn;
}

// k_q4k_gemm: four warps cooperate on each output row. Each warp owns a disjoint stride of
// 256-value super-blocks, preserving the exact f32-dequantized dot while exposing the short Qwen
// hidden-width K loop to the GPU. Two rows per block retain 256-thread occupancy; only the four
// warp totals cross shared memory at the final row reduction.
__global__ void k_q4k_gemm(const unsigned char *Q4K, const float2 *coeff, const float *X, float *Y, int out, int in, int P) {
  constexpr int warps_per_row = 4;
  constexpr int rows_per_block = 2;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int row_warp = warp & (warps_per_row - 1);
  int row = warp / warps_per_row;
  int o = blockIdx.x * rows_per_block + row;
  int t = blockIdx.y;
  bool active = o < out && t < P;
  int nsb = in / 256;
  const unsigned char *wrow = active ? Q4K + (size_t)o * nsb * 144 : Q4K;
  const float *xrow = active ? X + (size_t)t * in : X;
  float sum = 0.f;
  if (active) {
    for (int sb = row_warp; sb < nsb; sb += warps_per_row) {
      const unsigned char *blk = wrow + (size_t)sb * 144;
      const unsigned char *q = blk + 16;
      const float2 *cb = coeff + ((size_t)o * nsb + sb) * 8;
      const float *xb = xrow + (size_t)sb * 256;
#pragma unroll
      for (int group = 0; group < 8; group++) {
        int qi = (group >> 1) * 32 + lane;
        unsigned char packed = q[qi];
        int code = (group & 1) ? (packed >> 4) : (packed & 0x0f);
        float2 c = cb[group];
        float xval = xb[group * 32 + lane];
        sum = fmaf(c.x * (float)code, xval, sum);
        sum = fmaf(-c.y, xval, sum);
      }
    }
  }
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) sum += __shfl_down_sync(0xffffffff, sum, delta);
  __shared__ float row_sums[rows_per_block][warps_per_row];
  if (lane == 0) row_sums[row][row_warp] = sum;
  __syncthreads();
  if (active && row_warp == 0 && lane == 0) {
    float total = row_sums[row][0];
#pragma unroll
    for (int i = 1; i < warps_per_row; ++i) total += row_sums[row][i];
    Y[(size_t)t * out + o] = total;
  }
}

// k_q4k_gemm_panel reuses each dequantized Q4_K lane across a short token panel. Prefill
// otherwise rereads the same weight row once per token; keeping four token accumulators in each
// lane trades registers for a 4x reduction in weight traffic while preserving f32 dequantization.
template <int token_tile>
__global__ void k_q4k_gemm_panel(const unsigned char *Q4K, const float *X, float *Y,
                                 int out, int in, int P) {
  int lane = threadIdx.x;
  int o = blockIdx.x;
  int token0 = blockIdx.y * token_tile;
  int nsb = in / 256;
  const unsigned char *wrow = Q4K + (size_t)o * nsb * 144;
  float sums[token_tile] = {};
  for (int sb = 0; sb < nsb; ++sb) {
    const unsigned char *blk = wrow + (size_t)sb * 144;
    float d = __half2float(*(const __half *)(blk));
    float dmin = __half2float(*(const __half *)(blk + 2));
    const unsigned char *scales = blk + 4;
    const unsigned char *q = blk + 16;
#pragma unroll
    for (int group = 0; group < 8; ++group) {
      unsigned char sc, mn;
      getScaleMinK4_dev(group, scales, &sc, &mn);
      int qi = (group >> 1) * 32 + lane;
      unsigned char packed = q[qi];
      int code = (group & 1) ? (packed >> 4) : (packed & 0x0f);
      float w = d * (float)sc * (float)code - dmin * (float)mn;
      int k = sb * 256 + group * 32 + lane;
#pragma unroll
      for (int j = 0; j < token_tile; ++j) {
        int token = token0 + j;
        if (token < P) sums[j] = fmaf(w, X[(size_t)token * in + k], sums[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < token_tile; ++j) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
      sums[j] += __shfl_down_sync(0xffffffff, sums[j], delta);
    if (lane == 0 && token0 + j < P) Y[(size_t)(token0 + j) * out + o] = sums[j];
  }
}

struct q4_coeff_cache_entry { const uint8_t *q; float2 *coeff; int out; int in; int device; };
static q4_coeff_cache_entry g_q4_coeff_cache[512];
static int g_q4_coeff_cache_n = 0;
static void q4_coeff_cache_evict(const void *q) {
  int device = 0;
  CK(cudaGetDevice(&device));
  for (int i = 0; i < g_q4_coeff_cache_n; ++i) {
    if (g_q4_coeff_cache[i].q != q || g_q4_coeff_cache[i].device != device) continue;
    CK(cudaFree(g_q4_coeff_cache[i].coeff));
    g_q4_coeff_cache[i] = g_q4_coeff_cache[--g_q4_coeff_cache_n];
    g_q4_coeff_cache[g_q4_coeff_cache_n] = {};
    return;
  }
}
static float2 *q4_coeffs_for(const uint8_t *q, int out, int in) {
  int device = 0;
  CK(cudaGetDevice(&device));
  for (int i = 0; i < g_q4_coeff_cache_n; ++i) {
    q4_coeff_cache_entry &entry = g_q4_coeff_cache[i];
    if (entry.q == q && entry.out == out && entry.in == in && entry.device == device) return entry.coeff;
  }
  if (g_q4_coeff_cache_n >= 512) { fprintf(stderr, "fak-cuda: q4 coeff cache full\n"); abort(); }
  size_t n = (size_t)out * (in / 256) * 8;
  float2 *coeff = nullptr;
  CK(cudaMalloc(&coeff, n * sizeof(float2)));
  FCUDA_TS("k_q4k_coeffs", (n + 255) / 256, k_q4k_coeffs<<<(n + 255) / 256, 256, 0, g_stream>>>(q, coeff, out, in));
  g_q4_coeff_cache[g_q4_coeff_cache_n++] = {q, coeff, out, in, device};
  return coeff;
}

// g_q4k_prefill_arm is the P>=128 Q4_K GEMM arm the Go side resolved from FAK_CUDA_Q4K_PREFILL
// (cuda_backend.h): 0 compensated, 1 mmq, 2 experimental fp8lt. Any other arm returns
// FCUDA_Q4K_PREFILL_ARM_REFUSED.
#define FCUDA_Q4K_PREFILL_ARM_REFUSED 80003
static int g_q4k_prefill_arm = 0;
extern "C" void fcuda_set_q4k_prefill_arm(int arm) { g_q4k_prefill_arm = arm; }
static int fcuda_q4k_fp8lt_matmul(const uint8_t *, const float *, float *, int, int, int);

extern "C" int fcuda_q4k_matmul_f32(const uint8_t *dQ4K, const float *dX, float *dY,
                                    int out, int in, int P) {
  if (P >= 128) {
    if (g_q4k_prefill_arm == 1) return fcuda_q4k_mmq_matmul(dQ4K, dX, dY, out, in, P);
    if (g_q4k_prefill_arm == 2) return fcuda_q4k_fp8lt_matmul(dQ4K, dX, dY, out, in, P);
    if (g_q4k_prefill_arm != 0) return FCUDA_Q4K_PREFILL_ARM_REFUSED;
    float *wide = (float *)scratch_malloc((size_t)out * in * sizeof(float));
    if (!wide) return FCUDA_SCRATCH_ALLOC_FAILED;
    FCUDA_TS("k_q4k_dequant_transient", ((size_t)out * in + 255) / 256, k_q4k_dequant_transient<<<((size_t)out * in + 255) / 256, 256, 0, g_stream>>>(
        (const unsigned char *)dQ4K, wide, out, in));
    int status = fp16_compensated_matmul(wide, dX, dY, out, in, P);
    fcuda_free(wide);
    return status;
  }
  if (P < 4) {
    float2 *coeff = q4_coeffs_for(dQ4K, out, in);
    FCUDA_TS("k_q4k_gemm", dim3((out + 1) / 2, P), k_q4k_gemm<<<dim3((out + 1) / 2, P), 256, 0, g_stream>>>(
        (const unsigned char *)dQ4K, coeff, dX, dY, out, in, P));
    return 0;
  }
  constexpr int token_tile = 4;
  FCUDA_TS("k_q4k_gemm_panel", dim3(out, (P + token_tile - 1) / token_tile), k_q4k_gemm_panel<token_tile><<<dim3(out, (P + token_tile - 1) / token_tile), 32, 0, g_stream>>>(
      (const unsigned char *)dQ4K, dX, dY, out, in, P));
  return 0;
}

// k_q2_0_gemm: Y[t,o] = Σ_b (Σ_{i in block} tern(o,b,i)·X[t,b·block+i]) · Wscale[o,b], where
// each weight is a PACKED TERNARY code — 2 bits, 4 per byte, LSB-first, decoding u∈{0,1,2}
// to t=u-1∈{-1,0,+1} (the packed-ternary Q2_0 format, issue #4872). The weight is NEVER
// expanded to f32/f16 in VRAM: the kernel reads one code byte, unpacks the signed indicator,
// and accumulates X directly (a select/add, not a multiply), folding one f32 block scale at
// block end — the same per-block scheme as cpuref q2RowDot (only the reduction order differs,
// which is what makes the device lane Approx, not Reference). One block per (o,t); threads
// stride the ternary blocks; a shared-memory tree reduces the partials. in must divide block.
__global__ void k_q2_0_gemm(const unsigned char *Codes, const float *Wscale,
                            const float *X, float *Y, int out, int in, int P, int block) {
  int o = blockIdx.x, t = blockIdx.y;
  if (o >= out || t >= P) return;
  int nblk = in / block;
  int rowBytes = in / 4; // 2-bit codes, 4 per byte
  const unsigned char *crow = Codes + (size_t)o * rowBytes;
  const float *wsc = Wscale + (size_t)o * nblk;
  const float *xrow = X + (size_t)t * in;
  __shared__ float red[256];
  float local = 0.f;
  for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
    int off = b * block;
    float s = 0.f;
    for (int i = 0; i < block; i++) {
      int gi = off + i;
      int code = (int)((crow[gi >> 2] >> ((gi & 3) * 2)) & 0x3) - 1; // {-1,0,+1}
      s += (float)code * xrow[gi];
    }
    local += s * wsc[b];
  }
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) Y[(size_t)t * out + o] = red[0];
}

extern "C" void fcuda_q2_0_matmul_f32(const uint8_t *dCodes, const float *dScales,
                                      const float *dX, float *dY, int out, int in, int P,
                                      int block) {
  FCUDA_TS("k_q2_0_gemm", dim3(out, P), k_q2_0_gemm<<<dim3(out, P), 256, 0, g_stream>>>(
      (const unsigned char *)dCodes, dScales, dX, dY, out, in, P, block));
}

// Reference-order NVFP4 W4A16 path. This is intentionally one thread per output:
// it preserves the CPU oracle's serial K reduction while keeping the packed E2M1
// weights and E4M3FN block scales resident. It is a correctness baseline, not a
// tensor-core performance claim. X must be finite and remain finite after BF16 RNE;
// preserving stream asynchrony means this ABI does not scan device data first.
static __device__ __forceinline__ float nv4_bf16_rne(float x) {
  uint32_t bits = __float_as_uint(x);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return __uint_as_float(bits & 0xffff0000u);
}

static __device__ __forceinline__ float nv4_e2m1(uint32_t nib) {
  const uint32_t sign = (nib & 0x8u) << 28;
  const uint32_t e = (nib >> 1) & 0x3u;
  const uint32_t m = nib & 0x1u;
  const uint32_t mag = e ? (((e + 126u) << 23) | (m << 22))
                         : (m ? 0x3f000000u : 0u);
  return __uint_as_float(sign | mag);
}

static __device__ __forceinline__ float nv4_e4m3fn(uint32_t b) {
  const uint32_t sign = (b & 0x80u) << 24;
  const uint32_t e = (b >> 3) & 0x0fu;
  const uint32_t m = b & 0x07u;
  uint32_t mag;
  if (e != 0u) {
    mag = ((e + 120u) << 23) | (m << 20);
  } else {
    mag = __float_as_uint(__fmul_rn((float)m, 0.001953125f));
  }
  return __uint_as_float(sign | mag);
}

__global__ void k_nvfp4_w4a16_f32(const uint8_t *Payload, const float *X,
                                  float *Y, int out, int in, int P,
                                  float globalScale) {
  const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t outputs = (size_t)P * (size_t)out;
  if (index >= outputs) return;

  const int o = (int)(index % (size_t)out);
  const int p = (int)(index / (size_t)out);
  const size_t weightBytes = (size_t)out * (size_t)in / 2u;
  const uint8_t *wrow = Payload + (size_t)o * (size_t)in / 2u;
  const uint8_t *srow = Payload + weightBytes + (size_t)o * (size_t)(in / 16);
  const float *xrow = X + (size_t)p * (size_t)in;

  float sum = 0.0f;
  for (int k = 0; k < in; ++k) {
    const uint8_t pair = wrow[k >> 1];
    const uint32_t nib = (k & 1) ? (uint32_t)(pair >> 4) : (uint32_t)(pair & 0x0fu);
    const float weight = __fmul_rn(nv4_e2m1(nib), nv4_e4m3fn(srow[k >> 4]));
    const float product = __fmul_rn(weight, nv4_bf16_rne(xrow[k]));
    sum = __fadd_rn(sum, product);
  }
  Y[index] = __fmul_rn(sum, globalScale);
}

extern "C" int fcuda_nvfp4_w4a16_f32(const uint8_t *dPayload, const float *dX,
                                      float *dY, int out, int in, int P,
                                      float globalScale) {
  if (!dPayload || !dX || !dY || out <= 0 || in <= 0 || P <= 0 || (in & 15) != 0 ||
      globalScale < 0.0f || !isfinite(globalScale))
    return (int)cudaErrorInvalidValue;
  const size_t outputs = (size_t)P * (size_t)out;
  const size_t blocks = (outputs + 255u) / 256u;
  if (blocks > (size_t)UINT_MAX) return (int)cudaErrorInvalidConfiguration;
  FCUDA_TS("k_nvfp4_w4a16_f32", (unsigned int)blocks, k_nvfp4_w4a16_f32<<<(unsigned int)blocks, 256, 0, g_stream>>>(
      dPayload, dX, dY, out, in, P, globalScale));
  return (int)cudaGetLastError();
}

// Q5_K/Q6_K resident dequant-fused GEMV. One warp computes one output row.
__global__ void k_q5k_gemm(const unsigned char *W, const float *X, float *Y,
                           int out, int in, int P) {
  int lane = threadIdx.x & 31;
  int o = blockIdx.x * 8 + (threadIdx.x >> 5);
  int t = blockIdx.y;
  if (o >= out || t >= P) return;

  int nsb = in / 256;
  float sum = 0.f;
  const unsigned char *wr = W + (size_t)o * nsb * 176;
  const float *xr = X + (size_t)t * in;
  for (int sb = 0; sb < nsb; ++sb) {
    const unsigned char *b = wr + (size_t)sb * 176;
    float d = __half2float(*(const __half *)b);
    float dm = __half2float(*(const __half *)(b + 2));
    const unsigned char *sc = b + 4;
    const unsigned char *qh = b + 16;
    const unsigned char *ql = b + 48;
    const float *xb = xr + (size_t)sb * 256;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      unsigned char scale, mn;
      getScaleMinK4_dev(g, sc, &scale, &mn);
      int pair = g >> 1;
      int q = (g & 1) ? (ql[pair * 32 + lane] >> 4)
                      : (ql[pair * 32 + lane] & 15);
      if (qh[lane] & (1u << g)) q += 16;
      sum += (d * (float)scale * (float)q - dm * (float)mn) * xb[g * 32 + lane];
    }
  }
#pragma unroll
  for (int z = 16; z; z >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, z);
  if (lane == 0) Y[(size_t)t * out + o] = sum;
}

__global__ void k_q6k_gemm(const unsigned char *W, const float *X, float *Y,
                           int out, int in, int P) {
  int lane = threadIdx.x & 31;
  int o = blockIdx.x * 8 + (threadIdx.x >> 5);
  int t = blockIdx.y;
  if (o >= out || t >= P) return;

  int nsb = in / 256;
  float sum = 0.f;
  const unsigned char *wr = W + (size_t)o * nsb * 210;
  const float *xr = X + (size_t)t * in;
  for (int sb = 0; sb < nsb; ++sb) {
    const unsigned char *b = wr + (size_t)sb * 210;
    const unsigned char *ql = b;
    const unsigned char *qh = b + 128;
    const signed char *sc = (const signed char *)(b + 192);
    float d = __half2float(*(const __half *)(b + 208));
    const float *xb = xr + (size_t)sb * 256;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int qo = h * 64;
      int ho = h * 32;
      int so = h * 8;
      int base = h * 128;
      int is = lane >> 4;
      int q1 = (ql[qo + lane] & 15) | (((qh[ho + lane] >> 0) & 3) << 4);
      int q2 = (ql[qo + lane + 32] & 15) | (((qh[ho + lane] >> 2) & 3) << 4);
      int q3 = (ql[qo + lane] >> 4) | (((qh[ho + lane] >> 4) & 3) << 4);
      int q4 = (ql[qo + lane + 32] >> 4) | (((qh[ho + lane] >> 6) & 3) << 4);
      sum += d * (float)sc[so + is] * (q1 - 32) * xb[base + lane];
      sum += d * (float)sc[so + is + 2] * (q2 - 32) * xb[base + lane + 32];
      sum += d * (float)sc[so + is + 4] * (q3 - 32) * xb[base + lane + 64];
      sum += d * (float)sc[so + is + 6] * (q4 - 32) * xb[base + lane + 96];
    }
  }
#pragma unroll
  for (int z = 16; z; z >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, z);
  if (lane == 0) Y[(size_t)t * out + o] = sum;
}

extern "C" void fcuda_q5k_matmul_f32(const uint8_t *w, const float *x, float *y,
                                      int o, int i, int p) {
  FCUDA_TS("k_q5k_gemm", dim3((o + 7) / 8, p), k_q5k_gemm<<<dim3((o + 7) / 8, p), 256, 0, g_stream>>>(w, x, y, o, i, p));
}
extern "C" void fcuda_q6k_matmul_f32(const uint8_t *w, const float *x, float *y,
                                      int o, int i, int p) {
  FCUDA_TS("k_q6k_gemm", dim3((o + 7) / 8, p), k_q6k_gemm<<<dim3((o + 7) / 8, p), 256, 0, g_stream>>>(w, x, y, o, i, p));
}

// =================================================================================
// Q2_K Resident Dequant-Fused GEMV & Panel GEMM (sm_80/sm_89/sm_90, issue #11945)
// Super-block: 256 weights, 84 bytes (16 scales, 64 quants, 2 d, 2 dmin)
// =================================================================================

// k_q2k_dequant_transient: unpacks resident Q2_K weights to transient FP32 for Tensor Core GEMM (P >= 128).
__global__ void k_q2k_dequant_transient(const unsigned char *Q2K, float *W, int out, int in) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = out * in;
  if (i >= total) return;

  int o = i / in;
  int k = i % in;
  int sb = k / 256;
  int elem = k % 256;

  const unsigned char *blk = Q2K + ((size_t)o * (in / 256) + sb) * 84;
  float d = __half2float(*(const __half *)(blk + 80));
  float min = __half2float(*(const __half *)(blk + 82));

  int h = elem / 128;
  int rem = elem % 128;
  int j = rem / 32;
  int lane = rem % 32;
  int lane_half = lane >> 4;
  int is = h * 8 + 2 * j + lane_half;
  unsigned char sc = blk[is];
  float dl = d * (float)(sc & 0x0f);
  float ml = min * (float)(sc >> 4);

  unsigned char qb = blk[16 + h * 32 + lane];
  int code = (qb >> (2 * j)) & 3;

  W[i] = dl * (float)code - ml;
}

// k_q2k_gemv: 1 warp (32 threads) per output row.
// 256 threads per block = 8 rows per block. Zero shared memory, warp shuffle reduction.
// sm_80/sm_89/sm_90 native execution eliminating CPU decode offload.
__global__ void k_q2k_gemv(const unsigned char *W, const float *X, float *Y,
                           int out, int in, int P) {
  int lane = threadIdx.x & 31;
  int o = blockIdx.x * 8 + (threadIdx.x >> 5);
  int t = blockIdx.y;
  if (o >= out || t >= P) return;

  int nsb = in / 256;
  float sum = 0.f;
  const unsigned char *wrow = W + (size_t)o * nsb * 84;
  const float *xrow = X + (size_t)t * in;
  int lane_half = lane >> 4; // 0 for lane 0..15, 1 for lane 16..31

  for (int sb = 0; sb < nsb; ++sb) {
    const unsigned char *blk = wrow + (size_t)sb * 84;
    const unsigned char *scales = blk;
    const unsigned char *q = blk + 16;
    float d = __half2float(*(const __half *)(blk + 80));
    float min = __half2float(*(const __half *)(blk + 82));
    const float *xb = xrow + (size_t)sb * 256;

#pragma unroll
    for (int h = 0; h < 2; ++h) {
      unsigned char qb = q[h * 32 + lane];
      int so = h * 8;
      int base = h * 128;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        unsigned char sc = scales[so + 2 * j + lane_half];
        float dl = d * (float)(sc & 0x0f);
        float ml = min * (float)(sc >> 4);
        int code = (qb >> (2 * j)) & 3;
        float w = dl * (float)code - ml;
        sum = fmaf(w, xb[base + j * 32 + lane], sum);
      }
    }
  }

#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, delta);
  }
  if (lane == 0) {
    Y[(size_t)t * out + o] = sum;
  }
}

// k_q2k_gemm_panel: reuses resident Q2_K weights across token_tile tokens to amortize DRAM traffic.
template <int token_tile>
__global__ void k_q2k_gemm_panel(const unsigned char *W, const float *X, float *Y,
                                 int out, int in, int P) {
  int lane = threadIdx.x;
  int o = blockIdx.x;
  int token0 = blockIdx.y * token_tile;
  if (o >= out) return;

  int nsb = in / 256;
  const unsigned char *wrow = W + (size_t)o * nsb * 84;
  float sums[token_tile] = {0.f};
  int lane_half = lane >> 4;

  for (int sb = 0; sb < nsb; ++sb) {
    const unsigned char *blk = wrow + (size_t)sb * 84;
    const unsigned char *scales = blk;
    const unsigned char *q = blk + 16;
    float d = __half2float(*(const __half *)(blk + 80));
    float min = __half2float(*(const __half *)(blk + 82));

#pragma unroll
    for (int h = 0; h < 2; ++h) {
      unsigned char qb = q[h * 32 + lane];
      int so = h * 8;
      int base = sb * 256 + h * 128;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        unsigned char sc = scales[so + 2 * j + lane_half];
        float dl = d * (float)(sc & 0x0f);
        float ml = min * (float)(sc >> 4);
        int code = (qb >> (2 * j)) & 3;
        float w = dl * (float)code - ml;
        int k = base + j * 32 + lane;
#pragma unroll
        for (int m = 0; m < token_tile; ++m) {
          int token = token0 + m;
          if (token < P) {
            sums[m] = fmaf(w, X[(size_t)token * in + k], sums[m]);
          }
        }
      }
    }
  }

#pragma unroll
  for (int m = 0; m < token_tile; ++m) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      sums[m] += __shfl_down_sync(0xffffffff, sums[m], delta);
    }
    if (lane == 0 && token0 + m < P) {
      Y[(size_t)(token0 + m) * out + o] = sums[m];
    }
  }
}

extern "C" void fcuda_q2k_gemv(const uint8_t *dQ2K, const float *dX, float *dY,
                               int out, int in, int P) {
  FCUDA_TS("k_q2k_gemv", dim3((out + 7) / 8, P), k_q2k_gemv<<<dim3((out + 7) / 8, P), 256, 0, g_stream>>>(
      (const unsigned char *)dQ2K, dX, dY, out, in, P));
}

extern "C" void fcuda_q2k_gemm_panel(const uint8_t *dQ2K, const float *dX, float *dY,
                                     int out, int in, int P) {
  constexpr int token_tile = 4;
  FCUDA_TS("k_q2k_gemm_panel", dim3(out, (P + token_tile - 1) / token_tile), k_q2k_gemm_panel<token_tile><<<dim3(out, (P + token_tile - 1) / token_tile), 32, 0, g_stream>>>(
      (const unsigned char *)dQ2K, dX, dY, out, in, P));
}

extern "C" int fcuda_q2k_matmul_f32(const uint8_t *dQ2K, const float *dX, float *dY,
                                    int out, int in, int P) {
  if (P >= 128) {
    float *wide = (float *)scratch_malloc((size_t)out * in * sizeof(float));
    if (!wide) return FCUDA_SCRATCH_ALLOC_FAILED;
    FCUDA_TS("k_q2k_dequant_transient", ((size_t)out * in + 255) / 256, k_q2k_dequant_transient<<<((size_t)out * in + 255) / 256, 256, 0, g_stream>>>(
        (const unsigned char *)dQ2K, wide, out, in));
    int status = fp16_compensated_matmul(wide, dX, dY, out, in, P);
    fcuda_free(wide);
    return status;
  }
  if (P < 4) {
    fcuda_q2k_gemv(dQ2K, dX, dY, out, in, P);
    return 0;
  }
  fcuda_q2k_gemm_panel(dQ2K, dX, dY, out, in, P);
  return 0;
}
