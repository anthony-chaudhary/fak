// ---- AWQ (Activation-aware Weight Quantization) 4-bit kernels -------------------
// AWQ format: 4-bit weights packed 2 per byte (nibble-packed), per-channel scales.
// Dequantization: weight = scale[o] * (code - 8), where 8 is the zero-point.
// Kernels compute matmul directly on 4-bit weights without full dequantization.

// k_awq_dequant_row dequantizes one AWQ row: dst[i] = scale * (unpack4bit(src[i]) - 8)
// One block per row, 256 threads for dequantization.
__global__ void k_awq_dequant_row(const uint8_t *src, float scale, float *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // Each byte contains two 4-bit values: low nibble first, high nibble second
  int byteIdx = i / 2;
  uint8_t b = src[byteIdx];
  uint8_t code;
  if (i % 2 == 0) {
    code = b & 0x0f;  // low nibble
  } else {
    code = b >> 4;    // high nibble
  }
  dst[i] = scale * (float)(int32_t(code) - 8);
}

// k_awq_gemv computes y = A @ x where A is AWQ 4-bit [out, in], x is [in].
// One block per output row, threads collaborate on dot product.
__global__ void k_awq_gemv(const uint8_t *dW, const float *dScales, const float *dX, float *dY, int out, int in) {
  int o = blockIdx.x;
  if (o >= out) return;
  const uint8_t *row = dW + (size_t)o * (in / 2);
  float scale = dScales[o];

  __shared__ float red[256];
  float local = 0.f;

  // Each thread handles a portion of the dot product
  for (int i = threadIdx.x; i < in / 2; i += blockDim.x) {
    uint8_t b = row[i];
    // Low nibble
    float w0 = scale * (float)(int32_t(b & 0x0f) - 8);
    local += w0 * dX[i * 2];
    // High nibble
    float w1 = scale * (float)(int32_t(b >> 4) - 8);
    local += w1 * dX[i * 2 + 1];
  }

  // Block reduction
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    dY[o] = red[0];
  }
}

// k_awq_gemm computes Y[P, out] = X[P, in] @ W[out, in]^T where W is AWQ 4-bit.
// Grid: (P, out) blocks, 256 threads per block.
// Each block computes one output element by dotting one input row with one weight row.
__global__ void k_awq_gemm(const uint8_t *dW, const float *dScales, const float *dX, float *dY, int out, int in, int P) {
  int t = blockIdx.y;  // token index
  int o = blockIdx.x;  // output channel
  if (t >= P || o >= out) return;

  const uint8_t *row = dW + (size_t)o * (in / 2);
  const float *xRow = dX + (size_t)t * in;
  float scale = dScales[o];

  float acc = 0.f;
  for (int i = threadIdx.x; i < in / 2; i += blockDim.x) {
    uint8_t b = row[i];
    // Low nibble
    float w0 = scale * (float)(int32_t(b & 0x0f) - 8);
    acc += w0 * xRow[i * 2];
    // High nibble
    float w1 = scale * (float)(int32_t(b >> 4) - 8);
    acc += w1 * xRow[i * 2 + 1];
  }

  // Block reduction
  __shared__ float red[256];
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    dY[t * out + o] = red[0];
  }
}

// C API for AWQ kernels

// fcuda_awq_gemv: y[out] = AWQ[out, in] @ x[in]
extern "C" void fcuda_awq_gemv(const uint8_t *dW, const float *dScales, const float *dX, float *dY, int out, int in) {
  int threads = 256;
  FCUDA_TS("k_awq_gemv", out, k_awq_gemv<<<out, threads, 0, g_stream>>>(dW, dScales, dX, dY, out, in));
}

// fcuda_awq_gemm: Y[P, out] = X[P, in] @ AWQ[out, in]^T
extern "C" void fcuda_awq_gemm(const uint8_t *dW, const float *dScales, const float *dX, float *dY, int out, int in, int P) {
  dim3 threads(256, 1);
  dim3 blocks(out, P);
  FCUDA_TS("k_awq_gemm", blocks, k_awq_gemm<<<blocks, threads, 0, g_stream>>>(dW, dScales, dX, dY, out, in, P));
}

// ---- GPTQ (AutoGPTQ/GPTQModel) int32-packed weight-only kernels -----------------
// GPTQ format: 4/8-bit codes packed pack=32/bits per int32 along the INPUT dim
// (qweight [in/pack, out]), zero-points packed the same way along the OUTPUT dim
// (qzeros [nGroups, out/pack]), and per-group f32 scales [nGroups, out]. An optional
// int32 g_idx [in] carries the activation-order group of each input; when NULL the
// group is i/groupSize (clamped to nGroups-1). Dequant mirrors internal/model/gptq.go
// bit-for-bit: weight[o,i] = (code(i,o) - (zero(g,o)+1)) * scale[g,o], the AutoGPTQ
// convention (unpack(qzeros)+1 is the effective zero point). The dequant is fused into
// the GEMV tile and accumulated in F32 — no full dequant buffer is materialized.

// gptqCode unpacks the 4/8-bit code at (input i, output o) from int32-packed qweight.
__device__ __forceinline__ uint32_t gptqCode(const uint32_t *dW, int i, int o, int out,
                                             int pack, int bits, uint32_t mask) {
  uint32_t v = dW[(size_t)(i / pack) * out + o];
  return (v >> (uint32_t((i % pack) * bits))) & mask;
}

// gptqZero unpacks the effective zero point at (group g, output o): AutoGPTQ stores
// zero-1, so the +1 here matches the CPU resident oracle's qt.zero().
__device__ __forceinline__ uint32_t gptqZero(const uint32_t *dZeros, int g, int o, int outPack,
                                             int pack, int bits, uint32_t mask) {
  uint32_t v = dZeros[(size_t)g * outPack + (o / pack)];
  return ((v >> (uint32_t((o % pack) * bits))) & mask) + 1u;
}

// k_gptq_gemv computes y = W @ x where W is a GPTQ int32-packed [out, in] weight.
// One block per output row; threads collaborate on the dot product with a tree reduction.
__global__ void k_gptq_gemv(const uint32_t *dW, const uint32_t *dZeros, const float *dScales,
                            const int32_t *dGidx, const float *dX, float *dY,
                            int out, int in, int bits, int groupSize, int nGroups) {
  int o = blockIdx.x;
  if (o >= out) return;
  int pack = 32 / bits;
  uint32_t mask = (1u << bits) - 1u;
  int outPack = out / pack;

  __shared__ float red[256];
  float local = 0.f;

  for (int i = threadIdx.x; i < in; i += blockDim.x) {
    int g;
    if (dGidx) {
      g = dGidx[i];
    } else {
      g = i / groupSize;
      if (g >= nGroups) g = nGroups - 1;
    }
    uint32_t code = gptqCode(dW, i, o, out, pack, bits, mask);
    uint32_t zero = gptqZero(dZeros, g, o, outPack, pack, bits, mask);
    float w = ((float)(int32_t)code - (float)(int32_t)zero) * dScales[(size_t)g * out + o];
    local += w * dX[i];
  }

  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    dY[o] = red[0];
  }
}

// C API for GPTQ kernels

// fcuda_gptq_gemv: y[out] = GPTQ[out, in] @ x[in]. dGidx may be NULL (group = i/groupSize).
extern "C" void fcuda_gptq_gemv(const uint32_t *dW, const uint32_t *dZeros, const float *dScales,
                                const int32_t *dGidx, const float *dX, float *dY,
                                int out, int in, int bits, int groupSize, int nGroups) {
  int threads = 256;
  FCUDA_TS("k_gptq_gemv", out, k_gptq_gemv<<<out, threads, 0, g_stream>>>(dW, dZeros, dScales, dGidx, dX, dY,
                                             out, in, bits, groupSize, nGroups));
}
/* Draft insertion in cuda_kernels.cu. Clean-room use of the documented
 * cuBLASLt API; no NVIDIA sample source is copied. The official BSD-3 sample
 * was consulted only as an API cross-check:
 * NVIDIA/CUDALibrarySamples@a94482ebecf8b16d5b83ab276b7db3a84979f0e5,
 * cuBLASLt/LtNvfp4Matmul/sample_cublasLt_LtNvfp4Matmul.cu. */
#include <cublasLt.h>
#include <stdint.h>

static uint64_t g_nvfp4_lt_calls = 0;
static const size_t g_nvfp4_lt_workspace_bytes = (size_t)32 * 1024 * 1024;

static int nvfp4_lt_cublas_status(cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS) return FCUDA_NVFP4_LT_OK;
  if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH)
    return FCUDA_NVFP4_LT_RUNTIME_UNSUPPORTED;
  return FCUDA_NVFP4_LT_CUBLAS_ERROR_BASE - (int)status;
}

static void nvfp4_lt_cleanup_status(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    fprintf(stderr, "fak-cuda: NVFP4 cuBLASLt cleanup %s failed: %d\n",
            operation, (int)status);
}

static int nvfp4_lt_capture_status(void) {
  if (g_capture_open) return FCUDA_NVFP4_LT_CAPTURE;
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  cudaError_t status = cudaStreamIsCapturing(g_stream, &capture);
  if (status != cudaSuccess) return FCUDA_NVFP4_LT_CUDA_ERROR_BASE - (int)status;
  return capture == cudaStreamCaptureStatusNone ? FCUDA_NVFP4_LT_OK : FCUDA_NVFP4_LT_CAPTURE;
}

static bool nvfp4_lt_scale_bytes(int rows, int scale_cols, size_t *bytes) {
  if (rows <= 0 || scale_cols <= 0 || !bytes) return false;
  size_t padded_rows = ((size_t)rows + 127u) / 128u * 128u;
  size_t padded_cols = ((size_t)scale_cols + 3u) / 4u * 4u;
  if (padded_rows > SIZE_MAX / padded_cols) return false;
  *bytes = padded_rows * padded_cols;
  return true;
}

__global__ static void k_nvfp4_lt_swizzle_scales(const uint8_t *linear,
                                                  uint8_t *tiled,
                                                  int rows, int scale_cols) {
  size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t logical = (size_t)rows * (size_t)scale_cols;
  if (index >= logical) return;
  int row = (int)(index / (size_t)scale_cols);
  int col = (int)(index - (size_t)row * (size_t)scale_cols);
  int tile_cols = (scale_cols + 3) / 4;
  size_t tile = ((size_t)(row / 128) * (size_t)tile_cols + (size_t)(col / 4)) * 512u;
  size_t offset = tile + (size_t)(row % 32) * 16u +
                  (size_t)((row % 128) / 32) * 4u + (size_t)(col % 4);
  tiled[offset] = linear[index];
}

static int nvfp4_lt_swizzle_scales(const uint8_t *linear, uint8_t *tiled,
                                    size_t tiled_bytes, int rows,
                                    int scale_cols) {
  size_t expected = 0;
  if (!linear || !tiled || !nvfp4_lt_scale_bytes(rows, scale_cols, &expected) ||
      tiled_bytes != expected)
    return FCUDA_NVFP4_LT_GEOMETRY;
  size_t logical = (size_t)rows * (size_t)scale_cols;
  if (logical > (size_t)INT_MAX) return FCUDA_NVFP4_LT_GEOMETRY;
  cudaError_t error = cudaSuccess;
  FCUDA_TS("fill", dim3(0, 0), error = cudaMemsetAsync(tiled, 0, tiled_bytes, g_stream));
  if (error != cudaSuccess) return FCUDA_NVFP4_LT_CUDA_ERROR_BASE - (int)error;
  unsigned int blocks = (unsigned int)((logical + 255u) / 256u);
  FCUDA_TS("k_nvfp4_lt_swizzle_scales", blocks, k_nvfp4_lt_swizzle_scales<<<blocks, 256, 0, g_stream>>>(
      linear, tiled, rows, scale_cols));
  error = cudaPeekAtLastError();
  return error == cudaSuccess ? FCUDA_NVFP4_LT_OK
                              : FCUDA_NVFP4_LT_CUDA_ERROR_BASE - (int)error;
}

extern "C" int fcuda_nvfp4_w4a4_preflight(size_t *library_version) {
#if !defined(CUBLAS_VERSION) || CUBLAS_VERSION < 120800
  if (library_version) *library_version = 0;
  return FCUDA_NVFP4_LT_COMPILE_UNSUPPORTED;
#else
  if (library_version) *library_version = cublasLtGetVersion();
  if (!g_stream || !g_blas || !g_blas_workspace)
    return FCUDA_NVFP4_LT_WORKSPACE_UNAVAILABLE;
  int capture = nvfp4_lt_capture_status();
  if (capture != FCUDA_NVFP4_LT_OK) return capture;
  if (!g_cublas_lt) {
    cublasStatus_t status = cublasLtCreate(&g_cublas_lt);
    if (status != CUBLAS_STATUS_SUCCESS) return nvfp4_lt_cublas_status(status);
  }
  return FCUDA_NVFP4_LT_OK;
#endif
}

extern "C" int fcuda_nvfp4_w4a4_f32(const uint8_t *dW,
                                      const uint8_t *dWLinearScale,
                                      uint8_t *dWScaleScratch,
                                      size_t wScaleScratchBytes,
                                      const uint8_t *dA,
                                      const uint8_t *dALinearScale,
                                      uint8_t *dAScaleScratch,
                                      size_t aScaleScratchBytes,
                                      float *dY, int out, int k, int rows,
                                      float alpha, int skip_algo_66) {
#if !defined(CUBLAS_VERSION) || CUBLAS_VERSION < 120800
  (void)dW; (void)dWLinearScale; (void)dWScaleScratch; (void)wScaleScratchBytes;
  (void)dA; (void)dALinearScale; (void)dAScaleScratch; (void)aScaleScratchBytes; (void)dY;
  (void)out; (void)k; (void)rows; (void)alpha; (void)skip_algo_66;
  return FCUDA_NVFP4_LT_COMPILE_UNSUPPORTED;
#else
  size_t expectedWScaleBytes = 0, expectedAScaleBytes = 0;
  int scaleCols = k / 16;
  if (!dW || !dWLinearScale || !dWScaleScratch || !dA || !dALinearScale ||
      !dAScaleScratch || !dY || out <= 0 || k <= 0 ||
      rows <= 0 || (k % 32) != 0 || (out % 4) != 0 ||
      !isfinite(alpha) || alpha < 0.0f ||
      !nvfp4_lt_scale_bytes(out, scaleCols, &expectedWScaleBytes) ||
      !nvfp4_lt_scale_bytes(rows, scaleCols, &expectedAScaleBytes) ||
      wScaleScratchBytes != expectedWScaleBytes ||
      aScaleScratchBytes != expectedAScaleBytes ||
      (((uintptr_t)dW | (uintptr_t)dWLinearScale | (uintptr_t)dWScaleScratch |
        (uintptr_t)dA | (uintptr_t)dALinearScale | (uintptr_t)dAScaleScratch |
        (uintptr_t)dY) & 15u) != 0)
    return FCUDA_NVFP4_LT_GEOMETRY;
  int status = fcuda_nvfp4_w4a4_preflight(nullptr);
  if (status != FCUDA_NVFP4_LT_OK) return status;

  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t wLayout = nullptr, aLayout = nullptr;
  cublasLtMatrixLayout_t cLayout = nullptr, yLayout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasStatus_t cs;
  cublasOperation_t transW = CUBLAS_OP_T, transA = CUBLAS_OP_N;
  cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  cublasLtMatmulHeuristicResult_t candidates[8] = {};
  int returned = 0, viable[8] = {}, viableCount = 0;
  const float beta = 0.0f;

#define NVFP4_LT_CHECK(call) do { cs = (call); if (cs != CUBLAS_STATUS_SUCCESS) { \
  status = nvfp4_lt_cublas_status(cs); goto cleanup; } } while (0)

  NVFP4_LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  NVFP4_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &transW, sizeof(transW)));
  NVFP4_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &transA, sizeof(transA)));
  NVFP4_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                &scaleMode, sizeof(scaleMode)));
  NVFP4_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                &scaleMode, sizeof(scaleMode)));
  NVFP4_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                &dWScaleScratch, sizeof(dWScaleScratch)));
  NVFP4_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                &dAScaleScratch, sizeof(dAScaleScratch)));
  NVFP4_LT_CHECK(cublasLtMatrixLayoutCreate(&wLayout, CUDA_R_4F_E2M1, k, out, k));
  NVFP4_LT_CHECK(cublasLtMatrixLayoutCreate(&aLayout, CUDA_R_4F_E2M1, k, rows, k));
  NVFP4_LT_CHECK(cublasLtMatrixLayoutCreate(&cLayout, CUDA_R_32F, out, rows, out));
  NVFP4_LT_CHECK(cublasLtMatrixLayoutCreate(&yLayout, CUDA_R_32F, out, rows, out));
  NVFP4_LT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  NVFP4_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &g_nvfp4_lt_workspace_bytes, sizeof(g_nvfp4_lt_workspace_bytes)));
  NVFP4_LT_CHECK(cublasLtMatmulAlgoGetHeuristic(
      g_cublas_lt, op, wLayout, aLayout, cLayout, yLayout, preference,
      8, candidates, &returned));
  for (int i = 0; i < returned; ++i) {
    if (candidates[i].state != CUBLAS_STATUS_SUCCESS) continue;
    int algoID = -1;
    size_t written = 0;
    cs = cublasLtMatmulAlgoConfigGetAttribute(
        &candidates[i].algo, CUBLASLT_ALGO_CONFIG_ID,
        &algoID, sizeof(algoID), &written);
    if (cs != CUBLAS_STATUS_SUCCESS || written != sizeof(algoID)) continue;
    /* Go's source-pinned policy requests this only for affected SM103 library
     * windows; ordinary GEMM keeps every other returned candidate. */
    if (skip_algo_66 && algoID == 66) continue;
    viable[viableCount++] = i;
  }
  if (viableCount == 0) {
    status = FCUDA_NVFP4_LT_HEURISTIC_UNAVAILABLE;
    goto cleanup;
  }
  status = nvfp4_lt_swizzle_scales(dWLinearScale, dWScaleScratch,
                                    wScaleScratchBytes, out, scaleCols);
  if (status != FCUDA_NVFP4_LT_OK) goto cleanup;
  status = nvfp4_lt_swizzle_scales(dALinearScale, dAScaleScratch,
                                    aScaleScratchBytes, rows, scaleCols);
  if (status != FCUDA_NVFP4_LT_OK) goto cleanup;
  status = FCUDA_NVFP4_LT_RUNTIME_UNSUPPORTED;
  for (int i = 0; i < viableCount; ++i) {
    FCUDA_TS("cublaslt_matmul", dim3(out, rows), cs = cublasLtMatmul(
        g_cublas_lt, op, &alpha, dW, wLayout, dA, aLayout, &beta,
        dY, cLayout, dY, yLayout, &candidates[viable[i]].algo,
        g_blas_workspace, g_nvfp4_lt_workspace_bytes, g_stream));
    if (cs == CUBLAS_STATUS_SUCCESS) {
      ++g_nvfp4_lt_calls;
      status = FCUDA_NVFP4_LT_OK;
      break;
    }
    if (cs == CUBLAS_STATUS_NOT_SUPPORTED || cs == CUBLAS_STATUS_ARCH_MISMATCH)
      continue;
    status = nvfp4_lt_cublas_status(cs);
    break;
  }

cleanup:
  if (preference) nvfp4_lt_cleanup_status(cublasLtMatmulPreferenceDestroy(preference), "preference");
  if (yLayout) nvfp4_lt_cleanup_status(cublasLtMatrixLayoutDestroy(yLayout), "Y layout");
  if (cLayout) nvfp4_lt_cleanup_status(cublasLtMatrixLayoutDestroy(cLayout), "C layout");
  if (aLayout) nvfp4_lt_cleanup_status(cublasLtMatrixLayoutDestroy(aLayout), "A layout");
  if (wLayout) nvfp4_lt_cleanup_status(cublasLtMatrixLayoutDestroy(wLayout), "W layout");
  if (op) nvfp4_lt_cleanup_status(cublasLtMatmulDescDestroy(op), "operation descriptor");
#undef NVFP4_LT_CHECK
  return status;
#endif
}

extern "C" uint64_t fcuda_nvfp4_w4a4_lt_calls(void) {
  return g_nvfp4_lt_calls;
}
