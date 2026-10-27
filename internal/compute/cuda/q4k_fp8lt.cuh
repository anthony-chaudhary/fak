/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Q4_K prompt-panel FP8 GEMM descriptor flow adapted from NVIDIA/CUDALibrarySamples
// cuBLASLt/LtFp8Matmul/sample_cublasLt_LtFp8Matmul.cu at
// 9b347f161cce5cd2a51a0406b140f2a2d81df024 (Apache-2.0).
// Fak changes: TN geometry, E4M3 inputs with per-tensor device scales, F32 output,
// stream/pool/workspace ownership, bounded heuristic selection, and checked cleanup.
// The Q4_K dequant and scale/convert kernels below are Fak's adapter, not an upstream GEMM.
//
// Experimental sm_89/sm_90 arm only. Both operands are rounded to E4M3: this is NOT
// equivalent to the compensated Q4_K route. Keep the default compensated until the
// unchanged strict numerical and same-card full-model acceptance gates pass.

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 120800
#include <cuda_fp8.h>
#include <algorithm>

namespace fak_q4k_fp8lt {

static __device__ __forceinline__ float weight_value(const uint8_t *w, size_t i) {
    const uint8_t *b = w + (i / 256) * 144;
    const int k = (int)(i % 256), group = k / 32;
    unsigned char scale, minimum;
    getScaleMinK4_dev(group, b + 4, &scale, &minimum);
    const unsigned char packed = b[16 + (group / 2) * 32 + k % 32];
    const int code = group & 1 ? packed >> 4 : packed & 15;
    return __half2float(*(const __half *)b) * (float)scale * (float)code -
           __half2float(*(const __half *)(b + 2)) * (float)minimum;
}

// Bounded grid-stride reduction: one atomic per CTA, never one per weight.
// Nonfinite input makes the scale NaN rather than silently saturating it to E4M3.
template <bool quant_weight>
static __global__ void amax(const void *src, size_t n, float *maximum) {
    float local = 0.0f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (size_t)blockDim.x * gridDim.x) {
        const float v = quant_weight ? weight_value((const uint8_t *)src, i) : ((const float *)src)[i];
        local = fmaxf(local, isfinite(v) ? fabsf(v) : INFINITY);
    }
    for (int d = 16; d; d >>= 1)
        local = fmaxf(local, __shfl_down_sync(0xffffffff, local, d));
    __shared__ float partial[8];
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = local;
    __syncthreads();
    if (threadIdx.x < 32) {
        local = threadIdx.x < 8 ? partial[threadIdx.x] : 0.0f;
        for (int d = 16; d; d >>= 1)
            local = fmaxf(local, __shfl_down_sync(0xffffffff, local, d));
        if (threadIdx.x == 0) atomicMax((unsigned int *)maximum, __float_as_uint(local));
    }
}

static __global__ void finish_scales(float *scales) {
    const int i = threadIdx.x;
    if (i >= 2) return;
    const float a = scales[i];
    // Avoid zero/subnormal divisors. The all-zero tensor has scale 1 and codes 0.
    scales[i] = !isfinite(a) ? NAN : a == 0.0f ? 1.0f : fmaxf(a / 448.0f, FLT_MIN);
}

template <bool quant_weight>
static __global__ void convert(const void *src, __nv_fp8_e4m3 *dst,
                               size_t logical, size_t padded, const float *scale) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < padded; i += (size_t)blockDim.x * gridDim.x) {
        const float v = i >= logical ? 0.0f :
            quant_weight ? weight_value((const uint8_t *)src, i) : ((const float *)src)[i];
        dst[i] = __nv_fp8_e4m3(v / *scale);
    }
}

static unsigned int grid(size_t n) {
    return (unsigned int)std::min((size_t)4096, (n + 255) / 256);
}

static void cleanup_status(cublasStatus_t status, const char *what) {
    if (status != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "fak-cuda: fp8lt cleanup %s failed: %d\n", what, (int)status);
}

} // namespace fak_q4k_fp8lt
#endif

#define FCUDA_Q4K_FP8LT_REFUSED 80005
static unsigned long long g_q4k_fp8lt_launches = 0;
extern "C" unsigned long long fcuda_q4k_fp8lt_launch_count(void) {
    return g_q4k_fp8lt_launches;
}

static int fcuda_q4k_fp8lt_matmul(const uint8_t *w, const float *x, float *y,
                                 int out, int in, int rows) {
#if !defined(CUBLAS_VERSION) || CUBLAS_VERSION < 120800
    (void)w; (void)x; (void)y; (void)out; (void)in; (void)rows;
    return FCUDA_Q4K_FP8LT_REFUSED;
#else
    using namespace fak_q4k_fp8lt;
    // Capture is excluded: first-use Lt descriptors/heuristics must not allocate
    // inside a decode graph. The fp8lt arm is only for uncached large panels.
    if (!w || !x || !y || out <= 0 || in <= 0 || rows < 128 ||
        in % 256 != 0 || out % 16 != 0 || rows > INT_MAX - 15 ||
        (((uintptr_t)w | (uintptr_t)x | (uintptr_t)y) & 15u) != 0 ||
        !g_blas_workspace || g_capture_open)
        return FCUDA_Q4K_FP8LT_REFUSED;
    cudaStreamCaptureStatus capture;
    cudaError_t ce = cudaStreamIsCapturing(g_stream, &capture);
    if (ce != cudaSuccess) return 81000 + (int)ce;
    if (capture != cudaStreamCaptureStatusNone) return FCUDA_Q4K_FP8LT_REFUSED;
    int device = 0, major = 0, minor = 0;
    if ((ce = cudaGetDevice(&device)) != cudaSuccess ||
        (ce = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device)) != cudaSuccess ||
        (ce = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device)) != cudaSuccess)
        return 81000 + (int)ce;
    const int sm = major * 10 + minor;
    if (sm != 89 && sm != 90) return FCUDA_Q4K_FP8LT_REFUSED;

    // Pad only prompt rows; trimming the contiguous F32 output preserves Y[P,out].
    // Weight rows stay unpadded and must meet the explicit 16-element admission.
    const int padded_rows = (rows + 15) / 16 * 16;
    const size_t wn = (size_t)out * in, xn = (size_t)rows * in;
    const size_t xpad = (size_t)padded_rows * in, ypad = (size_t)padded_rows * out;
    if (wn > INT_MAX || xpad > INT_MAX || ypad > INT_MAX) return FCUDA_Q4K_FP8LT_REFUSED;
    const size_t trim_bytes = padded_rows == rows ? 0 : ypad * sizeof(float);
    const size_t bytes = wn + xpad + 16 + trim_bytes;
    void *storage = scratch_malloc(bytes);
    if (!storage) return FCUDA_SCRATCH_ALLOC_FAILED;
    __nv_fp8_e4m3 *w8 = (__nv_fp8_e4m3 *)storage;
    __nv_fp8_e4m3 *x8 = w8 + wn;
    float *scales = (float *)(x8 + xpad);
    float *result = trim_bytes ? scales + 4 : y;
    const float *wscale = scales, *xscale = scales + 1;
    const size_t workspace_bytes = (size_t)32 * 1024 * 1024;
    const float alpha = 1.0f, beta = 0.0f;
    const cublasOperation_t transw = CUBLAS_OP_T, transx = CUBLAS_OP_N;
    const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t wd = nullptr, xd = nullptr, yd = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic = {};
    int found = 0, status = 0;
    cublasStatus_t cs = CUBLAS_STATUS_SUCCESS;

#define Q4K_FP8LT_CALL(call) do { cs = (call); if (cs != CUBLAS_STATUS_SUCCESS) { \
    status = (cs == CUBLAS_STATUS_NOT_SUPPORTED || cs == CUBLAS_STATUS_ARCH_MISMATCH) \
        ? FCUDA_Q4K_FP8LT_REFUSED : 82000 + (int)cs; goto cleanup; } } while (0)
#define Q4K_FP8LT_CUDA(call) do { ce = (call); if (ce != cudaSuccess) { \
    status = 81000 + (int)ce; goto cleanup; } } while (0)

    if (!g_cublas_lt) Q4K_FP8LT_CALL(cublasLtCreate(&g_cublas_lt));
    Q4K_FP8LT_CALL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    Q4K_FP8LT_CALL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transw, sizeof(transw)));
    Q4K_FP8LT_CALL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transx, sizeof(transx)));
    Q4K_FP8LT_CALL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    Q4K_FP8LT_CALL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    Q4K_FP8LT_CALL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wscale, sizeof(wscale)));
    Q4K_FP8LT_CALL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &xscale, sizeof(xscale)));
    Q4K_FP8LT_CALL(cublasLtMatrixLayoutCreate(&wd, CUDA_R_8F_E4M3, in, out, in));
    Q4K_FP8LT_CALL(cublasLtMatrixLayoutCreate(&xd, CUDA_R_8F_E4M3, in, padded_rows, in));
    Q4K_FP8LT_CALL(cublasLtMatrixLayoutCreate(&yd, CUDA_R_32F, out, padded_rows, out));
    Q4K_FP8LT_CALL(cublasLtMatmulPreferenceCreate(&pref));
    Q4K_FP8LT_CALL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &workspace_bytes, sizeof(workspace_bytes)));
    // One vendor heuristic, no timed autotuning and no silent alternative arm.
    Q4K_FP8LT_CALL(cublasLtMatmulAlgoGetHeuristic(g_cublas_lt, op, wd, xd, yd, yd, pref, 1, &heuristic, &found));
    if (found != 1 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
        status = FCUDA_Q4K_FP8LT_REFUSED;
        goto cleanup;
    }
    Q4K_FP8LT_CUDA(cudaMemsetAsync(scales, 0, 2 * sizeof(float), g_stream));
    FCUDA_TS("q4k_fp8lt_weight_amax", grid(wn), amax<true><<<grid(wn), 256, 0, g_stream>>>(w, wn, scales));
    Q4K_FP8LT_CUDA(cudaPeekAtLastError());
    FCUDA_TS("q4k_fp8lt_activation_amax", grid(xn), amax<false><<<grid(xn), 256, 0, g_stream>>>(x, xn, scales + 1));
    Q4K_FP8LT_CUDA(cudaPeekAtLastError());
    FCUDA_TS("q4k_fp8lt_scales", 1, finish_scales<<<1, 32, 0, g_stream>>>(scales));
    Q4K_FP8LT_CUDA(cudaPeekAtLastError());
    FCUDA_TS("q4k_fp8lt_dequant", grid(wn), convert<true><<<grid(wn), 256, 0, g_stream>>>(w, w8, wn, wn, scales));
    Q4K_FP8LT_CUDA(cudaPeekAtLastError());
    FCUDA_TS("q4k_fp8lt_quantize", grid(xpad), convert<false><<<grid(xpad), 256, 0, g_stream>>>(x, x8, xn, xpad, scales + 1));
    Q4K_FP8LT_CUDA(cudaPeekAtLastError());
    FCUDA_TS("q4k_fp8lt_matmul", dim3(out, padded_rows), cs = cublasLtMatmul(
        g_cublas_lt, op, &alpha, w8, wd, x8, xd, &beta, result, yd, result, yd,
        &heuristic.algo, g_blas_workspace, workspace_bytes, g_stream));
    Q4K_FP8LT_CALL(cs);
    Q4K_FP8LT_CUDA(cudaPeekAtLastError());
    if (trim_bytes)
        Q4K_FP8LT_CUDA(cudaMemcpyAsync(y, result, (size_t)rows * out * sizeof(float), cudaMemcpyDeviceToDevice, g_stream));
    ++g_q4k_fp8lt_launches; // Enqueue witness only; callers must synchronize/read for numerical evidence.

cleanup:
    if (pref) cleanup_status(cublasLtMatmulPreferenceDestroy(pref), "preference");
    if (yd) cleanup_status(cublasLtMatrixLayoutDestroy(yd), "Y layout");
    if (xd) cleanup_status(cublasLtMatrixLayoutDestroy(xd), "X layout");
    if (wd) cleanup_status(cublasLtMatrixLayoutDestroy(wd), "W layout");
    if (op) cleanup_status(cublasLtMatmulDescDestroy(op), "operation");
    // The allocator recycles on this same serialized stream; it does not cudaFree
    // an in-flight temporary. Later users remain ordered after this operation.
    fcuda_free(storage);
#undef Q4K_FP8LT_CUDA
#undef Q4K_FP8LT_CALL
    return status;
#endif
}

