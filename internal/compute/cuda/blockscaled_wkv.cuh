/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Descriptor flow adapted from NVIDIA/CUDALibrarySamples (Apache-2.0):
// cuBLASLt/LtMxfp8Matmul/sample_cublasLt_LtMxfp8Matmul.cu at
// 9b347f161cce5cd2a51a0406b140f2a2d81df024.
// Fak changes: exact V4.1 wkv_b TN/F32 geometry; runtime-opt-in SM120-only
// capability; existing SIMT cast plus byte-only scale repack; checked owned
// buffers, stream/capture/lifetime/error boundaries. This is not a DeepGEMM
// port or the original B300 acceptance. All physical qualification is pending.
//
// CUDA 12.9.1 cuBLAS manual sections 3.1.4.3.2 and 3.4.17 specify the 128x4
// scale tiles and Blackwell GeForce TN block-scaled FP8 with F32 C/D:
// https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html
// No new enum/type/API references escape this compile-time toolkit floor.

enum {
    BSW_UNSUPPORTED = -72001, BSW_SHAPE = -72002, BSW_BUFFER = -72003,
    BSW_CAPTURE = -72004, BSW_NONFINITE = -72005, BSW_UNCERTAIN = -72006
};

#if CUDART_VERSION >= 12090 && defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 120901
namespace fak_blockscaled_wkv {

static constexpr size_t workspace_bytes = (size_t)32 * 1024 * 1024;
static bool shape(int rows) { return rows == 128 || rows == 512 || rows == 4096; }
static int blas_error(cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS ? 0 : 200000 + (int)s; }
static int prepare_error(cublasStatus_t s) {
    return s == CUBLAS_STATUS_NOT_SUPPORTED || s == CUBLAS_STATUS_ARCH_MISMATCH
        ? BSW_UNSUPPORTED : blas_error(s);
}

// Release does not require the opt-in. Turning the experiment off must not
// strand a valid owner. Every use still excludes capture and terminal peers.
static int state() {
    if (v41_candidates_stream_uncertain()) return BSW_UNCERTAIN;
    if (!g_blas || !g_blas_workspace || !g_stream) return BSW_UNSUPPORTED;
    if (g_capture_open) return BSW_CAPTURE;
    cudaStreamCaptureStatus capture;
    cudaError_t e = cudaStreamIsCapturing(g_stream, &capture);
    if (e != cudaSuccess) return (int)e;
    if (capture != cudaStreamCaptureStatusNone) return BSW_CAPTURE;
    int device = -1;
    e = cudaGetDevice(&device);
    if (e != cudaSuccess) return (int)e;
    return device == 0 ? 0 : BSW_UNSUPPORTED;
}

static int capability(int *runtime_version, size_t *blas_version) {
    const char *opt = getenv("FAK_CUDA_BLOCKSCALED_WKV");
    if (!opt || strcmp(opt, "1")) return BSW_UNSUPPORTED;
    int rc = state();
    if (rc) return rc;
    int runtime = 0;
    cudaError_t e = cudaRuntimeGetVersion(&runtime);
    if (e != cudaSuccess) return (int)e;
    cudaDeviceProp p{};
    e = cudaGetDeviceProperties(&p, 0);
    if (e != cudaSuccess) return (int)e;
    const size_t version = cublasLtGetVersion();
    if (runtime < 12090 || version < 120901 || p.major != 12 || p.minor != 0)
        return BSW_UNSUPPORTED;
    if (runtime_version) *runtime_version = runtime;
    if (blas_version) *blas_version = version;
    return 0;
}

// A failed fence/free retains the first terminal error and ALL native live
// allocations, including prior producer-stream inputs. Existing global release
// guards observe this flag; no successful peer can rehabilitate the context.
static int drain(int rc) {
    cudaError_t e = cudaStreamSynchronize(g_stream);
    if (e != cudaSuccess) {
        if (!g_cuda_cleanup_status) g_cuda_cleanup_status = (int)e;
        return (int)e;
    }
    return rc;
}

static int extent(const void *p, size_t n, size_t alignment) {
    const uintptr_t begin = (uintptr_t)p;
    if (!p || begin % alignment || n > UINTPTR_MAX - begin) return BSW_BUFFER;
    // A successful stream fence does not retire another producer owner.
    if (v41c_pending_contains(p)) return BSW_BUFFER;
    const auto it = g_live.find((void *)p);
    if (it == g_live.end() || it->second < n) return BSW_BUFFER;
    cudaPointerAttributes a{};
    cudaError_t e = cudaPointerGetAttributes(&a, p);
    if (e != cudaSuccess) return (int)e;
    return a.type == cudaMemoryTypeDevice && a.device == 0 ? 0 : BSW_BUFFER;
}

struct Operation {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t w = nullptr, x = nullptr, y = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic{};

    int close(int rc) {
        int cleanup = 0;
        auto check = [&](cublasStatus_t s) { if (!cleanup) cleanup = blas_error(s); };
        if (pref) { check(cublasLtMatmulPreferenceDestroy(pref)); pref = nullptr; }
        if (y) { check(cublasLtMatrixLayoutDestroy(y)); y = nullptr; }
        if (x) { check(cublasLtMatrixLayoutDestroy(x)); x = nullptr; }
        if (w) { check(cublasLtMatrixLayoutDestroy(w)); w = nullptr; }
        if (op) { check(cublasLtMatmulDescDestroy(op)); op = nullptr; }
        return cleanup ? cleanup : rc;
    }

    int prepare(int rows, const uint8_t *ws, const uint8_t *xs) {
        if (!shape(rows)) return BSW_SHAPE;
#define BSW_PREP(call) do { const cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) return close(prepare_error(s)); } while (0)
        if (!g_cublas_lt) {
            cublasLtHandle_t created = nullptr;
            BSW_PREP(cublasLtCreate(&created));
            g_cublas_lt = created;
        }
        const cublasOperation_t transw = CUBLAS_OP_T, transx = CUBLAS_OP_N;
        const cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        BSW_PREP(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        BSW_PREP(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transw, sizeof(transw)));
        BSW_PREP(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transx, sizeof(transx)));
        BSW_PREP(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
        BSW_PREP(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
        BSW_PREP(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ws, sizeof(ws)));
        BSW_PREP(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &xs, sizeof(xs)));
        // Column-major op(W[K,512])^T * X[K,P] = Y[512,P] is row-major
        // Y[P,512] = X[P,K] * W[512,K]^T without transposing any payload.
        BSW_PREP(cublasLtMatrixLayoutCreate(&w, CUDA_R_8F_E4M3, 5120, 512, 5120));
        BSW_PREP(cublasLtMatrixLayoutCreate(&x, CUDA_R_8F_E4M3, 5120, rows, 5120));
        BSW_PREP(cublasLtMatrixLayoutCreate(&y, CUDA_R_32F, 512, rows, 512));
        BSW_PREP(cublasLtMatmulPreferenceCreate(&pref));
        BSW_PREP(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &workspace_bytes, sizeof(workspace_bytes)));
        int found = 0;
        BSW_PREP(cublasLtMatmulAlgoGetHeuristic(g_cublas_lt, op, w, x, y, y, pref, 1, &heuristic, &found));
#undef BSW_PREP
        if (found != 1 || heuristic.state != CUBLAS_STATUS_SUCCESS || heuristic.workspaceSize > workspace_bytes)
            return close(BSW_UNSUPPORTED);
        return 0;
    }
};

// recipe32 byte (row,kb) lives at 4*((kb/4)*P+row)+kb%4.
// Vendor byte offset is ((row/128)*40+kb/4)*512 +
// (row%32)*16 + ((row%128)/32)*4 + kb%4. Copying each aligned word
// preserves all four E8M0 bytes exactly and gives every output word one writer.
// All admitted P and 160 scale columns exactly tile 128x4; no padding is read.
static __global__ void repack(const uint32_t *recipe, uint32_t *tiled, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 40) return;
    const int row = i / 40, chunk = i % 40;
    const int dst = ((row / 128) * 40 + chunk) * 128 +
                    (row % 32) * 4 + (row % 128) / 32;
    tiled[dst] = recipe[chunk * rows + row];
}

static __global__ void check_output(const float *y, uint32_t *status, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !isfinite(y[i])) atomicOr(status, 2u);
}

static int read_status(const uint32_t *status) {
    uint32_t host = 0;
    fcuda_ts_untimed_transfer();
    cudaError_t e = cudaMemcpy(&host, status, sizeof(host), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return drain((int)e);
    g_host_bytes += sizeof(host);
    return host ? BSW_NONFINITE : 0;
}
} // namespace fak_blockscaled_wkv

extern "C" int fcuda_blockscaled_wkv_support(int rows, int *runtime_version, size_t *blas_version) {
    using namespace fak_blockscaled_wkv;
    if (!shape(rows)) return BSW_SHAPE;
    int rc = capability(runtime_version, blas_version);
    if (rc) return rc;
    Operation op;
    // Heuristic query only: these aligned dummy scale pointers are never read
    // by a kernel. Run repeats selection with its actual owned scale buffers.
    rc = op.prepare(rows, (const uint8_t *)g_blas_workspace, (const uint8_t *)g_blas_workspace);
    return op.close(rc);
}

extern "C" int fcuda_blockscaled_wkv_upload(void *dst, const void *src, size_t bytes) {
    using namespace fak_blockscaled_wkv;
    int rc = capability(nullptr, nullptr);
    if (rc) return rc;
    if (!src || (bytes != 512 * 5120 && bytes != 512 * 160)) return BSW_BUFFER;
    rc = extent(dst, bytes, 16);
    if (rc) return rc;
    fcuda_ts_untimed_transfer();
    cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) g_h2d_bytes += bytes;
    return drain((int)e);
}

extern "C" int fcuda_blockscaled_wkv_release(void *ptr, size_t bytes) {
    using namespace fak_blockscaled_wkv;
    if (!ptr) return 0;
    int rc = state();
    if (rc) return rc;
    rc = extent(ptr, bytes, 4);
    if (rc) return rc;
    auto it = g_live.find(ptr);
    if (it->second != bytes) return BSW_BUFFER;
    rc = drain(0);
    if (rc) return rc;
    cudaError_t e = cudaFree(ptr);
    if (e != cudaSuccess) {
        if (!g_cuda_cleanup_status) g_cuda_cleanup_status = (int)e;
        return (int)e;
    }
    fak_alloc_untrack(ptr);
    g_live.erase(it);
    return 0;
}

extern "C" int fcuda_blockscaled_wkv_run(const float *x, const uint8_t *w, const uint8_t *ws,
    uint8_t *a, uint32_t *recipe, uint8_t *tiled, uint32_t *status, float *y, int rows) {
    using namespace fak_blockscaled_wkv;
    if (!shape(rows)) return BSW_SHAPE;
    int rc = capability(nullptr, nullptr);
    if (rc) return rc;
    const void *ptr[8] = {x, w, ws, a, recipe, tiled, status, y};
    const size_t sizes[8] = {(size_t)rows * 5120 * 4, 512 * 5120, 512 * 160,
        (size_t)rows * 5120, (size_t)rows * 160, (size_t)rows * 160, 4, (size_t)rows * 512 * 4};
    for (int i = 0; i < 8; ++i) {
        rc = extent(ptr[i], sizes[i], i == 6 ? 4 : 16);
        if (rc) return rc;
        for (int j = 0; j < i; ++j)
            if ((uintptr_t)ptr[i] < (uintptr_t)ptr[j] + sizes[j] &&
                (uintptr_t)ptr[j] < (uintptr_t)ptr[i] + sizes[i]) return BSW_BUFFER;
    }
    Operation op;
    rc = op.prepare(rows, ws, tiled);
    if (rc) return rc; // All capability/layout/heuristic checks precede mutation.
    auto finish = [&](int code) { return op.close(drain(code)); };
    cudaError_t e;
    FCUDA_TS("fill", dim3(0, 0), e = cudaMemsetAsync(status, 0, 4, g_stream));
    if (e != cudaSuccess) return finish((int)e);
    // Independent SM120 whole-operation contract. Reuse only the architecture-
    // neutral SIMT kernel, NOT the intentionally SM100/103-only exported API.
    FCUDA_TS("k_mxfp8_cast_f32", rows * 40,
        k_mxfp8_cast_f32<<<rows * 40, 128, 0, g_stream>>>(x, a, recipe, status, rows, 5120, rows));
    rc = drain((int)cudaGetLastError());
    if (rc) return op.close(rc);
    rc = read_status(status);
    if (rc) return op.close(rc);
    FCUDA_TS("repack", (rows * 40 + 255) / 256,
        repack<<<(rows * 40 + 255) / 256, 256, 0, g_stream>>>(recipe, (uint32_t *)tiled, rows));
    e = cudaGetLastError();
    if (e != cudaSuccess) return finish((int)e);
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t cs;
    FCUDA_TS("cublaslt_matmul", dim3(512, rows), cs = cublasLtMatmul(
        g_cublas_lt, op.op, &alpha, w, op.w, a, op.x, &beta, y, op.y,
        y, op.y, &op.heuristic.algo, g_blas_workspace, workspace_bytes, g_stream));
    // Even NOT_SUPPORTED here is an execution error: casting already happened.
    if (cs != CUBLAS_STATUS_SUCCESS) return finish(blas_error(cs));
    FCUDA_TS("check_output", (rows * 512 + 255) / 256,
        check_output<<<(rows * 512 + 255) / 256, 256, 0, g_stream>>>(y, status, rows * 512));
    rc = drain((int)cudaGetLastError());
    if (!rc) rc = read_status(status);
    return op.close(rc);
}
#else
// Keep ordinary cuda && cgo builds linkable with older toolkits. These stubs
// reference no FP8 scale enum, new library type, or SM120 device instruction.
extern "C" int fcuda_blockscaled_wkv_support(int, int *, size_t *) { return BSW_UNSUPPORTED; }
extern "C" int fcuda_blockscaled_wkv_upload(void *, const void *, size_t) { return BSW_UNSUPPORTED; }
extern "C" int fcuda_blockscaled_wkv_release(void *, size_t) { return BSW_UNSUPPORTED; }
extern "C" int fcuda_blockscaled_wkv_run(const float *, const uint8_t *, const uint8_t *,
    uint8_t *, uint32_t *, uint8_t *, uint32_t *, float *, int) { return BSW_UNSUPPORTED; }
#endif
