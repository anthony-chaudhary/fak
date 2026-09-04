// cuda_kernels.cu — the CUDA C++ hardware seam behind the typed compute.Backend.
//
// Compiled offline by nvcc into a static lib (libfakcuda.a) that the cgo wrapper
// (cuda.go, //go:build cuda) links. Every op here is f32: this first device backend is
// an *Approx* peer of the cpuref *Reference* — held to the argmax-exact + logit-cosine
// gate, NOT to max|Δ|=0. cuBLAS SGEMM (a different reduction order than the model's fdot
// tree) is what makes that distinction real and honest. Quantized weight dtypes are not
// implemented on device yet; the Go MatMul refuses them with a clear message so the
// boundary stays explicit (see the deferred-work GitHub issues).
//
// DIRECTION.md: this is a sanctioned hardware seam in a statically-typed compiled
// language, OFF the request path, behind a re-validated typed boundary (the flat C ABI
// in cuda_backend.h). The default `go build` excludes it; only `-tags cuda` links it.

#include "cuda_backend.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>  /* __half + __float2half — the fp16 compute path (#484) */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unordered_map>
#include <vector>

static cublasHandle_t g_blas = nullptr;
static void *g_blas_workspace = nullptr; // explicit cuBLAS workspace so it never cudaMalloc's mid-capture (#969)
static bool g_capture_open = false;      // true between fcuda_graph_begin and end/abort: arms the #969 dalloc guard
// All device work runs on g_stream (a blocking stream, so a synchronous cudaMemcpy on the
// legacy default stream still fences it). One stream is what makes the whole per-token op
// sequence CAPTURABLE into a CUDA graph — the only way to collapse ~600 WSL CUDA calls/
// token (proven floor ~12 tok/s) down to one graph launch.
static cudaStream_t g_stream = 0;

#define CK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
  fprintf(stderr, "fak-cuda: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); } } while (0)

// Caching device allocator. cudaMalloc/cudaFree are slow (~0.5 ms each on WSL and they
// implicitly serialize the device), and the forward loop allocates ~hundreds of small
// output buffers per token. A size-bucketed free list recycles freed buffers so a steady-
// state decode pays ~zero allocation cost after warm-up — the same arena trick llama.cpp's
// graph allocator uses, kept minimal. Single-threaded by the Go-side cudaMu mutex.
static std::unordered_map<size_t, std::vector<void *>> g_pool; // free buffers, by exact byte size
static std::unordered_map<void *, size_t> g_live;              // live ptr -> its byte size
static std::unordered_map<void *, size_t> g_managed_live;      // live cudaMallocManaged ptr -> byte size

// Host-transfer witnesses (#482/#4738): cumulative bytes in each direction. Every successful
// h2d/d2h copy adds to its own counter so a whole-operation test can prove zero traffic in both
// directions after initial upload. Monotonic; tests reset them around each measured step.
static size_t g_host_bytes = 0;
static size_t g_h2d_bytes = 0;

// Whole-operation Qwen3.5/3.6 GDN witness (#4725/#4738). This counts confirmed
// completed operations, not enqueues; the Go fixture reads it together with both
// transfer counters to prove one real recurrent operation ran without host staging.
static size_t g_qwen35_gdn_operations = 0;
// One-shot deterministic failure injection used only through the unexported Go test helper.
// Stages 2..6 model a launch check failure; stage 7 models final async execution failure.
static int g_qwen35_gdn_test_fault_stage = 0;

extern "C" int fcuda_init(char *name, int namelen, int *sm, size_t *total_mem) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 1;
  if (cudaSetDevice(0) != cudaSuccess) return 2;
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 3;
  if (name && namelen > 0) { strncpy(name, p.name, namelen - 1); name[namelen - 1] = 0; }
  if (sm) *sm = p.major * 10 + p.minor;
  if (total_mem) *total_mem = p.totalGlobalMem;
  if (!g_blas) { if (cublasCreate(&g_blas) != CUBLAS_STATUS_SUCCESS) return 4; }
  cublasSetPointerMode(g_blas, CUBLAS_POINTER_MODE_HOST);
  if (!g_stream) CK(cudaStreamCreate(&g_stream));
  cublasSetStream(g_blas, g_stream);
  // Pre-allocate an explicit cuBLAS workspace (#969). With NO user workspace, cuBLAS lazily
  // cudaMalloc's its own internal workspace the first time it sees a given GEMM problem — and
  // CUDA forbids cudaMalloc while g_stream is mid stream-capture. Because the decode GEMMs are
  // captured into the CUDA-graph (cuda_graph_test.go / the HAL fast path), that lazy alloc fired
  // INSIDE capture on a fresh A100 (sm_80, CUDA 12.8/13.0) and crashed graph replay with
  // "cudaMalloc(256 bytes) ... operation not permitted when stream is capturing". Handing cuBLAS
  // a user-owned workspace makes it use that buffer exclusively and never allocate during capture.
  // 32 MiB is NVIDIA's recommended size for Hopper and is safe (a superset) on Ampere/Ada too.
  if (!g_blas_workspace) {
    const size_t ws = (size_t)32 * 1024 * 1024;
    if (cudaMalloc(&g_blas_workspace, ws) == cudaSuccess)
      cublasSetWorkspace(g_blas, g_blas_workspace, ws);
  }
  return 0;
}

// fcuda_set_tf32 toggles the cuBLAS math mode for the f32 SGEMM path (Lever 4 of the
// H100-KERNEL-5X-ROADMAP). ON => CUBLAS_TF32_TENSOR_OP_MATH: the f32 GEMMs (the prefill
// projections — fcuda_matmul_f32 / cublasSgemm above) run on Hopper/Ampere TENSOR CORES at
// TF32 input precision (10-bit mantissa) with F32 accumulation, instead of the FP32 CUDA
// cores. That is a large prefill speedup (the compute-bound phase) at a small, DISCLOSED
// precision cost — TF32 keeps the f32 exponent, so the dynamic range is unchanged and only
// the mantissa narrows. OFF (default) => CUBLAS_DEFAULT_MATH: the pedantic FP32-core path the
// recorded device-vs-cpuref cosine floors were witnessed against, so an existing run's numerics
// never change unless TF32 is explicitly requested. The F16 HGEMM peers (fcuda_matmul_f16) are
// unaffected — they already select a tensor-op algorithm explicitly via cublasGemmEx. Idempotent
// and safe before fcuda_init (no handle yet => nothing to set; the Go init applies it after
// fcuda_init creates g_blas).
extern "C" void fcuda_set_tf32(int on) {
  if (!g_blas) return;
  cublasSetMathMode(g_blas, on ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
}

extern "C" int fcuda_mem_info(size_t *free_mem, size_t *total_mem) {
  size_t free_b = 0;
  size_t total_b = 0;
  cudaError_t e = cudaMemGetInfo(&free_b, &total_b);
  if (e != cudaSuccess) return (int)e;
  if (free_mem) *free_mem = free_b;
  if (total_mem) *total_mem = total_b;
  return 0;
}

extern "C" void *fcuda_malloc(size_t bytes) {
  if (bytes == 0) bytes = 1;
  auto it = g_pool.find(bytes);
  void *d = nullptr;
  if (it != g_pool.end() && !it->second.empty()) {
    d = it->second.back();
    it->second.pop_back();
  } else {
    if (g_capture_open) {
      // #969: a device alloc on the captured decode path is the exact bug this guard exists to
      // kill — CUDA forbids cudaMalloc between BeginCapture and EndCapture, so it would crash the
      // whole serve with the opaque "operation not permitted while the stream is capturing". Every
      // graphed-decode scratch (the cuBLAS workspace, the g_q8_* activation-quant buffers, the
      // argmax index) MUST be pre-warmed before fcuda_graph_begin. Fail LOUD naming the size so the
      // offending un-pre-warmed allocation is pinpointed instead of surfacing as a generic capture
      // error. Returning nullptr routes the Go side to its typed DeviceAllocError panic.
      fprintf(stderr, "fak-cuda: BUG #969: fcuda_malloc(%zu) on a pool miss while a CUDA graph capture is OPEN — pre-warm this scratch before BeginCapture\n", bytes);
      return nullptr;
    }
    cudaError_t _e = cudaMalloc(&d, bytes);
    if (_e != cudaSuccess) {
      // Report the TRUE reason instead of letting CK swallow it and the Go caller panic with no
      // cause: an out-of-memory says "out of memory", a context poisoned by a prior asynchronous
      // kernel/launch fault says e.g. "an illegal memory access was encountered". Returning nullptr
      // keeps a genuine OOM loud (dalloc still panics) — this EXPOSES the error, it does not mask it.
      fprintf(stderr, "fak-cuda: cudaMalloc(%zu bytes) failed: %s\n", bytes, cudaGetErrorString(_e));
      return nullptr;
    }
  }
  g_live[d] = bytes;
  return d;
}
extern "C" void *fcuda_malloc_managed(size_t bytes) {
  if (bytes == 0) bytes = 1;
  void *d = nullptr;
  cudaError_t _e = cudaMallocManaged(&d, bytes, cudaMemAttachGlobal);
  if (_e != cudaSuccess) {
    fprintf(stderr, "fak-cuda: cudaMallocManaged(%zu bytes) failed: %s\n", bytes, cudaGetErrorString(_e));
    return nullptr;
  }
  g_managed_live[d] = bytes;
  return d;
}
static void q4_coeff_cache_evict(const void *q);

extern "C" void fcuda_free(void *d) {
  if (!d) return;
  q4_coeff_cache_evict(d);
  auto mit = g_managed_live.find(d);
  if (mit != g_managed_live.end()) {
    g_managed_live.erase(mit);
    cudaFree(d);
    return;
  }
  auto it = g_live.find(d);
  if (it != g_live.end()) {
    g_pool[it->second].push_back(d); // return to the pool for reuse, don't cudaFree
    g_live.erase(it);
  } else {
    cudaFree(d);
  }
}

extern "C" size_t fcuda_live_allocations(void) {
  return g_live.size() + g_managed_live.size();
}

// fcuda_graph_prewarm deepens every pool bucket by `extra` spare buffers (#969). The warm
// forward before capture pools ONE set of each transient size, but a single captured decodeChain
// holds several same-size transients live AT ONCE (e.g. the per-layer RMSNorm outputs), so the
// pool can drain mid-forward and the next same-size devTr misses -> cudaMalloc, which is illegal
// while g_stream is capturing. Called once right before fcuda_graph_begin (outside capture), this
// guarantees headroom so the captured forward is served entirely from the free list. Idempotent
// and cheap: it only tops each bucket up to `extra` spares (a few hundred bytes each).
extern "C" void fcuda_graph_prewarm(int extra) {
  if (extra <= 0) return;
  // Snapshot the sizes first: pushing into g_pool while iterating it would invalidate iterators.
  std::vector<size_t> sizes;
  sizes.reserve(g_pool.size());
  for (auto &kv : g_pool) sizes.push_back(kv.first);
  for (size_t bytes : sizes) {
    auto &bucket = g_pool[bytes];
    while ((int)bucket.size() < extra) {
      void *d = nullptr;
      if (cudaMalloc(&d, bytes) != cudaSuccess || !d) break; // best-effort; a real OOM still fails loud later
      bucket.push_back(d);
    }
  }
}

extern "C" void fcuda_trim_pool_large(size_t max_keep_bytes) {
  for (auto it = g_pool.begin(); it != g_pool.end(); ) {
    if (it->first > max_keep_bytes) {
      for (void *p : it->second) {
        cudaFree(p);
      }
      it = g_pool.erase(it);
    } else {
      ++it;
    }
  }
}
extern "C" void fcuda_h2d(void *d, const void *h, size_t n) {
  cudaError_t e = cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
  if (e == cudaSuccess) g_h2d_bytes += n;
  else fprintf(stderr, "fak-cuda: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
}
extern "C" void fcuda_d2h(void *h, const void *d, size_t n) {
  cudaError_t e = cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost);
  if (e == cudaSuccess) g_host_bytes += n;
  else fprintf(stderr, "fak-cuda: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
}
// Device-to-device copies stay on the default stream but are ASYNC w.r.t. the host: a
// synchronous cudaMemcpy fences the whole device, and RoPE + every KV append issues one,
// so a 30-layer decode paid ~150 full device syncs per token (catastrophic on WSL, where a
// sync is ~1-2 ms). Stream ordering still serializes them against the kernels correctly;
// the only host fence we keep is the final logits d2h in Read.
extern "C" void fcuda_d2d(void *dst, const void *src, size_t n) { CK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, g_stream)); }
extern "C" void fcuda_sync(void) { CK(cudaDeviceSynchronize()); }

static cudaEvent_t g_trace_start;
extern "C" double fcuda_event_elapsed_ms_start(void) {
    CK(cudaEventCreate(&g_trace_start));
    CK(cudaEventRecord(g_trace_start, g_stream));
    return 0.0;
}
extern "C" double fcuda_event_elapsed_ms_end(void) {
    cudaEvent_t end;
    CK(cudaEventCreate(&end));
    CK(cudaEventRecord(end, g_stream));
    CK(cudaEventSynchronize(end));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, g_trace_start, end));
    CK(cudaEventDestroy(end));
    CK(cudaEventDestroy(g_trace_start));
    return (double)ms;
}

extern "C" int fcuda_set_device(int device) {
  cudaError_t e = cudaSetDevice(device);
  if (e != cudaSuccess) {
    fprintf(stderr, "fak-cuda: cudaSetDevice(%d) failed: %s\n", device, cudaGetErrorString(e));
    return (int)e + 1000;
  }
  return 0;
}

extern "C" void *fcuda_malloc_on(int device, size_t bytes) {
  if (bytes == 0) bytes = 1;
  if (fcuda_set_device(device) != 0) return nullptr;
  void *d = nullptr;
  cudaError_t e = cudaMalloc(&d, bytes);
  if (e != cudaSuccess) {
    fprintf(stderr, "fak-cuda: cudaMalloc(%zu bytes) on device %d failed: %s\n",
            bytes, device, cudaGetErrorString(e));
    return nullptr;
  }
  return d;
}

extern "C" void fcuda_free_on(int device, void *d) {
  if (!d) return;
  if (fcuda_set_device(device) != 0) return;
  q4_coeff_cache_evict(d);
  CK(cudaFree(d));
}

extern "C" void fcuda_h2d_on(int device, void *d, const void *h, size_t n) {
  if (fcuda_set_device(device) != 0) return;
  cudaError_t e = cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
  if (e == cudaSuccess) g_h2d_bytes += n;
  else fprintf(stderr, "fak-cuda: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
}

extern "C" void fcuda_d2h_on(int device, void *h, const void *d, size_t n) {
  if (fcuda_set_device(device) != 0) return;
  cudaError_t e = cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost);
  if (e == cudaSuccess) g_host_bytes += n;
  else fprintf(stderr, "fak-cuda: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
}

extern "C" void fcuda_d2d_on(int device, void *dst, const void *src, size_t n) {
  if (fcuda_set_device(device) != 0) return;
  CK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, g_stream));
}

// host-transfer witness accessors (#482/#4738): see g_host_bytes/g_h2d_bytes above.
extern "C" size_t fcuda_hostxfer_bytes(void) { return g_host_bytes; }
extern "C" void fcuda_hostxfer_reset(void) { g_host_bytes = 0; }
extern "C" size_t fcuda_h2dxfer_bytes(void) { return g_h2d_bytes; }
extern "C" void fcuda_h2dxfer_reset(void) { g_h2d_bytes = 0; }

// Private whole-sequence bindings. They deliberately do not extend cuda_backend.h:
// the public flat ABI remains the stable primitive surface, while cuda_kernels.go
// owns these cgo-local entry points and the complete resident-sequence contract.
__global__ void k_qwen35_embedding_gather(const float *embedding, const int *ids,
                                           float *out, int tokens, int hidden) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)tokens * hidden;
  if (i >= n) return;
  int token = (int)(i / hidden);
  int col = (int)(i - (size_t)token * hidden);
  out[i] = embedding[(size_t)ids[token] * hidden + col];
}

extern "C" int fak_qwen35_embedding_gather_f32(
    const float *dEmbedding, const int *hIDs, int *dIDs, float *dOut,
    int tokens, int hidden) {
  if (!dEmbedding || !hIDs || !dIDs || !dOut || tokens <= 0 || hidden <= 0) return -1;
  cudaGetLastError();
  size_t idsBytes = (size_t)tokens * sizeof(int);
  cudaError_t copy = cudaMemcpy(dIDs, hIDs, idsBytes, cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) return 10000 + (int)copy;
  g_h2d_bytes += idsBytes;
  size_t n = (size_t)tokens * hidden;
  k_qwen35_embedding_gather<<<(n + 255) / 256, 256, 0, g_stream>>>(
      dEmbedding, dIDs, dOut, tokens, hidden);
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 20000 + (int)launch;
}

extern "C" int fak_qwen35_pointer_is_device(const void *pointer) {
  if (!pointer) return 0;
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return attributes.type == cudaMemoryTypeDevice ? 1 : 0;
}

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

static void fp16_compensated_matmul(const float *W, const float *X, float *Y,
                                    int out, int in, int tokens) {
  size_t wn = (size_t)out * in, xn = (size_t)tokens * in;
  __half *wHi = (__half *)fcuda_malloc(wn * sizeof(__half));
  __half *wLo = (__half *)fcuda_malloc(wn * sizeof(__half));
  __half *xHi = (__half *)fcuda_malloc(xn * sizeof(__half));
  __half *xLo = (__half *)fcuda_malloc(xn * sizeof(__half));
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
}

__global__ void k_q8_dequant_transient(const signed char *codes, const float *scales,
                                         float *W, int rows, int cols, int block) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (i >= total) return;
  int row = i / cols, col = i % cols;
  W[i] = (float)codes[i] * scales[(size_t)row * (cols / block) + col / block];
}

static void q8_dequant_sgemm(const void *codes, const float *scales,
                            const float *X, float *Y, int rows, int cols, int tokens, int block) {
  float *wide = (float *)fcuda_malloc((size_t)rows * cols * sizeof(float));
  k_q8_dequant_transient<<<((size_t)rows * cols + 255) / 256, 256, 0, g_stream>>>(
      (const signed char *)codes, scales, wide, rows, cols, block);
  fp16_compensated_matmul(wide, X, Y, rows, cols, tokens);
  fcuda_free(wide);
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
    q8_dequant_sgemm(dInQKV, dInQKVScale, dX, dMixed, convDim, hidden, tokens, 32);
    q8_dequant_sgemm(dInZ, dInZScale, dX, dZ, nV * vHd, hidden, tokens, 32);
    q8_dequant_sgemm(dInB, dInBScale, dX, dB, nV, hidden, tokens, 32);
    q8_dequant_sgemm(dInA, dInAScale, dX, dA, nV, hidden, tokens, 32);
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
    q8_dequant_sgemm(dOutW, dOutWScale, dCore, dOut, hidden, nV * vHd, tokens, 32);
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
  cublasStatus_t st = cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N,
              out, P, in,
              &alpha,
              dW, in,
              dX, in,
              &beta,
              dY, out);
  if (st != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "fak-cuda: cublasSgemm(out=%d,in=%d,P=%d) failed: %d\n", out, in, P, (int)st);
    // FAIL-SAFE (#972): a failed SGEMM leaves dY untouched, and dY is a RECYCLED pool buffer,
    // so a stale (smaller) prior allocation's content reads back as a plausible-but-wrong vector
    // (the "out=257 -> got 64" ghost the issue reports). Zero the full out*P result so a failed
    // launch yields an unambiguous all-zero vector the cosine/shape gate rejects deterministically,
    // never a short/garbage read that masquerades as a real result.
    cudaMemsetAsync(dY, 0, (size_t)out * (size_t)P * sizeof(float), g_stream);
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
  k_f32_to_f16<<<(n + 255) / 256, 256, 0, g_stream>>>(dSrc, (__half *)dDstHalf, n);
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
  k_f32_to_f16_T<<<(total + 255) / 256, 256, 0, g_stream>>>(dSrc, (__half *)dDstHalf, out, in);
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
  k_f32_to_f16<<<(P * in + 255) / 256, 256, 0, g_stream>>>(dX, dXh, P * in);
  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st;
  if (colMajor) {
    st = cublasGemmEx(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, out, P, in,
                      &alpha, A, CUDA_R_16F, out, dXh, CUDA_R_16F, in,
                      &beta, dY, CUDA_R_32F, out,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  } else {
    st = cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, out, P, in,
                      &alpha, A, CUDA_R_16F, in, dXh, CUDA_R_16F, in,
                      &beta, dY, CUDA_R_32F, out,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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
extern "C" void fcuda_q8_matmul_f32(const int8_t *dCodes, const float *dScales, const float *dX,
                                    float *dY, int out, int in, int P, int block) {
  if (P >= 128) {
    q8_dequant_sgemm(dCodes, dScales, dX, dY, out, in, P, block);
    return;
  }
  int nblk = in / block;
  int needQX = P * in;          // int8 activation codes
  int needScale = P * nblk;     // per-block act scales (floats)
  if (needQX > g_q8_qX_cap) {
    if (g_q8_qX) cudaFree(g_q8_qX);
    CK(cudaMalloc(&g_q8_qX, (size_t)needQX));
    g_q8_qX_cap = needQX;
  }
  if (needScale > g_q8_xScale_cap) {
    if (g_q8_xScale) cudaFree(g_q8_xScale);
    CK(cudaMalloc(&g_q8_xScale, (size_t)needScale * sizeof(float)));
    g_q8_xScale_cap = needScale;
  }
  k_q8_quant_act<<<dim3(nblk, P), 64, 0, g_stream>>>(dX, g_q8_qX, g_q8_xScale, P, in, block);
  k_q8_gemm<<<dim3((out + 7) / 8, P), 256, 0, g_stream>>>((const signed char *)dCodes, dScales, g_q8_qX, g_q8_xScale,
                                                dY, out, in, P, block);
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
  k_q4k_coeffs<<<(n + 255) / 256, 256, 0, g_stream>>>(q, coeff, out, in);
  g_q4_coeff_cache[g_q4_coeff_cache_n++] = {q, coeff, out, in, device};
  return coeff;
}

// Persistent Q8_1 activation-quantization scratch for Q4_K MMVQ (#8635).
static signed char *g_q81_qX = nullptr;
static int g_q81_qX_cap = 0; // bytes
static float2 *g_q81_xScaleSum = nullptr;
static int g_q81_xScaleSum_cap = 0; // float2 elements

// k_q8_1_quant_act quantizes f32 activation X[P, in] into Q8_1 on-device:
// Per 32-element sub-block, computes scale d = amax/127, sum s = exact sum of x elements,
// and signed int8 quantized codes q = q8round(x / d).
__global__ void k_q8_1_quant_act(const float *X, signed char *qX, float2 *xScaleSum,
                                 int P, int in) {
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int b = blockIdx.x * 4 + warp_id;
  int t = blockIdx.y;
  int nblk = in / 32;
  if (t >= P || b >= nblk) return;

  size_t act_offset = (size_t)t * in + (size_t)b * 32 + lane;
  float val = X[act_offset];
  float a = fabsf(val);

#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    a = fmaxf(a, __shfl_down_sync(0xffffffff, a, delta));
  }
  float amax = __shfl_sync(0xffffffff, a, 0);
  float d = amax / 127.0f;
  float inv = d > 0.0f ? 1.0f / d : 0.0f;

  signed char q = d > 0.0f ? q8round_dev(val * inv) : (signed char)0;
  qX[act_offset] = q;

  float sumVal = val;
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    sumVal += __shfl_down_sync(0xffffffff, sumVal, delta);
  }

  if (lane == 0) {
    xScaleSum[(size_t)t * nblk + b] = make_float2(d, sumVal);
  }
}

// k_q4k_mmvq: signed DP4A Q4_K matrix-vector decode kernel (MMVQ) with Q8_1 activations (#8635).
// Each warp owns one output row o. The 32 lanes cooperate to load 128 bytes of nibbles coalesced,
// evaluate signed __dp4a on 4-element vectors, reduce across chunk warps, and combine scales and mins.
__global__ void k_q4k_mmvq(const unsigned char *Q4K, const signed char *qX,
                           const float2 *xScaleSum, float *Y, int out, int in, int P) {
  constexpr int warps_per_block = 8;
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int o = blockIdx.x * warps_per_block + warp;
  int t = blockIdx.y;
  if (o >= out || t >= P) return;

  int nsb = in / 256;
  const unsigned char *wrow = Q4K + (size_t)o * nsb * 144;
  const signed char *xrow = qX + (size_t)t * in;
  const float2 *xsc = xScaleSum + (size_t)t * (in / 32);

  float row_sum = 0.0f;
  for (int sb = 0; sb < nsb; ++sb) {
    const unsigned char *blk = wrow + (size_t)sb * 144;

    uint4 h;
    if (lane == 0) {
      h = *(const uint4 *)blk;
    }
    h.x = __shfl_sync(0xffffffff, h.x, 0);
    h.y = __shfl_sync(0xffffffff, h.y, 0);
    h.z = __shfl_sync(0xffffffff, h.z, 0);
    h.w = __shfl_sync(0xffffffff, h.w, 0);

    float d = __half2float(*(const __half *)&h.x);
    float dmin = __half2float(*((const __half *)&h.x + 1));
    const unsigned char *scales = (const unsigned char *)&h.y;

    const unsigned char *q = blk + 16;
    int v = ((const int *)q)[lane];

    int k = lane >> 3;       // 0..3 (chunk)
    int l_group = lane & 7;  // 0..7 (lane within chunk)
    int sA = sb * 8 + 2 * k;
    int sB = sA + 1;

    int qxA = *(const int *)(xrow + sA * 32 + l_group * 4);
    int qxB = *(const int *)(xrow + sB * 32 + l_group * 4);

    int low = v & 0x0f0f0f0f;
    int high = (v >> 4) & 0x0f0f0f0f;

    int dotA = __dp4a(low, qxA, 0);
    int dotB = __dp4a(high, qxB, 0);

    dotA += __shfl_down_sync(0xffffffff, dotA, 4);
    dotA += __shfl_down_sync(0xffffffff, dotA, 2);
    dotA += __shfl_down_sync(0xffffffff, dotA, 1);

    dotB += __shfl_down_sync(0xffffffff, dotB, 4);
    dotB += __shfl_down_sync(0xffffffff, dotB, 2);
    dotB += __shfl_down_sync(0xffffffff, dotB, 1);

    float chunk_sum = 0.0f;
    if (l_group == 0) {
      unsigned char scA, mnA, scB, mnB;
      getScaleMinK4_dev(2 * k, scales, &scA, &mnA);
      getScaleMinK4_dev(2 * k + 1, scales, &scB, &mnB);

      float2 actA = xsc[sA];
      float2 actB = xsc[sB];

      float termA = (d * (float)scA) * (actA.x * (float)dotA) - (dmin * (float)mnA) * actA.y;
      float termB = (d * (float)scB) * (actB.x * (float)dotB) - (dmin * (float)mnB) * actB.y;
      chunk_sum = termA + termB;
    }

    chunk_sum += __shfl_down_sync(0xffffffff, chunk_sum, 16);
    chunk_sum += __shfl_down_sync(0xffffffff, chunk_sum, 8);

    if (lane == 0) {
      row_sum += chunk_sum;
    }
  }

  if (lane == 0) {
    Y[(size_t)t * out + o] = row_sum;
  }
}

extern "C" void fcuda_q4k_mmvq_dp4a(const uint8_t *dQ4K, const float *dX, float *dY,
                                    int out, int in, int P) {
  int nblk = in / 32;
  int needQX = P * in;
  int needScaleSum = P * nblk;
  if (needQX > g_q81_qX_cap) {
    if (g_q81_qX) cudaFree(g_q81_qX);
    CK(cudaMalloc(&g_q81_qX, (size_t)needQX));
    g_q81_qX_cap = needQX;
  }
  if (needScaleSum > g_q81_xScaleSum_cap) {
    if (g_q81_xScaleSum) cudaFree(g_q81_xScaleSum);
    CK(cudaMalloc(&g_q81_xScaleSum, (size_t)needScaleSum * sizeof(float2)));
    g_q81_xScaleSum_cap = needScaleSum;
  }
  dim3 quantGrid((nblk + 3) / 4, P);
  k_q8_1_quant_act<<<quantGrid, 128, 0, g_stream>>>(dX, g_q81_qX, g_q81_xScaleSum, P, in);

  constexpr int warps_per_block = 8;
  dim3 mmvqGrid((out + warps_per_block - 1) / warps_per_block, P);
  k_q4k_mmvq<<<mmvqGrid, warps_per_block * 32, 0, g_stream>>>(
      (const unsigned char *)dQ4K, g_q81_qX, g_q81_xScaleSum, dY, out, in, P);
}

static bool use_q4k_mmvq(void) {
  const char *env = getenv("FAK_CUDA_Q4K_MMVQ");
  if (env && env[0] == '0' && env[1] == '\0') return false;
  return true;
}

extern "C" void fcuda_q4k_matmul_f32(const uint8_t *dQ4K, const float *dX, float *dY,
                                     int out, int in, int P) {
  if (P >= 128) {
    float *wide = (float *)fcuda_malloc((size_t)out * in * sizeof(float));
    k_q4k_dequant_transient<<<((size_t)out * in + 255) / 256, 256, 0, g_stream>>>(
        (const unsigned char *)dQ4K, wide, out, in);
    fp16_compensated_matmul(wide, dX, dY, out, in, P);
    fcuda_free(wide);
    return;
  }
  if (P < 4) {
    if (use_q4k_mmvq()) {
      fcuda_q4k_mmvq_dp4a(dQ4K, dX, dY, out, in, P);
      return;
    }
    float2 *coeff = q4_coeffs_for(dQ4K, out, in);
    k_q4k_gemm<<<dim3((out + 1) / 2, P), 256, 0, g_stream>>>(
        (const unsigned char *)dQ4K, coeff, dX, dY, out, in, P);
    return;
  }
  constexpr int token_tile = 4;
  k_q4k_gemm_panel<token_tile><<<dim3(out, (P + token_tile - 1) / token_tile), 32, 0, g_stream>>>(
      (const unsigned char *)dQ4K, dX, dY, out, in, P);
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
  k_q2_0_gemm<<<dim3(out, P), 256, 0, g_stream>>>(
      (const unsigned char *)dCodes, dScales, dX, dY, out, in, P, block);
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
  k_q5k_gemm<<<dim3((o + 7) / 8, p), 256, 0, g_stream>>>(w, x, y, o, i, p);
}
extern "C" void fcuda_q6k_matmul_f32(const uint8_t *w, const float *x, float *y,
                                      int o, int i, int p) {
  k_q6k_gemm<<<dim3((o + 7) / 8, p), 256, 0, g_stream>>>(w, x, y, o, i, p);
}

// ---- RMSNorm: one block per row -------------------------------------------------
__global__ void k_rmsnorm(const float *X, const float *W, float *Y, int rows, int n, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *x = X + (size_t)r * n;
  float *y = Y + (size_t)r * n;
  __shared__ float red[256];
  float local = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) local += x[i] * x[i];
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * W[i];
}
extern "C" void fcuda_rmsnorm_f32(const float *dX, const float *dW, float *dY, int rows, int n, float eps) {
  k_rmsnorm<<<rows, 256, 0, g_stream>>>(dX, dW, dY, rows, n, eps);
}

// ---- RMSNorm: warp-per-row (fitting widths <= 1024) -----------------------------
__global__ void k_rmsnorm_warp_per_row(const float *X, const float *W, float *Y, int rows, int n, float eps) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x % 32;
  if (warp_id >= rows) return;
  const float *x = X + (size_t)warp_id * n;
  float *y = Y + (size_t)warp_id * n;
  float local = 0.f;
  for (int i = lane; i < n; i += 32) local += x[i] * x[i];
  for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(0xffffffff, local, offset);
  }
  float mean = __shfl_sync(0xffffffff, local, 0) / (float)n;
  float inv = rsqrtf(mean + eps);
  for (int i = lane; i < n; i += 32) y[i] = x[i] * inv * W[i];
}

extern "C" void fcuda_rmsnorm_dispatched_f32(const float *dX, const float *dW, float *dY, int rows, int n, float eps) {
  if (n <= 1024) {
    int warps = 8;
    int blocks = (rows + warps - 1) / warps;
    k_rmsnorm_warp_per_row<<<blocks, warps * 32, 0, g_stream>>>(dX, dW, dY, rows, n, eps);
  } else {
    k_rmsnorm<<<rows, 256, 0, g_stream>>>(dX, dW, dY, rows, n, eps);
  }
}

// ---- Fused RMSNorm + Residual Add ----------------------------------------------
__global__ void k_rmsnorm_fused_residual_add(
    const float *X, const float *ResidualIn, const float *W,
    float *ResidualOut, float *Y, int rows, int n, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *x = X + (size_t)r * n;
  const float *rin = ResidualIn + (size_t)r * n;
  float *rout = ResidualOut ? (ResidualOut + (size_t)r * n) : nullptr;
  float *y = Y + (size_t)r * n;
  __shared__ float red[256];
  float local = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float val = x[i] + rin[i];
    if (rout) rout[i] = val;
    local += val * val;
  }
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float val = rout ? rout[i] : (x[i] + rin[i]);
    y[i] = val * inv * W[i];
  }
}
extern "C" void fcuda_rmsnorm_fused_residual_add_f32(
    const float *dX, const float *dResidualIn, const float *dW,
    float *dResidualOut, float *dY, int rows, int n, float eps) {
  k_rmsnorm_fused_residual_add<<<rows, 256, 0, g_stream>>>(
      dX, dResidualIn, dW, dResidualOut, dY, rows, n, eps);
}

// ---- RoPE (HF non-interleaved rotate_half) at one absolute position -------------
__global__ void k_rope(float *X, int pos, int nHeads, int headDim, double theta) {
  int half = headDim / 2;
  int t = blockIdx.x * blockDim.x + threadIdx.x; // one (head, j) pair
  if (t >= nHeads * half) return;
  int h = t / half, j = t % half;
  double inv = 1.0 / pow(theta, (double)(2 * j) / (double)headDim);
  double a = (double)pos * inv;
  float c = (float)cos(a), s = (float)sin(a);
  float *hv = X + (size_t)h * headDim;
  float x0 = hv[j], x1 = hv[j + half];
  hv[j]        = x0 * c - x1 * s;
  hv[j + half] = x1 * c + x0 * s;
}
extern "C" void fcuda_rope_f32(float *dX, int pos, int nHeads, int headDim, double theta) {
  int total = nHeads * (headDim / 2);
  k_rope<<<(total + 127) / 128, 128, 0, g_stream>>>(dX, pos, nHeads, headDim, theta);
}


__global__ void k_partial_rope_copy(float *out, const float *in, int pos,
                                    int nHeads, int headDim, int rotaryDim,
                                    double theta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = nHeads * headDim;
  if (i >= n) return;
  int d = i % headDim;
  int h = i / headDim;
  if (d >= rotaryDim) { out[i] = in[i]; return; }
  int half = rotaryDim / 2;
  int j = d < half ? d : d - half;
  double freq = pow(theta, -(2.0 * j) / rotaryDim);
  float cs = cos((double)pos * freq), sn = sin((double)pos * freq);
  float a = in[h * headDim + j], b = in[h * headDim + j + half];
  out[i] = d < half ? a * cs - b * sn : b * cs + a * sn;
}
extern "C" void fcuda_partial_rope_qk_f32(const float *dQ, const float *dK,
                                             float *dQOut, float *dKOut, int pos,
                                             int nQHeads, int nKHeads, int headDim,
                                             int rotaryDim, double theta) {
  int qn = nQHeads * headDim, kn = nKHeads * headDim;
  k_partial_rope_copy<<<(qn+255)/256,256,0,g_stream>>>(dQOut,dQ,pos,nQHeads,headDim,rotaryDim,theta);
  CK(cudaGetLastError());
  k_partial_rope_copy<<<(kn+255)/256,256,0,g_stream>>>(dKOut,dK,pos,nKHeads,headDim,rotaryDim,theta);
  CK(cudaGetLastError());
}
__global__ void k_sigmoid_mul(float *x, const float *gate, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}
extern "C" void fcuda_sigmoid_mul_f32(float *dX, const float *dGate, int n) {
  k_sigmoid_mul<<<(n+255)/256,256,0,g_stream>>>(dX,dGate,n);
  CK(cudaGetLastError());
}
__global__ void k_split_qwen35_qg(const float *qg, float *q, float *gate,
                                    int nHeads, int headDim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = nHeads * headDim;
  if (i >= n) return;
  int h = i / headDim, d = i % headDim;
  int src = h * 2 * headDim + d;
  q[i] = qg[src];
  gate[i] = qg[src + headDim];
}
extern "C" void fcuda_split_qwen35_qg_f32(const float *dQG, float *dQ, float *dGate,
                                             int nHeads, int headDim) {
  int n = nHeads * headDim;
  k_split_qwen35_qg<<<(n+255)/256,256,0,g_stream>>>(dQG,dQ,dGate,nHeads,headDim);
  CK(cudaGetLastError());
}

__global__ void k_qwen35_split_qg_panel(const float *qg, float *q, float *gate,
                                         int tokens, int nHeads, int headDim) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t width = (size_t)nHeads * headDim;
  size_t n = (size_t)tokens * width;
  if (i >= n) return;
  int token = (int)(i / width);
  int within = (int)(i - (size_t)token * width);
  int head = within / headDim;
  int dim = within - head * headDim;
  size_t src = (size_t)token * 2 * width + (size_t)head * 2 * headDim + dim;
  q[i] = qg[src];
  gate[i] = qg[src + headDim];
}

extern "C" int fak_qwen35_split_qg_panel_f32(
    const float *dQG, float *dQ, float *dGate,
    int tokens, int nHeads, int headDim) {
  if (!dQG || !dQ || !dGate || tokens <= 0 || nHeads <= 0 || headDim <= 0) return -1;
  cudaGetLastError();
  size_t n = (size_t)tokens * nHeads * headDim;
  k_qwen35_split_qg_panel<<<(n + 255) / 256, 256, 0, g_stream>>>(
      dQG, dQ, dGate, tokens, nHeads, headDim);
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 30000 + (int)launch;
}

__global__ void k_qwen35_partial_rope_panel(
    float *out, const float *in, int tokens, int startPos,
    int nHeads, int headDim, int rotaryDim, double theta) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t width = (size_t)nHeads * headDim;
  size_t n = (size_t)tokens * width;
  if (i >= n) return;
  int token = (int)(i / width);
  int within = (int)(i - (size_t)token * width);
  int head = within / headDim;
  int dim = within - head * headDim;
  if (dim >= rotaryDim) { out[i] = in[i]; return; }
  int half = rotaryDim / 2;
  int j = dim < half ? dim : dim - half;
  double freq = pow(theta, -(2.0 * j) / rotaryDim);
  double angle = (double)(startPos + token) * freq;
  float cs = (float)cos(angle), sn = (float)sin(angle);
  size_t base = (size_t)token * width + (size_t)head * headDim;
  float a = in[base + j], b = in[base + j + half];
  out[i] = dim < half ? a * cs - b * sn : b * cs + a * sn;
}

extern "C" int fak_qwen35_partial_rope_panel_f32(
    const float *dQ, const float *dK, float *dQOut, float *dKOut,
    int tokens, int startPos, int nQHeads, int nKHeads, int headDim,
    int rotaryDim, double theta) {
  if (!dQ || !dK || !dQOut || !dKOut || tokens <= 0 || startPos < 0 ||
      nQHeads <= 0 || nKHeads <= 0 || headDim <= 0 || rotaryDim <= 0 ||
      rotaryDim > headDim || (rotaryDim & 1) != 0 || !(theta > 0.0)) return -1;
  cudaGetLastError();
  size_t qn = (size_t)tokens * nQHeads * headDim;
  size_t kn = (size_t)tokens * nKHeads * headDim;
  k_qwen35_partial_rope_panel<<<(qn + 255) / 256, 256, 0, g_stream>>>(
      dQOut, dQ, tokens, startPos, nQHeads, headDim, rotaryDim, theta);
  cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) return 40000 + (int)launch;
  k_qwen35_partial_rope_panel<<<(kn + 255) / 256, 256, 0, g_stream>>>(
      dKOut, dK, tokens, startPos, nKHeads, headDim, rotaryDim, theta);
  launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 41000 + (int)launch;
}

// ---- SwiGLU / residual add / bias add -------------------------------------------
__global__ void k_swiglu(const float *G, const float *U, float *Y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = G[i];
  Y[i] = (g / (1.f + expf(-g))) * U[i];
}
extern "C" void fcuda_swiglu_f32(const float *dG, const float *dU, float *dY, int n) {
  k_swiglu<<<(n + 255) / 256, 256, 0, g_stream>>>(dG, dU, dY, n);
}
__global__ void k_add(float *D, const float *S, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) D[i] += S[i];
}
extern "C" void fcuda_add_f32(float *dDst, const float *dSrc, int n) {
  k_add<<<(n + 255) / 256, 256, 0, g_stream>>>(dDst, dSrc, n);
}
__global__ void k_add_bias(float *D, const float *B, int rows, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width) return;
  D[i] += B[i % width];
}
extern "C" void fcuda_add_bias_f32(float *dDst, const float *dBias, int rows, int width) {
  k_add_bias<<<(rows * width + 255) / 256, 256, 0, g_stream>>>(dDst, dBias, rows, width);
}

// k_copyrow appends a KV row into a fixed-base buffer at a SCALAR float offset. It replaces
// the AppendKV cudaMemcpyAsync, whose destination pointer grew every token — a moving
// pointer that cudaGraphExecUpdate cannot patch, so a captured decode graph could not be
// reused (it re-instantiated each token). A kernel with the offset as a scalar arg IS
// ExecUpdate-patchable, so one captured graph now serves the whole growing cache.
__global__ void k_copyrow(float *dstBase, const float *src, int offset, int n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) dstBase[(size_t)offset + i] = src[i];
}
extern "C" void fcuda_kv_write(float *dstBase, const float *src, int offset, int n) {
  if (n <= 0) return;
  unsigned blocks = ((unsigned)n + 255u) / 256u;
  k_copyrow<<<blocks, 256, 0, g_stream>>>(dstBase, src, offset, n);
}
extern "C" int fak_qwen35_kv_write_f32(
    float *dstBase, const float *src, int offset, int n) {
  if (!dstBase || !src || offset < 0 || n <= 0 || offset > INT_MAX - n) return -1;
  cudaError_t prior = cudaGetLastError();
  if (prior != cudaSuccess) return 51000 + (int)prior;
  unsigned blocks = ((unsigned)n + 255u) / 256u;
  k_copyrow<<<blocks, 256, 0, g_stream>>>(dstBase, src, offset, n);
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 52000 + (int)launch;
}

// ---- Decode attention: NAIVE one-block-per-head (the #486 baseline) --------------
// q[nH*hd]; K,V [nPos, nKV*hd]; out[nH*hd]; scores scratch [nH*nPos]. This is the
// original correct-but-naive kernel (commit 54f8b58): it materializes a FULL scores[nPos]
// row per head in GLOBAL memory (g_attn_scratch) and makes four passes over it (raw score
// write, max, exp-in-place, weighted-V read). The flash kernel below replaces it on the
// live Attention path; this one is RETAINED only as the fused-vs-naive microbench baseline
// (#486) — fcuda_attention_f32 keeps it reachable from the Go side's attentionNaive.
__global__ void k_attention(const float *Q, const float *K, const float *V, float *Out,
                            float *Scores, int nPos, int nH, int nKV, int hd, float scale) {
  int h = blockIdx.x;
  if (h >= nH) return;
  int grp = nH / nKV;
  int kvh = h / grp;
  int w = nKV * hd;
  const float *qh = Q + (size_t)h * hd;
  float *sc = Scores + (size_t)h * nPos;
  __shared__ float red[128];
  // phase 1: raw scores = scale * dot(qh, K[j, kvh])
  for (int j = threadIdx.x; j < nPos; j += blockDim.x) {
    const float *kh = K + (size_t)j * w + kvh * hd;
    float d = 0.f;
    for (int e = 0; e < hd; e++) d += qh[e] * kh[e];
    sc[j] = d * scale;
  }
  __syncthreads();
  // phase 2: block-reduce max
  float lm = -1e30f;
  for (int j = threadIdx.x; j < nPos; j += blockDim.x) lm = fmaxf(lm, sc[j]);
  red[threadIdx.x] = lm; __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
  float mx = red[0];
  __syncthreads();
  // phase 3: exp in place + block-reduce sum
  float ls = 0.f;
  for (int j = threadIdx.x; j < nPos; j += blockDim.x) { float e = expf(sc[j] - mx); sc[j] = e; ls += e; }
  red[threadIdx.x] = ls; __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
  float sum = red[0];
  __syncthreads();
  // phase 4: out[d] = Σ_j (sc[j]/sum) * V[j, kvh, d]
  for (int d = threadIdx.x; d < hd; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < nPos; j++) acc += sc[j] * V[(size_t)j * w + kvh * hd + d];
    Out[(size_t)h * hd + d] = acc / sum;
  }
}
// Persistent attention scratch, sized to nH*maxPos and grown ONCE (outside any graph
// capture, since maxPos is fixed from the first call). A per-call pooled scratch would
// change size as nPos grows each token -> a cudaMalloc mid-capture (illegal). Reused across
// layers/tokens; each k_attention writes-then-reads it within one call, and calls are
// serialized on g_stream, so sharing is safe.
static float *g_attn_scratch = nullptr;
static int g_attn_scratch_cap = 0; // floats
extern "C" void fcuda_attention_f32(const float *dQ, const float *dK, const float *dV, float *dOut,
                                    int nPos, int maxPos, int nH, int nKV, int hd, float scale) {
  int need = nH * (maxPos > nPos ? maxPos : nPos);
  if (need > g_attn_scratch_cap) {
    if (g_attn_scratch) cudaFree(g_attn_scratch);
    CK(cudaMalloc(&g_attn_scratch, (size_t)need * sizeof(float)));
    g_attn_scratch_cap = need;
  }
  k_attention<<<nH, 128, 0, g_stream>>>(dQ, dK, dV, dOut, g_attn_scratch, nPos, nH, nKV, hd, scale);
}

// ---- Flash / online-softmax decode attention (#486) ------------------------------
// The fused replacement for k_attention. One block per query head; the FLASH_THREADS
// threads of a block SPLIT the head dimension. The KV window is streamed in-place with a
// RUNNING (max m, sum l, output accumulator acc) — the FlashAttention online-softmax — so:
//   • NO scores[nPos] buffer is ever materialized (the naive kernel's g_attn_scratch is
//     gone on this path); the only scratch is per-block SHARED memory (the query row +
//     a reduction staging row), allocated by the launch and reused for every key, every
//     layer and every token — there is NO per-call global allocation at all.
//   • one streaming pass over K and V replaces the naive kernel's four passes over a
//     global scores row, cutting the HBM traffic the decode attention pays.
// causal/grp/scale are consumed as kernel PARAMS (grp = nH/nKV selects the KV head; the
// cache holds exactly the attendable keys, so causality is by construction; scale folds
// into the score). The math is the cpuref softmax(scale·q·k)·V reordered into the online
// form — only the f32 reduction order differs, which is what keeps the lane Approx (the
// recorded cudaFlashAttnCosineMin floor), not bit-identity.
//
// Online-softmax recurrence, per key j (every thread runs it identically off the reduced
// score, so m and l stay replicated and consistent across the block):
//   s      = scale · dot(q, K_j)
//   m'     = max(m, s)
//   corr   = exp(m − m')           // rescales the running sum/acc onto the new max
//   p      = exp(s − m')
//   l      = l·corr + p
//   acc[d] = acc[d]·corr + p·V_j[d]   // each thread owns the dims d it strides
// and out[d] = acc[d]/l after the window. Bit-faithful to softmax then ΣwV; no full row.
#define FLASH_THREADS 128
// FLASH_ACC_MAX bounds the head dims one thread carries = ceil(hd / FLASH_THREADS). 8 covers
// hd ≤ 1024 (every real attention head dim is ≤ 256), so the per-thread acc lives in
// registers/local memory, never a global scores row.
#define FLASH_ACC_MAX 8
__global__ void k_flash_attention(const float *Q, const float *K, const float *V, float *Out,
                                  int nPos, int nH, int nKV, int hd, float scale) {
  int h = blockIdx.x;
  if (h >= nH) return;
  int grp = nH / nKV;       // query heads per KV head (GQA/MQA group)
  int kvh = h / grp;        // the KV head this query head reads
  int w = nKV * hd;         // per-position stride in K/V
  const float *qh = Q + (size_t)h * hd;
  extern __shared__ float smem[];
  float *qs = smem;                 // [hd]            : the query row, cached once in shared
  float *red = smem + hd;           // [FLASH_THREADS] : dot-product reduction staging
  int tid = threadIdx.x;
  // cache the query row in shared memory (threads stride hd).
  for (int d = tid; d < hd; d += FLASH_THREADS) qs[d] = qh[d];
  __syncthreads();
  // online-softmax running state. acc[k] is this thread's accumulator for owned dim
  // d = tid + k*FLASH_THREADS; m and l are replicated across the block.
  float m = -1e30f, l = 0.f;
  float acc[FLASH_ACC_MAX];
#pragma unroll
  for (int k = 0; k < FLASH_ACC_MAX; k++) acc[k] = 0.f;
  for (int j = 0; j < nPos; j++) {
    const float *kj = K + (size_t)j * w + (size_t)kvh * hd;
    // partial dot over this thread's strided dims, then a block reduction -> full score.
    float partial = 0.f;
    for (int d = tid; d < hd; d += FLASH_THREADS) partial += qs[d] * kj[d];
    red[tid] = partial;
    __syncthreads();
    for (int s = FLASH_THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float sc = red[0] * scale;        // every thread reads the reduced dot
    __syncthreads();                  // WAR: finish all red[0] reads before next key reuses red
    float mNew = fmaxf(m, sc);
    float corr = expf(m - mNew);      // 0 on the first key (m = -inf): clears the empty acc
    float p = expf(sc - mNew);        // expf (not __expf) so the only divergence from the
    l = l * corr + p;                 // reference is f32 reduction order, not a faster-exp ulp

    const float *vj = V + (size_t)j * w + (size_t)kvh * hd;
    int k = 0;
    for (int d = tid; d < hd; d += FLASH_THREADS, k++) acc[k] = acc[k] * corr + p * vj[d];
    m = mNew;
  }
  float invL = l > 0.f ? 1.f / l : 0.f;
  int k = 0;
  for (int d = tid; d < hd; d += FLASH_THREADS, k++) Out[(size_t)h * hd + d] = acc[k] * invL;
}
// fcuda_flash_attention_f32 launches the flash kernel: one block per query head, FLASH_THREADS
// threads, and just enough dynamic shared memory for the query row + the reduction row. Unlike
// the naive entrypoint there is NO g_attn_scratch — the online form needs no nPos-sized buffer,
// so nothing is allocated or grown per call. maxPos is accepted for a signature parallel to the
// naive baseline (the microbench calls both) but is unused here.
extern "C" void fcuda_flash_attention_f32(const float *dQ, const float *dK, const float *dV, float *dOut,
                                          int nPos, int maxPos, int nH, int nKV, int hd, float scale) {
  (void)maxPos;
  size_t shmem = ((size_t)hd + FLASH_THREADS) * sizeof(float);
  k_flash_attention<<<nH, FLASH_THREADS, shmem, g_stream>>>(dQ, dK, dV, dOut, nPos, nH, nKV, hd, scale);
}

// Prompt-panel counterpart of k_flash_attention. K/V already contain the
// complete appended panel; query row t is restricted to prefix+t+1, preserving
// causality without materializing a scores panel or replaying rows from Go. The
// first visible value seeds (m,l,acc) exactly; later values use the online-softmax
// recurrence, avoiding any finite sentinel assumption about the score range.
__global__ void k_qwen35_causal_attention_panel(
    const float *Q, const float *K, const float *V, float *Out,
    int tokens, int prefix, int nH, int nKV, int hd, float scale) {
  int flat = blockIdx.x;
  int h = flat % nH, token = flat / nH;
  if (h >= nH || token >= tokens) return;
  int tid = threadIdx.x, kvh = h / (nH / nKV), width = nKV * hd;
  int nPos = prefix + token + 1;
  const float *qh = Q + (size_t)token * nH * hd + (size_t)h * hd;
  extern __shared__ float smem[];
  float *qs = smem;
  float *red = smem + hd;
  for (int d = tid; d < hd; d += FLASH_THREADS) qs[d] = qh[d];
  __syncthreads();
  float m = 0.f, l = 0.f;
  float acc[FLASH_ACC_MAX];
#pragma unroll
  for (int k = 0; k < FLASH_ACC_MAX; k++) acc[k] = 0.f;
  for (int j = 0; j < nPos; j++) {
    const float *kj = K + (size_t)j * width + (size_t)kvh * hd;
    float partial = 0.f;
    for (int d = tid; d < hd; d += FLASH_THREADS) partial += qs[d] * kj[d];
    red[tid] = partial;
    __syncthreads();
    for (int s = FLASH_THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float score = red[0] * scale;
    __syncthreads();
    const float *vj = V + (size_t)j * width + (size_t)kvh * hd;
    if (j == 0) {
      m = score;
      l = 1.f;
      int k = 0;
      for (int d = tid; d < hd; d += FLASH_THREADS, k++) acc[k] = vj[d];
      continue;
    }
    float nextM = fmaxf(m, score);
    float correction = expf(m - nextM), probability = expf(score - nextM);
    l = l * correction + probability;
    int k = 0;
    for (int d = tid; d < hd; d += FLASH_THREADS, k++) {
      acc[k] = acc[k] * correction + probability * vj[d];
    }
    m = nextM;
  }
  float invL = l > 0.f ? 1.f / l : 0.f;
  int k = 0;
  for (int d = tid; d < hd; d += FLASH_THREADS, k++) {
    Out[(size_t)token * nH * hd + (size_t)h * hd + d] = acc[k] * invL;
  }
}

// Qwen3.8 27B uses a 256-wide attention head. Keep that production geometry on
// a named source path so a source/ABI audit can prove it does not accidentally
// regress to the legacy warp-only body that returned without writing hd > 32.
// The recurrence deliberately matches k_qwen35_causal_attention_panel above:
// one 128-thread block owns a row, each thread owns two dimensions, and shared
// memory holds the query plus the block-reduction lane.
#define QWEN38_PROMPT_HEAD_DIM 256
__global__ void k_qwen38_causal_attention_panel_hd256(
    const float *Q, const float *K, const float *V, float *Out,
    int tokens, int prefix, int nH, int nKV, float scale) {
  int flat = blockIdx.x;
  int h = flat % nH, token = flat / nH;
  if (h >= nH || token >= tokens) return;
  int tid = threadIdx.x, kvh = h / (nH / nKV);
  int width = nKV * QWEN38_PROMPT_HEAD_DIM;
  int nPos = prefix + token + 1;
  const float *qh = Q + (size_t)token * nH * QWEN38_PROMPT_HEAD_DIM +
                    (size_t)h * QWEN38_PROMPT_HEAD_DIM;
  extern __shared__ float smem[];
  float *qs = smem;
  float *red = smem + QWEN38_PROMPT_HEAD_DIM;
  for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS) qs[d] = qh[d];
  __syncthreads();
  float m = 0.f, l = 0.f;
  float acc[QWEN38_PROMPT_HEAD_DIM / FLASH_THREADS];
#pragma unroll
  for (int k = 0; k < QWEN38_PROMPT_HEAD_DIM / FLASH_THREADS; k++) acc[k] = 0.f;
  for (int j = 0; j < nPos; j++) {
    const float *kj = K + (size_t)j * width +
                      (size_t)kvh * QWEN38_PROMPT_HEAD_DIM;
    float partial = 0.f;
    for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS) {
      partial += qs[d] * kj[d];
    }
    red[tid] = partial;
    __syncthreads();
    for (int s = FLASH_THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float score = red[0] * scale;
    __syncthreads();
    const float *vj = V + (size_t)j * width +
                      (size_t)kvh * QWEN38_PROMPT_HEAD_DIM;
    if (j == 0) {
      m = score;
      l = 1.f;
      int k = 0;
      for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS, k++) acc[k] = vj[d];
      continue;
    }
    float nextM = fmaxf(m, score);
    float correction = expf(m - nextM), probability = expf(score - nextM);
    l = l * correction + probability;
    int k = 0;
    for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS, k++) {
      acc[k] = acc[k] * correction + probability * vj[d];
    }
    m = nextM;
  }
  float invL = l > 0.f ? 1.f / l : 0.f;
  int k = 0;
  for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS, k++) {
    Out[(size_t)token * nH * QWEN38_PROMPT_HEAD_DIM +
        (size_t)h * QWEN38_PROMPT_HEAD_DIM + d] = acc[k] * invL;
  }
}

extern "C" int fak_qwen35_causal_attention_panel_f32(
    const float *dQ, const float *dK, const float *dV, float *dOut,
    int tokens, int prefix, int nH, int nKV, int hd, float scale) {
  if (!dQ || !dK || !dV || !dOut ||
      tokens <= 0 || prefix < 0 || nH != 24 || nKV != 4 ||
      hd != QWEN38_PROMPT_HEAD_DIM ||
      tokens > INT_MAX / nH || prefix > INT_MAX - tokens ||
      !isfinite(scale) || scale <= 0.f) return -1;
  // width is an int inside the kernel and the Go sequence cache uses int-sized
  // element offsets at its C ABI. Refuse every product that cannot be represented
  // before launch, rather than allowing an overflowed address calculation to no-op
  // or write an unrelated allocation.
  if (nKV > INT_MAX / hd || nH > INT_MAX / hd) return -1;
  int kvWidth = nKV * hd, qWidth = nH * hd;
  int positions = prefix + tokens;
  if (positions > INT_MAX / kvWidth || tokens > INT_MAX / qWidth) return -1;
  cudaGetLastError();
  size_t shmem = ((size_t)QWEN38_PROMPT_HEAD_DIM + FLASH_THREADS) * sizeof(float);
  k_qwen38_causal_attention_panel_hd256<<<tokens * nH, FLASH_THREADS, shmem, g_stream>>>(
      dQ, dK, dV, dOut, tokens, prefix, nH, nKV, scale);
  cudaError_t launch = cudaGetLastError();
  return launch == cudaSuccess ? 0 : 50000 + (int)launch;
}

extern "C" int fak_qwen35_sequence_sync(void) {
  cudaError_t completed = cudaStreamSynchronize(g_stream);
  return completed == cudaSuccess ? 0 : 60000 + (int)completed;
}

// ---- GLM-MoE-DSA sparse attention over the host-selected key set ------------------
// model.glmDsaAttendCached's inner loop on the device. GLM-5.2's attention is SPARSE: a learned
// indexer picks the top-k keys a query attends, and the softmax(scale·q·k)·ΣwV runs over only
// that selected set. The selection (the f64 index scores + top-k) is computed HOST-side and the
// selected K/V rows are gathered contiguous, so this kernel attends exactly the same keys the
// host loop would — its only divergence is the f32 reduction order (Approx, cudaDsaSparseAttnCosineMin).
// Two things differ from k_flash_attention: (1) it streams the nSel GATHERED selected rows (per
// position the gather laid all nH heads contiguous: head h at i*nH*kd + h*kd for K, i*nH*vd + h*vd
// for V), not a contiguous causal window; (2) MLA's key width (kd = qkNope+qkRope) and value width
// (vd) DIFFER, so it carries both instead of one hd. Same online-softmax form (running max/sum/acc),
// so no scores[nSel] row is ever materialized. One block per query head; FLASH_THREADS threads split
// the dims; the only scratch is per-block shared memory (the query row + a reduction row).
__global__ void k_dsa_sparse_attend(const float *Q, const float *selK, const float *selV, float *Out,
                                    int nSel, int nH, int kd, int vd, float scale) {
  int h = blockIdx.x;
  if (h >= nH) return;
  const float *qh = Q + (size_t)h * kd;
  extern __shared__ float smem[];
  float *qs = smem;            // [kd]            : query row cached once in shared
  float *red = smem + kd;      // [FLASH_THREADS] : dot-product reduction staging
  int tid = threadIdx.x;
  for (int d = tid; d < kd; d += FLASH_THREADS) qs[d] = qh[d];
  __syncthreads();
  // online-softmax running state; acc[k] owns value dim d = tid + k*FLASH_THREADS, m/l replicated.
  float m = -1e30f, l = 0.f;
  float acc[FLASH_ACC_MAX];
#pragma unroll
  for (int k = 0; k < FLASH_ACC_MAX; k++) acc[k] = 0.f;
  for (int i = 0; i < nSel; i++) {
    const float *ki = selK + (size_t)i * nH * kd + (size_t)h * kd;
    float partial = 0.f;
    for (int d = tid; d < kd; d += FLASH_THREADS) partial += qs[d] * ki[d];
    red[tid] = partial;
    __syncthreads();
    for (int s = FLASH_THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float sc = red[0] * scale;        // every thread reads the reduced dot
    __syncthreads();                  // WAR: finish all red[0] reads before the next key reuses red
    float mNew = fmaxf(m, sc);
    float corr = expf(m - mNew);      // 0 on the first key (m = -inf): clears the empty acc
    float p = expf(sc - mNew);
    l = l * corr + p;
    const float *vi = selV + (size_t)i * nH * vd + (size_t)h * vd;
    int k = 0;
    for (int d = tid; d < vd; d += FLASH_THREADS, k++) acc[k] = acc[k] * corr + p * vi[d];
    m = mNew;
  }
  float invL = l > 0.f ? 1.f / l : 0.f;
  int k = 0;
  for (int d = tid; d < vd; d += FLASH_THREADS, k++) Out[(size_t)h * vd + d] = acc[k] * invL;
}
// fcuda_dsa_sparse_attend_f32 launches the sparse-attend kernel: one block per query head,
// FLASH_THREADS threads, dynamic shared memory for the query row + the reduction row (sized on
// the KEY width kd). No per-call global scratch (the online form needs no nSel-sized buffer).
extern "C" void fcuda_dsa_sparse_attend_f32(const float *dQ, const float *dSelK, const float *dSelV,
                                            float *dOut, int nSel, int nH, int kd, int vd, float scale) {
  size_t shmem = ((size_t)kd + FLASH_THREADS) * sizeof(float);
  k_dsa_sparse_attend<<<nH, FLASH_THREADS, shmem, g_stream>>>(dQ, dSelK, dSelV, dOut, nSel, nH, kd, vd, scale);
}

// ---- GLM-MoE-DSA learned-indexer score + top-k selection ---------------------------
// The LAST GLM-5.2 compute that was host-resident even after the dense projections and the
// sparse-attention compute moved to the kernel: the learned indexer that picks WHICH keys a query
// attends. For each cached key k (position k, valid iff k<=queryPos) and the one query, score(k) =
// Σ_h weights[h]·relu(scale·dot(indexQ_h, indexK_k)). The per-head dot is accumulated in DOUBLE so
// the device score equals the host f64 score bit-closely — the indexer drives a DISCRETE top-k, so
// it must be reduction-FAITHFUL (selection-stable), not just cosine-close like the f32 GEMM lanes. A
// single flipped selection would diverge the forward far past any cosine floor, so f64 here is what
// lets the device attend the SAME keys the host would and keeps the downstream witness argmax-exact.
//
// k_dsa_index_score: one block per key, blockDim threads stride the (head,dim) work; the per-key
// score lands in dScores[k]. Masked keys (k>queryPos) get -inf so the top-k never picks them.
__global__ void k_dsa_index_score(const float *indexQ, const float *indexK, const float *weights,
                                  double *dScores, int nKeys, int nH, int indexDim, int queryPos,
                                  float scale) {
  int k = blockIdx.x;
  if (k >= nKeys) return;
  if (k > queryPos) { if (threadIdx.x == 0) dScores[k] = -1e300; return; }
  const float *key = indexK + (size_t)k * indexDim;
  __shared__ double red[256];
  double acc = 0.0;
  // Each thread sums a slice of the (head,dim) grid: head h, dim d, contributing
  //   weights[h]·relu(scale·Σ_d q·k) — but relu is per-HEAD, so accumulate per-head dots first.
  // With nH and indexDim both small, one thread owns whole heads (stride blockDim over heads).
  for (int h = threadIdx.x; h < nH; h += blockDim.x) {
    const float *qh = indexQ + (size_t)h * indexDim;
    double hd = 0.0;
    for (int d = 0; d < indexDim; d++) hd += (double)qh[d] * (double)key[d];
    double hs = hd * (double)scale;
    if (hs < 0.0) hs = 0.0;
    acc += (double)weights[h] * hs;
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) dScores[k] = red[0];
}

// k_dsa_index_topk: single block. Repeats a max-reduction topK times, each pass picking the highest
// remaining score (ties by LOWER position — the dsaTopKIndices order) and masking it out for the
// next pass. nKeys is small (one decode step's causal window) and topK tiny, so the O(topK·nKeys)
// host-equivalent selection is cheap on one block. Writes the selected positions to dSel[0..ret-1].
__global__ void k_dsa_index_topk(const double *dScores, int nKeys, int queryPos, int topK,
                                 int *dSel, int *dCount) {
  __shared__ double vbest[256];
  __shared__ int ibest[256];
  __shared__ char taken[4096]; // == DSA_TOPK_MAX_KEYS; the host wrapper declines nKeys past it, so
                               // every key the top-k sees here is maskable (no un-masked re-select tail).
  int nValid = (queryPos + 1 < nKeys) ? queryPos + 1 : nKeys;
  for (int i = threadIdx.x; i < nKeys && i < 4096; i += blockDim.x) taken[i] = 0;
  __syncthreads();
  int picked = 0;
  int want = topK < nValid ? topK : nValid;
  for (int pass = 0; pass < want; pass++) {
    double bv = -1e300; int bi = -1;
    for (int i = threadIdx.x; i < nValid; i += blockDim.x) {
      if (i < 4096 && taken[i]) continue;
      double v = dScores[i];
      if (bi < 0 || v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    vbest[threadIdx.x] = bv; ibest[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        double ov = vbest[threadIdx.x + s]; int oi = ibest[threadIdx.x + s];
        int cur = ibest[threadIdx.x];
        if (oi >= 0 && (cur < 0 || ov > vbest[threadIdx.x] || (ov == vbest[threadIdx.x] && oi < cur))) {
          vbest[threadIdx.x] = ov; ibest[threadIdx.x] = oi;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      int sel = ibest[0];
      dSel[picked] = sel;
      if (sel >= 0 && sel < 4096) taken[sel] = 1;
    }
    __syncthreads();
    picked++;
  }
  if (threadIdx.x == 0) *dCount = picked;
}

// DSA_TOPK_MAX_KEYS is the causal-window cap k_dsa_index_topk's shared `taken[]` mask can cover. A
// decode step's nKeys is the causal window (one position's history), well under this for every context
// the device path serves today; but to keep the boundary HONEST rather than silently degrading, the
// host wrapper refuses (sentinel -1 → caller falls back to the host f64 loop) when nKeys exceeds it,
// instead of running a topk whose un-maskable tail could re-select a position. Must match the literal
// shared-array size in k_dsa_index_topk.
#define DSA_TOPK_MAX_KEYS 4096

// fcuda_dsa_index_select_f32 scores all keys (k_dsa_index_score, f64-accumulated) then selects the
// top-k positions (k_dsa_index_topk) on the device, copying back only the small index list. The
// f64 score scratch is allocated internally (sized [nKeys] doubles) so the caller never sees the
// double dtype. Returns the number of positions written into host outIdx (= min(topK, #valid keys)),
// or -1 to DECLINE (nKeys past the shared-mem top-k cap) so the caller keeps the host selection.
extern "C" int fcuda_dsa_index_select_f32(const float *dIndexQ, const float *dIndexK,
                                          const float *dWeights, int nKeys, int nH,
                                          int indexDim, int queryPos, int topK, float scale,
                                          int *outIdx) {
  if (nKeys <= 0 || topK <= 0) return 0;
  if (nKeys > DSA_TOPK_MAX_KEYS) return -1; // window past the maskable top-k tail: decline, host falls back
  int threads = 256;
  double *dScores = (double *)fcuda_malloc(sizeof(double) * (size_t)nKeys);
  k_dsa_index_score<<<nKeys, threads, 0, g_stream>>>(dIndexQ, dIndexK, dWeights,
                                                     dScores, nKeys, nH, indexDim,
                                                     queryPos, scale);
  int *dSel = (int *)fcuda_malloc(sizeof(int) * (size_t)topK);
  int *dCount = (int *)fcuda_malloc(sizeof(int));
  k_dsa_index_topk<<<1, threads, 0, g_stream>>>(dScores, nKeys, queryPos, topK, dSel, dCount);
  int hCount = 0;
  CK(cudaMemcpy(&hCount, dCount, sizeof(int), cudaMemcpyDeviceToHost));
  if (hCount < 0) hCount = 0;
  if (hCount > topK) hCount = topK;
  if (hCount > 0) {
    CK(cudaMemcpy(outIdx, dSel, sizeof(int) * (size_t)hCount, cudaMemcpyDeviceToHost));
  }
  g_host_bytes += sizeof(int) * (size_t)(hCount + 1); // only the index list crosses host-ward
  fcuda_free(dScores);
  fcuda_free(dSel);
  fcuda_free(dCount);
  return hCount;
}

// ---- argmax (first index of the max value — cpuref tie-break) -------------------
__global__ void k_argmax(const float *L, int n, int *outIdx) {
  __shared__ float vbest[256];
  __shared__ int   ibest[256];
  float bv = -1e30f; int bi = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = L[i];
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  vbest[threadIdx.x] = bv; ibest[threadIdx.x] = bi; __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      float ov = vbest[threadIdx.x + s]; int oi = ibest[threadIdx.x + s];
      if (ov > vbest[threadIdx.x] || (ov == vbest[threadIdx.x] && oi < ibest[threadIdx.x])) {
        vbest[threadIdx.x] = ov; ibest[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *outIdx = ibest[0];
}
// ---- CUDA graph capture/replay (collapse ~600 calls/token -> 1 launch) -----------
// The decode op-graph has IDENTICAL topology every token (same ops); only kernel/memcpy
// PARAMS change (the RoPE/attention position, nPos, the KV-append destination offset). So
// we instantiate ONCE and, every later token, cudaGraphExecUpdate the kept exec with the
// freshly-captured graph — patching just those params, NOT recompiling. That removes the
// per-token instantiate cost that made naive per-token capture a no-win.
static cudaGraphExec_t g_exec = nullptr;

extern "C" void fcuda_graph_reset(void) {
  if (g_exec) { cudaGraphExecDestroy(g_exec); g_exec = nullptr; }
}

// fcuda_graph_abort ends an open capture and throws the captured graph away WITHOUT launching
// it — the recovery half of fcuda_graph_begin for the case where the Go side unwound (panicked)
// mid-capture. Without this the stream stays in capture mode and every subsequent op fails with
// "operation not permitted while the stream is capturing", cascading the whole serve. Clearing
// any sticky error keeps the context usable for the next request. Best-effort: a failed
// EndCapture still leaves cudaGetLastError cleared so the next op is not poisoned by it.
extern "C" void fcuda_graph_abort(void) {
  cudaGraph_t graph = nullptr;
  g_capture_open = false; // #969: capture torn down; the dalloc guard is disarmed again
  cudaStreamEndCapture(g_stream, &graph);
  if (graph) cudaGraphDestroy(graph);
  cudaGetLastError(); // swallow the capture-state error so the next op starts clean
}

extern "C" int fcuda_graph_begin(void) {
  // Global mode: capture every op submitted to g_stream regardless of thread (the Go
  // caller LockOSThread-pins the token, and cudaMu serializes, so nothing else submits).
  if (cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeGlobal) != cudaSuccess) return 1;
  g_capture_open = true; // #969: arm the dalloc guard for the captured region
  return 0;
}
extern "C" int fcuda_graph_end_launch(void) {
  cudaGraph_t graph = nullptr;
  g_capture_open = false; // #969: capture closed; device allocation is permitted again
  cudaError_t e = cudaStreamEndCapture(g_stream, &graph);
  if (e != cudaSuccess || !graph) { fprintf(stderr, "fak-cuda: EndCapture: %s\n", cudaGetErrorString(e)); return 1; }
  if (g_exec == nullptr) {
    e = cudaGraphInstantiate(&g_exec, graph, 0);
    if (e != cudaSuccess) { fprintf(stderr, "fak-cuda: Instantiate: %s\n", cudaGetErrorString(e)); cudaGraphDestroy(graph); g_exec = nullptr; return 2; }
  } else {
    cudaGraphExecUpdateResultInfo info;
    e = cudaGraphExecUpdate(g_exec, graph, &info);
    if (e != cudaSuccess) {
      // topology drifted (shouldn't, but be safe): re-instantiate from scratch.
      static int warned = 0;
      if (warned < 3) { fprintf(stderr, "fak-cuda: ExecUpdate failed: %s (result=%d) -> re-instantiate\n", cudaGetErrorString(e), (int)info.result); warned++; }
      cudaGetLastError();
      cudaGraphExecDestroy(g_exec);
      e = cudaGraphInstantiate(&g_exec, graph, 0);
      if (e != cudaSuccess) { fprintf(stderr, "fak-cuda: Re-instantiate: %s\n", cudaGetErrorString(e)); cudaGraphDestroy(graph); g_exec = nullptr; return 2; }
    }
  }
  int rc = 0;
  e = cudaGraphLaunch(g_exec, g_stream);
  if (e != cudaSuccess) { fprintf(stderr, "fak-cuda: Launch: %s\n", cudaGetErrorString(e)); rc = 3; }
  CK(cudaStreamSynchronize(g_stream));
  cudaGraphDestroy(graph); // keep g_exec for the next token's ExecUpdate
  return rc;
}

// Persistent argmax index buffer (#969), grown ONCE and reused, exactly like g_attn_scratch
// above. The previous code did fcuda_malloc(sizeof(int)) + fcuda_free per call; the pooled
// allocator misses (-> cudaMalloc) on the first argmax of any size class, and CUDA forbids
// cudaMalloc while g_stream is mid-capture ("operation not permitted when stream is capturing").
// Since argmax is the final op of the captured decode graph, that lazy alloc crashed graph
// capture/replay. A single persistent device int touched on every token never allocates inside
// capture. ensure_argmax_idx() runs on the FIRST (uncaptured) fcuda_argmax_f32 call — the same
// pre-warm role the explicit cuBLAS workspace prealloc in fcuda_init plays for the GEMM scratch
// and the uncaptured halLogitsWarm step (internal/model/hal.go) plays for every pooled op output —
// so by the time the first captured token runs, this buffer is already resident. There is no
// separate fcuda_graph_warm() entry point; warming is exactly these three pre-capture allocations,
// and the #969 g_capture_open guard in fcuda_malloc fails loud if any scratch was missed.
static int *g_argmax_idx = nullptr;
static void ensure_argmax_idx(void) {
  if (!g_argmax_idx) CK(cudaMalloc(&g_argmax_idx, sizeof(int)));
}

extern "C" int fcuda_argmax_f32(const float *dLogits, int n) {
  int hIdx = 0;
  ensure_argmax_idx(); // no-op after the first call; never allocates inside a capture
  k_argmax<<<1, 256, 0, g_stream>>>(dLogits, n, g_argmax_idx);
  CK(cudaMemcpy(&hIdx, g_argmax_idx, sizeof(int), cudaMemcpyDeviceToHost));
  g_host_bytes += sizeof(int); // only the token id crosses host-ward — the #482 witness
  return hIdx;
}

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
  k_awq_gemv<<<out, threads, 0, g_stream>>>(dW, dScales, dX, dY, out, in);
}

// fcuda_awq_gemm: Y[P, out] = X[P, in] @ AWQ[out, in]^T
extern "C" void fcuda_awq_gemm(const uint8_t *dW, const float *dScales, const float *dX, float *dY, int out, int in, int P) {
  dim3 threads(256, 1);
  dim3 blocks(out, P);
  k_awq_gemm<<<blocks, threads, 0, g_stream>>>(dW, dScales, dX, dY, out, in, P);
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
  k_gptq_gemv<<<out, threads, 0, g_stream>>>(dW, dZeros, dScales, dGidx, dX, dY,
                                             out, in, bits, groupSize, nGroups);
}
