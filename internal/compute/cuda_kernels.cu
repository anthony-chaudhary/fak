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
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unordered_map>
#include <vector>

static cublasHandle_t g_blas = nullptr;
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

// host-transfer witness (#482): cumulative device->host bytes. Every d2h copy adds to it — the
// Read fence (fcuda_d2h) adds the vector bytes, the single token-id copy in fcuda_argmax_f32
// adds sizeof(int) — so the Go-side async test can prove a greedy step pulls only the argmax id
// host-ward, not the full logits vector. Monotonic; the test resets it around each step.
static size_t g_host_bytes = 0;

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
  return 0;
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
extern "C" void fcuda_free(void *d) {
  if (!d) return;
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
extern "C" void fcuda_h2d(void *d, const void *h, size_t n) { CK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice)); }
extern "C" void fcuda_d2h(void *h, const void *d, size_t n) { CK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost)); g_host_bytes += n; }
// Device-to-device copies stay on the default stream but are ASYNC w.r.t. the host: a
// synchronous cudaMemcpy fences the whole device, and RoPE + every KV append issues one,
// so a 30-layer decode paid ~150 full device syncs per token (catastrophic on WSL, where a
// sync is ~1-2 ms). Stream ordering still serializes them against the kernels correctly;
// the only host fence we keep is the final logits d2h in Read.
extern "C" void fcuda_d2d(void *dst, const void *src, size_t n) { CK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, g_stream)); }
extern "C" void fcuda_sync(void) { CK(cudaDeviceSynchronize()); }

// async host-transfer witness accessors (#482): see g_host_bytes above.
extern "C" size_t fcuda_hostxfer_bytes(void) { return g_host_bytes; }
extern "C" void fcuda_hostxfer_reset(void) { g_host_bytes = 0; }

// y[P,out] = x[P,in] @ W[out,in]^T, all row-major. Column-major cuBLAS recipe:
// treat row-major W[out,in] as col-major [in,out] (op=T), row-major X[P,in] as col-major
// [in,P] (op=N); the col-major out[out,P] result IS row-major Y[P,out]. Verified by index.
extern "C" void fcuda_matmul_f32(const float *dW, const float *dX, float *dY, int out, int in, int P) {
  const float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N,
              out, P, in,
              &alpha,
              dW, in,
              dX, in,
              &beta,
              dY, out);
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

// k_q8_gemm: Y[t,o] = Σ_b (Σ_i qW[o,b,i]·qX[t,b,i]) · dW[o,b] · dX[t,b]. One block per (o,t);
// threads stride the blocks, each forms its block's integer dot (int32) then scales by the
// weight·activation block scales — the same per-block scheme as cpuref qdot8scalar (only the
// reduction order differs, which is what makes the lane Approx, not Reference).
__global__ void k_q8_gemm(const signed char *W, const float *Wscale,
                          const signed char *qX, const float *xScale,
                          float *Y, int out, int in, int P, int block) {
  int o = blockIdx.x, t = blockIdx.y;
  if (o >= out || t >= P) return;
  int nblk = in / block;
  const signed char *wrow = W + (size_t)o * in;
  const float *wsc = Wscale + (size_t)o * nblk;
  const signed char *xrow = qX + (size_t)t * in;
  const float *xsc = xScale + (size_t)t * nblk;
  __shared__ float red[256];
  float local = 0.f;
  for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
    const signed char *wb = wrow + (size_t)b * block;
    const signed char *xb = xrow + (size_t)b * block;
    int acc = 0;
    for (int i = 0; i < block; i++) acc += (int)wb[i] * (int)xb[i];
    local += (float)acc * wsc[b] * xsc[b];
  }
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) Y[(size_t)t * out + o] = red[0];
}

extern "C" void fcuda_q8_matmul_f32(const int8_t *dCodes, const float *dScales, const float *dX,
                                    float *dY, int out, int in, int P, int block) {
  int nblk = in / block;
  signed char *qX = (signed char *)fcuda_malloc((size_t)P * in);          // int8 activation codes
  float *xScale = (float *)fcuda_malloc((size_t)P * nblk * sizeof(float)); // per-block act scales
  k_q8_quant_act<<<dim3(nblk, P), 64, 0, g_stream>>>(dX, qX, xScale, P, in, block);
  k_q8_gemm<<<dim3(out, P), 256, 0, g_stream>>>((const signed char *)dCodes, dScales, qX, xScale,
                                                dY, out, in, P, block);
  fcuda_free(qX);
  fcuda_free(xScale); // pooled, stream-ordered: the GEMM reads them before any reuse
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

// k_q4k_gemm: Y[t,o] = Σ over the row's Q4_K super-blocks of the dequant-fused dot. Each 256-elem
// super-block is 144 bytes — f16 d (0..1), f16 dmin (2..3), 12 packed sub-scale bytes (4..15), 128
// 4-bit code bytes (16..143). The weight is dequantized FUSED into the tile: w = d·scale·code −
// dmin·min, with (scale,min) per 32-elem sub-block from getScaleMinK4_dev — exactly the GGUF
// loader's dequantQ4K — and dotted with the f32 activation, F32 accumulate. No activation quant on
// this path (the weight, not the activation, is the narrow operand). One block per (o,t); threads
// stride the super-blocks.
__global__ void k_q4k_gemm(const unsigned char *Q4K, const float *X, float *Y, int out, int in, int P) {
  int o = blockIdx.x, t = blockIdx.y;
  if (o >= out || t >= P) return;
  int nsb = in / 256; // super-blocks per row
  const unsigned char *wrow = Q4K + (size_t)o * nsb * 144;
  const float *xrow = X + (size_t)t * in;
  __shared__ float red[256];
  float local = 0.f;
  for (int sb = threadIdx.x; sb < nsb; sb += blockDim.x) {
    const unsigned char *blk = wrow + (size_t)sb * 144;
    float d = __half2float(*(const __half *)(blk));
    float dmin = __half2float(*(const __half *)(blk + 2));
    const unsigned char *scales = blk + 4;  // 12 packed sub-scale bytes
    const unsigned char *q = blk + 16;      // 128 4-bit code bytes
    const float *xb = xrow + (size_t)sb * 256;
    int qi = 0, is = 0;
    float acc = 0.f;
    for (int j = 0; j < 256; j += 64) {
      unsigned char sc, mn;
      getScaleMinK4_dev(is, scales, &sc, &mn);
      float d1 = d * sc, m1 = dmin * mn;
      getScaleMinK4_dev(is + 1, scales, &sc, &mn);
      float d2 = d * sc, m2 = dmin * mn;
      for (int l = 0; l < 32; l++) {
        float w = d1 * (float)(q[qi + l] & 0x0f) - m1;
        acc += w * xb[j + l];
      }
      for (int l = 0; l < 32; l++) {
        float w = d2 * (float)(q[qi + l] >> 4) - m2;
        acc += w * xb[j + 32 + l];
      }
      qi += 32;
      is += 2;
    }
    local += acc;
  }
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) Y[(size_t)t * out + o] = red[0];
}

extern "C" void fcuda_q4k_matmul_f32(const uint8_t *dQ4K, const float *dX, float *dY,
                                     int out, int in, int P) {
  k_q4k_gemm<<<dim3(out, P), 256, 0, g_stream>>>((const unsigned char *)dQ4K, dX, dY, out, in, P);
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
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dstBase[offset + i] = src[i];
}
extern "C" void fcuda_kv_write(float *dstBase, const float *src, int offset, int n) {
  k_copyrow<<<(n + 255) / 256, 256, 0, g_stream>>>(dstBase, src, offset, n);
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

extern "C" int fcuda_graph_begin(void) {
  // Global mode: capture every op submitted to g_stream regardless of thread (the Go
  // caller LockOSThread-pins the token, and cudaMu serializes, so nothing else submits).
  return cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeGlobal) == cudaSuccess ? 0 : 1;
}
extern "C" int fcuda_graph_end_launch(void) {
  cudaGraph_t graph = nullptr;
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

extern "C" int fcuda_argmax_f32(const float *dLogits, int n) {
  int hIdx = 0;
  int *dIdx = (int *)fcuda_malloc(sizeof(int));
  k_argmax<<<1, 256, 0, g_stream>>>(dLogits, n, dIdx);
  CK(cudaMemcpy(&hIdx, dIdx, sizeof(int), cudaMemcpyDeviceToHost));
  g_host_bytes += sizeof(int); // only the token id crosses host-ward — the #482 witness
  fcuda_free(dIdx);
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
