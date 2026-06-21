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

extern "C" void *fcuda_malloc(size_t bytes) {
  if (bytes == 0) bytes = 1;
  auto it = g_pool.find(bytes);
  void *d = nullptr;
  if (it != g_pool.end() && !it->second.empty()) {
    d = it->second.back();
    it->second.pop_back();
  } else {
    CK(cudaMalloc(&d, bytes));
  }
  g_live[d] = bytes;
  return d;
}
extern "C" void fcuda_free(void *d) {
  if (!d) return;
  auto it = g_live.find(d);
  if (it != g_live.end()) {
    g_pool[it->second].push_back(d); // return to the pool for reuse, don't cudaFree
    g_live.erase(it);
  } else {
    cudaFree(d);
  }
}
extern "C" void fcuda_h2d(void *d, const void *h, size_t n) { CK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice)); }
extern "C" void fcuda_d2h(void *h, const void *d, size_t n) { CK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost)); }
// Device-to-device copies stay on the default stream but are ASYNC w.r.t. the host: a
// synchronous cudaMemcpy fences the whole device, and RoPE + every KV append issues one,
// so a 30-layer decode paid ~150 full device syncs per token (catastrophic on WSL, where a
// sync is ~1-2 ms). Stream ordering still serializes them against the kernels correctly;
// the only host fence we keep is the final logits d2h in Read.
extern "C" void fcuda_d2d(void *dst, const void *src, size_t n) { CK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, g_stream)); }
extern "C" void fcuda_sync(void) { CK(cudaDeviceSynchronize()); }

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

// ---- Decode attention: one block per query head ---------------------------------
// q[nH*hd]; K,V [nPos, nKV*hd]; out[nH*hd]; scores scratch [nH*nPos].
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
