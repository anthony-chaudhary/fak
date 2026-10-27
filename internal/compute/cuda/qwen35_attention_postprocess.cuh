// Private Qwen dense prefill seam. Arithmetic is adapted ONLY from this repo's
// k_rmsnorm and k_qwen35_partial_rope_panel in transformer_ops.cuh at
// 133fc57c9bf4f209390031ed31833a420d87faf6 (Apache-2.0). No new numerical algorithm.
// One block per Q head, K head, or V-copy head; exactly 256 threads per block.
#include <stdint.h>

static unsigned long long g_qwen35_attention_postprocess_launches = 0;

__global__ void k_qwen35_attention_postprocess(
    const float *Q, const float *K, const float *V,
    const float *QW, const float *KW, float *QOut,
    float *KRawCache, float *KCache, float *VCache,
    int start, int rotary, float eps, double theta) {
  const int token = (int)blockIdx.x / 32;
  const int group = (int)blockIdx.x % 32;
  const int dim = (int)threadIdx.x;
  if (group >= 28) {
    const size_t src = (size_t)token * 1024 + (group - 28) * 256 + dim;
    const size_t dst = (size_t)(start + token) * 1024 + (group - 28) * 256 + dim;
    VCache[dst] = V[src];
    return; // Whole CTA takes this branch; no barrier divergence.
  }
  const bool isQ = group < 24;
  const int head = isQ ? group : group - 24;
  const int width = isQ ? 6144 : 1024;
  const size_t base = (size_t)token * width + head * 256;
  const float *x = (isQ ? Q : K) + base;
  const float *weight = isQ ? QW : KW;
  __shared__ float red[256];
  // Explicit f32 store/load boundary replaces the prior normalized global
  // tensor. Volatile prevents contraction through this inter-kernel boundary.
  __shared__ volatile float normalized[256];
  float local = 0.f;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) local += x[i] * x[i];
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)256 + eps);
  for (int i = threadIdx.x; i < 256; i += blockDim.x) normalized[i] = x[i] * inv * weight[i];
  __syncthreads();
  const size_t dst = isQ ? base + dim : (size_t)(start + token) * 1024 + head * 256 + dim;
  float *out = isQ ? QOut : KCache;
  if (!isQ) KRawCache[dst] = normalized[dim];
  if (dim >= rotary) { out[dst] = normalized[dim]; return; }
  int half = rotary / 2;
  int j = dim < half ? dim : dim - half;
  double freq = pow(theta, -(2.0 * j) / rotary);
  double angle = (double)(start + token) * freq;
  float cs = (float)cos(angle), sn = (float)sin(angle);
  float a = normalized[j], b = normalized[j + half];
  out[dst] = dim < half ? a * cs - b * sn : b * cs + a * sn;
}

// 1 supported, 0 known unsupported, negative runtime status. No enqueue or
// allocation here: the caller can retain the original composition on a decline.
extern "C" int fak_qwen35_attention_postprocess_capability(int tokens) {
  if (tokens <= 0 || tokens > INT_MAX / 6144 || g_capture_open) return 0;
  int device = -1;
  cudaError_t e = cudaGetDevice(&device);
  if (e != cudaSuccess) return -(int)e;
  if (device != 0) return 0; // Existing sequence tensors are strict device zero.
  cudaStreamCaptureStatus capture;
  e = cudaStreamIsCapturing(g_stream, &capture);
  if (e != cudaSuccess) return -(int)e;
  if (capture != cudaStreamCaptureStatusNone) return 0;
  cudaDeviceProp prop;
  e = cudaGetDeviceProperties(&prop, device);
  if (e != cudaSuccess) return -(int)e;
  return prop.maxThreadsPerBlock >= 256 && prop.maxThreadsDim[0] >= 256 &&
      prop.sharedMemPerBlock >= 2048 && (size_t)tokens * 32 <= (size_t)prop.maxGridSize[0];
}

static bool qwen35_postprocess_extent(const void *p, size_t bytes) {
  if (!p || bytes == 0 || (uintptr_t)p > UINTPTR_MAX - bytes) return false;
  auto it = g_live.find((void *)p);
  if (it != g_live.end()) return it->second >= bytes;
  auto panel = g_panel_live.find((void *)p);
  return panel != g_panel_live.end() && panel->second >= bytes;
}

extern "C" int fak_qwen35_attention_postprocess_f32(
    const float *Q, const float *K, const float *V,
    const float *QW, const float *KW, float *QOut,
    float *KRawCache, float *KCache, float *VCache,
    int tokens, int start, int rotary, float eps, double theta) {
  if (tokens <= 0 || tokens > INT_MAX / 6144 || start < 0 ||
      start > INT_MAX / 1024 - tokens || rotary <= 0 || rotary > 256 ||
      (rotary & 1) || !(eps > 0.f) || !isfinite(eps) ||
      !(theta > 0.0) || !isfinite(theta)) return -1;
  const int supported = fak_qwen35_attention_postprocess_capability(tokens);
  if (supported != 1) return supported < 0 ? 84000 - supported : -2;
  const size_t qb = (size_t)tokens * 6144 * sizeof(float);
  const size_t kb = (size_t)tokens * 1024 * sizeof(float);
  const size_t cacheBytes = (size_t)(start + tokens) * 1024 * sizeof(float);
  const void *ptrs[9] = {Q, K, V, QW, KW, QOut, KRawCache, KCache, VCache};
  const size_t bytes[9] = {qb, kb, kb, 256*sizeof(float), 256*sizeof(float), qb, cacheBytes, cacheBytes, cacheBytes};
  for (int i = 0; i < 9; ++i) {
    if (!qwen35_postprocess_extent(ptrs[i], bytes[i])) return -3;
    cudaPointerAttributes attrs;
    const cudaError_t e = cudaPointerGetAttributes(&attrs, ptrs[i]);
    if (e != cudaSuccess) return 87000 + (int)e;
    if (attrs.type != cudaMemoryTypeDevice || attrs.device != 0) return -5;
  }
  // Read-only operands may alias each other, but no output may alias any input
  // or another output, including a cache prefix. All are exact allocation bases.
  for (int out = 5; out < 9; ++out) {
    for (int other = 0; other < out; ++other) {
      const uintptr_t a = (uintptr_t)ptrs[out], b = (uintptr_t)ptrs[other];
      if (a < b + bytes[other] && b < a + bytes[out]) return -4;
    }
  }
  // Never erase a prior asynchronous error and reinterpret it as a decline.
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return 85000 + (int)e;
  FCUDA_TS("k_qwen35_attention_postprocess", tokens * 32,
      k_qwen35_attention_postprocess<<<tokens * 32, 256, 0, g_stream>>>(
          Q, K, V, QW, KW, QOut, KRawCache, KCache, VCache, start, rotary, eps, theta));
  e = cudaGetLastError();
  if (e != cudaSuccess) return 86000 + (int)e;
  ++g_qwen35_attention_postprocess_launches; // Enqueue, NOT completion witness.
  return 0;
}

extern "C" unsigned long long fak_qwen35_attention_postprocess_launch_count(void) {
  return g_qwen35_attention_postprocess_launches;
}
