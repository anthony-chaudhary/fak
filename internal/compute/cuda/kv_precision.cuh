// Native CUDA KV storage: F32 shadow, F16/BF16 attended planes. No host round trip.
#include "../cuda_kv_precision.h"
__device__ float kv_load(const void *p, size_t i, int dtype) {
  if (dtype == 1) return __half2float(((const __half *)p)[i]);
  if (dtype == 2) return __uint_as_float(((unsigned)((const unsigned short *)p)[i]) << 16);
  return ((const float *)p)[i];
}
__global__ void k_kv_store_typed(void *dst, const float *src, int offset, int n, int dtype) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = src[i];
  if (dtype == 1) ((__half *)dst)[(size_t)offset+i] = __float2half_rn(x);
  else if (dtype == 2) {
    unsigned bits = __float_as_uint(x);
    // Round ties to even; preserve NaNs rather than rounding them to infinity.
    unsigned short b = ((bits & 0x7fffffffU) > 0x7f800000U) ? (unsigned short)((bits >> 16) | 0x40U) : (unsigned short)((bits + 0x7fffU + ((bits >> 16)&1U)) >> 16);
    ((unsigned short *)dst)[(size_t)offset+i] = b;
  } else ((float *)dst)[(size_t)offset+i] = x;
}
__global__ void k_kv_load_typed(float *dst, const void *src, int n, int dtype) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = kv_load(src, i, dtype);
}
extern "C" int fcuda_kv_store_typed(void *dst, const float *src, int offset, int n, int dtype) {
  if (!dst || !src || offset < 0 || n <= 0 || dtype < 0 || dtype > 2 || (size_t)offset+(size_t)n > 2147483647ULL) return (int)cudaErrorInvalidValue;
  k_kv_store_typed<<<((size_t)n+255)/256,256,0,g_stream>>>(dst,src,offset,n,dtype);
  return (int)cudaGetLastError();
}
extern "C" int fcuda_kv_load_typed(float *dst, const void *src, int n, int dtype) {
  if (!dst || !src || n <= 0 || dtype < 0 || dtype > 2) return (int)cudaErrorInvalidValue;
  k_kv_load_typed<<<((size_t)n+255)/256,256,0,g_stream>>>(dst,src,n,dtype);
  return (int)cudaGetLastError();
}
__global__ void k_kv_attention_typed(
    const float *Q, const void *K, const void *V, float *Out,
    int tokens, int prefix, int nH, int nKV, int hd, float scale, int dtype) {
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
    size_t row = (size_t)j * width + (size_t)kvh * hd;
    float partial = 0.f;
    for (int d = tid; d < hd; d += FLASH_THREADS) partial += qs[d] * kv_load(K, row+d, dtype);
    red[tid] = partial;
    __syncthreads();
    for (int s = FLASH_THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float score = red[0] * scale;
    __syncthreads();
    if (j == 0) {
      m = score;
      l = 1.f;
      int k = 0;
      for (int d = tid; d < hd; d += FLASH_THREADS, k++) acc[k] = kv_load(V, row+d, dtype);
      continue;
    }
    float nextM = fmaxf(m, score);
    float correction = expf(m - nextM), probability = expf(score - nextM);
    l = l * correction + probability;
    int k = 0;
    for (int d = tid; d < hd; d += FLASH_THREADS, k++) {
      acc[k] = acc[k] * correction + probability * kv_load(V, row+d, dtype);
    }
    m = nextM;
  }
  float invL = l > 0.f ? 1.f / l : 0.f;
  int k = 0;
  for (int d = tid; d < hd; d += FLASH_THREADS, k++) {
    Out[(size_t)token * nH * hd + (size_t)h * hd + d] = acc[k] * invL;
  }
}

extern "C" int fcuda_kv_attention_typed(const float *q, const void *k, const void *v, float *out, int tokens, int prefix, int heads, int kvheads, int hd, float scale, int dtype) {
  if (!q || !k || !v || !out || tokens <= 0 || (prefix < 0 && !(prefix == -1 && tokens == 1)) || heads <= 0 || kvheads <= 0 || heads % kvheads || hd <= 0 || hd > FLASH_THREADS*FLASH_ACC_MAX || dtype < 0 || dtype > 2 || (size_t)tokens*(size_t)heads > 2147483647ULL || (long long)prefix+tokens > 2147483647LL || (size_t)kvheads*(size_t)hd > 2147483647ULL) return (int)cudaErrorInvalidValue;
  k_kv_attention_typed<<<tokens*heads,FLASH_THREADS,((size_t)hd+FLASH_THREADS)*sizeof(float),g_stream>>>(q,k,v,out,tokens,prefix,heads,kvheads,hd,scale,dtype);
  return (int)cudaGetLastError();
}
