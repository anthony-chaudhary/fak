// Native CUDA KV storage: F32 shadow, F16/BF16 attended planes. No host round trip.
#include "../cuda_kv_precision.h"
#include <stdlib.h>
#include <string.h>
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
  FCUDA_TS("k_kv_store_typed", ((size_t)n+255)/256, k_kv_store_typed<<<((size_t)n+255)/256,256,0,g_stream>>>(dst,src,offset,n,dtype));
  return (int)cudaGetLastError();
}
extern "C" int fcuda_kv_load_typed(float *dst, const void *src, int n, int dtype) {
  if (!dst || !src || n <= 0 || dtype < 0 || dtype > 2) return (int)cudaErrorInvalidValue;
  FCUDA_TS("k_kv_load_typed", ((size_t)n+255)/256, k_kv_load_typed<<<((size_t)n+255)/256,256,0,g_stream>>>(dst,src,n,dtype));
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

// Four query warps share each widened K/V tile. The final partial query group
// must still participate in both block barriers: inactive warps help load KV,
// but never read a query or publish an output. Each warp's causal limit is
// independent, including when a prefix ends inside a shared tile.
__global__ void k_kv_attention_prompt4_typed(
    const float *Q, const void *K, const void *V, float *Out,
    int tokens, int prefix, int nH, int nKV, int hd, float scale, int dtype) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int first = (int)blockIdx.x * 4;
  const int token = first + warp, h = (int)blockIdx.y;
  const bool active = token < tokens;
  const int kvh = h / (nH / nKV), width = nKV * hd;
  const int end = prefix + (tokens - first < 4 ? tokens : first + 4);
  const int limit = active ? prefix + token + 1 : 0;
  float q[8], acc[8];
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    const int col = lane + d * 32;
    q[d] = active && col < hd ? Q[((size_t)token*nH+h)*hd+col] : 0.f;
    acc[d] = 0.f;
  }
  extern __shared__ float tile[];
  float *keys = tile, *values = tile + 8*hd;
  float m = 0.f, l = 0.f;
  for (int begin = 0; begin < end;) {
    const int rows = end - begin < 8 ? end - begin : 8;
    for (int i = threadIdx.x; i < rows*hd; i += 128) {
      const size_t src = (size_t)(begin+i/hd)*width+(size_t)kvh*hd+i%hd;
      keys[i] = kv_load(K, src, dtype);
      values[i] = kv_load(V, src, dtype);
    }
    __syncthreads();
    if (active) {
      for (int row = 0; row < rows && begin+row < limit; ++row) {
        float score = 0.f;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
          const int col = lane + d*32;
          if (col < hd) score += q[d]*keys[row*hd+col];
        }
#pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1)
          score += __shfl_down_sync(0xffffffffU, score, stride);
        score = __shfl_sync(0xffffffffU, score, 0)*scale;
        const float nextM = l == 0.f ? score : fmaxf(m, score);
        const float correction = l == 0.f ? 0.f : expf(m-nextM);
        const float probability = expf(score-nextM);
        l = l*correction+probability;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
          const int col = lane + d*32;
          if (col < hd) acc[d] = acc[d]*correction+probability*values[row*hd+col];
        }
        m = nextM;
      }
    }
    __syncthreads();
    begin += rows;
  }
  if (active) {
    const float invL = l > 0.f ? 1.f/l : 0.f;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      const int col = lane + d*32;
      if (col < hd) Out[((size_t)token*nH+h)*hd+col] = acc[d]*invL;
    }
  }
}

extern "C" int fcuda_kv_attention_typed(const float *q, const void *k, const void *v, float *out, int tokens, int prefix, int heads, int kvheads, int hd, float scale, int dtype) {
  if (!q || !k || !v || !out || tokens <= 0 || (prefix < 0 && !(prefix == -1 && tokens == 1)) || heads <= 0 || kvheads <= 0 || heads % kvheads || hd <= 0 || hd > FLASH_THREADS*FLASH_ACC_MAX || dtype < 0 || dtype > 2 || (size_t)tokens*(size_t)heads > 2147483647ULL || (long long)prefix+tokens > 2147483647LL || (size_t)kvheads*(size_t)hd > 2147483647ULL) return (int)cudaErrorInvalidValue;
  // The different dot reduction needs physical qualification before promotion.
  // Decode, F32 and larger heads retain their current implementation. Read the
  // opt-in on the host; CUDA graph replay retains the kernel chosen at capture.
  const char *prefill = getenv("FAK_CUDA_KV16_PREFILL");
  if (prefill && strcmp(prefill, "tile4") == 0 && dtype != 0 && tokens >= 4 && hd <= 256 && heads <= 65535) {
    const dim3 grid((unsigned)((tokens+3ULL)/4), (unsigned)heads);
    FCUDA_TS("k_kv_attention_prompt4_typed", grid, k_kv_attention_prompt4_typed<<<grid,128,(size_t)16*hd*sizeof(float),g_stream>>>(q,k,v,out,tokens,prefix,heads,kvheads,hd,scale,dtype));
    return (int)cudaGetLastError();
  }
  FCUDA_TS("k_kv_attention_typed", tokens*heads, k_kv_attention_typed<<<tokens*heads,FLASH_THREADS,((size_t)hd+FLASH_THREADS)*sizeof(float),g_stream>>>(q,k,v,out,tokens,prefix,heads,kvheads,hd,scale,dtype));
  return (int)cudaGetLastError();
}
