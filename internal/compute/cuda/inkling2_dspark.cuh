// inkling2_dspark.cuh: device ops of the Inkling-2 DSpark draft (the checkpoint's
// mtp head) beside the resident target forward. The draft runs its dense layers over
// one block of rows (anchor then masks) and attends to a position-indexed ring of
// context keys built from target features. Positions come from device memory so the
// draft and its context commit replay inside captured graphs.
// Included after inkling2_prefill.cuh by cuda_kernels.cu.

#include "inkling2_fp8.cuh"

#define INK2D_MAX_TAPS 4

struct Ink2DTaps {
  const float *f[INK2D_MAX_TAPS];
  float a[INK2D_MAX_TAPS];
};

__device__ __forceinline__ double ink2d_block_sum(double v, double *sh) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = (blockDim.x + 31) >> 5;
  for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xffffffff, v, d);
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  double t = 0;
  for (int i = 0; i < nwarp; ++i) t += sh[i];
  __syncthreads();
  return t;
}

// out[row] = rms(sum_t a[t] * f_t[row]) * w, one block per row.
__global__ void k_ink2d_fuse(Ink2DTaps tp, int taps, const float *w, float *out, int n, float eps) {
  ink2_pdl_enter();
  __shared__ double sh[32];
  const size_t base = (size_t)blockIdx.x * n;
  double ss = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = 0.f;
    for (int t = 0; t < taps; ++t) v += tp.a[t] * tp.f[t][base + i];
    out[base + i] = v;
    ss += (double)v * (double)v;
  }
  ss = ink2d_block_sum(ss, sh);
  const float s = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
  for (int i = threadIdx.x; i < n; i += blockDim.x) out[base + i] = out[base + i] * s * w[i];
}

__device__ __forceinline__ float ink2d_block_max(float v, float *sh) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = (blockDim.x + 31) >> 5;
  v = ink2_warp_max(v);
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  float mx = 0.f;
  for (int i = 0; i < nwarp; ++i) mx = fmaxf(mx, sh[i]);
  __syncthreads();
  return mx;
}

__device__ __forceinline__ void ink2d_quantize_fp8_head(int i, int dim, float val, uint8_t *dst, float *scDst,
                                                        int *overflow, float *shMax, int *shBad) {
  if (threadIdx.x == 0) *shBad = 0;
  __syncthreads();
  const bool active = i < dim;
  const bool finite = !active || isfinite(val);
  if (!finite) *shBad = 1;
  const float mag = (active && finite) ? fabsf(val) : 0.f;
  const float amax = ink2d_block_max(mag, shMax);
  if (*shBad != 0) {
    if (threadIdx.x == 0) {
      *scDst = 0.f;
      if (overflow) atomicOr(overflow, 1);
    }
    if (active) dst[i] = 0u;
    return;
  }
  if (amax < kInk2FP8MinAbsmax) {
    if (threadIdx.x == 0) *scDst = 0.f;
    if (active) dst[i] = 0u;
    return;
  }
  const float sc = amax / kInk2FP8Max;
  const float inv = kInk2FP8Max / amax;
  if (!isfinite(sc) || !isfinite(inv)) {
    if (threadIdx.x == 0) {
      *scDst = 0.f;
      if (overflow) atomicOr(overflow, 1);
    }
    if (active) dst[i] = 0u;
    return;
  }
  if (threadIdx.x == 0) *scDst = sc;
  if (active) dst[i] = ink2_e4m3fn_encode(val * inv);
}

// Per-head RMS norm (weight w over dim) then neox RoPE at position *posDev + row.
// ring writes row r to dst slot (pos % cap), else in row order; x may equal dst
// when not ring. Block (head, row), dim threads.
template <bool FP8>
__device__ __forceinline__ void ink2d_qk_body(const float *x, const float *w, void *dstVoid, float *scaleDst,
                                              int *overflow, int heads, int dim, const int *posDev, int cap,
                                              int ring, float eps, double theta) {
  __shared__ double sh[32];
  __shared__ float v[256];
  __shared__ float shMax[32];
  __shared__ int shBad;
  const int h = blockIdx.x, row = blockIdx.y, i = threadIdx.x;
  const int width = heads * dim;
  const int pos = *posDev + row;
  const float xi = x[(size_t)row * width + (size_t)h * dim + i];
  const double ss = ink2d_block_sum((double)xi * (double)xi, sh);
  const float s = (float)(1.0 / sqrt(ss / (double)dim + (double)eps));
  v[i] = xi * s * w[i];
  __syncthreads();
  const int half = dim / 2;
  const int j = i < half ? i : i - half;
  const double angle = (double)pos / pow(theta, (double)(2 * j) / (double)dim);
  const float c = (float)cos(angle), sn = (float)sin(angle);
  const float o = i < half ? v[i] * c - v[i + half] * sn : v[j] * sn + v[i] * c;
  const size_t slot = ring ? (size_t)(pos % cap) : (size_t)row;
  if constexpr (!FP8) {
    float *d = static_cast<float *>(dstVoid) + slot * width + (size_t)h * dim;
    d[i] = o;
  } else {
    uint8_t *d = static_cast<uint8_t *>(dstVoid) + slot * width + (size_t)h * dim;
    float *scDst = scaleDst + slot * heads + h;
    ink2d_quantize_fp8_head(i, dim, o, d, scDst, overflow, shMax, &shBad);
  }
}

__global__ void k_ink2d_qk(const float *x, const float *w, float *dst, int heads, int dim, const int *posDev, int cap,
                           int ring, float eps, double theta) {
  ink2_pdl_enter();
  ink2d_qk_body<false>(x, w, dst, nullptr, nullptr, heads, dim, posDev, cap, ring, eps, theta);
}

__global__ void k_ink2d_qk_fp8(const float *x, const float *w, uint8_t *dst, float *scale, int *overflow, int heads,
                               int dim, const int *posDev, int cap, int ring, float eps, double theta) {
  ink2_pdl_enter();
  ink2d_qk_body<true>(x, w, dst, scale, overflow, heads, dim, posDev, cap, ring, eps, theta);
}

// ring[(pos + row) % cap] = src[row] for rows rows of width floats.
__global__ void k_ink2d_ring_store(const float *src, float *ring, int width, const int *posDev, int cap) {
  ink2_pdl_enter();
  const int row = blockIdx.y;
  const int pos = *posDev + row;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += gridDim.x * blockDim.x)
    ring[(size_t)(pos % cap) * width + i] = src[(size_t)row * width + i];
}

// ring[(pos + row) % cap] = fp8(src[row]) per (row, kvHead), scale[(pos + row) % cap, kvHead] = absmax / 448.
// Block (kvHead, row), ((dim + 31) & ~31) threads.
__global__ void k_ink2d_ring_store_fp8(const float *src, uint8_t *ring, float *scale, int *overflow, int kvHeads,
                                       int dim, const int *posDev, int cap) {
  ink2_pdl_enter();
  __shared__ float shMax[32];
  __shared__ int shBad;
  const int kvh = blockIdx.x, row = blockIdx.y, i = threadIdx.x;
  const int pos = *posDev + row;
  const size_t slot = (size_t)(pos % cap);
  const int width = kvHeads * dim;
  const float val = i < dim ? src[(size_t)row * width + (size_t)kvh * dim + i] : 0.f;
  uint8_t *dst = ring + slot * width + (size_t)kvh * dim;
  float *scDst = scale + slot * kvHeads + kvh;
  ink2d_quantize_fp8_head(i, dim, val, dst, scDst, overflow, shMax, &shBad);
}

// Block attention of the draft: query row r (anchor position pos = *posDev, row at
// pos + r) sees the context ring positions [lo, pos) and every block key, without a
// causal mask inside the block. local layers keep context positions >= pos - window.
// Block (head, row), 128 threads; scores in dynamic shared memory.
template <bool FP8>
__device__ __forceinline__ void ink2d_attend_body(const float *q, const float *kb, const float *vb, const void *kcVoid,
                                                  const void *vcVoid, const float *ks, const float *vs, float *out,
                                                  int heads, int kvHeads, int dim, const int *posDev, int window,
                                                  int cap) {
  extern __shared__ float sc[];
  __shared__ float red[32];
  const int h = blockIdx.x, r = blockIdx.y;
  const int kvh = h / (heads / kvHeads);
  const int kvDim = kvHeads * dim;
  const int pos = *posDev;
  int lo = 0;
  if (window > 0 && pos - window > lo) lo = pos - window;
  if (pos - cap > lo) lo = pos - cap;
  const int nctx = pos - lo;
  const int n = nctx + gridDim.y;
  const float *qh = q + (size_t)r * heads * dim + (size_t)h * dim;
  const float scale = rsqrtf((float)dim);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
  for (int j = warp; j < n; j += nwarp) {
    float d = 0.f;
    float keyScale = scale;
    if constexpr (!FP8) {
      const float *kr = j < nctx ? static_cast<const float *>(kcVoid) + (size_t)((lo + j) % cap) * kvDim
                                   : kb + (size_t)(j - nctx) * kvDim;
      kr += (size_t)kvh * dim;
      for (int i = lane; i < dim; i += 32) d = fmaf(qh[i], kr[i], d);
    } else if (j < nctx) {
      const size_t slot = (size_t)((lo + j) % cap);
      const uint8_t *kr = static_cast<const uint8_t *>(kcVoid) + slot * kvDim + (size_t)kvh * dim;
      for (int i = lane; i < dim; i += 32) d = fmaf(qh[i], ink2_e4m3fn_decode(kr[i]), d);
      keyScale *= ks[slot * kvHeads + kvh];
    } else {
      const float *kr = kb + (size_t)(j - nctx) * kvDim + (size_t)kvh * dim;
      for (int i = lane; i < dim; i += 32) d = fmaf(qh[i], kr[i], d);
    }
    for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
    if (lane == 0) sc[j] = d * keyScale;
  }
  __syncthreads();
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, sc[j]);
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  mx = red[0];
  for (int i = 1; i < nwarp; ++i) mx = fmaxf(mx, red[i]);
  __syncthreads();
  float sum = 0.f;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    float e = __expf(sc[j] - mx);
    sum += e;
    if constexpr (FP8) {
      if (j < nctx) e *= vs[(size_t)((lo + j) % cap) * kvHeads + kvh];
    }
    sc[j] = e;
  }
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
  if (lane == 0) red[warp] = sum;
  __syncthreads();
  sum = 0.f;
  for (int i = 0; i < nwarp; ++i) sum += red[i];
  const float inv = 1.f / sum;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float acc = 0.f;
    if constexpr (!FP8) {
      for (int j = 0; j < n; ++j) {
        const float *vr = j < nctx ? static_cast<const float *>(vcVoid) + (size_t)((lo + j) % cap) * kvDim
                                     : vb + (size_t)(j - nctx) * kvDim;
        acc = fmaf(sc[j], vr[(size_t)kvh * dim + i], acc);
      }
    } else {
      const uint8_t *vcBase = static_cast<const uint8_t *>(vcVoid) + (size_t)kvh * dim + i;
      int j = 0;
      for (; j < nctx; ++j) {
        const uint8_t c = vcBase[(size_t)((lo + j) % cap) * kvDim];
        acc = fmaf(sc[j], ink2_e4m3fn_decode(c), acc);
      }
      const float *vbBase = vb + (size_t)kvh * dim + i;
      for (; j < n; ++j) {
        acc = fmaf(sc[j], vbBase[(size_t)(j - nctx) * kvDim], acc);
      }
    }
    out[(size_t)r * heads * dim + (size_t)h * dim + i] = acc * inv;
  }
}

__global__ void k_ink2d_attend(const float *q, const float *kb, const float *vb, const float *kc, const float *vc,
                               float *out, int heads, int kvHeads, int dim, int rows, const int *posDev, int window,
                               int cap) {
  ink2_pdl_enter();
  ink2d_attend_body<false>(q, kb, vb, kc, vc, nullptr, nullptr, out, heads, kvHeads, dim, posDev, window, cap);
}

__global__ void k_ink2d_attend_fp8(const float *q, const float *kb, const float *vb, const uint8_t *kc,
                                   const uint8_t *vc, const float *ks, const float *vs, float *out, int heads,
                                   int kvHeads, int dim, const int *posDev, int window, int cap) {
  ink2_pdl_enter();
  ink2d_attend_body<true>(q, kb, vb, kc, vc, ks, vs, out, heads, kvHeads, dim, posDev, window, cap);
}

__global__ void k_ink2d_silu_mul(const float *g, const float *u, float *out, int n) {
  ink2_pdl_enter();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const float x = g[i];
    out[i] = x / (1.f + __expf(-x)) * u[i];
  }
}

extern "C" int fcuda_ink2d_fuse(const float *const *f, const float *a, int taps, const float *w, float *out, int rows,
                                int n, float eps) {
  if (taps < 1 || taps > INK2D_MAX_TAPS) return (int)cudaErrorInvalidValue;
  Ink2DTaps tp = {};
  for (int t = 0; t < taps; ++t) {
    tp.f[t] = f[t];
    tp.a[t] = a[t];
  }
  FCUDA_TS("k_ink2d_fuse", rows, k_ink2d_fuse<<<rows, 256, 0, g_stream>>>(tp, taps, w, out, n, eps));
  return ink2_last_error();
}

extern "C" int fcuda_ink2d_qk(const float *x, const float *w, float *dst, int rows, int heads, int dim,
                              const int *posDev, int cap, int ring, float eps, double theta) {
  if (dim > 256 || dim % 64 != 0) return (int)cudaErrorInvalidValue;
  const dim3 grid(heads, rows);
  FCUDA_TS("k_ink2d_qk", grid, k_ink2d_qk<<<grid, dim, 0, g_stream>>>(x, w, dst, heads, dim, posDev, cap, ring, eps, theta));
  return ink2_last_error();
}

extern "C" int fcuda_ink2d_ring_store(const float *src, float *ring, int rows, int width, const int *posDev, int cap) {
  const dim3 grid((width + 255) / 256, rows);
  FCUDA_TS("k_ink2d_ring_store", grid, k_ink2d_ring_store<<<grid, 256, 0, g_stream>>>(src, ring, width, posDev, cap));
  return ink2_last_error();
}

extern "C" int fcuda_ink2d_attend(const float *q, const float *kb, const float *vb, const float *kc, const float *vc,
                                  float *out, int rows, int heads, int kvHeads, int dim, const int *posDev, int window,
                                  int cap) {
  int span = cap;
  if (window > 0 && window < span) span = window;
  const size_t smem = (size_t)(span + rows) * sizeof(float);
  const int oe = ink2p_smem_optin((const void *)k_ink2d_attend, smem);
  if (oe) return oe;
  const dim3 grid(heads, rows);
  FCUDA_TS("k_ink2d_attend", grid, k_ink2d_attend<<<grid, 128, smem, g_stream>>>(q, kb, vb, kc, vc, out, heads, kvHeads,
                                                                                dim, rows, posDev, window, cap));
  return ink2_last_error();
}

extern "C" int fcuda_ink2d_silu_mul(const float *g, const float *u, float *out, int n) {
  FCUDA_TS("k_ink2d_silu_mul", (n + 255) / 256, k_ink2d_silu_mul<<<(n + 255) / 256, 256, 0, g_stream>>>(g, u, out, n));
  return ink2_last_error();
}

extern "C" int fcuda_ink2d_qk_fp8(const float *x, const float *w, uint8_t *dst, float *scale, int *overflow, int rows,
                                  int heads, int dim, const int *posDev, int cap, int ring, float eps, double theta) {
  if (rows <= 0 || heads <= 0 || dim > 256 || dim % 64 != 0 || (ring && cap <= 0)) return (int)cudaErrorInvalidValue;
  const dim3 grid(heads, rows);
  FCUDA_TS("k_ink2d_qk_fp8", grid,
           k_ink2d_qk_fp8<<<grid, dim, 0, g_stream>>>(x, w, dst, scale, overflow, heads, dim, posDev, cap, ring, eps, theta));
  return ink2_last_error();
}

extern "C" int fcuda_ink2d_ring_store_fp8(const float *src, uint8_t *ring, float *scale, int *overflow, int rows,
                                            int kvHeads, int dim, const int *posDev, int cap) {
  if (rows <= 0 || kvHeads <= 0 || dim <= 0 || dim > 256 || cap <= 0) return (int)cudaErrorInvalidValue;
  const int nt = (dim + 31) & ~31;
  const dim3 grid(kvHeads, rows);
  FCUDA_TS("k_ink2d_ring_store_fp8", grid,
           k_ink2d_ring_store_fp8<<<grid, nt, 0, g_stream>>>(src, ring, scale, overflow, kvHeads, dim, posDev, cap));
  return ink2_last_error();
}

extern "C" int fcuda_ink2d_attend_fp8(const float *q, const float *kb, const float *vb, const uint8_t *kc,
                                      const uint8_t *vc, const float *ks, const float *vs, float *out, int rows,
                                      int heads, int kvHeads, int dim, const int *posDev, int window, int cap) {
  int span = cap;
  if (window > 0 && window < span) span = window;
  const size_t smem = (size_t)(span + rows) * sizeof(float);
  const int oe = ink2p_smem_optin((const void *)k_ink2d_attend_fp8, smem);
  if (oe) return oe;
  const dim3 grid(heads, rows);
  FCUDA_TS("k_ink2d_attend_fp8", grid,
           k_ink2d_attend_fp8<<<grid, 128, smem, g_stream>>>(q, kb, vb, kc, vc, ks, vs, out, heads, kvHeads, dim,
                                                              posDev, window, cap));
  return ink2_last_error();
}
