//go:build linux && cgo && rocm

#include "rocm_backend.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

namespace {

char last_error[512] = "";
hipStream_t stream = nullptr;
int active_device = -1;
size_t host_transfer_bytes = 0;
size_t kernel_launches = 0;

int failf(int code, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(last_error, sizeof(last_error), fmt, args);
  va_end(args);
  return code ? code : -1;
}

int hip_fail(const char *what, hipError_t status) {
  return failf(1000 + static_cast<int>(status), "%s: %s", what,
               hipGetErrorString(status));
}

int ready(const char *what) {
  if (active_device < 0 || stream == nullptr)
    return failf(-2, "%s: ROCm backend is not initialized", what);
  last_error[0] = '\0';
  hipError_t status = hipSetDevice(active_device);
  if (status != hipSuccess) return hip_fail("hipSetDevice", status);
  return 0;
}

int launch_status(const char *what) {
  hipError_t status = hipGetLastError();
  if (status != hipSuccess) return hip_fail(what, status);
  status = hipStreamSynchronize(stream);
  if (status != hipSuccess) return hip_fail(what, status);
  ++kernel_launches;
  return 0;
}

__device__ float fp16_at(const uint8_t *p) {
  return __half2float(*reinterpret_cast<const __half *>(p));
}

__device__ void q4_scale_min(int group, const uint8_t *q, uint8_t *scale,
                             uint8_t *minimum) {
  if (group < 4) {
    *scale = q[group] & 63;
    *minimum = q[group + 4] & 63;
  } else {
    *scale = (q[group + 4] & 0x0f) | ((q[group - 4] >> 6) << 4);
    *minimum = (q[group + 4] >> 4) | ((q[group] >> 6) << 4);
  }
}

__global__ void f32_matmul(const float *w, const float *x, float *y,
                           int out, int in, int rows) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (o >= out || row >= rows) return;
  const float *wr = w + static_cast<size_t>(o) * in;
  const float *xr = x + static_cast<size_t>(row) * in;
  float sum = 0.0f;
  for (int i = 0; i < in; ++i) sum = fmaf(wr[i], xr[i], sum);
  y[static_cast<size_t>(row) * out + o] = sum;
}

__global__ void q8_matmul(const int8_t *codes, const float *scales,
                          const float *x, float *y, int out, int in, int rows,
                          int block) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (o >= out || row >= rows) return;
  int blocks = in / block;
  const int8_t *wr = codes + static_cast<size_t>(o) * in;
  const float *ws = scales + static_cast<size_t>(o) * blocks;
  const float *xr = x + static_cast<size_t>(row) * in;
  float sum = 0.0f;
  for (int b = 0; b < blocks; ++b) {
    float partial = 0.0f;
    int base = b * block;
    for (int i = 0; i < block; ++i)
      partial = fmaf(static_cast<float>(wr[base + i]), xr[base + i], partial);
    sum = fmaf(ws[b], partial, sum);
  }
  y[static_cast<size_t>(row) * out + o] = sum;
}

__device__ float q4k_value(const uint8_t *blk, int elem) {
  int group = elem / 32;
  int lane = elem & 31;
  uint8_t scale, minimum;
  q4_scale_min(group, blk + 4, &scale, &minimum);
  uint8_t packed = blk[16 + (group >> 1) * 32 + lane];
  int code = (group & 1) ? packed >> 4 : packed & 15;
  return fp16_at(blk) * static_cast<float>(scale * code) -
         fp16_at(blk + 2) * static_cast<float>(minimum);
}

__device__ float q5k_value(const uint8_t *blk, int elem) {
  int group = elem / 32;
  int lane = elem & 31;
  uint8_t scale, minimum;
  q4_scale_min(group, blk + 4, &scale, &minimum);
  const uint8_t *qh = blk + 16;
  const uint8_t *ql = blk + 48;
  int code = (group & 1) ? (ql[(group >> 1) * 32 + lane] >> 4)
                         : (ql[(group >> 1) * 32 + lane] & 15);
  if (qh[lane] & (1u << group)) code += 16;
  return fp16_at(blk) * static_cast<float>(scale * code) -
         fp16_at(blk + 2) * static_cast<float>(minimum);
}

__device__ float q6k_value(const uint8_t *blk, int elem) {
  const uint8_t *ql = blk;
  const uint8_t *qh = blk + 128;
  const int8_t *scales = reinterpret_cast<const int8_t *>(blk + 192);
  int half = elem / 128;
  int within = elem & 127;
  int lane = within & 31;
  int quarter = within / 32;
  int qoff = half * 64;
  int hoff = half * 32;
  int code;
  if (quarter == 0)
    code = (ql[qoff + lane] & 15) | (((qh[hoff + lane] >> 0) & 3) << 4);
  else if (quarter == 1)
    code = (ql[qoff + lane + 32] & 15) | (((qh[hoff + lane] >> 2) & 3) << 4);
  else if (quarter == 2)
    code = (ql[qoff + lane] >> 4) | (((qh[hoff + lane] >> 4) & 3) << 4);
  else
    code = (ql[qoff + lane + 32] >> 4) | (((qh[hoff + lane] >> 6) & 3) << 4);
  int scale_index = half * 8 + (lane >> 4) + quarter * 2;
  return fp16_at(blk + 208) * static_cast<float>(scales[scale_index]) *
         static_cast<float>(code - 32);
}

enum KQuantKind { KQ4 = 4, KQ5 = 5, KQ6 = 6 };

template <KQuantKind kind>
__global__ void kquant_matmul(const uint8_t *weights, const float *x, float *y,
                              int out, int in, int rows) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (o >= out || row >= rows) return;
  constexpr int block_bytes = kind == KQ4 ? 144 : (kind == KQ5 ? 176 : 210);
  int superblocks = in / 256;
  const uint8_t *wr = weights + static_cast<size_t>(o) * superblocks * block_bytes;
  const float *xr = x + static_cast<size_t>(row) * in;
  float sum = 0.0f;
  for (int sb = 0; sb < superblocks; ++sb) {
    const uint8_t *blk = wr + static_cast<size_t>(sb) * block_bytes;
    for (int i = 0; i < 256; ++i) {
      float w = kind == KQ4 ? q4k_value(blk, i)
                            : (kind == KQ5 ? q5k_value(blk, i)
                                           : q6k_value(blk, i));
      sum = fmaf(w, xr[sb * 256 + i], sum);
    }
  }
  y[static_cast<size_t>(row) * out + o] = sum;
}

__global__ void rmsnorm(const float *x, const float *weight, float *y,
                        int rows, int width, float eps) {
  int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float reduction[256];
  const float *xr = x + static_cast<size_t>(row) * width;
  float local = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    local = fmaf(xr[i], xr[i], local);
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (int step = blockDim.x / 2; step > 0; step >>= 1) {
    if (threadIdx.x < step) reduction[threadIdx.x] += reduction[threadIdx.x + step];
    __syncthreads();
  }
  float inv = rsqrtf(reduction[0] / static_cast<float>(width) + eps);
  float *yr = y + static_cast<size_t>(row) * width;
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    yr[i] = xr[i] * inv * weight[i];
}

__global__ void rope(const float *src, float *dst, int pos, int heads,
                     int head_dim, double theta) {
  int half = head_dim / 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= heads * half) return;
  int head = index / half;
  int d = index % half;
  double angle = static_cast<double>(pos) /
                 pow(theta, static_cast<double>(2 * d) / head_dim);
  float c = static_cast<float>(cos(angle));
  float s = static_cast<float>(sin(angle));
  size_t base = static_cast<size_t>(head) * head_dim;
  float a = src[base + d];
  float b = src[base + d + half];
  dst[base + d] = a * c - b * s;
  dst[base + d + half] = b * c + a * s;
}

__global__ void swiglu(const float *gate, const float *up, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
  }
}

__global__ void add(float *dst, const float *src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

__global__ void add_bias(float *dst, const float *bias, int n, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += bias[i % width];
}

/* Correctness-first native attention: one GPU thread owns one query head and
 * performs online softmax. It keeps all math and KV traffic device-resident;
 * later tuning can parallelize within the head without changing the ABI. */
__global__ void attention(const float *q, const float *k, const float *v,
                          float *out, int q_heads, int kv_heads, int positions,
                          int head_dim, int group_size, float scale) {
  int qh = blockIdx.x * blockDim.x + threadIdx.x;
  if (qh >= q_heads) return;
  int kvh = qh / group_size;
  int stride = kv_heads * head_dim;
  const float *qr = q + static_cast<size_t>(qh) * head_dim;
  float max_score = -INFINITY;
  float denom = 0.0f;
  for (int d = 0; d < head_dim; ++d)
    out[static_cast<size_t>(qh) * head_dim + d] = 0.0f;
  for (int p = 0; p < positions; ++p) {
    const float *kr = k + static_cast<size_t>(p) * stride + kvh * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) dot = fmaf(qr[d], kr[d], dot);
    float score = dot * scale;
    float next_max = fmaxf(max_score, score);
    float correction = expf(max_score - next_max);
    float probability = expf(score - next_max);
    denom = denom * correction + probability;
    const float *vr = v + static_cast<size_t>(p) * stride + kvh * head_dim;
    float *yr = out + static_cast<size_t>(qh) * head_dim;
    for (int d = 0; d < head_dim; ++d)
      yr[d] = yr[d] * correction + probability * vr[d];
    max_score = next_max;
  }
  float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
  for (int d = 0; d < head_dim; ++d)
    out[static_cast<size_t>(qh) * head_dim + d] *= inv;
}

__global__ void argmax(const float *values, int n, int *result) {
  if (isnan(values[0])) {
    if (threadIdx.x == 0) *result = 0;
    return;
  }
  __shared__ float best_value[256];
  __shared__ int best_index[256];
  float value = values[0];
  int index = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float candidate = values[i];
    if (candidate > value || (candidate == value && i < index)) {
      value = candidate;
      index = i;
    }
  }
  best_value[threadIdx.x] = value;
  best_index[threadIdx.x] = index;
  __syncthreads();
  for (int step = blockDim.x / 2; step > 0; step >>= 1) {
    if (threadIdx.x < step) {
      float other = best_value[threadIdx.x + step];
      int other_index = best_index[threadIdx.x + step];
      if (other > best_value[threadIdx.x] ||
          (other == best_value[threadIdx.x] && other_index < best_index[threadIdx.x])) {
        best_value[threadIdx.x] = other;
        best_index[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *result = best_index[0];
}

__global__ void kv_rerope(const float *raw_keys, float *keys, int start,
                          int positions, int kv_heads, int head_dim,
                          double theta) {
  int half = head_dim / 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (positions - start) * kv_heads * half;
  if (index >= total) return;
  int d = index % half;
  int pair = index / half;
  int head = pair % kv_heads;
  int pos = start + pair / kv_heads;
  size_t base = (static_cast<size_t>(pos) * kv_heads + head) * head_dim;
  double angle = static_cast<double>(pos) /
                 pow(theta, static_cast<double>(2 * d) / head_dim);
  float c = static_cast<float>(cos(angle));
  float s = static_cast<float>(sin(angle));
  float a = raw_keys[base + d];
  float b = raw_keys[base + d + half];
  keys[base + d] = a * c - b * s;
  keys[base + d + half] = b * c + a * s;
}

}  // namespace

extern "C" const char *frocm_last_error(void) { return last_error; }

extern "C" int frocm_init(int device, char *name, int name_len, char *gfx,
                           int gfx_len, size_t *total_mem) {
  last_error[0] = '\0';
  int count = 0;
  hipError_t hs = hipGetDeviceCount(&count);
  if (hs != hipSuccess) return hip_fail("hipGetDeviceCount", hs);
  if (device < 0 || device >= count)
    return failf(-1, "device %d outside [0,%d)", device, count);
  if ((hs = hipSetDevice(device)) != hipSuccess) return hip_fail("hipSetDevice", hs);
  hipDeviceProp_t prop{};
  if ((hs = hipGetDeviceProperties(&prop, device)) != hipSuccess)
    return hip_fail("hipGetDeviceProperties", hs);
  if (stream == nullptr && (hs = hipStreamCreate(&stream)) != hipSuccess)
    return hip_fail("hipStreamCreate", hs);
  active_device = device;
  if (name && name_len > 0) {
    strncpy(name, prop.name, static_cast<size_t>(name_len - 1));
    name[name_len - 1] = '\0';
  }
  if (gfx && gfx_len > 0) {
    strncpy(gfx, prop.gcnArchName, static_cast<size_t>(gfx_len - 1));
    gfx[gfx_len - 1] = '\0';
  }
  if (total_mem) *total_mem = prop.totalGlobalMem;
  return 0;
}

extern "C" int frocm_mem_info(size_t *free_mem, size_t *total_mem) {
  if (int rc = ready("mem_info")) return rc;
  hipError_t status = hipMemGetInfo(free_mem, total_mem);
  return status == hipSuccess ? 0 : hip_fail("hipMemGetInfo", status);
}

extern "C" int frocm_malloc(void **ptr, size_t bytes) {
  if (int rc = ready("malloc")) return rc;
  if (!ptr || bytes == 0) return failf(-1, "malloc: invalid output or zero bytes");
  hipError_t status = hipMalloc(ptr, bytes);
  return status == hipSuccess ? 0 : hip_fail("hipMalloc", status);
}

extern "C" int frocm_free(void *ptr) {
  if (!ptr) return 0;
  if (int rc = ready("free")) return rc;
  hipError_t status = hipFree(ptr);
  return status == hipSuccess ? 0 : hip_fail("hipFree", status);
}

extern "C" int frocm_h2d(void *dst, const void *src, size_t bytes) {
  if (int rc = ready("h2d")) return rc;
  if (!dst || (!src && bytes)) return failf(-1, "h2d: invalid pointer");
  hipError_t status = hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
  if (status != hipSuccess) return hip_fail("hipMemcpy H2D", status);
  host_transfer_bytes += bytes;
  return 0;
}

extern "C" int frocm_d2h(void *dst, const void *src, size_t bytes) {
  if (int rc = ready("d2h")) return rc;
  if ((!dst && bytes) || !src) return failf(-1, "d2h: invalid pointer");
  hipError_t status = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
  if (status != hipSuccess) return hip_fail("hipMemcpy D2H", status);
  host_transfer_bytes += bytes;
  return 0;
}

extern "C" int frocm_d2d(void *dst, const void *src, size_t bytes) {
  if (int rc = ready("d2d")) return rc;
  if ((!dst || !src) && bytes) return failf(-1, "d2d: invalid pointer");
  hipError_t status = hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream);
  if (status != hipSuccess) return hip_fail("hipMemcpyAsync D2D", status);
  status = hipStreamSynchronize(stream);
  return status == hipSuccess ? 0 : hip_fail("hipMemcpyAsync D2D completion", status);
}

extern "C" int frocm_sync(void) {
  if (int rc = ready("sync")) return rc;
  hipError_t status = hipStreamSynchronize(stream);
  return status == hipSuccess ? 0 : hip_fail("hipStreamSynchronize", status);
}

extern "C" size_t frocm_host_transfer_bytes(void) { return host_transfer_bytes; }
extern "C" void frocm_host_transfer_reset(void) { host_transfer_bytes = 0; }
extern "C" size_t frocm_kernel_launches(void) { return kernel_launches; }
extern "C" void frocm_kernel_launches_reset(void) { kernel_launches = 0; }

extern "C" int frocm_matmul_f32(const float *w, const float *x, float *y,
                                 int out, int in, int rows) {
  if (int rc = ready("matmul_f32")) return rc;
  if (!w || !x || !y || out <= 0 || in <= 0 || rows <= 0)
    return failf(-1, "matmul_f32: invalid pointer or geometry");
  f32_matmul<<<dim3((out + 127) / 128, rows), 128, 0, stream>>>(
      w, x, y, out, in, rows);
  return launch_status("matmul_f32 launch");
}

extern "C" int frocm_q8_matmul_f32(const int8_t *codes, const float *scales,
                                    const float *x, float *y, int out, int in,
                                    int rows, int block) {
  if (int rc = ready("q8_matmul_f32")) return rc;
  if (!codes || !scales || !x || !y || out <= 0 || in <= 0 || rows <= 0 ||
      block <= 0 || in % block != 0)
    return failf(-1, "q8_matmul_f32: invalid pointer or geometry");
  q8_matmul<<<dim3((out + 127) / 128, rows), 128, 0, stream>>>(
      codes, scales, x, y, out, in, rows, block);
  return launch_status("q8_matmul_f32 launch");
}

template <KQuantKind kind>
int launch_kquant(const char *what, const uint8_t *weights, const float *x,
                  float *y, int out, int in, int rows) {
  if (int rc = ready(what)) return rc;
  if (!weights || !x || !y || out <= 0 || in <= 0 || rows <= 0 || in % 256 != 0)
    return failf(-1, "%s: invalid pointer or geometry", what);
  kquant_matmul<kind><<<dim3((out + 127) / 128, rows), 128, 0, stream>>>(
      weights, x, y, out, in, rows);
  return launch_status(what);
}

extern "C" int frocm_q4k_matmul_f32(const uint8_t *w, const float *x, float *y,
                                     int out, int in, int rows) {
  return launch_kquant<KQ4>("q4k_matmul_f32 launch", w, x, y, out, in, rows);
}
extern "C" int frocm_q5k_matmul_f32(const uint8_t *w, const float *x, float *y,
                                     int out, int in, int rows) {
  return launch_kquant<KQ5>("q5k_matmul_f32 launch", w, x, y, out, in, rows);
}
extern "C" int frocm_q6k_matmul_f32(const uint8_t *w, const float *x, float *y,
                                     int out, int in, int rows) {
  return launch_kquant<KQ6>("q6k_matmul_f32 launch", w, x, y, out, in, rows);
}

extern "C" int frocm_rmsnorm_f32(const float *x, const float *weight, float *y,
                                  int rows, int width, float eps) {
  if (int rc = ready("rmsnorm_f32")) return rc;
  if (!x || !weight || !y || rows <= 0 || width <= 0 || eps < 0)
    return failf(-1, "rmsnorm_f32: invalid pointer or geometry");
  rmsnorm<<<rows, 256, 0, stream>>>(x, weight, y, rows, width, eps);
  return launch_status("rmsnorm_f32 launch");
}

extern "C" int frocm_rope_f32(const float *src, float *dst, int pos, int heads,
                               int head_dim, double theta) {
  if (int rc = ready("rope_f32")) return rc;
  if (!src || !dst || pos < 0 || heads <= 0 || head_dim <= 0 || head_dim % 2 || theta <= 0)
    return failf(-1, "rope_f32: invalid pointer or geometry");
  int pairs = heads * (head_dim / 2);
  rope<<<(pairs + 127) / 128, 128, 0, stream>>>(src, dst, pos, heads, head_dim, theta);
  return launch_status("rope_f32 launch");
}

extern "C" int frocm_swiglu_f32(const float *gate, const float *up, float *out, int n) {
  if (int rc = ready("swiglu_f32")) return rc;
  if (!gate || !up || !out || n <= 0) return failf(-1, "swiglu_f32: invalid input");
  swiglu<<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
  return launch_status("swiglu_f32 launch");
}

extern "C" int frocm_add_f32(float *dst, const float *src, int n) {
  if (int rc = ready("add_f32")) return rc;
  if (!dst || !src || n <= 0) return failf(-1, "add_f32: invalid input");
  add<<<(n + 255) / 256, 256, 0, stream>>>(dst, src, n);
  return launch_status("add_f32 launch");
}

extern "C" int frocm_add_bias_f32(float *dst, const float *bias, int rows, int width) {
  if (int rc = ready("add_bias_f32")) return rc;
  if (!dst || !bias || rows <= 0 || width <= 0) return failf(-1, "add_bias_f32: invalid input");
  int n = rows * width;
  add_bias<<<(n + 255) / 256, 256, 0, stream>>>(dst, bias, n, width);
  return launch_status("add_bias_f32 launch");
}

extern "C" int frocm_attention_f32(const float *q, const float *k, const float *v,
                                    float *out, int q_heads, int kv_heads, int positions,
                                    int head_dim, int group_size, float scale) {
  if (int rc = ready("attention_f32")) return rc;
  if (!q || !k || !v || !out || q_heads <= 0 || kv_heads <= 0 || positions <= 0 ||
      head_dim <= 0 || group_size <= 0 || q_heads != kv_heads * group_size || !isfinite(scale))
    return failf(-1, "attention_f32: invalid pointer or geometry");
  attention<<<(q_heads + 63) / 64, 64, 0, stream>>>(
      q, k, v, out, q_heads, kv_heads, positions, head_dim, group_size, scale);
  return launch_status("attention_f32 launch");
}

extern "C" int frocm_argmax_f32(const float *values, int n, int *index) {
  if (int rc = ready("argmax_f32")) return rc;
  if (!values || !index || n <= 0) return failf(-1, "argmax_f32: invalid input");
  int *device_index = nullptr;
  hipError_t hs = hipMalloc(reinterpret_cast<void **>(&device_index), sizeof(int));
  if (hs != hipSuccess) return hip_fail("argmax hipMalloc", hs);
  argmax<<<1, 256, 0, stream>>>(values, n, device_index);
  int rc = launch_status("argmax_f32 launch");
  if (!rc) {
    hs = hipMemcpy(index, device_index, sizeof(int), hipMemcpyDeviceToHost);
    if (hs != hipSuccess) rc = hip_fail("argmax D2H", hs);
  }
  hipError_t free_status = hipFree(device_index);
  if (!rc && free_status != hipSuccess) rc = hip_fail("argmax hipFree", free_status);
  return rc;
}

extern "C" int frocm_kv_append_f32(float *keys, float *raw_keys, float *values,
                                    const float *key, const float *raw_key,
                                    const float *value, int slot, int row_width) {
  if (int rc = ready("kv_append_f32")) return rc;
  if (!keys || !raw_keys || !values || !key || !raw_key || !value || slot < 0 || row_width <= 0)
    return failf(-1, "kv_append_f32: invalid pointer or geometry");
  size_t bytes = static_cast<size_t>(row_width) * sizeof(float);
  size_t offset = static_cast<size_t>(slot) * row_width;
  hipError_t hs = hipMemcpyAsync(keys + offset, key, bytes, hipMemcpyDeviceToDevice, stream);
  if (hs != hipSuccess) return hip_fail("kv append key", hs);
  hs = hipMemcpyAsync(raw_keys + offset, raw_key, bytes, hipMemcpyDeviceToDevice, stream);
  if (hs != hipSuccess) return hip_fail("kv append raw key", hs);
  hs = hipMemcpyAsync(values + offset, value, bytes, hipMemcpyDeviceToDevice, stream);
  if (hs != hipSuccess) return hip_fail("kv append value", hs);
  hs = hipStreamSynchronize(stream);
  return hs == hipSuccess ? 0 : hip_fail("kv append completion", hs);
}

extern "C" int frocm_kv_evict_f32(float *keys, float *raw_keys, float *values,
                                   int positions, int from, int count, int kv_heads,
                                   int head_dim, double theta) {
  if (int rc = ready("kv_evict_f32")) return rc;
  if (!keys || !raw_keys || !values || positions <= 0 || from < 0 || count <= 0 ||
      from >= positions || kv_heads <= 0 || head_dim <= 0 || head_dim % 2 || theta <= 0)
    return failf(-1, "kv_evict_f32: invalid pointer or geometry");
  int end = from + count < positions ? from + count : positions;
  int survivors = positions - end;
  int new_positions = positions - (end - from);
  size_t row_bytes = static_cast<size_t>(kv_heads) * head_dim * sizeof(float);
  if (survivors > 0) {
    size_t bytes = static_cast<size_t>(survivors) * row_bytes;
    void *tmp = nullptr;
    hipError_t hs = hipMalloc(&tmp, bytes * 3);
    if (hs != hipSuccess) return hip_fail("kv evict hipMalloc", hs);
    uint8_t *base = static_cast<uint8_t *>(tmp);
    float *tmp_keys = reinterpret_cast<float *>(base);
    float *tmp_raw = reinterpret_cast<float *>(base + bytes);
    float *tmp_values = reinterpret_cast<float *>(base + bytes * 2);
    hs = hipMemcpyAsync(tmp_keys, keys + static_cast<size_t>(end) * kv_heads * head_dim,
                        bytes, hipMemcpyDeviceToDevice, stream);
    if (hs == hipSuccess)
      hs = hipMemcpyAsync(tmp_raw, raw_keys + static_cast<size_t>(end) * kv_heads * head_dim,
                          bytes, hipMemcpyDeviceToDevice, stream);
    if (hs == hipSuccess)
      hs = hipMemcpyAsync(tmp_values, values + static_cast<size_t>(end) * kv_heads * head_dim,
                          bytes, hipMemcpyDeviceToDevice, stream);
    if (hs == hipSuccess)
      hs = hipMemcpyAsync(keys + static_cast<size_t>(from) * kv_heads * head_dim,
                          tmp_keys, bytes, hipMemcpyDeviceToDevice, stream);
    if (hs == hipSuccess)
      hs = hipMemcpyAsync(raw_keys + static_cast<size_t>(from) * kv_heads * head_dim,
                          tmp_raw, bytes, hipMemcpyDeviceToDevice, stream);
    if (hs == hipSuccess)
      hs = hipMemcpyAsync(values + static_cast<size_t>(from) * kv_heads * head_dim,
                          tmp_values, bytes, hipMemcpyDeviceToDevice, stream);
    if (hs == hipSuccess) hs = hipStreamSynchronize(stream);
    hipError_t free_status = hipFree(tmp);
    if (hs != hipSuccess) return hip_fail("kv evict compact", hs);
    if (free_status != hipSuccess) return hip_fail("kv evict hipFree", free_status);
  }
  int pairs = (new_positions - from) * kv_heads * (head_dim / 2);
  if (pairs > 0) {
    kv_rerope<<<(pairs + 127) / 128, 128, 0, stream>>>(
        raw_keys, keys, from, new_positions, kv_heads, head_dim, theta);
    return launch_status("kv evict re-RoPE launch");
  }
  return 0;
}
