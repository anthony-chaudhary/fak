// rmsnorm_fp8_quant.cu — fused RMSNorm + static-scale FP8 E4M3 activation
// quantization. Included by cuda_kernels.cu (one-line #include, build wiring
// owned by the successor policy-hook packet) so the kernel shares g_stream,
// the pooled allocator and the transfer witnesses of the single libfakcuda
// object.
//
// ADAPT of vllm-project/vllm csrc/layernorm_quant_kernels.cu:23,88,161
// (rms_norm_static_fp8_quant_kernel, fused_add_rms_norm_static_fp8_quant_kernel)
// at pin ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0): one-pass
// RMSNorm writing OCP e4m3fn codes under a static per-tensor scale, for
// layers whose next op consumes FP8 activations. Re-expressed as a CUDA
// grid-stride kernel for fak's flat C ABI. Divergences from upstream, all
// deliberate:
//   - The f32 plane is the sole input dtype (upstream dispatches
//     fp16/bf16/fp32); rounding the normalized value through scalar_t becomes
//     the identity, and the existing f32 RMSNorm kernels and current-serving
//     precision stay untouched.
//   - One thread owns one row and accumulates the squared sum sequentially in
//     index order (upstream's cub BlockReduce order is implementation-
//     dependent); the fixed order is what lets the Go oracle pin the codes
//     byte for byte.
//   - inv = 1.0f / sqrtf(sumsq/width + eps) instead of rsqrtf (up to 1.5 ulp
//     error), keeping the device twin bit-identical to the Go reference.
//   - Every critical multiply/add/divide goes through explicit __f*_rn
//     intrinsics so the compiler cannot form FMAs and diverge from the oracle.
//   - Zero-variance rows (sumsq == 0) keep the graceful all-zero-code fallback
//     on device (a bad row mid-batch must not abort the forward pass); the
//     strict typed refusal lives in the host Go API
//     (ErrRmsNormFp8ZeroVariance).
//   - The fused variant stores the add result in a dedicated out_residual
//     buffer instead of mutating the residual argument in place.
//
// Contract (mirrored by the Go oracle RmsNormFp8Quant in
// rmsnorm_fp8_quant.go):
//   - Input: row-major f32 x [rows][width], f32 weight [width], static
//     per-tensor scale (> 0, finite), epsilon (>= 0, finite).
//   - Per row: sumsq = Σ x[i]² in index order; inv = 1/sqrt(sumsq/width + eps);
//     code[i] = e4m3((x[i]*inv)*w[i] * scale_inv), ties to even, signed
//     saturation at ±448, the 0x7F NaN code never emitted (in-file encoder
//     rfq_e4m3_encode on the OCP grid, pinned by rmsnorm_fp8_quant_test.go).
//   - Fused variant: res = x + residual is stored first, then normalized and
//     quantized from res.
// Statuses: 0 ok, -1 null pointer or refused geometry, -2 graph capture open,
// -3 device allocation failure, positive cudaError_t for copy/launch/execution.

#include <cmath>
#include <cstdint>
#include <limits.h>

#define RFQ_THREADS 256
#define RFQ_MAX_BLOCKS 1024

union RfqBits {
  float f;
  uint32_t u;
};

static __device__ __forceinline__ uint32_t rfq_float_bits(float x) {
  RfqBits b;
  b.f = x;
  return b.u;
}

static __device__ __forceinline__ float rfq_bits_float(uint32_t u) {
  RfqBits b;
  b.u = u;
  return b.f;
}

// OCP e4m3fn encode: round-to-nearest-even with signed saturation at +/-448
// (codes 0x7E/0xFE). The 0x7F NaN code is never emitted. Subnormals follow
// the OCP grid m*2^-9 (m in 0..7); a rounded-up top subnormal promotes to
// the smallest normal 2^-6 (code 0x08), keeping the grid gap-free at 2^-6.
// Mirrors the Go reference rfqEncodeE4M3 bit for bit. This deliberately does
// NOT reuse dsv41_kq_e4m3_encode, whose subnormal grid is m*2^-13: the
// pinned vLLM conversion is the hardware cvt.rn.satfinite.e4m3x2.f32, i.e.
// the OCP grid.
static __device__ __forceinline__ uint8_t rfq_e4m3_encode(float x) {
  if (x >= 448.0f) return 0x7Eu;
  if (x <= -448.0f) return 0xFEu;
  const uint32_t bits = rfq_float_bits(x);
  const uint32_t sign = bits >> 31;
  const uint32_t mag = bits & 0x7FFFFFFFu;
  if (mag == 0) return (uint8_t)(sign << 7);
  const int exp = int((mag >> 23) & 0xFFu) - 127;
  if (exp >= -6) {
    const uint32_t frac = mag & 0x7FFFFFu;
    uint32_t m = frac >> 20;
    const uint32_t rem = frac & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) ++m;
    int e = exp + 7;
    if (m == 8u) {
      m = 0;
      ++e;
    }
    if (e > 15) return (uint8_t)((sign << 7) | 0x7Eu);
    return (uint8_t)((sign << 7) | ((uint32_t)e << 3) | m);
  }
  // Subnormal target: q = |x| * 2^9 is exact in f32 and lies in [0, 8).
  const float q = rfq_bits_float(mag) * 512.0f;
  uint32_t m = (uint32_t)q;
  const float r = q - (float)m;
  if (r > 0.5f || (r == 0.5f && (m & 1u))) ++m;
  if (m > 7u) return (uint8_t)((sign << 7) | 0x08u);  // promote to smallest normal
  return (uint8_t)((sign << 7) | m);
}

// One thread owns one row, grid-strided: accumulate the squared sum in index
// order, invert the RMS, and write one e4m3 code per dim.
__global__ void k_rfq_rmsnorm_fp8_quant(
    const float *__restrict__ x,
    const float *__restrict__ weight,
    uint8_t *__restrict__ out,
    float scale_inv,
    float epsilon,
    int32_t rows,
    int32_t width) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += stride) {
    const float *xr = x + row * width;
    float sumsq = 0.0f;
    for (int32_t i = 0; i < width; ++i) {
      const float v = __ldg(xr + i);
      sumsq = __fadd_rn(sumsq, __fmul_rn(v, v));
    }
    uint8_t *orow = out + row * width;
    if (sumsq == 0.0f) {
      for (int32_t i = 0; i < width; ++i) orow[i] = 0;
      continue;
    }
    const float denom = __fadd_rn(__fdiv_rn(sumsq, (float)width), epsilon);
    const float inv = __fdiv_rn(1.0f, sqrtf(denom));
    for (int32_t i = 0; i < width; ++i) {
      const float n = __fmul_rn(__fmul_rn(__ldg(xr + i), inv), __ldg(weight + i));
      orow[i] = rfq_e4m3_encode(__fmul_rn(n, scale_inv));
    }
  }
}

// Fused add-first variant: store res = x + residual, reduce over res, then
// normalize and quantize from the stored res (the second pass re-reads the
// value it just wrote, exactly like upstream re-reading its residual).
__global__ void k_rfq_fused_add_rmsnorm_fp8_quant(
    const float *__restrict__ x,
    const float *__restrict__ residual,
    const float *__restrict__ weight,
    float *__restrict__ out_residual,
    uint8_t *__restrict__ out,
    float scale_inv,
    float epsilon,
    int32_t rows,
    int32_t width) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += stride) {
    const float *xr = x + row * width;
    const float *rr = residual + row * width;
    float *residual_row = out_residual + row * width;
    float sumsq = 0.0f;
    for (int32_t i = 0; i < width; ++i) {
      const float s = __fadd_rn(__ldg(xr + i), __ldg(rr + i));
      residual_row[i] = s;
      sumsq = __fadd_rn(sumsq, __fmul_rn(s, s));
    }
    uint8_t *orow = out + row * width;
    if (sumsq == 0.0f) {
      for (int32_t i = 0; i < width; ++i) orow[i] = 0;
      continue;
    }
    const float denom = __fadd_rn(__fdiv_rn(sumsq, (float)width), epsilon);
    const float inv = __fdiv_rn(1.0f, sqrtf(denom));
    for (int32_t i = 0; i < width; ++i) {
      const float n = __fmul_rn(__fmul_rn(residual_row[i], inv), __ldg(weight + i));
      orow[i] = rfq_e4m3_encode(__fmul_rn(n, scale_inv));
    }
  }
}

// rfq_geom is the validated byte geometry of one rmsnorm-fp8-quant call.
struct rfq_geom {
  size_t x, weight, residual, out;
};

// rfq_geometry validates shape and sizes every operand; false means status -1.
static bool rfq_geometry(int32_t rows, int32_t width, bool fused, rfq_geom *g) {
  if (rows < 1 || width < 1) return false;
  const int64_t total = (int64_t)rows * width;
  if (total > INT32_MAX) return false;
  g->x = (size_t)total * sizeof(float);
  g->weight = (size_t)width * sizeof(float);
  g->residual = fused ? g->x : 0;
  g->out = (size_t)total;  // one e4m3 byte per dim
  return true;
}

// rfq_params admits the static scale and epsilon: finite, scale strictly
// positive with a finite reciprocal, epsilon non-negative.
static bool rfq_params(float scale, float epsilon) {
  return std::isfinite(scale) && scale > 0.0f && std::isfinite(1.0f / scale) &&
         std::isfinite(epsilon) && epsilon >= 0.0f;
}

static size_t g_rfq_launches = 0;
static size_t g_rfq_fused_launches = 0;

// One synchronous rmsnorm+fp8-quant: upload x and weight, launch, fence, read
// the code panel back. Host buffers must carry exactly the documented sizes
// (validated against the recomputed geometry). Statuses as the file header.
extern "C" int fcuda_rmsnorm_fp8_quant(
    const float *x, size_t x_bytes,
    const float *weight, size_t weight_bytes,
    float scale, float epsilon,
    int32_t rows, int32_t width,
    uint8_t *out_codes, size_t out_codes_bytes) {
  if (!x || !weight || !out_codes) return -1;
  rfq_geom g;
  if (!rfq_geometry(rows, width, false, &g)) return -1;
  if (x_bytes != g.x || weight_bytes != g.weight || out_codes_bytes != g.out) return -1;
  if (!rfq_params(scale, epsilon)) return -1;
  if (g_capture_open) return -2;
  void *dX = fcuda_malloc(g.x);
  void *dW = fcuda_malloc(g.weight);
  void *dOut = fcuda_malloc(g.out);
  int status = 0;
  cudaError_t e = cudaSuccess;
  if (!dX || !dW || !dOut) {
    fcuda_free(dX); fcuda_free(dW); fcuda_free(dOut);
    return -3;
  }
  (void)cudaGetLastError(); // discard stale status before attributing this call
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dX, x, g.x, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += g.x;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dW, weight, g.weight, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += g.weight;
  {
    unsigned int grid = (unsigned int)(((int64_t)rows + RFQ_THREADS - 1) / RFQ_THREADS);
    if (grid > RFQ_MAX_BLOCKS) grid = RFQ_MAX_BLOCKS;
    FCUDA_TS("k_rfq_rmsnorm_fp8_quant", grid,
             k_rfq_rmsnorm_fp8_quant<<<grid, RFQ_THREADS, 0, g_stream>>>(
                 (const float *)dX, (const float *)dW, (uint8_t *)dOut,
                 1.0f / scale, epsilon, rows, width));
    if ((e = cudaGetLastError()) != cudaSuccess) { status = (int)e; goto done; }
  }
  if ((e = cudaStreamSynchronize(g_stream)) != cudaSuccess) { status = (int)e; goto done; }
  ++g_rfq_launches;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(out_codes, dOut, g.out, cudaMemcpyDeviceToHost)) != cudaSuccess) { status = (int)e; goto done; }
  g_host_bytes += g.out;
done:
  fcuda_free(dX);
  fcuda_free(dW);
  fcuda_free(dOut);
  return status;
}

// One synchronous fused add + rmsnorm + fp8-quant: upload x, residual and
// weight, launch, fence, read both the code panel and the add result back.
extern "C" int fcuda_rmsnorm_fp8_quant_fused_add(
    const float *x, size_t x_bytes,
    const float *residual, size_t residual_bytes,
    const float *weight, size_t weight_bytes,
    float scale, float epsilon,
    int32_t rows, int32_t width,
    uint8_t *out_codes, size_t out_codes_bytes,
    float *out_residual, size_t out_residual_bytes) {
  if (!x || !residual || !weight || !out_codes || !out_residual) return -1;
  rfq_geom g;
  if (!rfq_geometry(rows, width, true, &g)) return -1;
  if (x_bytes != g.x || residual_bytes != g.residual || weight_bytes != g.weight ||
      out_codes_bytes != g.out || out_residual_bytes != g.x) return -1;
  if (!rfq_params(scale, epsilon)) return -1;
  if (g_capture_open) return -2;
  void *dX = fcuda_malloc(g.x);
  void *dR = fcuda_malloc(g.residual);
  void *dW = fcuda_malloc(g.weight);
  void *dOut = fcuda_malloc(g.out);
  void *dOR = fcuda_malloc(g.x);
  int status = 0;
  cudaError_t e = cudaSuccess;
  if (!dX || !dR || !dW || !dOut || !dOR) {
    fcuda_free(dX); fcuda_free(dR); fcuda_free(dW); fcuda_free(dOut); fcuda_free(dOR);
    return -3;
  }
  (void)cudaGetLastError(); // discard stale status before attributing this call
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dX, x, g.x, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += g.x;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dR, residual, g.residual, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += g.residual;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(dW, weight, g.weight, cudaMemcpyHostToDevice)) != cudaSuccess) { status = (int)e; goto done; }
  g_h2d_bytes += g.weight;
  {
    unsigned int grid = (unsigned int)(((int64_t)rows + RFQ_THREADS - 1) / RFQ_THREADS);
    if (grid > RFQ_MAX_BLOCKS) grid = RFQ_MAX_BLOCKS;
    FCUDA_TS("k_rfq_fused_add_rmsnorm_fp8_quant", grid,
             k_rfq_fused_add_rmsnorm_fp8_quant<<<grid, RFQ_THREADS, 0, g_stream>>>(
                 (const float *)dX, (const float *)dR, (const float *)dW,
                 (float *)dOR, (uint8_t *)dOut,
                 1.0f / scale, epsilon, rows, width));
    if ((e = cudaGetLastError()) != cudaSuccess) { status = (int)e; goto done; }
  }
  if ((e = cudaStreamSynchronize(g_stream)) != cudaSuccess) { status = (int)e; goto done; }
  ++g_rfq_fused_launches;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(out_codes, dOut, g.out, cudaMemcpyDeviceToHost)) != cudaSuccess) { status = (int)e; goto done; }
  g_host_bytes += g.out;
  fcuda_ts_untimed_transfer();
  if ((e = cudaMemcpy(out_residual, dOR, g.x, cudaMemcpyDeviceToHost)) != cudaSuccess) { status = (int)e; goto done; }
  g_host_bytes += g.x;
done:
  fcuda_free(dX);
  fcuda_free(dR);
  fcuda_free(dW);
  fcuda_free(dOut);
  fcuda_free(dOR);
  return status;
}

// Witness: cumulative successful operation completions.
extern "C" size_t fcuda_rmsnorm_fp8_quant_launches(void) {
  return g_rfq_launches;
}

extern "C" size_t fcuda_rmsnorm_fp8_quant_fused_launches(void) {
  return g_rfq_fused_launches;
}
