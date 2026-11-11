// silu_mul_fp8_quant.cu — fused SiLU-mul + per-token-group FP8 (e4m3fn,
// ue8m0 scale) quantization of expert intermediate activations.
//
// ADAPTED from vllm-project/vllm,
// csrc/libtorch_stable/quantization/activation_kernels.cu:276
// (silu_mul_fp8_quant_deep_gemm_kernel) with the act_and_mul_kernel structure
// (csrc/libtorch_stable/activation_kernels.cu:102) at
// b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd (Apache-2.0; attribution in
// THIRD_PARTY_NOTICES.md): swiglu(gate, up) followed by per-128-group e4m3fn
// quantization whose ue8m0 scale is driven by the group amax, producing the
// packed codes+scales a stage-2 down-projection GEMM would consume.
// Re-expressed against fak conventions (extern "C" fcuda_* entries, int
// status returns, explicit stream argument); no upstream code copied.
// Ref oss-port-vllmtop-silu-mul-fp8-quant-20261103.
//
// Not yet #included into cuda_kernels.cu: that one-line build wiring is a
// follow-on ticket beyond this candidate's path cap. The Go oracle in
// silu_mul_fp8_quant.go pins the observable contract (amax-driven ue8m0
// scale, e4m3fn RNE payload, ragged 128-wide group boundary, non-finite
// refusal).
//
// Deltas from upstream:
//   - f32 input and intermediates (upstream's deep-gemm path runs bf16);
//     silu uses expf (device precision) where the host oracle evaluates in
//     f64 — qualification must bound it against the oracle. No GPU or
//     throughput claim attaches to this candidate.
//   - Ragged tails: hidden need not be a multiple of 128; the last group
//     covers hidden%128 elements (upstream requires H % 128 == 0).
//   - The scale exponent is computed exactly in f32 rather than upstream's
//     bf16 log2/ceil pipeline.
//   - Upstream leaves non-finite propagation unguarded; this kernel records
//     the first fault index in the status word so the host can refuse.

#include <cuda_runtime.h>
#include <cstdint>

#define SMFQ_OK 0
#define SMFQ_BAD_ARGS 1
#define SMFQ_CUDA_ERROR 4

// Sanity caps for the library contract (upstream carries none): they keep
// the launch math in int64 range without bounding any realistic expert row.
#define SMFQ_MAX_TOKENS (1 << 20)
#define SMFQ_MAX_HIDDEN (1 << 16)
#define SMFQ_MAX_CELLS (1LL << 30)

#define SMFQ_GROUP_SIZE 128
#define SMFQ_THREADS 128
#define SMFQ_WARPS (SMFQ_THREADS / 32)

// Status word layout (four device words, 16 bytes):
//   [0] magic 0x51464D53 ('SMFQ'), written by block 0 thread 0
//   [1] tokens
//   [2] first operand fault flat index (token*2H + i, gate side i < H, up
//       side i >= H), UINT_MAX when clean
//   [3] first intermediate fault index (i < H, finite operands whose fused
//       product overflows f32), UINT_MAX when clean
// On any fault the affected lanes publish 0x00 codes; the host must not
// consume yq/ys unless both fault words are UINT_MAX.
#define SMFQ_STATUS_MAGIC 0x51464D53u
#define SMFQ_NO_FAULT 0xFFFFFFFFu

__device__ __forceinline__ bool smfq_nonfinite(float v) {
  return (__float_as_uint(v) & 0x7f800000u) == 0x7f800000u;
}

// Host-oracle mirror of siluMulFp8Silu: z/(1+exp(-z)). Finite inputs only;
// expf is device precision, as permitted here — qualification must bound it
// against the f64 host evaluation.
__device__ __forceinline__ float smfq_silu(float z) {
  return z / (1.f + expf(-z));
}

// Bit-exact device mirror of siluMulFp8GroupExponent: smallest integer e
// with 2^e >= amax/448, amax floored at the f32 widening of upstream's EPS
// (bf16 raw 0x2E5C). raw stays a positive normal, so ceil(log2(raw))
// resolves from its bits without a transcendental.
__device__ __forceinline__ int smfq_group_exponent(float amax) {
  const float floor = __uint_as_float(0x2E5C0000u);
  if (amax < floor) amax = floor;
  const float raw = __fdiv_rn(amax, 448.0f);
  const unsigned bits = __float_as_uint(raw);
  int exp = (int)(bits >> 23) - 127;
  if (bits & 0x7fffffu) ++exp;
  return exp;
}

// Biased ue8m0 byte; the clamp is a closed-contract guard (finite inputs
// stay inside [84, 247]).
__device__ __forceinline__ unsigned char smfq_encode_scale(int exp) {
  int enc = exp + 127;
  if (enc < 0) enc = 0;
  if (enc > 255) enc = 255;
  return (unsigned char)enc;
}

// Bit-exact device mirror of siluMulFp8EncodeE4M3: nearest e4m3fn, ties to
// even, signed saturation at +/-448 (0x7E/0xFE); the 0x7F/0xFF NaN codes are
// never emitted on a clean call.
__device__ __forceinline__ unsigned char smfq_e4m3_encode(float x) {
  if (x >= 448.0f) return 0x7Eu;
  if (x <= -448.0f) return 0xFEu;
  const unsigned bits = __float_as_uint(x);
  const unsigned sign = bits >> 31;
  const unsigned mag = bits & 0x7fffffffu;
  if (mag == 0) return (unsigned char)(sign << 7);
  int exp = (int)(mag >> 23) - 127;
  if (exp >= -6) {
    const unsigned frac = mag & 0x7fffffu;
    unsigned m = frac >> 20;
    const unsigned rem = frac & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) ++m;
    int e = exp + 7;
    if (m == 8u) {
      m = 0u;
      ++e;
    }
    if (e > 15) return (unsigned char)((sign << 7) | 0x7Eu);
    return (unsigned char)((sign << 7) | ((unsigned)e << 3) | m);
  }
  // Subnormal target: q = |x| * 512 lies in [0, 8); the power-of-two scale
  // is exact, so the residual is exact in f32 too.
  const float q = __fmul_rn(__uint_as_float(mag), 512.0f);
  unsigned m = (unsigned)q;
  const float r = __fsub_rn(q, (float)m);
  if (r > 0.5f || (r == 0.5f && (m & 1u))) ++m;
  if (m == 8u) return (unsigned char)((sign << 7) | 0x08u);  // roll to 2^-6
  return (unsigned char)((sign << 7) | m);
}

// One block owns one token per grid-stride pass, act_and_mul style: the row
// base is x + token*2*hidden, gate half then up half. Each warp owns one
// 128-wide group per iteration; lane l covers local slots 4l..4l+3, so the
// warp xor reduction yields the group amax without shared memory. All 32
// lanes reach every shuffle; inactive tail lanes contribute 0.
__global__ void k_silu_mul_fp8_quant(const float* __restrict__ x,
                                     unsigned char* __restrict__ yq,
                                     unsigned char* __restrict__ ys,
                                     unsigned* __restrict__ status,
                                     const int tokens, const int hidden) {
  const int groups = (hidden + SMFQ_GROUP_SIZE - 1) / SMFQ_GROUP_SIZE;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status[0] = SMFQ_STATUS_MAGIC;
    status[1] = (unsigned)tokens;
    status[2] = SMFQ_NO_FAULT;
    status[3] = SMFQ_NO_FAULT;
  }
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int token = blockIdx.x; token < tokens; token += gridDim.x) {
    const float* row = x + (size_t)token * 2 * hidden;
    unsigned char* cq = yq + (size_t)token * hidden;
    unsigned char* cs = ys + (size_t)token * groups;
    for (int grp = warp; grp < groups; grp += SMFQ_WARPS) {
      const int lo = grp * SMFQ_GROUP_SIZE;
      const int hi = lo + SMFQ_GROUP_SIZE;
      const int valid = hi < hidden ? SMFQ_GROUP_SIZE : hidden - lo;
      float v[4];
      unsigned op_fault = SMFQ_NO_FAULT;  // flat row index into the 2H row
      unsigned mid_fault = SMFQ_NO_FAULT; // element index within H
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int i = lo + lane * 4 + k;
        if (i >= valid) {
          v[k] = 0.f;
          continue;
        }
        const float g = row[i];
        const float u = row[hidden + i];
        if (smfq_nonfinite(g)) {
          const unsigned flat = (unsigned)(token * 2 * hidden + i);
          op_fault = flat < op_fault ? flat : op_fault;
          v[k] = 0.f;
          continue;
        }
        if (smfq_nonfinite(u)) {
          const unsigned flat = (unsigned)(token * 2 * hidden + hidden + i);
          op_fault = flat < op_fault ? flat : op_fault;
          v[k] = 0.f;
          continue;
        }
        const float r = smfq_silu(g) * u;
        if (smfq_nonfinite(r)) {
          mid_fault = (unsigned)i < mid_fault ? (unsigned)i : mid_fault;
          v[k] = 0.f;
          continue;
        }
        v[k] = r;
      }
      float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])),
                         fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask));
      const int exp = smfq_group_exponent(amax);
      if (lane == 0) cs[grp] = smfq_encode_scale(exp);
      // Quarantine marker: on any fault publish zero codes and record the
      // first one; the host refuses the call from the status word.
      if (op_fault != SMFQ_NO_FAULT) atomicMin(status + 2, op_fault);
      if (mid_fault != SMFQ_NO_FAULT) atomicMin(status + 3, mid_fault);
      const bool bad = op_fault != SMFQ_NO_FAULT || mid_fault != SMFQ_NO_FAULT;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int i = lo + lane * 4 + k;
        if (i < valid)
          cq[i] = bad ? 0x00u : smfq_e4m3_encode(__ldexpf(v[k], -exp));
      }
    }
  }
}

// Admission mirrors the Go oracle's shape contract: non-null pointers,
// positive geometry, extents inside the sanity caps. The caller provides 16
// status bytes of device memory; they are pre-filled with 0xFF so a missing
// magic or a non-UINT_MAX fault word is always visible after sync.
extern "C" int fcuda_silu_mul_fp8_quant(const float* x, unsigned char* yq,
                                        unsigned char* ys, unsigned* status,
                                        int tokens, int hidden,
                                        cudaStream_t stream) {
  if (!x || !yq || !ys || !status || tokens < 1 || hidden < 1 ||
      tokens > SMFQ_MAX_TOKENS || hidden > SMFQ_MAX_HIDDEN) {
    return SMFQ_BAD_ARGS;
  }
  const int64_t rows = (int64_t)tokens * 2 * hidden;
  const int64_t codes = (int64_t)tokens * hidden;
  const int64_t scales =
      (int64_t)tokens * ((hidden + SMFQ_GROUP_SIZE - 1) / SMFQ_GROUP_SIZE);
  if (rows > SMFQ_MAX_CELLS || codes > SMFQ_MAX_CELLS || scales > SMFQ_MAX_CELLS)
    return SMFQ_BAD_ARGS;
  cudaError_t err = cudaMemsetAsync(status, 0xff, 16, stream);
  if (err != cudaSuccess) return SMFQ_CUDA_ERROR;
  unsigned blocks = (unsigned)tokens;
  if (blocks > 65535u) blocks = 65535u;
  err = k_silu_mul_fp8_quant<<<blocks, SMFQ_THREADS, 0, stream>>>(
      x, yq, ys, status, tokens, hidden);
  return err == cudaSuccess ? SMFQ_OK : SMFQ_CUDA_ERROR;
}

#undef SMFQ_OK
#undef SMFQ_BAD_ARGS
#undef SMFQ_CUDA_ERROR
#undef SMFQ_MAX_TOKENS
#undef SMFQ_MAX_HIDDEN
#undef SMFQ_MAX_CELLS
#undef SMFQ_GROUP_SIZE
#undef SMFQ_THREADS
#undef SMFQ_WARPS
#undef SMFQ_STATUS_MAGIC
#undef SMFQ_NO_FAULT
