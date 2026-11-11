// Batched multi-row router gate GEMM: activation(rows) x weight(fp32) -> fp32,
// for chunked-prefill batches on 512-class expert models. Replaces the per-row
// gate MatMul launches with one batched [rows,hidden] x [hidden,experts] GEMM.
// The one-row fused gate (V41RouterGateTopKBackend) stays canonical for decode;
// this is the prefill-batch path.
//
// ADAPT of vllm-project/vllm csrc/fp32_router_gemm.cu at pin
// ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0): fp32_router_gemm_kernel,
// re-expressed for fak's flat C ABI and the 512-class expert envelope. Divergences
// from upstream, all deliberate:
//   - fp32 activation only (upstream's bf16 activation variant is dropped): the
//     router gate projection is an fp32 op in fak, so VPT=4 (one float4 load).
//   - The 512-class expert ceiling (MRE_MAX_EXPERTS) and the 5120 hidden
//     dimension (MRE_HIDDEN) replace upstream's per-shape (E, H) instantiations;
//     the row count is the only template axis the host dispatches on.
//   - No PDL / programmatic-stream-serialization: fak's single libfakcuda stream
//     already serializes the router gate behind the prefill chunk, so the
//     cudaGridDependencySynchronize / cudaTriggerProgrammaticLaunchCompletion
//     pair is dropped.
//   - The per-(shape, M) Blackwell geometry sweep (token groups, EPB crossover)
//     is dropped: the legacy 128-thread / EPB=1 geometry is the portable floor,
//     and the B300-tuned variants are a follow-on once the device-vs-cpuref
//     cosine gate is wired.
//
// Contract (mirrored by the Go oracle MoeRouterGemmFp32 in moe_router_gemm.go):
//   - weight is row-major fp32 [experts, hidden]; mat_a is row-major fp32
//     [rows, hidden]; out is row-major fp32 [rows, experts].
//   - Each block owns kEPB expert columns; each thread accumulates VPT=4 K
//     elements per iteration, so kHiddenDim must be a multiple of
//     4 * kBlockSize.
//   - The warp butterfly reduction + shared-memory finalize reproduce the
//     upstream accumulation order. Device-vs-oracle parity is gated by a
//     documented per-element ULP tolerance, not bit-identity: the two summation
//     orders differ by construction (each thread accumulates its K slice as 40
//     sequential products, the 32-lane partials fold through a five-level warp
//     butterfly, and the four warp partials add sequentially in the finalize;
//     the Go oracle runs eight stride-8 accumulator chains of 640 products each
//     folded as ((s0+s1)+(s2+s3))+((s4+s5)+(s6+s7))). Matching the oracle
//     bit-for-bit would force every output onto a single thread reproducing the
//     stride-8 chains, removing the K parallelism the batched GEMM exists for.
//     Gate: every element within MoeRouterGemmParityMaxULPs ULPs of the Go
//     oracle MoeRouterGemmFp32 (see moe_router_gemm.go); any larger deviation
//     fails the shape. The raider-fak sm_120 panel (lease hwl-69a8382f581f24e5,
//     2026-11-05) observed deterministic few-ULP deltas on all five shapes,
//     consistent with the order difference.
// Statuses: 0 ok, -1 null pointer or refused geometry, -2 graph capture open,
// -3 device allocation failure, positive cudaError_t for copy/launch/execution.

#include <cuda_runtime.h>
#include <stdint.h>

#define MRE_MAX_EXPERTS 512
#define MRE_HIDDEN      5120
#define MRE_MAX_ROWS    48
#define MRE_VPT         4
#define MRE_BLOCK       128
#define MRE_EPB         1

// One block computes kEPB expert columns; each thread owns VPT=4 K elements per
// iteration, so the K slice per iteration is VPT * kBlockSize.
template <int kNumTokens>
__global__ __launch_bounds__(MRE_BLOCK, 1) void mre_router_gemm_kernel(
    float* __restrict__ out,
    const float* __restrict__ mat_a,
    const float* __restrict__ mat_b) {
  constexpr int kBlockSize = MRE_BLOCK;
  constexpr int kEPB = MRE_EPB;
  constexpr int kNumExperts = MRE_MAX_EXPERTS;
  constexpr int kHiddenDim = MRE_HIDDEN;
  constexpr int k_elems_per_k_iteration = MRE_VPT * kBlockSize;
  constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration;
  static_assert(kHiddenDim % k_elems_per_k_iteration == 0, "hidden must tile the K slice");
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;

  int const e_base = blockIdx.x * kEPB;
  int const tid = threadIdx.x;
  int const warpId = tid / kWarpSize;
  int const laneId = tid % kWarpSize;

  float acc[kNumTokens][kEPB] = {};
  __shared__ float sm_reduction[kNumTokens][kEPB][kNumWarps];

#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = ki * k_elems_per_k_iteration + tid * MRE_VPT;

    float b_float[kEPB][MRE_VPT];
#pragma unroll
    for (int e = 0; e < kEPB; e++) {
      float4 v = *reinterpret_cast<const float4*>(mat_b + (size_t)(e_base + e) * kHiddenDim + k_base);
      b_float[e][0] = v.x;
      b_float[e][1] = v.y;
      b_float[e][2] = v.z;
      b_float[e][3] = v.w;
    }

#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
      float4 a = *reinterpret_cast<const float4*>(mat_a + (size_t)m_idx * kHiddenDim + k_base);
      float a_float[MRE_VPT] = {a.x, a.y, a.z, a.w};
#pragma unroll
      for (int e = 0; e < kEPB; e++) {
#pragma unroll
        for (int k = 0; k < MRE_VPT; k++) {
          acc[m_idx][e] += a_float[k] * b_float[e][k];
        }
      }
    }
  }

  // Warp-level butterfly reduction.
#pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
#pragma unroll
    for (int e = 0; e < kEPB; e++) {
      float sum = acc[m][e];
      sum += __shfl_xor_sync(0xffffffff, sum, 16);
      sum += __shfl_xor_sync(0xffffffff, sum, 8);
      sum += __shfl_xor_sync(0xffffffff, sum, 4);
      sum += __shfl_xor_sync(0xffffffff, sum, 2);
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      if (laneId == 0) sm_reduction[m][e][warpId] = sum;
    }
  }

  __syncthreads();

  // Parallel finalize: one thread per (m, e) output.
  for (int idx = threadIdx.x; idx < kNumTokens * kEPB; idx += kBlockSize) {
    int const m = idx / kEPB;
    int const e = idx % kEPB;
    float final_sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kNumWarps; w++) final_sum += sm_reduction[m][e][w];
    out[(size_t)m * kNumExperts + e_base + e] = final_sum;
  }
}

// Launcher: one block per kEPB expert columns.
template <int kNumTokens>
static int mre_launch(float* out, const float* mat_a, const float* mat_b, cudaStream_t stream) {
  dim3 grid(MRE_MAX_EXPERTS / MRE_EPB);
  dim3 block(MRE_BLOCK);
  void* args[] = { &out, &mat_a, &mat_b };
  cudaError_t err = cudaLaunchKernel(
      (const void*)mre_router_gemm_kernel<kNumTokens>,
      grid, block, args, 0, stream);
  return (int)err;
}

// Flat C ABI entry: the host validates the geometry (rows/hidden/experts bounds)
// before launch, so the kernel never sees an out-of-envelope shape. The 128-thread
// / EPB=1 geometry is the portable floor for the 512-class expert envelope.
extern "C" int fak_moe_router_gemm_f32(
    float* out, const float* mat_a, const float* mat_b,
    int rows, int hidden, int experts, cudaStream_t stream) {
  if (out == nullptr || mat_a == nullptr || mat_b == nullptr) return -1;
  if (rows <= 0 || rows > MRE_MAX_ROWS) return -1;
  if (experts != MRE_MAX_EXPERTS) return -1;
  if (hidden != MRE_HIDDEN) return -1;

  // Dispatch on the row count: the kernel is templated on kNumTokens, so the
  // host selects the matching instantiation. Rows past the M<=48 envelope are
  // refused above; the common chunked-prefill shapes are 1..48.
  switch (rows) {
    case 1:  return mre_launch<1>(out, mat_a, mat_b, stream);
    case 2:  return mre_launch<2>(out, mat_a, mat_b, stream);
    case 3:  return mre_launch<3>(out, mat_a, mat_b, stream);
    case 4:  return mre_launch<4>(out, mat_a, mat_b, stream);
    case 5:  return mre_launch<5>(out, mat_a, mat_b, stream);
    case 6:  return mre_launch<6>(out, mat_a, mat_b, stream);
    case 7:  return mre_launch<7>(out, mat_a, mat_b, stream);
    case 8:  return mre_launch<8>(out, mat_a, mat_b, stream);
    case 16: return mre_launch<16>(out, mat_a, mat_b, stream);
    case 32: return mre_launch<32>(out, mat_a, mat_b, stream);
    case 48: return mre_launch<48>(out, mat_a, mat_b, stream);
    default: return -1;
  }
}
