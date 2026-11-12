/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Adapted from vllm-project/vllm at fa3b9d1377219247060d3c8b49eae199c9e7e154
 * (Apache-2.0). License and notices: THIRD_PARTY_NOTICES.md ("vLLM"); see also
 * third_party/vllm-gdn-decode/NOTICE.
 * FAK uses FP32 activations and one canonical [head,key,value] carried state.
 * The upstream 1..8-token warp-row recurrence is retained; shared-memory
 * transposition adapts the state without a persistent layout or allocation change.
 * FAK's independent-lane specialization reuses that arithmetic with one row and
 * one canonical carried state per lane; it does not reinterpret lanes as time.
 */

namespace fak_gdn_vllm {

constexpr int kDim = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxTokens = 8;
constexpr int kTileStride = kDim + 4;
static_assert((kDim * kChunkV + kChunkV * kTileStride +
               3 * kMaxTokens * kDim + 2 * kMaxTokens) * sizeof(float) <= 48 * 1024,
              "GDN post-conv shared storage must fit the default 48 KiB limit");

__device__ __forceinline__ void copy16(float* shared, const float* global) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               : : "r"(address), "l"(global));
#else
  *reinterpret_cast<float4*>(shared) = *reinterpret_cast<const float4*>(global);
#endif
}

__device__ __forceinline__ void commit_copy() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void wait_copy() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void copy_state_chunk(
    float* shared, const float* state, int chunk, int tid) {
  for (int copy = tid; copy < kDim * kChunkV / 4; copy += kThreads) {
    const int key = copy / (kChunkV / 4);
    const int value = (copy % (kChunkV / 4)) * 4;
    copy16(shared + key * kChunkV + value,
           state + key * kDim + chunk * kChunkV + value);
  }
  commit_copy();
}

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

struct Sum2 { float x, y; };

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
  }
  return x;
}

// Bounded by-value kernel arguments: no device pointer to a host pointer array.
// The caller already owns and validates one distinct recurrent state per lane.
struct IndependentLaneStates { float* recurrent[kMaxTokens]; };

template <int ValueHeadsPerKeyHead, bool IndependentLanes = false>
__global__ __launch_bounds__(kThreads, 1) void k_qwen35_gdn_post_conv_vllm(
    const float* __restrict__ mixed_qkv, const float* __restrict__ a,
    const float* __restrict__ b, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ state,
    const float* __restrict__ output_gate, const float* __restrict__ norm_weight,
    float* __restrict__ out, int tokens, int nK, int nV, int convDim, float norm_eps,
    IndependentLaneStates lane_states = {}) {
  // Independent decode lanes share weights, not carried state or time. The
  // default specialization retains the original contiguous-token addressing.
  if (IndependentLanes) {
    const size_t batch = static_cast<size_t>(blockIdx.y);
    mixed_qkv += batch * convDim;
    a += batch * nV;
    b += batch * nV;
    output_gate += batch * nV * kDim;
    out += batch * nV * kDim;
    state = lane_states.recurrent[batch];
    tokens = 1;
  }
  const int value_head = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int key_head = value_head / ValueHeadsPerKeyHead;
  __shared__ __align__(16) float canonical[kDim][kChunkV];
  __shared__ __align__(16) float tile[kChunkV][kTileStride];
  __shared__ __align__(16) float shared_q[kMaxTokens][kDim];
  __shared__ __align__(16) float shared_k[kMaxTokens][kDim];
  __shared__ float shared_out[kMaxTokens][kDim];
  __shared__ float shared_decay[kMaxTokens];
  __shared__ float shared_beta[kMaxTokens];
  float* head_state = state + static_cast<size_t>(value_head) * kDim * kDim;
  copy_state_chunk(&canonical[0][0], head_state, 0, tid);

  if (warp < tokens) {
    const int t = warp;
    const size_t base = static_cast<size_t>(t) * convDim;
    float q_values[4], k_values[4];
    float q_square = 0.0f, k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] = mixed_qkv[base + key_head * kDim + dim];
      k_values[i] = mixed_qkv[base + nK * kDim + key_head * kDim + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = rsqrtf(sums.x + 1.0e-6f) * rsqrtf(static_cast<float>(kDim));
    const float k_scale = rsqrtf(sums.y + 1.0e-6f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      shared_k[t][dim] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const size_t gate = static_cast<size_t>(t) * nV + value_head;
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a[gate] + dt_bias[value_head]);
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b[gate]);
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) rows[row] = warp + row * kWarps;

#pragma unroll
  for (int chunk = 0; chunk < kDim / kChunkV; ++chunk) {
    wait_copy();
    __syncthreads();
    for (int linear = tid; linear < kDim * kChunkV; linear += kThreads) {
      const int key = linear / kChunkV, value = linear % kChunkV;
      tile[value][key] = canonical[key][value];
    }
    __syncthreads();
    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float4 value = *reinterpret_cast<const float4*>(&tile[rows[row]][k_base]);
      h[row][0] = value.x; h[row][1] = value.y;
      h[row][2] = value.z; h[row][3] = value.w;
    }
    // Every reader is finished before next-chunk prefetch reuses canonical.
    __syncthreads();
    if (chunk + 1 < kDim / kChunkV) {
      copy_state_chunk(&canonical[0][0], head_state, chunk + 1, tid);
    }

    for (int t = 0; t < tokens; ++t) {
      const float4 q4 = *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
      const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};
      float dot_hk[kRowsPerWarp] = {};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= shared_decay[t];
          dot_hk[row] += h[row][i] * k_values[i];
        }
      }
      const Sum2 hk01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
      const Sum2 hk23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
      const float reduced_hk[kRowsPerWarp] = {hk01.x, hk01.y, hk23.x, hk23.y};
      float dot_hq[kRowsPerWarp] = {};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        const size_t v_offset = static_cast<size_t>(t) * convDim +
                                2 * nK * kDim + value_head * kDim + value;
        const float delta = (mixed_qkv[v_offset] - reduced_hk[row]) * shared_beta[t];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += k_values[i] * delta;
          dot_hq[row] += h[row][i] * q_values[i];
        }
      }
      const Sum2 hq01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
      const Sum2 hq23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
      if (lane == 0) {
        shared_out[t][chunk * kChunkV + rows[0]] = hq01.x;
        shared_out[t][chunk * kChunkV + rows[1]] = hq01.y;
        shared_out[t][chunk * kChunkV + rows[2]] = hq23.x;
        shared_out[t][chunk * kChunkV + rows[3]] = hq23.y;
      }
    }
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      *reinterpret_cast<float4*>(&tile[rows[row]][k_base]) =
          make_float4(h[row][0], h[row][1], h[row][2], h[row][3]);
    }
    __syncthreads();
    for (int copy = tid; copy < kDim * kChunkV / 4; copy += kThreads) {
      const int key = copy / (kChunkV / 4), value = (copy % (kChunkV / 4)) * 4;
      *reinterpret_cast<float4*>(head_state + key * kDim + chunk * kChunkV + value) =
          make_float4(tile[value][key], tile[value + 1][key],
                      tile[value + 2][key], tile[value + 3][key]);
    }
    // No next-chunk transpose may overwrite tile before canonical stores finish.
    __syncthreads();
  }

  if (warp < tokens) {
    const int t = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      output_values[i] = shared_out[t][lane + i * 32];
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd = rsqrtf(sum_square / static_cast<float>(kDim) + norm_eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const size_t offset = (static_cast<size_t>(t) * nV + value_head) * kDim + value;
      const float z = output_gate[offset];
      const float gate = z * sigmoid_fast(z);
      out[offset] = output_values[i] * rstd * norm_weight[value] * gate;
    }
  }
}

static bool eligible(const char* mode, int tokens, int nK, int nV,
                     int kHd, int vHd, bool capturing) {
  if (!mode || strcmp(mode, "vllm") != 0 || capturing || tokens < 1 ||
      tokens > kMaxTokens || nK <= 0 || nV <= 0 || nV % nK != 0 ||
      kHd != kDim || vHd != kDim) return false;
  const int ratio = nV / nK;
  return (ratio >= 1 && ratio <= 4) || ratio == 8;
}

template <int Ratio>
static int launch(const float* mixed, const float* a, const float* b,
                  const float* aLog, const float* dtBias, float* state,
                  const float* z, const float* norm, float* core,
                  int tokens, int nK, int nV, int convDim, float eps) {
  int status = 0;
  FCUDA_TS("k_qwen35_gdn_post_conv_vllm", nV,
      k_qwen35_gdn_post_conv_vllm<Ratio><<<nV, kThreads, 0, g_stream>>>(
          mixed, a, b, aLog, dtBias, state, z, norm, core,
          tokens, nK, nV, convDim, eps);
      status = qwen35_gdn_launch_status(5));
  return status;
}

static int dispatch(const float* mixed, const float* a, const float* b,
                    const float* aLog, const float* dtBias, float* state,
                    const float* z, const float* norm, float* core,
                    int tokens, int nK, int nV, int convDim, float eps) {
#define FAK_GDN_VLLM_CASE(ratio) case ratio: return launch<ratio>( \
    mixed, a, b, aLog, dtBias, state, z, norm, core, tokens, nK, nV, convDim, eps)
  switch (nV / nK) {
    FAK_GDN_VLLM_CASE(1);
    FAK_GDN_VLLM_CASE(2);
    FAK_GDN_VLLM_CASE(3);
    FAK_GDN_VLLM_CASE(4);
    FAK_GDN_VLLM_CASE(8);
    default: return -1;
  }
#undef FAK_GDN_VLLM_CASE
}

// Independent-lane adaptation of the retained Apache-2.0 vLLM port above.
// Grid.y selects a lane-owned state and one activation row; no allocation or
// public ABI change. Keep the ordinary contiguous dispatch separate.
template <int Ratio>
static int launch_lanes(const float* mixed, const float* a, const float* b,
                        const float* aLog, const float* dtBias, IndependentLaneStates states,
                        const float* z, const float* norm, float* core,
                        int lanes, int nK, int nV, int convDim, float eps) {
  const dim3 grid((unsigned)nV, (unsigned)lanes);
  int status = 0;
  FCUDA_TS("k_qwen35_gdn_post_conv_vllm", grid,
      k_qwen35_gdn_post_conv_vllm<Ratio, true><<<grid, kThreads, 0, g_stream>>>(
          mixed, a, b, aLog, dtBias, nullptr, z, norm, core,
          1, nK, nV, convDim, eps, states);
      status = qwen35_gdn_launch_status(5));
  return status;
}

static int dispatch_lanes(const float* mixed, const float* a, const float* b,
                          const float* aLog, const float* dtBias, IndependentLaneStates states,
                          const float* z, const float* norm, float* core,
                          int lanes, int nK, int nV, int convDim, float eps) {
  if (lanes < 1 || lanes > kMaxTokens || nK <= 0 || nV <= 0 || nV % nK != 0)
    return -1;
  for (int lane = 0; lane < lanes; ++lane) if (!states.recurrent[lane]) return -1;
#define FAK_GDN_VLLM_LANES_CASE(ratio) case ratio: return launch_lanes<ratio>( \
    mixed, a, b, aLog, dtBias, states, z, norm, core, lanes, nK, nV, convDim, eps)
  switch (nV / nK) {
    FAK_GDN_VLLM_LANES_CASE(1);
    FAK_GDN_VLLM_LANES_CASE(2);
    FAK_GDN_VLLM_LANES_CASE(3);
    FAK_GDN_VLLM_LANES_CASE(4);
    FAK_GDN_VLLM_LANES_CASE(8);
    default: return -1;
  }
#undef FAK_GDN_VLLM_LANES_CASE
}

}  // namespace fak_gdn_vllm
