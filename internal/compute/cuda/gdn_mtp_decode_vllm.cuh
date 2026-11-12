/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from vllm-project/vllm at ae065918640a198a56f6f17eb45e251c3229fe41
 * (Apache-2.0). License and notices: THIRD_PARTY_NOTICES.md ("vLLM"). The
 * accepted-token replay idea is re-expressed below with no copied code blocks.
 * FAK keeps FP32 activations and one canonical [head,key,value] carried state
 * (upstream is BF16), the SiLU output gate, the cp.async pipeline on SM80+
 * with a synchronous vector copy below, and head ratios 1,2,3,4,8.
 *
 * Behavior: the state pool holds [slot][head][key][value] FP32 and slot 0 is
 * the checkpoint carried state. snapshots[t] == 0 rejects panel position t:
 * the carried state and the output row stay untouched. snapshots[t] > 0
 * accepts it: the carried state advances over that token only, and the
 * post-update state is snapshotted into pool slot snapshots[t]. Slot 0 is
 * written back with the post-panel carried state. See
 * third_party/vllm-gdn-decode/NOTICE for the base-kernel notice.
 */

namespace fak_gdn_vllm_mtp {

constexpr int kDim = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxTokens = 8;
constexpr int kTileStride = kDim + 4;
static_assert((kDim * kChunkV + kChunkV * kTileStride +
               3 * kMaxTokens * kDim + 2 * kMaxTokens) * sizeof(float) +
               kMaxTokens * sizeof(int) <= 48 * 1024,
              "GDN MTP post-conv shared storage must fit the default 48 KiB limit");

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

// Selection gate for the MTP form: the contiguous-panel shape gate plus the
// accepted-count bound. Env opt-in, capture exclusion and shape admission
// match the base gate; 0 <= accepted <= tokens admits the replay form
// (accepted == 0 launches a no-op that touches neither state nor outputs).
static bool eligible_mtp(const char* mode, int tokens, int nK, int nV,
                         int kHd, int vHd, bool capturing, int accepted) {
  if (!mode || strcmp(mode, "vllm") != 0 || capturing || tokens < 1 ||
      tokens > kMaxTokens || nK <= 0 || nV <= 0 || nV % nK != 0 ||
      kHd != kDim || vHd != kDim || accepted < 0 || accepted > tokens)
    return false;
  const int ratio = nV / nK;
  return (ratio >= 1 && ratio <= 4) || ratio == 8;
}

// One block per value head replays the verification panel over the checkpoint
// carried state. Only accepted positions advance the state; each accepted
// position snapshots its post-update state into its destination pool slot.
// The arithmetic mirrors k_qwen35_gdn_post_conv_vllm element-for-element so an
// all-accepted panel stays bitwise the contiguous-panel path.
template <int ValueHeadsPerKeyHead>
__global__ __launch_bounds__(kThreads, 1) void k_qwen35_gdn_mtp_post_conv_vllm(
    const float* __restrict__ mixed_qkv, const float* __restrict__ a,
    const float* __restrict__ b, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ state_pool,
    const float* __restrict__ output_gate, const float* __restrict__ norm_weight,
    float* __restrict__ out, int tokens, int nK, int nV, int convDim,
    float norm_eps, const int* __restrict__ snapshots) {
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
  __shared__ int shared_snap[kMaxTokens];
  float* head_state = state_pool + static_cast<size_t>(value_head) * kDim * kDim;
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
      shared_snap[t] = snapshots[t];
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
      if (shared_snap[t] == 0) continue;  // rejected: carried state untouched
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
      // Destination snapshot of this chunk's rows for the accepted token.
      // The pool is [slot][head][key][value], so each lane stores its four
      // key columns at the fixed value row rather than one packed float4.
      float* dest = state_pool +
          static_cast<size_t>(shared_snap[t]) * nV * kDim * kDim +
          static_cast<size_t>(value_head) * kDim * kDim;
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
#pragma unroll
        for (int i = 0; i < 4; ++i) dest[(k_base + i) * kDim + value] = h[row][i];
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

  if (warp < tokens && shared_snap[warp] != 0) {
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

template <int Ratio>
static int launch_mtp(const float* mixed, const float* a, const float* b,
                      const float* aLog, const float* dtBias, float* statePool,
                      const float* z, const float* norm, float* core,
                      int tokens, int nK, int nV, int convDim, float eps,
                      const int* snapshots) {
  int status = 0;
  FCUDA_TS("k_qwen35_gdn_mtp_post_conv_vllm", nV,
      k_qwen35_gdn_mtp_post_conv_vllm<Ratio><<<nV, kThreads, 0, g_stream>>>(
          mixed, a, b, aLog, dtBias, statePool, z, norm, core,
          tokens, nK, nV, convDim, eps, snapshots);
      status = qwen35_gdn_launch_status(8));
  return status;
}

static int dispatch_mtp(const float* mixed, const float* a, const float* b,
                        const float* aLog, const float* dtBias, float* statePool,
                        const float* z, const float* norm, float* core,
                        int tokens, int nK, int nV, int convDim, float eps,
                        const int* snapshots) {
#define FAK_GDN_MTP_VLLM_CASE(ratio) case ratio: return launch_mtp<ratio>( \
    mixed, a, b, aLog, dtBias, statePool, z, norm, core, tokens, nK, nV, \
    convDim, eps, snapshots)
  switch (nV / nK) {
    FAK_GDN_MTP_VLLM_CASE(1);
    FAK_GDN_MTP_VLLM_CASE(2);
    FAK_GDN_MTP_VLLM_CASE(3);
    FAK_GDN_MTP_VLLM_CASE(4);
    FAK_GDN_MTP_VLLM_CASE(8);
    default: return -1;
  }
#undef FAK_GDN_MTP_VLLM_CASE
}

}  // namespace fak_gdn_vllm_mtp
