// Group-constrained MoE top-k routing (noaux_tc style) CUDA kernel.
//
// UNWIRED: standalone CUDA kernel fragment not #included into cuda_kernels.cu
// and not declared in cuda_backend.h. Remains gated on qualification before
// wiring into libfakcuda.
//
// ADAPT of vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41
// (csrc/moe/grouped_topk_kernels.cu:480,639,922,1095, Apache-2.0):
// grouped_topk_fused_kernel family re-expressed for fak's flat C ABI.
// Divergences from upstream, deliberate:
//   - Ties in both group score and expert choice score resolve deterministically
//     by ASCENDING index (lower group / lower expert ID wins).
//   - Maximum/selection accumulators initialize to -INFINITY (never a clamped
//     finite sentinel).
//   - Grouped score sums top-2 values per group when experts_per_group >= 2
//     (and takes the lone value when experts_per_group == 1), selects top
//     topk_group groups, then selects top_k experts across kept groups.
//
// Statuses: 0 ok, -1 invalid pointer/geometry, positive cudaError_t on launch.

#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define MGT_SCORING_SOFTMAX 0
#define MGT_SCORING_SIGMOID 1
#define MGT_THREADS 256
#define MGT_MAX_EXPERTS 1024
#define MGT_MAX_GROUPS 64
#define MGT_MAX_TOPK 64

__device__ __forceinline__ static bool mgt_better(float score_a, int idx_a, float score_b, int idx_b) {
  return score_a > score_b || (score_a == score_b && idx_a < idx_b);
}

// k_mgt_grouped_topk_f32 runs group-constrained top-k routing per token row.
__global__ void __launch_bounds__(MGT_THREADS) k_mgt_grouped_topk_f32(
    const float* __restrict__ logits,
    const float* __restrict__ correction_bias,
    float* __restrict__ weights_out,
    int32_t* __restrict__ indices_out,
    int32_t* __restrict__ status_out,
    int32_t num_tokens,
    int32_t num_experts,
    int32_t num_groups,
    int32_t topk_group,
    int32_t top_k,
    int32_t scoring,
    int32_t renormalize,
    float routed_scaling_factor) {
  __shared__ float s_scores[MGT_MAX_EXPERTS];
  __shared__ float s_choice[MGT_MAX_EXPERTS];
  __shared__ float s_group_score[MGT_MAX_GROUPS];
  __shared__ uint8_t s_keep_group[MGT_MAX_GROUPS];
  __shared__ float s_warp_val[MGT_THREADS / 32];
  __shared__ int32_t s_bad;

  const int32_t row = (int32_t)blockIdx.x;
  if (row >= num_tokens) return;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = MGT_THREADS / 32;
  const float* x = logits + (size_t)row * (size_t)num_experts;

  if (tid == 0) s_bad = 0;
  __syncthreads();

  float local_max = -INFINITY;
  for (int32_t e = tid; e < num_experts; e += MGT_THREADS) {
    const float v = x[e];
    if (!isfinite(v)) {
      atomicOr(&s_bad, 1);
    } else {
      local_max = fmaxf(local_max, v);
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, off));
  }
  if (lane == 0) s_warp_val[warp] = local_max;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < nwarps) ? s_warp_val[lane] : -INFINITY;
    for (int off = 16; off > 0; off >>= 1) {
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    if (lane == 0) s_warp_val[0] = v;
  }
  __syncthreads();
  if (s_bad != 0) {
    if (tid == 0 && status_out) status_out[row] = 1;
    return;
  }
  const float row_max = s_warp_val[0];

  if (scoring == MGT_SCORING_SOFTMAX) {
    float local_sum = 0.0f;
    for (int32_t e = tid; e < num_experts; e += MGT_THREADS) {
      const float ex = expf(x[e] - row_max);
      s_scores[e] = ex;
      local_sum += ex;
    }
    for (int off = 16; off > 0; off >>= 1) {
      local_sum += __shfl_xor_sync(0xffffffffu, local_sum, off);
    }
    if (lane == 0) s_warp_val[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
      float v = (lane < nwarps) ? s_warp_val[lane] : 0.0f;
      for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
      }
      if (lane == 0) s_warp_val[0] = v;
    }
    __syncthreads();
    const float inv_sum = 1.0f / s_warp_val[0];
    for (int32_t e = tid; e < num_experts; e += MGT_THREADS) {
      s_scores[e] *= inv_sum;
    }
  } else {
    for (int32_t e = tid; e < num_experts; e += MGT_THREADS) {
      s_scores[e] = 1.0f / (1.0f + expf(-x[e]));
    }
  }
  __syncthreads();

  for (int32_t e = tid; e < num_experts; e += MGT_THREADS) {
    s_choice[e] = correction_bias ? (s_scores[e] + correction_bias[e]) : s_scores[e];
  }
  __syncthreads();

  const int32_t experts_per_group = num_experts / num_groups;
  if (tid < num_groups) {
    const int32_t g = tid;
    float best = -INFINITY;
    float second = -INFINITY;
    const int32_t base = g * experts_per_group;
    for (int32_t i = 0; i < experts_per_group; ++i) {
      const float c = s_choice[base + i];
      if (c > best) {
        second = best;
        best = c;
      } else if (c > second) {
        second = c;
      }
    }
    s_group_score[g] = (experts_per_group >= 2) ? (best + second) : best;
    s_keep_group[g] = (num_groups == 1) ? 1u : 0u;
  }
  __syncthreads();

  if (tid == 0) {
    if (num_groups > 1) {
      for (int32_t kg = 0; kg < topk_group; ++kg) {
        float best_gs = -INFINITY;
        int32_t best_g = INT32_MAX;
        for (int32_t g = 0; g < num_groups; ++g) {
          if (s_keep_group[g]) continue;
          if (mgt_better(s_group_score[g], g, best_gs, best_g)) {
            best_gs = s_group_score[g];
            best_g = g;
          }
        }
        s_keep_group[best_g] = 1u;
      }
    }

    int32_t picked[MGT_MAX_TOPK];
    float picked_w[MGT_MAX_TOPK];
    float wsum = 0.0f;
    for (int32_t k = 0; k < top_k; ++k) {
      float best_c = -INFINITY;
      int32_t best_e = INT32_MAX;
      for (int32_t e = 0; e < num_experts; ++e) {
        if (!s_keep_group[e / experts_per_group]) continue;
        bool taken = false;
        for (int32_t prev = 0; prev < k; ++prev) {
          if (picked[prev] == e) {
            taken = true;
            break;
          }
        }
        if (!taken && mgt_better(s_choice[e], e, best_c, best_e)) {
          best_c = s_choice[e];
          best_e = e;
        }
      }
      picked[k] = best_e;
      picked_w[k] = s_scores[best_e];
      wsum += picked_w[k];
    }

    const float norm = (renormalize && wsum > 0.0f) ? (1.0f / wsum) : 1.0f;
    float* w_out = weights_out + (size_t)row * (size_t)top_k;
    int32_t* i_out = indices_out + (size_t)row * (size_t)top_k;
    for (int32_t k = 0; k < top_k; ++k) {
      i_out[k] = picked[k];
      w_out[k] = picked_w[k] * norm * routed_scaling_factor;
    }
    if (status_out) status_out[row] = 0;
  }
}

// Flat C ABI entry point: validates grouped routing invariants before launch.
extern "C" int fcuda_moe_grouped_topk_f32(
    const float* d_logits,
    const float* d_correction_bias,
    float* d_weights_out,
    int32_t* d_indices_out,
    int32_t* d_status_out,
    int32_t num_tokens,
    int32_t num_experts,
    int32_t num_groups,
    int32_t topk_group,
    int32_t top_k,
    int32_t scoring,
    int32_t renormalize,
    float routed_scaling_factor,
    cudaStream_t stream) {
  if (!d_logits || !d_weights_out || !d_indices_out) return -1;
  if (num_tokens <= 0 || num_experts <= 0 || num_experts > MGT_MAX_EXPERTS) return -1;
  if (num_groups <= 0 || num_groups > MGT_MAX_GROUPS || (num_experts % num_groups) != 0) return -1;
  if (topk_group <= 0 || topk_group > num_groups) return -1;
  const int32_t experts_per_group = num_experts / num_groups;
  if (top_k <= 0 || top_k > topk_group * experts_per_group || top_k > MGT_MAX_TOPK) return -1;
  if (scoring != MGT_SCORING_SOFTMAX && scoring != MGT_SCORING_SIGMOID) return -1;
  if (!isfinite(routed_scaling_factor) || routed_scaling_factor < 0.0f) return -1;

  (void)cudaGetLastError();
  k_mgt_grouped_topk_f32<<<(unsigned int)num_tokens, MGT_THREADS, 0, stream>>>(
      d_logits, d_correction_bias, d_weights_out, d_indices_out, d_status_out,
      num_tokens, num_experts, num_groups, topk_group, top_k,
      scoring, renormalize, routed_scaling_factor);
  const cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? 0 : (int)err;
}
