// General device top-k gating with softmax/sigmoid scoring and optional
// renormalization + bias-assisted selection for non-V4.1 MoE routers.
//
// UNWIRED: standalone CUDA kernel fragment not #included into cuda_kernels.cu
// and not declared in cuda_backend.h. Remains gated behind qualification and a
// served non-V4.1 MoE model landing before wiring into libfakcuda.
//
// moeSigmoid, moeTopK, and topkGating re-expressed for fak's flat C ABI.
// Divergences from upstream, deliberate:
//   - Ties in selection score resolve deterministically by ASCENDING expert
//     index (lower expert ID wins), matching the Go host oracle.
//   - Row max reduction initializes to -INFINITY (never a clamped finite
//     sentinel) so arbitrary finite negative logits normalize without clamping.
//   - Supports optional correction_bias for selection while emitting unbiased
//     scores (noaux style) and optional top-k weight renormalization.
//
// Contract (mirrored by the Go oracle MoeTopKGating in moe_topk_gating.go):
//   - logits is row-major float32 [num_tokens, num_experts].
//   - weights_out is row-major float32 [num_tokens, top_k].
//   - indices_out is row-major int32_t [num_tokens, top_k].
//   - status_out (optional, may be null) records per-row non-finite refusals.
// Statuses: 0 ok, -1 invalid pointer/geometry, positive cudaError_t on launch.

#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define MTG_SCORING_SOFTMAX 0
#define MTG_SCORING_SIGMOID 1
#define MTG_THREADS 256
#define MTG_MAX_TOPK 64

__device__ __forceinline__ static bool mtg_better(float score_a, int idx_a, float score_b, int idx_b) {
  return score_a > score_b || (score_a == score_b && idx_a < idx_b);
}

// k_mtg_softmax_f32 computes row-wise numerically stable softmax in place or
// out-of-place across [num_tokens, num_experts].
__global__ void __launch_bounds__(MTG_THREADS) k_mtg_softmax_f32(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    int32_t num_tokens,
    int32_t num_experts) {
  __shared__ float s_warp[MTG_THREADS / 32];
  const int32_t row = (int32_t)blockIdx.x;
  if (row >= num_tokens) return;

  const float* x = logits + (size_t)row * (size_t)num_experts;
  float* y = probs + (size_t)row * (size_t)num_experts;
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = MTG_THREADS / 32;

  float local_max = -INFINITY;
  for (int32_t e = tid; e < num_experts; e += MTG_THREADS) {
    local_max = fmaxf(local_max, x[e]);
  }
  for (int off = 16; off > 0; off >>= 1) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, off));
  }
  if (lane == 0) s_warp[warp] = local_max;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < nwarps) ? s_warp[lane] : -INFINITY;
    for (int off = 16; off > 0; off >>= 1) {
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    if (lane == 0) s_warp[0] = v;
  }
  __syncthreads();
  const float row_max = s_warp[0];

  float local_sum = 0.0f;
  for (int32_t e = tid; e < num_experts; e += MTG_THREADS) {
    const float ex = expf(x[e] - row_max);
    y[e] = ex;
    local_sum += ex;
  }
  for (int off = 16; off > 0; off >>= 1) {
    local_sum += __shfl_xor_sync(0xffffffffu, local_sum, off);
  }
  if (lane == 0) s_warp[warp] = local_sum;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < nwarps) ? s_warp[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    if (lane == 0) s_warp[0] = v;
  }
  __syncthreads();
  const float inv_sum = (s_warp[0] > 0.0f) ? (1.0f / s_warp[0]) : 0.0f;
  for (int32_t e = tid; e < num_experts; e += MTG_THREADS) {
    y[e] *= inv_sum;
  }
}

// k_mtg_sigmoid_f32 computes elementwise sigmoid scores over [num_tokens, num_experts].
__global__ void __launch_bounds__(MTG_THREADS) k_mtg_sigmoid_f32(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    int64_t total) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
    probs[i] = 1.0f / (1.0f + expf(-logits[i]));
  }
}

// k_mtg_topk_gating_f32 performs fused score evaluation + deterministic top-k
// selection + optional weight renormalization and routed scaling per token row.
__global__ void __launch_bounds__(MTG_THREADS) k_mtg_topk_gating_f32(
    const float* __restrict__ logits,
    const float* __restrict__ correction_bias,
    float* __restrict__ weights_out,
    int32_t* __restrict__ indices_out,
    int32_t* __restrict__ status_out,
    int32_t num_tokens,
    int32_t num_experts,
    int32_t top_k,
    int32_t scoring,
    int32_t renormalize,
    float routed_scaling_factor) {
  __shared__ float s_warp_val[MTG_THREADS / 32];
  __shared__ int32_t s_warp_idx[MTG_THREADS / 32];
  __shared__ int32_t s_bad;
  __shared__ int32_t s_picked_idx[MTG_MAX_TOPK];
  __shared__ float s_picked_w[MTG_MAX_TOPK];

  const int32_t row = (int32_t)blockIdx.x;
  if (row >= num_tokens) return;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = MTG_THREADS / 32;
  const float* x = logits + (size_t)row * (size_t)num_experts;

  if (tid == 0) s_bad = 0;
  __syncthreads();

  float local_max = -INFINITY;
  for (int32_t e = tid; e < num_experts; e += MTG_THREADS) {
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

  float softmax_sum = 1.0f;
  if (scoring == MTG_SCORING_SOFTMAX) {
    float local_sum = 0.0f;
    for (int32_t e = tid; e < num_experts; e += MTG_THREADS) {
      local_sum += expf(x[e] - row_max);
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
    softmax_sum = s_warp_val[0];
  }

  // Iterative top-k selection with deterministic ascending-index tie-break.
  for (int32_t k = 0; k < top_k; ++k) {
    float best_choice = -INFINITY;
    int32_t best_idx = INT32_MAX;
    for (int32_t e = tid; e < num_experts; e += MTG_THREADS) {
      bool taken = false;
      for (int32_t prev = 0; prev < k; ++prev) {
        if (s_picked_idx[prev] == e) {
          taken = true;
          break;
        }
      }
      if (taken) continue;
      const float score = (scoring == MTG_SCORING_SOFTMAX)
                              ? (expf(x[e] - row_max) / softmax_sum)
                              : (1.0f / (1.0f + expf(-x[e])));
      const float choice = correction_bias ? (score + correction_bias[e]) : score;
      if (mtg_better(choice, e, best_choice, best_idx)) {
        best_choice = choice;
        best_idx = e;
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      const float o_val = __shfl_xor_sync(0xffffffffu, best_choice, off);
      const int32_t o_idx = __shfl_xor_sync(0xffffffffu, best_idx, off);
      if (mtg_better(o_val, o_idx, best_choice, best_idx)) {
        best_choice = o_val;
        best_idx = o_idx;
      }
    }
    if (lane == 0) {
      s_warp_val[warp] = best_choice;
      s_warp_idx[warp] = best_idx;
    }
    __syncthreads();
    if (warp == 0) {
      float v = (lane < nwarps) ? s_warp_val[lane] : -INFINITY;
      int32_t idx = (lane < nwarps) ? s_warp_idx[lane] : INT32_MAX;
      for (int off = 16; off > 0; off >>= 1) {
        const float o_val = __shfl_xor_sync(0xffffffffu, v, off);
        const int32_t o_idx = __shfl_xor_sync(0xffffffffu, idx, off);
        if (mtg_better(o_val, o_idx, v, idx)) {
          v = o_val;
          idx = o_idx;
        }
      }
      if (lane == 0) {
        s_picked_idx[k] = idx;
        const float raw_score = (scoring == MTG_SCORING_SOFTMAX)
                                    ? (expf(x[idx] - row_max) / softmax_sum)
                                    : (1.0f / (1.0f + expf(-x[idx])));
        s_picked_w[k] = raw_score;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    float wsum = 0.0f;
    for (int32_t k = 0; k < top_k; ++k) wsum += s_picked_w[k];
    const float norm = (renormalize && wsum > 0.0f) ? (1.0f / wsum) : 1.0f;
    float* w_out = weights_out + (size_t)row * (size_t)top_k;
    int32_t* i_out = indices_out + (size_t)row * (size_t)top_k;
    for (int32_t k = 0; k < top_k; ++k) {
      i_out[k] = s_picked_idx[k];
      w_out[k] = s_picked_w[k] * norm * routed_scaling_factor;
    }
    if (status_out) status_out[row] = 0;
  }
}

// Flat C ABI entry point: validates geometry before launching k_mtg_topk_gating_f32.
extern "C" int fcuda_moe_topk_gating_f32(
    const float* d_logits,
    const float* d_correction_bias,
    float* d_weights_out,
    int32_t* d_indices_out,
    int32_t* d_status_out,
    int32_t num_tokens,
    int32_t num_experts,
    int32_t top_k,
    int32_t scoring,
    int32_t renormalize,
    float routed_scaling_factor,
    cudaStream_t stream) {
  if (!d_logits || !d_weights_out || !d_indices_out) return -1;
  if (num_tokens <= 0 || num_experts <= 0) return -1;
  if (top_k <= 0 || top_k > num_experts || top_k > MTG_MAX_TOPK) return -1;
  if (scoring != MTG_SCORING_SOFTMAX && scoring != MTG_SCORING_SIGMOID) return -1;
  if (!isfinite(routed_scaling_factor) || routed_scaling_factor < 0.0f) return -1;

  (void)cudaGetLastError();
  k_mtg_topk_gating_f32<<<(unsigned int)num_tokens, MTG_THREADS, 0, stream>>>(
      d_logits, d_correction_bias, d_weights_out, d_indices_out, d_status_out,
      num_tokens, num_experts, top_k, scoring, renormalize, routed_scaling_factor);
  const cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? 0 : (int)err;
}
