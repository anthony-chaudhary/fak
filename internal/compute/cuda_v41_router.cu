// Included by cuda_kernels.cu, not a separate translation unit.
//
// DeepSeek-V4.1 MoE router top-k on CUDA, scored and token-hashed layers.
// Adapted from antirez/ds4 ds4_cuda.cu router_select_warp_topk_kernel and
// router_select_kernel at 80ebbc396aee40eedc1d829222f3362d10fa4c6c (MIT, The
// ds4.c authors; see THIRD_PARTY_NOTICES.md): one warp per token row, four rows
// per block, k rounds of a __shfl_xor_sync argmax with the lower expert first
// on equal values, and a hash mode that takes the row's experts from a
// token-to-expert table instead of selecting them.
//
// Changed for Fak: ds4 hard-codes 256 experts, k=6, scale 1.5, softplus via
// log1pf(expf) and a fmaxf(sum, 2^-14) floor. Here experts, k (<=8) and scale
// are parameters; each lane keeps its own insertion-ordered top-k over a
// strided expert subset, so any expert count works and the k shuffle rounds
// merge lane heads; the score is v41_router_topk.comp's sqrt(softplus) formula;
// the in-order F32 sum (+1e-20 when full) and the four-word fault record follow
// decodeV41RouterTopK. Every arithmetic step is a separately rounded __f*_rn
// so nvcc never contracts it.

#define V41RT_LANES 32
#define V41RT_ROWS 4
#define V41RT_MAX_K 8
#define V41RT_NO_FAULT 0x7fffffff

__device__ __forceinline__ static bool v41rt_finite(float x) { return isfinite(x); }

// log1p(e) for e in [0,1]: log(u) * e / (u - 1) cancels the rounding of u = 1 + e.
__device__ __forceinline__ static float v41rt_log1p_unit(float e) {
  const float u = __fadd_rn(1.0f, e);
  if (u == 1.0f) return e;
  return __fmul_rn(logf(u), __fdiv_rn(e, __fsub_rn(u, 1.0f)));
}

// Below -20, log1p(e^z) = e^z to F32 precision; sqrt(e^z) = e^(z/2) avoids
// underflowing e^z before the root.
__device__ __forceinline__ static float v41rt_score(float z) {
  if (z < -20.0f) return expf(__fmul_rn(0.5f, z));
  const float e = expf(-fabsf(z));
  return __fsqrt_rn(__fadd_rn(fmaxf(z, 0.0f), v41rt_log1p_unit(e)));
}

// Selection order: larger value first, lower expert first on equal values.
__device__ __forceinline__ static bool v41rt_before(float av, int ae, float bv, int be) {
  return av > bv || (av == bv && ae < be);
}

__device__ static void v41rt_fault(unsigned *status, unsigned stage, int expert, float value) {
  status[0] = stage;
  status[1] = (unsigned)(expert + 1);
  status[2] = __float_as_uint(value);
  status[3] = 0u;
}

__global__ void k_v41_router_argmax_topk(const float *logits, const float *bias,
    const int *hash, unsigned *experts_out, float *weights_out, unsigned *status_out,
    int rows, int experts, int top_k, int full, int has_bias, int hash_mode,
    float route_scale) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x * V41RT_ROWS + threadIdx.y;
  if (row >= rows) return;
  const float *z = logits + (size_t)row * experts;
  unsigned *status = status_out + (size_t)row * 4u;
  unsigned *pick_e = experts_out + (size_t)row * top_k;
  float *pick_w = weights_out + (size_t)row * top_k;
  if (lane == 0) {
    status[0] = status[1] = status[2] = status[3] = 0u;
    for (int s = 0; s < top_k; ++s) { pick_e[s] = 0u; pick_w[s] = 0.0f; }
  }

  int held = 0;
  int picked[V41RT_MAX_K];
  float sel_value[V41RT_MAX_K], sel_score[V41RT_MAX_K];
  int fault_expert = V41RT_NO_FAULT;
  unsigned fault_stage = 0u;
  float fault_value = 0.0f;
  // Host order per expert: logit, score, bias, then selection score. A lane
  // walks its experts ascending, so its first fault is its lowest one.
  for (int e = lane; e < experts; e += V41RT_LANES) {
    const float v = z[e];
    if (!v41rt_finite(v)) { fault_expert = e; fault_stage = 1u; fault_value = v; break; }
    const float s = v41rt_score(v);
    if (!v41rt_finite(s)) { fault_expert = e; fault_stage = 2u; fault_value = s; break; }
    if (hash_mode) continue;  // hash rows select from the table, never from scores
    float c = s;
    if (has_bias) {
      const float b = bias[e];
      if (!v41rt_finite(b)) { fault_expert = e; fault_stage = 3u; fault_value = b; break; }
      c = __fadd_rn(s, b);
      if (!v41rt_finite(c)) { fault_expert = e; fault_stage = 4u; fault_value = c; break; }
    }
    int insert = held;
    for (int slot = 0; slot < held; ++slot) {
      // Strict > keeps earlier (lower) experts first on corrected ties.
      if (c > sel_value[slot]) { insert = slot; break; }
    }
    if (insert < top_k) {
      for (int slot = min(held, top_k - 1); slot > insert; --slot) {
        picked[slot] = picked[slot - 1];
        sel_value[slot] = sel_value[slot - 1];
        sel_score[slot] = sel_score[slot - 1];
      }
      picked[insert] = e;
      sel_value[insert] = c;
      sel_score[insert] = s;
      held = min(held + 1, top_k);
    }
  }

  // The row's first fault is the lowest faulting expert across the warp.
  int lowest = fault_expert;
  for (int mask = 16; mask > 0; mask >>= 1) lowest = min(lowest, __shfl_xor_sync(0xffffffffu, lowest, mask));
  if (lowest != V41RT_NO_FAULT) {
    if (fault_expert == lowest) v41rt_fault(status, fault_stage, fault_expert, fault_value);
    return;
  }

  int row_pick[V41RT_MAX_K];
  float row_score[V41RT_MAX_K];
  if (hash_mode) {
    if (lane != 0) return;
    // Ids were validated on the host: in range and distinct per row.
    for (int slot = 0; slot < top_k; ++slot) {
      const int e = hash[(size_t)row * top_k + slot];
      row_pick[slot] = e;
      row_score[slot] = v41rt_score(z[e]);
    }
  } else {
    // ds4 warp argmax round, over lane heads instead of per-lane registers.
    int head = 0;
    for (int round = 0; round < top_k; ++round) {
      const bool has = head < held;
      float best_v = has ? sel_value[head] : -INFINITY;
      float best_s = has ? sel_score[head] : 0.0f;
      int best_e = has ? picked[head] : V41RT_NO_FAULT;
      int best_l = lane;
      for (int mask = 16; mask > 0; mask >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best_v, mask);
        const float os = __shfl_xor_sync(0xffffffffu, best_s, mask);
        const int oe = __shfl_xor_sync(0xffffffffu, best_e, mask);
        const int ol = __shfl_xor_sync(0xffffffffu, best_l, mask);
        if (v41rt_before(ov, oe, best_v, best_e)) { best_v = ov; best_s = os; best_e = oe; best_l = ol; }
      }
      if (best_l == lane) ++head;
      row_pick[round] = best_e;
      row_score[round] = best_s;
    }
    if (lane != 0) return;
  }

  float sum = 0.0f;
  for (int slot = 0; slot < top_k; ++slot) sum = __fadd_rn(sum, row_score[slot]);
  if (!v41rt_finite(sum) || sum < 0.0f || (!full && sum == 0.0f)) { v41rt_fault(status, 5u, -1, sum); return; }
  const float denominator = full ? __fadd_rn(sum, 1e-20f) : sum;
  for (int slot = 0; slot < top_k; ++slot) {
    const float w = __fmul_rn(__fdiv_rn(row_score[slot], denominator), route_scale);
    if (!v41rt_finite(w)) { v41rt_fault(status, 6u, row_pick[slot], w); return; }
    pick_e[slot] = (unsigned)row_pick[slot];
    pick_w[slot] = w;
  }
}

static uint64_t g_v41_router_completed = 0;
static bool g_v41_router_uncertain = false;

static int v41_router_owned(const void *p, uint64_t bytes) {
  if (!p || bytes == 0) return -2;
  auto found = g_live.find(const_cast<void *>(p));
  if (found == g_live.end() || found->second < (size_t)bytes) return -2;
  cudaPointerAttributes attributes;
  cudaError_t error = cudaPointerGetAttributes(&attributes, p);
  if (error != cudaSuccess) return 11000 + (int)error;
  return attributes.type == cudaMemoryTypeDevice && attributes.device == 0 ? 0 : -2;
}

extern "C" int fak_v41_router_state(void) {
  if (!g_stream || g_capture_open || g_v41_router_uncertain) return -3;
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) return 10000 + (int)error;
  int device = -1;
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) return 10000 + (int)error;
  if (device != 0) return -3;
  cudaStreamCaptureStatus capture;
  error = cudaStreamIsCapturing(g_stream, &capture);
  if (error != cudaSuccess) return 10000 + (int)error;
  return capture == cudaStreamCaptureStatusNone ? 0 : -3;
}

static int v41_router_drain(int status) {
  cudaError_t drained = cudaStreamSynchronize(g_stream);
  if (drained != cudaSuccess) {
    // The kernel may still own its outputs: admission closes permanently.
    g_v41_router_uncertain = true;
    return 50000 + (int)drained;
  }
  return status;
}

// Routes rows on the device and copies back only the picks and status words.
// hash_host is rows*top_k host ids (hash mode) staged into hash_dev; scratch
// holds experts, weights and status back to back (rows*(2*top_k+4) words).
extern "C" int fak_v41_router_topk_f32(const float *logits, const float *bias,
    const int *hash_host, int *hash_dev, unsigned *scratch, unsigned *host_experts,
    float *host_weights, unsigned *host_status, int rows, int experts, int top_k,
    int full, int has_bias, int hash_mode, float route_scale) {
  if (rows <= 0 || experts <= 0 || top_k <= 0 || top_k > V41RT_MAX_K || top_k > experts ||
      !host_experts || !host_weights || !host_status || (hash_mode && (has_bias || !hash_host)))
    return 2;
  const uint64_t logit_count = (uint64_t)rows * experts;
  const uint64_t pick_count = (uint64_t)rows * top_k;
  if (logit_count > INT_MAX || pick_count > INT_MAX / 4) return 2;
  int state = fak_v41_router_state();
  if (state != 0) return state;
  const uint64_t scratch_bytes = (pick_count * 2u + (uint64_t)rows * 4u) * 4u;
  int owned = v41_router_owned(logits, logit_count * 4u);
  if (owned == 0) owned = v41_router_owned(bias, has_bias ? (uint64_t)experts * 4u : 4u);
  if (owned == 0) owned = v41_router_owned(scratch, scratch_bytes);
  if (owned == 0 && hash_mode) owned = v41_router_owned(hash_dev, pick_count * 4u);
  if (owned != 0) return owned;
  unsigned *dev_experts = scratch;
  float *dev_weights = (float *)(scratch + pick_count);
  unsigned *dev_status = scratch + pick_count * 2u;
  cudaError_t error;
  if (hash_mode) {
    fcuda_ts_untimed_transfer();
    error = cudaMemcpy(hash_dev, hash_host, pick_count * 4u, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return v41_router_drain(30000 + (int)error);
    g_h2d_bytes += pick_count * 4u;
  }
  const dim3 block(V41RT_LANES, V41RT_ROWS);
  const unsigned blocks = (unsigned)((rows + V41RT_ROWS - 1) / V41RT_ROWS);
  FCUDA_TS("k_v41_router_argmax_topk", blocks,
    k_v41_router_argmax_topk<<<blocks, block, 0, g_stream>>>(logits, bias, hash_mode ? hash_dev : nullptr,
      dev_experts, dev_weights, dev_status, rows, experts, top_k, full, has_bias, hash_mode, route_scale));
  error = cudaGetLastError();
  if (error != cudaSuccess) return v41_router_drain(30000 + (int)error);
  error = cudaStreamSynchronize(g_stream);
  if (error != cudaSuccess) return v41_router_drain(40000 + (int)error);
  int copied = fcuda_d2h_checked(host_status, dev_status, (size_t)rows * 16u);
  if (copied == 0) copied = fcuda_d2h_checked(host_experts, dev_experts, pick_count * 4u);
  if (copied == 0) copied = fcuda_d2h_checked(host_weights, dev_weights, pick_count * 4u);
  if (copied != 0) return v41_router_drain(59000 + copied);
  ++g_v41_router_completed;
  return 0;
}

extern "C" uint64_t fak_v41_router_completed(void) { return g_v41_router_completed; }
