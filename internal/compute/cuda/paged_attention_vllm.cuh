// ---- Paged decode attention over a block table (fak cuda-paged-kv) ------------------
// Adapted from vLLM csrc/attention/attention_kernels.cuh, attention_generic.cuh,
// attention_utils.cuh, dtype_float32.cuh and csrc/cuda_compat.h, plus the
// reshape_and_cache_kernel of csrc/cache_kernels.cu, at tag v0.11.0
// (b8b302cde434df8c9289a2b465406b47ebab1c2d), Apache-2.0.
// Copyright (c) 2023, The vLLM team. Copyright (c) 2020-2023, NVIDIA CORPORATION.
// The kernel itself is adapted from NVIDIA FasterTransformer
// decoder_masked_multihead_attention_template.hpp (release/v5.3_tag), Apache-2.0.
// License and notice: THIRD_PARTY_NOTICES.md ("vLLM paged attention v1/v2").
//
// The upstream bodies are kept; the deltas for Fak are:
//   * Only the scalar_t = cache_t = float, KV_DTYPE = kAuto, IS_BLOCK_SPARSE = false
//     instantiation is kept, so every `if constexpr` arm on those parameters is resolved
//     (this translation unit compiles as C++14) and the fp8/bf16/fp16 dtype headers,
//     ALiBi slopes and block-sparse parameters are dropped.
//   * torch launchers are replaced by the extern "C" adapters at the bottom, which keep
//     upstream's v1/v2 selection rule and PARTITION_SIZE = 512.
//   * reshape_and_cache takes one contiguous run (block, offset, tokens) instead of a
//     slot_mapping tensor: Fak appends rows in logical order, a run never crosses a page.
//   * The namespace is fak_vllm_pa (the TU is shared with other adapted kernels).
// The Fak-native glue below the upstream code (row gather out of the paged layout and the
// tiny device scalar store) has no upstream counterpart: vLLM never reads its cache back
// into row-major form.
//
// Cache layout (upstream): K [num_blocks, nKV, hd/x, block, x] with x = 16/sizeof(float),
// V [num_blocks, nKV, hd, block]. One physical block id addresses the same block in every
// layer's arena, so a block table is per-session, not per-layer.

namespace fak_vllm_pa {

#define FAK_PA_WARP_SIZE 32
#define FAK_PA_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#define FAK_PA_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)
#define FAK_PA_MAX(a, b) ((a) > (b) ? (a) : (b))
#define FAK_PA_MIN(a, b) ((a) < (b) ? (a) : (b))
#define FAK_PA_DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

// ---- attention_generic.cuh ----
template <typename T, int VEC_SIZE>
struct Vec {};
template <typename T>
struct FloatVec {};
template <typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b);
template <typename T>
inline __device__ float sum(T v);
template <typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}
template <typename A, typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

// ---- dtype_float32.cuh (float only) ----
template <>
struct Vec<float, 1> {
  using Type = float;
};
template <>
struct Vec<float, 2> {
  using Type = float2;
};
template <>
struct Vec<float, 4> {
  using Type = float4;
};
template <>
struct FloatVec<float> {
  using Type = float;
};
template <>
struct FloatVec<float2> {
  using Type = float2;
};
template <>
struct FloatVec<float4> {
  using Type = float4;
};
template <>
inline __device__ float mul<float, float>(float a, float b) {
  return a * b;
}
template <>
inline __device__ float2 mul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}
template <>
inline __device__ float4 mul(float4 a, float4 b) {
  float4 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  c.z = a.z * b.z;
  c.w = a.w * b.w;
  return c;
}
inline __device__ float fma(float a, float b, float c) { return a * b + c; }
inline __device__ float2 fma(float2 a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}
inline __device__ float4 fma(float4 a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}
template <>
inline __device__ float sum(float v) {
  return v;
}
template <>
inline __device__ float sum(float2 v) {
  return v.x + v.y;
}
template <>
inline __device__ float sum(float4 v) {
  return v.x + v.y + v.z + v.w;
}
inline __device__ void from_float(float& dst, float src) { dst = src; }
inline __device__ void from_float(float4& dst, float4 src) { dst = src; }
inline __device__ float to_float(float u) { return u; }
inline __device__ void zero(float& dst) { dst = 0.f; }

// ---- attention_utils.cuh ----
template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline __device__ float qk_dot_(const Vec (&q)[N], const Vec (&k)[N]) {
  using A_vec = typename FloatVec<Vec>::Type;
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fak_vllm_pa::fma(q[ii], k[ii], qk_vec);
  }
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += FAK_PA_SHFL_XOR_SYNC(qk, mask);
  }
  return qk;
}
template <typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template <typename Vec, int N>
  static inline __device__ float dot(const Vec (&q)[N], const Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

// ---- attention_kernels.cuh ----
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  int warp = threadIdx.x / FAK_PA_WARP_SIZE;
  int lane = threadIdx.x % FAK_PA_WARP_SIZE;
#pragma unroll
  for (int mask = FAK_PA_WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += FAK_PA_SHFL_XOR_SYNC(sum, mask);
  }
  if (lane == 0) {
    red_smem[warp] = sum;
  }
  __syncthreads();
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += FAK_PA_SHFL_XOR_SYNC(sum, mask);
  }
  return FAK_PA_SHFL_SYNC(sum, 0);
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums, float* __restrict__ max_logits,
    scalar_t* __restrict__ out, const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache, const cache_t* __restrict__ v_cache,
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq, const int q_stride,
    const int kv_block_stride, const int kv_head_stride) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = seq_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    return;
  }

  const int num_seq_blocks = FAK_PA_DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      FAK_PA_MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      FAK_PA_MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = FAK_PA_MAX(FAK_PA_WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;
  static_assert(NUM_THREADS % THREAD_GROUP_SIZE == 0, "thread groups must tile the block");
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      FAK_PA_DIVIDE_ROUND_UP(BLOCK_SIZE, FAK_PA_WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / FAK_PA_WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / FAK_PA_WARP_SIZE;
  const int lane = thread_idx % FAK_PA_WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;

  constexpr int VEC_SIZE = FAK_PA_MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();

  extern __shared__ char shared_mem[];
  float* logits = reinterpret_cast<float*>(shared_mem);
  __shared__ float red_smem[2 * NUM_WARPS];

  constexpr int x = 16 / sizeof(cache_t);
  float qk_max = -FLT_MAX;

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * FAK_PA_WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const cache_t* k_ptr =
            k_cache + physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[j] = *reinterpret_cast<const K_vec*>(
            k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                             q_vecs[thread_group_offset], k_vecs);

      if (thread_group_offset == 0) {
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

#pragma unroll
  for (int mask = FAK_PA_WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, FAK_PA_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, FAK_PA_SHFL_XOR_SYNC(qk_max, mask));
  }
  qk_max = FAK_PA_SHFL_SYNC(qk_max, 0);

  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits +
                            seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                          head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  constexpr int V_VEC_SIZE = FAK_PA_MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = FAK_PA_WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      FAK_PA_DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx -
                                                           start_token_idx));

    const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride +
                           kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        if (block_idx == num_seq_blocks - 1) {
          // Out-of-context lanes of the tail block may hold stale values (NaN in
          // upstream issue 641); zero them explicitly.
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += FAK_PA_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
  }

  __syncthreads();

  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  if (warp_idx == 0) {
    scalar_t* out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Grid: (num_heads, num_seqs, 1).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache, const cache_t* __restrict__ v_cache,
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq, const int q_stride,
    const int kv_block_stride, const int kv_head_stride) {
  paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>(
      nullptr, nullptr, out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq, q_stride,
      kv_block_stride, kv_head_stride);
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel(
    float* __restrict__ exp_sums, float* __restrict__ max_logits,
    scalar_t* __restrict__ tmp_out, const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache, const cache_t* __restrict__ v_cache,
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq, const int q_stride,
    const int kv_block_stride, const int kv_head_stride) {
  paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq, q_stride,
      kv_block_stride, kv_head_stride);
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS, int PARTITION_SIZE>
__global__ void paged_attention_v2_reduce_kernel(
    scalar_t* __restrict__ out, const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits, const scalar_t* __restrict__ tmp_out,
    const int* __restrict__ seq_lens, const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = seq_lens[seq_idx];
  const int num_partitions = FAK_PA_DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    scalar_t* out_ptr =
        out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr =
        tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / FAK_PA_WARP_SIZE;
  const int warp_idx = threadIdx.x / FAK_PA_WARP_SIZE;
  const int lane = threadIdx.x % FAK_PA_WARP_SIZE;

  extern __shared__ char shared_mem[];
  __shared__ float red_smem[2 * NUM_WARPS];

  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

#pragma unroll
  for (int mask = FAK_PA_WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, FAK_PA_SHFL_XOR_SYNC(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, FAK_PA_SHFL_XOR_SYNC(max_logit, mask));
  }
  max_logit = FAK_PA_SHFL_SYNC(max_logit, 0);

  float* shared_exp_sums =
      reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums +
                              seq_idx * num_heads * max_num_partitions +
                              head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr =
      out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] *
             inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }
}

// ---- cache_kernels.cu reshape_and_cache_kernel (one contiguous run) ----
// Writes `tokens` rows of key/value [tokens, num_heads*head_size] into block
// `block_idx` starting at in-block offset `block_offset`. A second (optional) key-layout
// plane `key2` writes `key2_src` the same way (Fak keeps pre-RoPE K beside post-RoPE K).
__global__ void reshape_and_cache_run_kernel(
    const float* __restrict__ key, const float* __restrict__ value,
    const float* __restrict__ key2_src, float* __restrict__ key_cache,
    float* __restrict__ value_cache, float* __restrict__ key2_cache,
    const int block_idx_in, const int block_offset_in, const int num_heads,
    const int head_size, const int block_size, const int x) {
  const int64_t token_idx = blockIdx.x;
  const int64_t block_idx = block_idx_in;
  const int64_t block_offset = block_offset_in + token_idx;
  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_idx = token_idx * n + i;
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;
    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        block_offset;
    key_cache[tgt_key_idx] = key[src_idx];
    value_cache[tgt_value_idx] = value[src_idx];
    if (key2_cache != nullptr) key2_cache[tgt_key_idx] = key2_src[src_idx];
  }
}

// ---- Fak-native glue (no upstream counterpart) ----
// Gathers logical rows [0, tokens) of one paged plane back into row-major
// dst[tokens, num_heads*head_size]. key_layout selects the K ([.., hd/x, block, x]) or the
// V ([.., hd, block]) block layout.
__global__ void paged_gather_rows_kernel(float* __restrict__ dst,
                                         const float* __restrict__ cache,
                                         const int* __restrict__ block_table,
                                         int tokens, int num_heads, int head_size,
                                         int block_size, int x, int key_layout) {
  const int64_t token = blockIdx.x;
  if (token >= tokens) return;
  const int64_t block_idx = block_table[token / block_size];
  const int64_t block_offset = token % block_size;
  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    int64_t src;
    if (key_layout) {
      src = block_idx * num_heads * (head_size / x) * block_size * x +
            head_idx * (head_size / x) * block_size * x +
            (head_offset / x) * block_size * x + block_offset * x + head_offset % x;
    } else {
      src = block_idx * num_heads * head_size * block_size +
            head_idx * head_size * block_size + head_offset * block_size + block_offset;
    }
    dst[token * n + i] = cache[src];
  }
}

__global__ void paged_store_int_kernel(int* dst, int value) { *dst = value; }

}  // namespace fak_vllm_pa

#define FAK_PA_PARTITION_SIZE 512
#define FAK_PA_NUM_THREADS 128
#define FAK_PA_BLOCK_SIZE 16

template <int HEAD_SIZE>
static int fak_paged_attention_launch(float* out, const float* q, const float* k_cache,
                                      const float* v_cache, const int* block_tables,
                                      const int* seq_lens, int num_seqs, int num_heads,
                                      int num_kv_heads, int max_num_blocks_per_seq,
                                      int max_seq_len, float scale, float* scratch,
                                      long long scratch_floats) {
  using namespace fak_vllm_pa;
  constexpr int BLOCK_SIZE = FAK_PA_BLOCK_SIZE;
  constexpr int NUM_THREADS = FAK_PA_NUM_THREADS;
  constexpr int PARTITION_SIZE = FAK_PA_PARTITION_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / FAK_PA_WARP_SIZE;
  const int q_stride = num_heads * HEAD_SIZE;
  const int kv_head_stride = HEAD_SIZE * BLOCK_SIZE;
  const int kv_block_stride = num_kv_heads * kv_head_stride;
  const int max_num_partitions = FAK_PA_DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
  // Upstream's v1/v2 selection (vllm/attention/ops/paged_attn.py).
  const bool use_v1 = max_seq_len <= 8192 &&
                      (max_num_partitions == 1 || num_seqs * num_heads > 512);
  if (use_v1) {
    const int padded_max_seq_len =
        FAK_PA_DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE) * BLOCK_SIZE;
    const int logits_size = padded_max_seq_len * (int)sizeof(float);
    const int outputs_size = (NUM_WARPS / 2) * HEAD_SIZE * (int)sizeof(float);
    const int shared_mem_size = FAK_PA_MAX(logits_size, outputs_size);
    auto kern = paged_attention_v1_kernel<float, float, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>;
    if (cudaFuncSetAttribute((const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem_size) != cudaSuccess) {
      return 3;
    }
    dim3 grid(num_heads, num_seqs, 1);
    dim3 block(NUM_THREADS);
    FCUDA_TS("fak_paged_attention_v1", num_heads,
             (kern<<<grid, block, shared_mem_size, g_stream>>>(
                 out, q, k_cache, v_cache, num_kv_heads, scale, block_tables, seq_lens,
                 max_num_blocks_per_seq, q_stride, kv_block_stride, kv_head_stride)));
    return cudaGetLastError() == cudaSuccess ? 0 : 4;
  }
  // v2: scratch holds exp_sums, max_logits [seqs*heads*partitions] and tmp_out
  // [seqs*heads*partitions*hd].
  const long long rows = (long long)num_seqs * num_heads * max_num_partitions;
  const long long need = rows * 2 + rows * HEAD_SIZE;
  if (scratch == nullptr || scratch_floats < need) return 5;
  float* exp_sums = scratch;
  float* max_logits = scratch + rows;
  float* tmp_out = scratch + rows * 2;
  const int shared_mem_size = FAK_PA_MAX(PARTITION_SIZE * (int)sizeof(float),
                                         (NUM_WARPS / 2) * HEAD_SIZE * (int)sizeof(float));
  dim3 grid(num_heads, num_seqs, max_num_partitions);
  dim3 block(NUM_THREADS);
  auto kern2 = paged_attention_v2_kernel<float, float, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>;
  FCUDA_TS("fak_paged_attention_v2", num_heads,
           (kern2<<<grid, block, shared_mem_size, g_stream>>>(
               exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
               block_tables, seq_lens, max_num_blocks_per_seq, q_stride, kv_block_stride,
               kv_head_stride)));
  const int reduce_shared_mem_size = 2 * max_num_partitions * (int)sizeof(float);
  dim3 reduce_grid(num_heads, num_seqs);
  FCUDA_TS("fak_paged_attention_v2_reduce", num_heads,
           (paged_attention_v2_reduce_kernel<float, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE>
            <<<reduce_grid, block, reduce_shared_mem_size, g_stream>>>(
                out, exp_sums, max_logits, tmp_out, seq_lens, max_num_partitions)));
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

extern "C" long long fcuda_paged_attention_scratch_floats(int num_seqs, int num_heads,
                                                         int head_size, int max_seq_len) {
  const long long parts = (max_seq_len + FAK_PA_PARTITION_SIZE - 1) / FAK_PA_PARTITION_SIZE;
  const long long rows = (long long)num_seqs * num_heads * parts;
  return rows * 2 + rows * head_size;
}

extern "C" int fcuda_paged_attention_f32(float* out, const float* q, const float* k_cache,
                                         const float* v_cache, const int* block_tables,
                                         const int* seq_lens, int num_seqs, int num_heads,
                                         int num_kv_heads, int head_size,
                                         int max_num_blocks_per_seq, int max_seq_len,
                                         float scale, float* scratch,
                                         long long scratch_floats) {
  if (num_seqs <= 0 || num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0 ||
      max_seq_len <= 0 || max_num_blocks_per_seq <= 0) {
    return 1;
  }
#define FAK_PA_CASE(HS)                                                                  \
  case HS:                                                                               \
    return fak_paged_attention_launch<HS>(out, q, k_cache, v_cache, block_tables,        \
                                          seq_lens, num_seqs, num_heads, num_kv_heads,   \
                                          max_num_blocks_per_seq, max_seq_len, scale,    \
                                          scratch, scratch_floats);
  switch (head_size) {
    FAK_PA_CASE(32)
    FAK_PA_CASE(64)
    FAK_PA_CASE(80)
    FAK_PA_CASE(96)
    FAK_PA_CASE(112)
    FAK_PA_CASE(120)
    FAK_PA_CASE(128)
    FAK_PA_CASE(192)
    FAK_PA_CASE(256)
    default:
      return 2;
  }
#undef FAK_PA_CASE
}

extern "C" int fcuda_paged_attention_supported(int head_size, int block_size) {
  if (block_size != FAK_PA_BLOCK_SIZE) return 0;
  switch (head_size) {
    case 32: case 64: case 80: case 96: case 112: case 120: case 128: case 192: case 256:
      return 1;
    default:
      return 0;
  }
}

extern "C" int fcuda_paged_kv_write_f32(float* k_cache, float* v_cache, float* k2_cache,
                                        const float* k, const float* v, const float* k2,
                                        int block_idx, int block_offset, int tokens,
                                        int num_kv_heads, int head_size, int block_size) {
  if (tokens <= 0 || block_idx < 0 || block_offset < 0 || block_offset + tokens > block_size ||
      head_size % 4 != 0) {
    return 1;
  }
  const int threads = FAK_PA_MIN(512, num_kv_heads * head_size);
  FCUDA_TS("fak_paged_kv_write", tokens,
           (fak_vllm_pa::reshape_and_cache_run_kernel<<<tokens, threads, 0, g_stream>>>(
               k, v, k2, k_cache, v_cache, k2_cache, block_idx, block_offset, num_kv_heads,
               head_size, block_size, 4)));
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

extern "C" int fcuda_paged_kv_gather_f32(float* dst, const float* cache, const int* block_table,
                                         int tokens, int num_kv_heads, int head_size,
                                         int block_size, int key_layout) {
  if (tokens <= 0) return 0;
  if (head_size % 4 != 0) return 1;
  const int threads = FAK_PA_MIN(512, num_kv_heads * head_size);
  FCUDA_TS("fak_paged_kv_gather", tokens,
           (fak_vllm_pa::paged_gather_rows_kernel<<<tokens, threads, 0, g_stream>>>(
               dst, cache, block_table, tokens, num_kv_heads, head_size, block_size, 4,
               key_layout)));
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

extern "C" int fcuda_paged_store_int(int* dst, int value) {
  FCUDA_TS("paged_store_int_kernel", 1,
           fak_vllm_pa::paged_store_int_kernel<<<1, 1, 0, g_stream>>>(dst, value));
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

#undef FAK_PA_PARTITION_SIZE
#undef FAK_PA_NUM_THREADS
#undef FAK_PA_BLOCK_SIZE
