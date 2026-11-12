// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Adapted from vllm-project/vllm csrc/fused_qknorm_rope_cache_kernel.cu
// at fa3b9d1377219247060d3c8b49eae199c9e7e154 (Apache-2.0). License and notice:
// THIRD_PARTY_NOTICES.md ("vLLM").

static unsigned long long g_qwen35_vllm_qk_launches = 0;

template <int HEAD_DIM, bool WIDE_192_ROTARY = false>
__global__ void k_qwen35_qknorm_rope_cache_vllm(
    const float *Q, const float *K, const float *V,
    const float *QW, const float *KW, float *QOut,
    float *KRawCache, float *KCache, float *VCache,
    int row, int position, int num_q_heads, int num_kv_heads,
    int rotary_pairs, float epsilon, double theta) {
  constexpr int kWarpSize = 32;
  constexpr int kThreadsPerBlock = 128;
  static_assert(!WIDE_192_ROTARY || HEAD_DIM == 192);
  constexpr int kThreadsPerHead =
      WIDE_192_ROTARY ? 32 : (HEAD_DIM <= 64 ? 8 : (HEAD_DIM <= 192 ? 16 : 32));
  constexpr int kValuesPerVector =
      WIDE_192_ROTARY ? 8 : (HEAD_DIM == 192 ? 4 : 8);
  constexpr int kValuesPerTile = kThreadsPerHead * kValuesPerVector;
  constexpr int kNumVectors = (HEAD_DIM + kValuesPerTile - 1) / kValuesPerTile;
  constexpr int kValuesPerThread = kValuesPerVector * kNumVectors;
  constexpr int kTileSize = kValuesPerTile * kNumVectors;
  constexpr int kHeadsPerBlock = kThreadsPerBlock / kThreadsPerHead;
  const int head_in_block = threadIdx.x / kThreadsPerHead;
  const int lane = threadIdx.x % kThreadsPerHead;
  const int local_head = blockIdx.x * kHeadsPerBlock + head_in_block;
  if (local_head >= num_q_heads + 2 * num_kv_heads) return;
  const bool is_q = local_head < num_q_heads;
  const bool is_k = !is_q && local_head < num_q_heads + num_kv_heads;
  const int head = is_q ? local_head :
      (is_k ? local_head - num_q_heads : local_head - num_q_heads - num_kv_heads);
  const size_t input_offset = (size_t)head * HEAD_DIM;
  const size_t cache_offset = ((size_t)row * num_kv_heads + head) * HEAD_DIM;
  if (!is_q && !is_k) {
#pragma unroll
    for (int block = 0; block < kNumVectors; ++block) {
#pragma unroll
      for (int element = 0; element < kValuesPerVector; ++element) {
        const int dim = block * kThreadsPerHead * kValuesPerVector +
                        lane * kValuesPerVector + element;
        if constexpr (kTileSize > HEAD_DIM) {
          if (dim >= HEAD_DIM) continue;
        }
        VCache[cache_offset + dim] = V[input_offset + dim];
      }
    }
    return;
  }
  const float *input = (is_q ? Q : K) + input_offset;
  float values[kValuesPerThread];
  float sum_squares = 0.0f;
#pragma unroll
  for (int block = 0; block < kNumVectors; ++block) {
#pragma unroll
    for (int element = 0; element < kValuesPerVector; ++element) {
      const int index = block * kValuesPerVector + element;
      const int dim = block * kThreadsPerHead * kValuesPerVector +
                      lane * kValuesPerVector + element;
      float value = 0.0f;
      if constexpr (kTileSize > HEAD_DIM) {
        if (dim < HEAD_DIM) value = input[dim];
      } else {
        value = input[dim];
      }
      values[index] = value;
      sum_squares = __fmaf_rn(value, value, sum_squares);
    }
  }
  const int warp_lane = threadIdx.x % kWarpSize;
  const int subwarp_base = (warp_lane / kThreadsPerHead) * kThreadsPerHead;
  constexpr uint32_t kSubwarpBits = 0xffffffffu >> (kWarpSize - kThreadsPerHead);
  const uint32_t subwarp_mask = kSubwarpBits << subwarp_base;
#pragma unroll
  for (int offset = 1; offset < kThreadsPerHead; offset <<= 1) {
    sum_squares += __shfl_xor_sync(subwarp_mask, sum_squares, offset, kWarpSize);
  }
  const float inverse_rms =
      rsqrtf(__fdiv_rn(sum_squares, static_cast<float>(HEAD_DIM)) + epsilon);
  const float *weight = is_q ? QW : KW;
  __shared__ float normalized[kHeadsPerBlock * HEAD_DIM];
  float *normalized_head = normalized + head_in_block * HEAD_DIM;
#pragma unroll
  for (int block = 0; block < kNumVectors; ++block) {
#pragma unroll
    for (int element = 0; element < kValuesPerVector; ++element) {
      const int index = block * kValuesPerVector + element;
      const int dim = block * kThreadsPerHead * kValuesPerVector +
                      lane * kValuesPerVector + element;
      if constexpr (kTileSize > HEAD_DIM) {
        if (dim >= HEAD_DIM) continue;
      }
      // normWeightHAL already applied NormGain1p. Do not add a second bias.
      normalized_head[dim] = (values[index] * inverse_rms) * weight[dim];
      if (is_k) KRawCache[cache_offset + dim] = normalized_head[dim];
    }
  }
  __syncwarp(subwarp_mask);
  constexpr int kRotaryPairsPerThread = 4;
  if (lane < rotary_pairs / kRotaryPairsPerThread) {
#pragma unroll
    for (int element = 0; element < kRotaryPairsPerThread; ++element) {
      const int pair = lane * kRotaryPairsPerThread + element;
      const int dim1 = pair + rotary_pairs;
      const float value0 = normalized_head[pair];
      const float value1 = normalized_head[dim1];
      const double angle = (double)position * pow(theta, -(double)pair / rotary_pairs);
      const float cos_value = (float)cos(angle), sin_value = (float)sin(angle);
      normalized_head[pair] = __fsub_rn(__fmul_rn(value0, cos_value),
                                             __fmul_rn(value1, sin_value));
      normalized_head[dim1] =
          __fmaf_rn(value0, sin_value, __fmul_rn(value1, cos_value));
    }
  }
  __syncwarp(subwarp_mask);
  float *output = is_q ? QOut + input_offset : KCache + cache_offset;
#pragma unroll
  for (int block = 0; block < kNumVectors; ++block) {
#pragma unroll
    for (int element = 0; element < kValuesPerVector; ++element) {
      const int dim = block * kThreadsPerHead * kValuesPerVector +
                      lane * kValuesPerVector + element;
      if constexpr (kTileSize > HEAD_DIM) {
        if (dim >= HEAD_DIM) continue;
      }
      output[dim] = normalized_head[dim];
    }
  }
}

// 1 is supported, 0 is an effect-free decline, negative is a runtime failure.
extern "C" int fak_qwen35_vllm_qk_capability(int nq, int nk, int hd, int rotary) {
  if (nq <= 0 || nk <= 0 || nq % nk || nq > INT_MAX - 2LL * nk ||
      (hd != 64 && hd != 128 && hd != 192 && hd != 256) ||
      nq > INT_MAX / hd || nk > INT_MAX / hd ||
      rotary < 0 || rotary > hd || rotary % 8) return 0;
  if (v41_candidates_stream_uncertain()) return -90001;
  if (!g_blas || !g_stream) return -90002;
  if (g_capture_open) return 0;
  cudaError_t e = cudaPeekAtLastError();
  if (e != cudaSuccess) return -10000 - (int)e;
  int device = -1;
  e = cudaGetDevice(&device);
  if (e != cudaSuccess) return -11000 - (int)e;
  if (device != 0) return 0;
  cudaStreamCaptureStatus capture;
  e = cudaStreamIsCapturing(g_stream, &capture);
  if (e != cudaSuccess) return -12000 - (int)e;
  if (capture != cudaStreamCaptureStatusNone) return 0;
  cudaDeviceProp prop;
  e = cudaGetDeviceProperties(&prop, device);
  if (e != cudaSuccess) return -13000 - (int)e;
  const int threads_per_head = hd == 192 && rotary > 128 ? 32 :
      (hd <= 64 ? 8 : (hd <= 192 ? 16 : 32));
  const int heads_per_block = 128 / threads_per_head;
  const int64_t blocks = ((int64_t)nq + 2LL * nk + heads_per_block - 1) / heads_per_block;
  return prop.warpSize == 32 && prop.maxThreadsPerBlock >= 128 &&
      prop.maxThreadsDim[0] >= 128 && prop.sharedMemPerBlock >=
      (size_t)heads_per_block * hd * sizeof(float) && blocks <= prop.maxGridSize[0];
}

static int qwen35_vllm_qk_owned(const void *p, size_t bytes) {
  if (!p || bytes == 0 || (uintptr_t)p > UINTPTR_MAX - bytes) return -3;
  auto live = g_live.find(const_cast<void *>(p));
  auto panel = g_panel_live.find(const_cast<void *>(p));
  if ((live == g_live.end() || live->second < bytes) &&
      (panel == g_panel_live.end() || panel->second < bytes)) return -3;
  cudaPointerAttributes attr;
  cudaError_t e = cudaPointerGetAttributes(&attr, p);
  if (e != cudaSuccess) return 14000 + (int)e;
  return attr.type == cudaMemoryTypeDevice && attr.device == 0 ? 0 : -5;
}

extern "C" int fak_qwen35_vllm_qk_f32(
    const float *Q, const float *K, const float *V,
    const float *QW, const float *KW, float *QOut,
    float *KRawCache, float *KCache, float *VCache,
    int row, int position, int nq, int nk, int hd, int rotary,
    float eps, double theta) {
  const int capability = fak_qwen35_vllm_qk_capability(nq, nk, hd, rotary);
  if (capability != 1) return capability < 0 ? -capability : -2;
  if (row < 0 || position < 0 || row >= INT_MAX / (nk * hd) ||
      !(eps > 0.f) || !isfinite(eps) || !(theta > 0.0) || !isfinite(theta)) return -1;
  const size_t qb = (size_t)nq * hd * sizeof(float);
  const size_t kb = (size_t)nk * hd * sizeof(float);
  const size_t cb = (size_t)(row + 1) * kb;
  const void *ptrs[] = {Q, K, V, QW, KW, QOut, KRawCache, KCache, VCache};
  const size_t sizes[] = {qb, kb, kb, (size_t)hd * 4, (size_t)hd * 4, qb, cb, cb, cb};
  for (int i = 0; i < 9; ++i) {
    int status = qwen35_vllm_qk_owned(ptrs[i], sizes[i]);
    if (status != 0) return status;
  }
  for (int out = 5; out < 9; ++out) {
    for (int other = 0; other < out; ++other) {
      const uintptr_t a = (uintptr_t)ptrs[out], b = (uintptr_t)ptrs[other];
      if (a < b + sizes[other] && b < a + sizes[out]) return -4;
    }
  }
#define FAK_VLLM_QK_LAUNCH(HD, WIDE) do { \
    constexpr int threads = WIDE ? 32 : (HD <= 64 ? 8 : (HD <= 192 ? 16 : 32)); \
    constexpr int heads = 128 / threads; \
    const unsigned blocks = ((int64_t)nq + 2LL * nk + heads - 1) / heads; \
    FCUDA_TS("k_qwen35_qknorm_rope_cache_vllm", blocks, \
        (k_qwen35_qknorm_rope_cache_vllm<HD, WIDE><<<blocks, 128, 0, g_stream>>>( \
            Q, K, V, QW, KW, QOut, KRawCache, KCache, VCache, \
            row, position, nq, nk, rotary / 2, eps, theta))); \
  } while (false)
  switch (hd) {
    case 64: FAK_VLLM_QK_LAUNCH(64, false); break;
    case 128: FAK_VLLM_QK_LAUNCH(128, false); break;
    case 192:
      if (rotary > 128) { FAK_VLLM_QK_LAUNCH(192, true); }
      else { FAK_VLLM_QK_LAUNCH(192, false); }
      break;
    case 256: FAK_VLLM_QK_LAUNCH(256, false); break;
  }
#undef FAK_VLLM_QK_LAUNCH
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    const cudaError_t drained = cudaStreamSynchronize(g_stream);
    if (drained != cudaSuccess && g_cuda_cleanup_status == 0)
      g_cuda_cleanup_status = (int)drained; // Existing terminal cleanup latch.
    return 15000 + (int)e;
  }
  ++g_qwen35_vllm_qk_launches; // Submission count, never completion evidence.
  return 0;
}

extern "C" unsigned long long fak_qwen35_vllm_qk_launch_count(void) {
  return g_qwen35_vllm_qk_launches;
}
