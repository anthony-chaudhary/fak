// Included by cuda_kernels.cu, not a separate translation unit. Ref #13994.
#ifdef FAK_CUDA_V41_INDEXER
// Each thread owns a complete row. Never parallelize either reduction: the
// contract is separately rounded binary32 products/sums in dimension/head order.
// Physical qualification must bind --ftz=false and this exact archive/device.
__global__ void k_v41_indexer_score_f32(const float *q, const float *keys,
    const float *weights, float *out, int rows, int heads, int dim) {
  for (size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       row < (size_t)rows; row += (size_t)gridDim.x * blockDim.x) {
    float score = 0.f;
    for (int head = 0; head < heads; ++head) {
      float dot = 0.f;
      for (int d = 0; d < dim; ++d)
        dot = __fadd_rn(dot, __fmul_rn(q[(size_t)head * dim + d], keys[row * dim + d]));
      if (dot < 0.f) dot = 0.f;
      score = __fadd_rn(score, __fmul_rn(dot, weights[head]));
    }
    out[row] = score;
  }
}

// Bounded SIMT reuse variant: the cooperative load is contiguous in global
// memory; padding avoids the power-of-two transpose stride, and each row's
// serial reduction accesses adjacent shared words across lanes. This changes data movement,
// never the dimension/head reduction order or separately rounded arithmetic.
// All 64 threads, including the final partial row tile, reach every barrier.
__global__ void k_v41_indexer_score_tiled_f32(const float *q, const float *keys,
    const float *weights, float *out, int rows, int heads, int dim) {
  __shared__ float key_tile[128 * 65];
  __shared__ float query[128];
  __shared__ float weight;
  const unsigned lane = threadIdx.x;
  for (size_t base = (size_t)blockIdx.x * 64; base < (size_t)rows;
       base += (size_t)gridDim.x * 64) {
    for (unsigned i = lane; i < 64u * (unsigned)dim; i += 64u) {
      const unsigned local_row = i / (unsigned)dim;
      const unsigned d = i % (unsigned)dim;
      const size_t row = base + local_row;
      key_tile[d * 65u + local_row] = row < (size_t)rows ? keys[row * dim + d] : 0.f;
    }
    __syncthreads();
    const size_t row = base + lane;
    float score = 0.f;
    for (int head = 0; head < heads; ++head) {
      for (unsigned d = lane; d < (unsigned)dim; d += 64u)
        query[d] = q[(size_t)head * dim + d];
      if (lane == 0) weight = weights[head];
      __syncthreads();
      if (row < (size_t)rows) {
        float dot = 0.f;
        for (int d = 0; d < dim; ++d)
          dot = __fadd_rn(dot, __fmul_rn(query[d], key_tile[(unsigned)d * 65u + lane]));
        if (dot < 0.f) dot = 0.f;
        score = __fadd_rn(score, __fmul_rn(dot, weight));
      }
      // Protect query/weight reuse and the next tile's key overwrite.
      __syncthreads();
    }
    if (row < (size_t)rows) out[row] = score;
  }
}

extern "C" int fak_v41_indexer_state(void) {
  if (!g_blas || !g_stream || g_capture_open || v41_candidates_stream_uncertain()) return -3;
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) return 10000 + (int)error;
  int device = -1;
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) return 10000 + (int)error;
  if (device != 0) return -3;
  cudaStreamCaptureStatus capture;
  error = cudaStreamIsCapturing(g_stream, &capture);
  if (error != cudaSuccess) return 10000 + (int)error;
  if (capture != cudaStreamCaptureStatusNone) return -3;
  // Do not attribute or drain unknown previous submissions. The Go lock keeps
  // the idle stream exclusive throughout validation, launch and completion.
  error = cudaStreamQuery(g_stream);
  if (error == cudaErrorNotReady) return -3;
  return error == cudaSuccess ? 0 : 10000 + (int)error;
}

static int v41_indexer_owned(const void *p, uint64_t count) {
  if (!p || count == 0 || count > INT_MAX || count > SIZE_MAX / sizeof(float)) return -2;
  auto found = g_live.find(const_cast<void *>(p));
  if (found == g_live.end() || found->second < (size_t)count * sizeof(float)) return -2;
  cudaPointerAttributes attributes;
  cudaError_t error = cudaPointerGetAttributes(&attributes, p);
  if (error != cudaSuccess) return 11000 + (int)error;
  return attributes.type == cudaMemoryTypeDevice && attributes.device == 0 ? 0 : -2;
}

extern "C" int fak_v41_indexer_input(const float *p, uint64_t count) {
  int state = fak_v41_indexer_state();
  return state == 0 ? v41_indexer_owned(p, count) : state;
}

static int v41_indexer_drain(int status, const float *q, const float *keys,
    const float *weights, const float *out) {
  cudaError_t drained = cudaStreamSynchronize(g_stream);
  if (drained != cudaSuccess) {
    // Admission closes permanently once uncertain; these slots cannot be
    // overwritten. Both free entry points retain the exact allocation bases.
    g_v41_indexer_quarantine[0] = q;
    g_v41_indexer_quarantine[1] = keys;
    g_v41_indexer_quarantine[2] = weights;
    g_v41_indexer_quarantine[3] = out;
    g_v41_indexer_uncertain = true;
    return 50000 + (int)drained;
  }
  return status;
}

extern "C" int fak_v41_indexer_score_f32(const float *q, const float *keys,
    const float *weights, float *out, int rows, int heads, int dim) {
  if (rows <= 0 || heads <= 0 || dim <= 0) return -1;
  const uint64_t qCount = (uint64_t)heads * dim, kCount = (uint64_t)rows * dim;
  if (qCount > INT_MAX || kCount > INT_MAX) return -1;
  int state = fak_v41_indexer_state();
  if (state != 0) return state;
  if (out == q || out == keys || out == weights) return -2;
  const void *pointers[] = {q, keys, weights, out};
  const uint64_t counts[] = {qCount, kCount, (uint64_t)heads, (uint64_t)rows};
  for (int i = 0; i < 4; ++i) {
    int owned = v41_indexer_owned(pointers[i], counts[i]);
    if (owned != 0) return owned;
  }
  const bool tiled = rows >= 64 && heads > 1 && dim <= 128;
  const unsigned row_tile = tiled ? 64u : 128u;
  unsigned blocks = ((unsigned)rows + row_tile - 1u) / row_tile;
  if (blocks > 65535u) blocks = 65535u;
  if (tiled) {
    FCUDA_TS("k_v41_indexer_score_tiled_f32", blocks,
      k_v41_indexer_score_tiled_f32<<<blocks, 64, 0, g_stream>>>(q, keys, weights, out, rows, heads, dim));
  } else {
    FCUDA_TS("k_v41_indexer_score_f32", blocks,
      k_v41_indexer_score_f32<<<blocks, 128, 0, g_stream>>>(q, keys, weights, out, rows, heads, dim));
  }
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return v41_indexer_drain(30000 + (int)error, q, keys, weights, out);
  error = cudaStreamSynchronize(g_stream);
  if (error != cudaSuccess) return v41_indexer_drain(40000 + (int)error, q, keys, weights, out);
  ++g_v41_indexer_completed;
  return 0;
}

extern "C" int fak_v41_indexer_readback(float *host, const float *device, uint64_t count) {
  if (!host) return -2;
  int status = fak_v41_indexer_input(device, count);
  if (status != 0) return status;
  // The shared checked ABI owns copy completion and successful-byte accounting.
  // Its status is 1000+cudaError_t; preserve our existing 60000+error encoding.
  const int copied = fcuda_d2h_checked(host, device, (size_t)count * sizeof(float));
  if (copied != 0) return v41_indexer_drain(59000 + copied, device, nullptr, nullptr, nullptr);
  return 0;
}

extern "C" uint64_t fak_v41_indexer_completed(void) { return g_v41_indexer_completed; }
#endif
