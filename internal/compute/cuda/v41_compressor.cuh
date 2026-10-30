// Included by cuda_kernels.cu, not a separate translation unit.
//
// DeepSeek V4.1 compressor ring and gated pool on CUDA (opt-in
// FAK_CUDA_V41_COMPRESSOR). The two device kernels are ported from antirez/ds4
// ds4_cuda.cu at 80ebbc396aee40eedc1d829222f3362d10fa4c6c (MIT, see
// THIRD_PARTY_NOTICES.md "antirez/ds4"):
//   compressor_store_kernel        -> v41c_compressor_store_kernel
//   compressor_update_pool_kernel  -> v41c_compressor_update_pool_kernel
// Changes for Fak, none to the pooling arithmetic: one token per store into an
// explicit caller slot (the ring is a plain [ratio,width] pair, so ds4's ratio-4
// overlap row offset and the APE add stay with the caller, which already folds
// APE into the score row); the ratio-4 two-half candidate split is the caller's
// width layout, so only ds4's generic branch is kept; and the 128-entry local
// candidate arrays are replaced by direct strided reads of the same values in
// the same order, which lifts the 128-row cap to V41CompressorPoolMaxRatio
// without changing any rounded operation. The per-dimension non-finite status
// word (V41CompressorRingKVNonFinite / ...ScoreNonFinite) is a Fak post-pass.
#ifdef FAK_CUDA_V41_COMPRESSOR

#define V41C_THREADS 256

__global__ static void v41c_compressor_store_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        uint32_t width,
        uint32_t slot) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= width) return;
    uint32_t j = (uint32_t)gid;
    state_kv[(uint64_t)slot * width + j] = kv[j];
    state_score[(uint64_t)slot * width + j] = sc[j];
}

__global__ static void v41c_compressor_update_pool_kernel(
        float *row,
        const float *state_kv,
        const float *state_score,
        uint32_t width,
        uint32_t ratio) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= width) return;
    float max_s = -INFINITY;
    for (uint32_t r = 0; r < ratio; r++) {
        max_s = fmaxf(max_s, state_score[(uint64_t)r * width + d]);
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t r = 0; r < ratio; r++) {
        float w = expf(state_score[(uint64_t)r * width + d] - max_s);
        den += w;
        acc += state_kv[(uint64_t)r * width + d] * w;
    }
    row[d] = den != 0.0f ? acc / den : 0.0f;
}

// Fak post-pass: one status word per latent dimension, written as the float
// value DecodeV41CompressorRing reads (0, 1, 2 or 3).
__global__ static void v41c_status_kernel(
        float *status,
        const float *state_kv,
        const float *state_score,
        uint32_t width,
        uint32_t ratio) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= width) return;
    unsigned bits = 0u;
    for (uint32_t r = 0; r < ratio; r++) {
        if ((__float_as_uint(state_kv[(uint64_t)r * width + d]) & 0x7f800000u) == 0x7f800000u) bits |= 1u;
        if ((__float_as_uint(state_score[(uint64_t)r * width + d]) & 0x7f800000u) == 0x7f800000u) bits |= 2u;
    }
    status[d] = (float)bits;
}

static float *g_v41c_status = nullptr;
static size_t g_v41c_status_bytes = 0;

extern "C" int fak_v41_compressor_state(void) {
  if (!g_stream || g_capture_open || v41_candidates_stream_uncertain()) return -3;
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
  return 0;
}

static int v41c_owned(const void *p, uint64_t count) {
  if (!p || count == 0 || count > INT_MAX || count > SIZE_MAX / sizeof(float)) return -2;
  auto found = g_live.find(const_cast<void *>(p));
  if (found == g_live.end() || found->second < (size_t)count * sizeof(float)) return -2;
  cudaPointerAttributes attributes;
  cudaError_t error = cudaPointerGetAttributes(&attributes, p);
  if (error != cudaSuccess) return 11000 + (int)error;
  return attributes.type == cudaMemoryTypeDevice && attributes.device == 0 ? 0 : -2;
}

static int v41c_drain(int status) {
  cudaError_t drained = cudaStreamSynchronize(g_stream);
  if (drained != cudaSuccess) {
    g_v41_compressor_uncertain = true;
    return 50000 + (int)drained;
  }
  return status;
}

static int v41c_admit(int ratio, int width, const void *const *pointers, const uint64_t *counts, int n) {
  if (ratio < 2 || width <= 0 || (int64_t)ratio * width > INT_MAX) return -1;
  int state = fak_v41_compressor_state();
  if (state != 0) return state;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j)
      if (pointers[i] == pointers[j]) return -2;
    int owned = v41c_owned(pointers[i], counts[i]);
    if (owned != 0) return owned;
  }
  return 0;
}

// Pool plus status for one [ratio,width] group into pooled[width] and
// status[width]. Caller has admitted every operand.
static int v41c_launch_pool(float *pooled, float *status, const float *kv, const float *sc, int ratio, int width) {
  const unsigned blocks = (unsigned)((width + V41C_THREADS - 1) / V41C_THREADS);
  FCUDA_TS("v41c_compressor_update_pool_kernel", blocks,
    v41c_compressor_update_pool_kernel<<<blocks, V41C_THREADS, 0, g_stream>>>(pooled, kv, sc,
      (uint32_t)width, (uint32_t)ratio));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return v41c_drain(30000 + (int)error);
  FCUDA_TS("v41c_status_kernel", blocks,
    v41c_status_kernel<<<blocks, V41C_THREADS, 0, g_stream>>>(status, kv, sc, (uint32_t)width, (uint32_t)ratio));
  error = cudaGetLastError();
  if (error != cudaSuccess) return v41c_drain(30000 + (int)error);
  return 0;
}

// Ring store. Without emit the store is only enqueued on g_stream. With emit
// out holds [2,width] (pooled row, status row) and the call drains.
extern "C" int fak_v41_compressor_ring_store_f32(const float *kv_row, const float *score_row,
    float *ring_kv, float *ring_score, float *out, int ratio, int width, int slot, int emit) {
  if (slot < 0 || slot >= ratio) return -1;
  const uint64_t ringCount = (uint64_t)ratio * width;
  const void *pointers[] = {kv_row, score_row, ring_kv, ring_score, out};
  const uint64_t counts[] = {(uint64_t)width, (uint64_t)width, ringCount, ringCount, 2ull * width};
  int admitted = v41c_admit(ratio, width, pointers, counts, emit ? 5 : 4);
  if (admitted != 0) return admitted;
  const unsigned blocks = (unsigned)((width + V41C_THREADS - 1) / V41C_THREADS);
  FCUDA_TS("v41c_compressor_store_kernel", blocks,
    v41c_compressor_store_kernel<<<blocks, V41C_THREADS, 0, g_stream>>>(kv_row, score_row,
      ring_kv, ring_score, (uint32_t)width, (uint32_t)slot));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return v41c_drain(30000 + (int)error);
  if (!emit) return 0;
  int launched = v41c_launch_pool(out, out + width, ring_kv, ring_score, ratio, width);
  if (launched != 0) return launched;
  error = cudaStreamSynchronize(g_stream);
  if (error != cudaSuccess) return v41c_drain(40000 + (int)error);
  ++g_v41_compressor_completed;
  return 0;
}

// Gated pool of one uploaded group. host_status receives width status words.
extern "C" int fak_v41_compressor_pool_f32(const float *kv, const float *scores, float *out,
    float *host_status, int ratio, int width) {
  if (!host_status) return -1;
  const uint64_t ringCount = (uint64_t)ratio * width;
  const void *pointers[] = {kv, scores, out};
  const uint64_t counts[] = {ringCount, ringCount, (uint64_t)width};
  int admitted = v41c_admit(ratio, width, pointers, counts, 3);
  if (admitted != 0) return admitted;
  const size_t statusBytes = (size_t)width * sizeof(float);
  if (g_v41c_status_bytes < statusBytes) {
    void *next = nullptr;
    if (fcuda_alloc_status(cudaMalloc(&next, statusBytes)) != cudaSuccess) return -1;
    fak_alloc_track(next, FAK_ALLOC_ACTIVATIONS_SCRATCH, FAK_ALLOC_DEVICE_LOCAL,
                    FAK_ALLOC_SITE("v41-compressor-pool-status"), statusBytes);
    if (g_v41c_status) {
      fak_alloc_untrack(g_v41c_status);
      cudaFree(g_v41c_status);
    }
    g_v41c_status = (float *)next;
    g_v41c_status_bytes = statusBytes;
  }
  int launched = v41c_launch_pool(out, g_v41c_status, kv, scores, ratio, width);
  if (launched != 0) return launched;
  cudaError_t error = cudaStreamSynchronize(g_stream);
  if (error != cudaSuccess) return v41c_drain(40000 + (int)error);
  const int copied = fcuda_d2h_checked(host_status, g_v41c_status, statusBytes);
  if (copied != 0) return v41c_drain(59000 + copied);
  ++g_v41_compressor_completed;
  return 0;
}

extern "C" uint64_t fak_v41_compressor_completed(void) { return g_v41_compressor_completed; }
#endif
