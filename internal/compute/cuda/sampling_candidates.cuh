// Raw bounded candidate transport. No float arithmetic, softmax, CDF or RNG.
// Included in cuda_kernels.cu after runtime services. Selection is opt-in at the
// serial host consumer, so ordinary CUDA builds retain their current behavior.
struct sampling_candidates_workspace {
  unsigned long long prefix;
  unsigned rank;
  unsigned histogram[256];
  unsigned words[2 + 2 * 65];
};

static int sampling_candidates_failure(int status) {
  if (status != 0 && g_cuda_cleanup_status == 0) g_cuda_cleanup_status = status;
  return status;
}

static int sampling_candidates_terminal_status(void) {
  return g_cuda_cleanup_status != 0 ? g_cuda_cleanup_status : (int)cudaErrorNotReady;
}

// The direct operation status was captured and latched before timing end.
// Profiling can itself fail or drain a full event ring; neither may hide that
// first status or permit another submission on a closed shared stream.
static int sampling_candidates_post_submit(cudaError_t operation) {
  const cudaError_t consumed = cudaGetLastError();
  if (operation != cudaSuccess) return sampling_candidates_failure((int)operation);
  if (v41_candidates_stream_uncertain()) return sampling_candidates_terminal_status();
  return sampling_candidates_failure((int)consumed);
}

extern "C" size_t fcuda_sampling_candidates_workspace_bytes(void) {
  return sizeof(sampling_candidates_workspace);
}

// The generic device allocator consumes ordinary OOM but leaves execution
// errors pending. Read those before any unwind can pool earlier live owners,
// including a successful pool allocation that exposes a prior async failure.
extern "C" int fcuda_sampling_candidates_allocation_status(void) {
  const cudaError_t pending = cudaGetLastError();
  if (v41_candidates_stream_uncertain()) return sampling_candidates_terminal_status();
  return sampling_candidates_failure((int)pending);
}

// Native release preserves its own first failure and owner maps. Publish that
// outcome before a caller can report a successful candidate transaction.
extern "C" int fcuda_sampling_candidates_release(void *owned) {
  const int pending = fcuda_sampling_candidates_allocation_status();
  if (pending != 0) return pending;
  if (owned) fcuda_free(owned);
  if (v41_candidates_stream_uncertain())
    return g_cuda_cleanup_status != 0 ? g_cuda_cleanup_status : (int)cudaErrorNotReady;
  return 0;
}

// 1 is supported, 0 is an unsubmitted envelope decline, negative is a checked
// runtime failure. A peer's terminal state can never turn into a host fallback.
extern "C" int fcuda_sampling_candidates_capability(void) {
  if (v41_candidates_stream_uncertain()) {
    const int status = g_cuda_cleanup_status != 0 ? g_cuda_cleanup_status : (int)cudaErrorNotReady;
    return status < 0 ? status : -status;
  }
  if (g_capture_open) return 0;
  int device = -1;
  cudaError_t e = cudaGetDevice(&device);
  if (e != cudaSuccess) return -sampling_candidates_failure((int)e);
  if (device != 0) return 0;
  cudaStreamCaptureStatus capture;
  e = cudaStreamIsCapturing(g_stream, &capture);
  if (e != cudaSuccess) return -sampling_candidates_failure((int)e);
  if (capture != cudaStreamCaptureStatusNone) return 0;
  cudaDeviceProp prop;
  e = cudaGetDeviceProperties(&prop, device);
  if (e != cudaSuccess) return -sampling_candidates_failure((int)e);
  return prop.maxThreadsPerBlock >= 256 && prop.maxThreadsDim[0] >= 256 &&
      prop.maxGridSize[0] >= 256 && prop.sharedMemPerBlock >= 256 * sizeof(unsigned);
}

// Numeric F32 order plus deterministic lower-ID ties. Normalize signed zero
// only for the key; emitted values come from the unchanged uint32 input.
__device__ unsigned long long sampling_candidates_key(unsigned bits, unsigned id) {
  unsigned rankBits = (bits & 0x7fffffffU) == 0 ? 0 : bits;
  const unsigned ordered = (rankBits & 0x80000000U) ? ~rankBits : rankBits ^ 0x80000000U;
  return ((unsigned long long)ordered << 32) | (0xffffffffU - id);
}

__global__ void k_sampling_candidates_histogram(const unsigned *row, int vocab,
    sampling_candidates_workspace *w, int shift) {
  __shared__ unsigned local[256];
  local[threadIdx.x] = 0;
  __syncthreads();
  const unsigned long long mask = shift == 56 ? 0ULL : (~0ULL << (shift + 8));
  const unsigned long long prefix = w->prefix;
  unsigned flags = 0;
  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < (size_t)vocab; id += (size_t)gridDim.x * blockDim.x) {
    const unsigned bits = row[id];
    if (shift == 56) {
      const unsigned magnitude = bits & 0x7fffffffU;
      if (magnitude > 0x7f800000U || bits == 0x7f800000U) flags |= 1;
      if (magnitude < 0x7f800000U) flags |= 2;
    }
    const unsigned long long key = sampling_candidates_key(bits, (unsigned)id);
    if ((key & mask) == prefix) atomicAdd(local + ((key >> shift) & 255), 1U);
  }
  if (flags) atomicOr(w->words, flags);
  __syncthreads();
  if (local[threadIdx.x]) atomicAdd(w->histogram + threadIdx.x, local[threadIdx.x]);
}

__global__ void k_sampling_candidates_cutoff(sampling_candidates_workspace *w,
    int shift, unsigned count) {
  if (threadIdx.x == 0) {
    unsigned rank = shift == 56 ? count : w->rank;
    for (int bucket = 255; bucket >= 0; --bucket) {
      const unsigned n = w->histogram[bucket];
      if (rank <= n) {
        w->prefix |= (unsigned long long)(unsigned)bucket << shift;
        w->rank = rank;
        break;
      }
      rank -= n;
    }
  }
  __syncthreads();
  w->histogram[threadIdx.x] = 0;
}

__global__ void k_sampling_candidates_emit(const unsigned *row, int vocab,
    sampling_candidates_workspace *w, unsigned count) {
  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < (size_t)vocab; id += (size_t)gridDim.x * blockDim.x) {
    const unsigned bits = row[id];
    if (sampling_candidates_key(bits, (unsigned)id) < w->prefix) continue;
    const unsigned slot = atomicAdd(w->words + 1, 1U);
    if (slot < count) {
      w->words[2 + 2 * slot] = (unsigned)id;
      w->words[3 + 2 * slot] = bits;
    }
  }
}

static bool sampling_candidates_extent(const void *p, size_t bytes) {
  if (!p || (uintptr_t)p % sizeof(unsigned) || bytes == 0 || (uintptr_t)p > UINTPTR_MAX - bytes) return false;
  auto live = g_live.find((void *)p);
  if (live != g_live.end()) return live->second >= bytes;
  auto panel = g_panel_live.find((void *)p);
  return panel != g_panel_live.end() && panel->second >= bytes;
}

static bool sampling_candidates_overlap(const void *a, size_t an, const void *b, size_t bn) {
  return (uintptr_t)a < (uintptr_t)b + bn && (uintptr_t)b < (uintptr_t)a + an;
}

// Caller supplies exact-base device owners and pointer-free host words. Only a
// zero return publishes output. Any CUDA failure closes shared-stream cleanup
// before Go can unwind, protecting source, workspace and optional retained row.
extern "C" int fcuda_sampling_candidates(const uint32_t *row, int vocab, int k,
    void *workspace, uint32_t *retained, uint32_t *host_words) {
  if (!host_words || vocab <= 1 || k < 1 || k > 64 || k >= vocab) return -1;
  const int capability = fcuda_sampling_candidates_capability();
  if (capability != 1) return capability < 0 ? -capability : -2;
  const size_t row_bytes = (size_t)vocab * sizeof(uint32_t);
  const size_t workspace_bytes = sizeof(sampling_candidates_workspace);
  if (!sampling_candidates_extent(row, row_bytes) ||
      !sampling_candidates_extent(workspace, workspace_bytes) ||
      (retained && !sampling_candidates_extent(retained, row_bytes))) return -3;
  if ((uintptr_t)workspace % alignof(sampling_candidates_workspace) ||
      sampling_candidates_overlap(row, row_bytes, workspace, workspace_bytes) ||
      (retained && (sampling_candidates_overlap(row, row_bytes, retained, row_bytes) ||
                    sampling_candidates_overlap(workspace, workspace_bytes, retained, row_bytes)))) return -4;
  const void *owners[3] = {row, workspace, retained};
  for (int i = 0; i < 3; ++i) {
    if (!owners[i]) continue;
    cudaPointerAttributes attrs;
    const cudaError_t e = cudaPointerGetAttributes(&attrs, owners[i]);
    if (e != cudaSuccess) return sampling_candidates_failure((int)e);
    if (attrs.type != cudaMemoryTypeDevice || attrs.device != 0) return -5;
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return sampling_candidates_failure((int)e);
  FCUDA_TS("sampling_candidates_fill", dim3(0, 0),
    if (v41_candidates_stream_uncertain()) return sampling_candidates_terminal_status();
    e = cudaMemsetAsync(workspace, 0, workspace_bytes, g_stream);
    if (e != cudaSuccess) (void)sampling_candidates_failure((int)e);
  );
  int submitted = sampling_candidates_post_submit(e);
  if (submitted != 0) return submitted;
  auto *w = (sampling_candidates_workspace *)workspace;
  const unsigned blocks = (unsigned)(((size_t)vocab + 255) / 256 < 256 ? ((size_t)vocab + 255) / 256 : 256);
  for (int shift = 56; shift >= 0; shift -= 8) {
    FCUDA_TS("k_sampling_candidates_histogram", blocks,
      if (v41_candidates_stream_uncertain()) return sampling_candidates_terminal_status();
      k_sampling_candidates_histogram<<<blocks, 256, 0, g_stream>>>(row, vocab, w, shift);
      e = cudaPeekAtLastError();
      if (e != cudaSuccess) (void)sampling_candidates_failure((int)e);
    );
    submitted = sampling_candidates_post_submit(e);
    if (submitted != 0) return submitted;
    FCUDA_TS("k_sampling_candidates_cutoff", 1,
      if (v41_candidates_stream_uncertain()) return sampling_candidates_terminal_status();
      k_sampling_candidates_cutoff<<<1, 256, 0, g_stream>>>(w, shift, (unsigned)k + 1);
      e = cudaPeekAtLastError();
      if (e != cudaSuccess) (void)sampling_candidates_failure((int)e);
    );
    submitted = sampling_candidates_post_submit(e);
    if (submitted != 0) return submitted;
  }
  FCUDA_TS("k_sampling_candidates_emit", blocks,
    if (v41_candidates_stream_uncertain()) return sampling_candidates_terminal_status();
    k_sampling_candidates_emit<<<blocks, 256, 0, g_stream>>>(row, vocab, w, (unsigned)k + 1);
    e = cudaPeekAtLastError();
    if (e != cudaSuccess) (void)sampling_candidates_failure((int)e);
  );
  submitted = sampling_candidates_post_submit(e);
  if (submitted != 0) return submitted;
  if (retained) {
    FCUDA_TS("sampling_candidates_retain", dim3(0, 0),
      if (v41_candidates_stream_uncertain()) return sampling_candidates_terminal_status();
      e = cudaMemcpyAsync(retained, row, row_bytes, cudaMemcpyDeviceToDevice, g_stream);
      if (e != cudaSuccess) (void)sampling_candidates_failure((int)e);
    );
    submitted = sampling_candidates_post_submit(e);
    if (submitted != 0) return submitted;
  }
  // The existing checked synchronous copy fences this device-zero blocking
  // stream. Never issue async D2H into borrowed Go memory and then release it
  // after an uncertain final fence. Copy exactly k+1 ID/raw pairs plus flags.
  const size_t bytes = (2 + 2 * ((size_t)k + 1)) * sizeof(uint32_t);
  const int copied = fcuda_d2h_checked(host_words, w->words, bytes);
  if (copied != 0) return sampling_candidates_failure(copied);
  if (v41_candidates_stream_uncertain())
    return g_cuda_cleanup_status != 0 ? g_cuda_cleanup_status : (int)cudaErrorNotReady;
  return 0;
}
