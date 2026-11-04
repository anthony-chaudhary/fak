// Private host-prefill input inventory. All entry points require cudaMu.
// Allocation publishes the sole owner before copying; no input is returned to
// an idle pool, upload cache, graph, or generic transient inventory.
static void *g_prefill_input = nullptr;
static size_t g_prefill_input_bytes = 0;
static int g_prefill_input_failure = 0;

static int fcuda_prefill_input_terminal(int status) {
  if (status && !g_prefill_input_failure) g_prefill_input_failure = status;
  if (status && !g_cuda_cleanup_status) g_cuda_cleanup_status = status;
  return g_prefill_input_failure;
}

// Read-only admission before any host-route weight upload; caller holds cudaMu.
extern "C" int fcuda_prefill_input_admit(void) {
  if (g_prefill_input_failure) return g_prefill_input_failure;
  if (g_prefill_input || g_capture_open) return -76001;
  if (g_cuda_cleanup_status) return g_cuda_cleanup_status;
  if (v41_candidates_stream_uncertain()) return (int)cudaErrorNotReady;
  if (!g_stream) return (int)cudaErrorInitializationError;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return (int)status;
  if (device != 0) return (int)cudaErrorInvalidDevice;
  return 0;
}

// -76001 is invalid admission; -76002 is allocation failure (the allocator
// reports its native cause). Positive values preserve actual CUDA statuses.
// On copy failure *out still owns storage; callers must retain it, not Free it.
extern "C" int fcuda_prefill_input_acquire(const void *host, size_t bytes,
                                           int panel, void **out, int *managed) {
  if (!out || !managed) return -76001;
  *out = nullptr;
  *managed = 0;
  if (!host || !bytes) return -76001;
  const int admission = fcuda_prefill_input_admit();
  if (admission != 0) return admission;
  const int cls = panel ? FAK_ALLOC_PREFILL_SCRATCH : FAK_ALLOC_ACTIVATIONS_SCRATCH;
  const int site = FAK_ALLOC_SITE("prefill-owned-input");
  void *p = panel ? fcuda_malloc_panel_tagged(bytes, cls, site)
                  : fcuda_malloc_tagged(bytes, cls, site);
  if (!p && !panel) {
    p = fcuda_malloc_managed_tagged(bytes, cls, site);
    if (p) *managed = 1;
  }
  if (!p) return -76002;
  g_prefill_input = p;
  g_prefill_input_bytes = bytes;
  *out = p;
  const int copy = fcuda_h2d_checked(p, host, bytes);
  if (copy != 0) return fcuda_prefill_input_terminal(copy);
  return 0;
}

// Preserve an observed completion failure before any deferred retirement.
// Status is opaque: checked D2H may return its documented encoded status.
extern "C" void fcuda_prefill_input_failed(int status) {
  if (g_prefill_input && status) (void)fcuda_prefill_input_terminal(status);
}

extern "C" int fcuda_prefill_input_retire(void *expected, size_t bytes) {
  if (g_prefill_input_failure) return g_prefill_input_failure;
  if (!expected || expected != g_prefill_input || bytes != g_prefill_input_bytes)
    return -76001;
  if (g_cuda_cleanup_status) return fcuda_prefill_input_terminal(g_cuda_cleanup_status);
  if (v41_candidates_stream_uncertain() || v41c_pending_contains(expected))
    return fcuda_prefill_input_terminal((int)cudaErrorNotReady);
  if (g_capture_open) return fcuda_prefill_input_terminal((int)cudaErrorInvalidValue);
  if (!g_stream) return fcuda_prefill_input_terminal((int)cudaErrorInitializationError);
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return fcuda_prefill_input_terminal((int)status);
  if (device != 0) return fcuda_prefill_input_terminal((int)cudaErrorInvalidDevice);
  auto ordinary = g_live.find(expected);
  auto managed = g_managed_live.find(expected);
  auto panel = g_panel_live.find(expected);
  const int matches = (ordinary != g_live.end()) +
                      (managed != g_managed_live.end()) +
                      (panel != g_panel_live.end());
  if (matches != 1 ||
      (ordinary != g_live.end() && ordinary->second != bytes) ||
      (managed != g_managed_live.end() && managed->second != bytes) ||
      (panel != g_panel_live.end() && (panel->second != bytes || g_panel_live_bytes < bytes)))
    return fcuda_prefill_input_terminal((int)cudaErrorInvalidValue);
  status = cudaStreamSynchronize(g_stream);
  if (status != cudaSuccess) return fcuda_prefill_input_terminal((int)status);
  status = cudaFree(expected);
  if (status != cudaSuccess) return fcuda_prefill_input_terminal((int)status);
  fak_alloc_untrack(expected);
  if (ordinary != g_live.end()) g_live.erase(ordinary);
  if (managed != g_managed_live.end()) g_managed_live.erase(managed);
  if (panel != g_panel_live.end()) {
    g_panel_live_bytes -= bytes;
    g_panel_live.erase(panel);
  }
  g_prefill_input = nullptr;
  g_prefill_input_bytes = 0;
  return 0;
}
