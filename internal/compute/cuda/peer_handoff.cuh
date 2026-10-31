// peer_handoff.cuh — the cross-device hidden-state handoff between layer bands on
// different CUDA devices of one process (compute.CUDAHop, model.PlanDeviceBands).
// The peer route is cudaMemcpyPeer with peer access enabled from the source device;
// the bounce route is D2H into pinned (portable) host staging, then H2D.
//
// Every entry point restores the caller's current device on exit, so a handoff never
// leaves the device-0 stream and cuBLAS handle bound to a foreign device. All copies
// are synchronous: the receiving band reads a complete hidden state.

struct fcuda_handoff_device_guard {
  int prev = 0;
  bool ok = false;
  fcuda_handoff_device_guard() { ok = cudaGetDevice(&prev) == cudaSuccess; }
  ~fcuda_handoff_device_guard() {
    if (ok) cudaSetDevice(prev);
  }
};

static int fcuda_handoff_status(cudaError_t e, const char *what) {
  if (e == cudaSuccess) return 0;
  cudaGetLastError(); // clear the sticky-free error so the next call starts clean
  fprintf(stderr, "fak-cuda: handoff %s: %s\n", what, cudaGetErrorString(e));
  return (int)e + 1000;
}

extern "C" int fcuda_handoff_device_count(int *n) {
  if (!n) return -1;
  *n = 0;
  return fcuda_handoff_status(cudaGetDeviceCount(n), "cudaGetDeviceCount");
}

extern "C" int fcuda_handoff_peer_probe(int src, int dst, int *can) {
  if (!can) return -1;
  *can = 0;
  return fcuda_handoff_status(cudaDeviceCanAccessPeer(can, src, dst), "cudaDeviceCanAccessPeer");
}

extern "C" int fcuda_handoff_peer_enable(int src, int dst) {
  fcuda_handoff_device_guard guard;
  cudaError_t e = cudaSetDevice(src);
  if (e == cudaSuccess) {
    e = cudaDeviceEnablePeerAccess(dst, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) {
      cudaGetLastError();
      e = cudaSuccess;
    }
  }
  return fcuda_handoff_status(e, "cudaDeviceEnablePeerAccess");
}

extern "C" void *fcuda_handoff_alloc(int device, size_t bytes) {
#ifdef FAK_CUDA_V41_CHECKED_STREAM
  if (v41_candidates_stream_uncertain()) return nullptr;
#endif
  fcuda_handoff_device_guard guard;
  void *d = nullptr;
  cudaError_t e = cudaSetDevice(device);
  if (e == cudaSuccess) e = fcuda_alloc_status(cudaMalloc(&d, bytes ? bytes : 1));
  if (fcuda_handoff_status(e, "cudaMalloc") != 0) return nullptr;
  fak_alloc_track(d, FAK_ALLOC_OTHER, FAK_ALLOC_DEVICE_LOCAL, FAK_ALLOC_SITE("peer-handoff"), bytes ? bytes : 1);
  return d;
}

extern "C" int fcuda_handoff_free(int device, void *d) {
  if (!d) return 0;
#ifdef FAK_CUDA_V41_CHECKED_STREAM
  if (v41_candidates_stream_uncertain()) return 1000 + (int)cudaErrorNotReady;
#endif
  fcuda_handoff_device_guard guard;
  cudaError_t e = cudaSetDevice(device);
  if (e == cudaSuccess) e = cudaFree(d);
  if (e == cudaSuccess) fak_alloc_untrack(d);
  return fcuda_handoff_status(e, "cudaFree");
}

extern "C" void *fcuda_handoff_pinned_alloc(size_t bytes) {
#ifdef FAK_CUDA_V41_CHECKED_STREAM
  if (v41_candidates_stream_uncertain()) return nullptr;
#endif
  void *h = nullptr;
  if (fcuda_handoff_status(cudaHostAlloc(&h, bytes ? bytes : 1, cudaHostAllocPortable), "cudaHostAlloc") != 0)
    return nullptr;
  fak_alloc_track(h, FAK_ALLOC_OTHER, FAK_ALLOC_PINNED_HOST, FAK_ALLOC_SITE("peer-handoff-pinned"), bytes ? bytes : 1);
  return h;
}

extern "C" int fcuda_handoff_pinned_free(void *h) {
  if (!h) return 0;
#ifdef FAK_CUDA_V41_CHECKED_STREAM
  if (v41_candidates_stream_uncertain()) return 1000 + (int)cudaErrorNotReady;
#endif
  cudaError_t e = cudaFreeHost(h);
  if (e == cudaSuccess) fak_alloc_untrack(h);
  return fcuda_handoff_status(e, "cudaFreeHost");
}

extern "C" int fcuda_handoff_h2d(int device, void *d, const void *h, size_t bytes) {
  fcuda_handoff_device_guard guard;
  cudaError_t e = cudaSetDevice(device);
  fcuda_ts_untimed_transfer();
  if (e == cudaSuccess) e = cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
  return fcuda_handoff_status(e, "h2d");
}

extern "C" int fcuda_handoff_d2h(int device, void *h, const void *d, size_t bytes) {
  fcuda_handoff_device_guard guard;
  cudaError_t e = cudaSetDevice(device);
  fcuda_ts_untimed_transfer();
  if (e == cudaSuccess) e = cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
  return fcuda_handoff_status(e, "d2h");
}

extern "C" int fcuda_handoff_peer_copy(int dst_device, void *dst, int src_device, const void *src, size_t bytes) {
  fcuda_handoff_device_guard guard;
  fcuda_ts_untimed_transfer();
  return fcuda_handoff_status(cudaMemcpyPeer(dst, dst_device, src, src_device, bytes), "cudaMemcpyPeer");
}
