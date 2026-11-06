// async_upload.cuh — the CUDA half of compute.AsyncUploader (spec-prefetch TICKET-02).
//
// One non-blocking copy stream plus a bounded pool of pinned staging slots. A transfer copies
// the caller's pageable bytes into a free slot on the host, then issues cudaMemcpyAsync from that
// slot on the copy stream and records the slot's event. The copy stream first waits on an event
// recorded on g_stream at issue time, so a pooled device buffer that earlier g_stream work still
// reads is never overwritten early. Work enqueued on g_stream AFTER the issue is free to run
// underneath the copy; that is the overlap the expert ring measures.
//
// Consumers must satisfy the fence (fcuda_upload_async_fence with block=1) before reading the
// destination. Each slot carries a generation: a slot is reused only after its event completed,
// so a fence whose generation is stale has landed by construction.
//
// Exhaustion is never growth: when no slot is free, the request exceeds the slot cap, or g_stream
// is mid-capture, the call returns 1 and the Go side falls back to the synchronous fcuda_h2d.
// Pinned memory is bounded by FCUDA_ASYNC_SLOTS * FCUDA_ASYNC_SLOT_CAP.

#define FCUDA_ASYNC_SLOTS 8
#define FCUDA_ASYNC_SLOT_CAP ((size_t)64 << 20)

struct fcuda_async_slot {
  void *host = nullptr;    // cudaHostAlloc staging, grown on demand up to the slot cap
  size_t cap = 0;
  cudaEvent_t done = nullptr;
  uint64_t gen = 0;        // bumped on every issue
  bool inflight = false;
};

static cudaStream_t g_copy_stream = nullptr;
static cudaEvent_t g_copy_order = nullptr; // g_stream position at issue time
static fcuda_async_slot g_async_slots[FCUDA_ASYNC_SLOTS];
static uint64_t g_async_issued = 0, g_async_fallback = 0;
static size_t g_async_pinned_bytes = 0;

static int fcuda_async_status(cudaError_t e, const char *what) {
  if (e == cudaSuccess) return 0;
  cudaGetLastError();
  fprintf(stderr, "fak-cuda: async upload %s: %s\n", what, cudaGetErrorString(e));
  return (int)e + 1000;
}

static int fcuda_async_init(void) {
  if (g_copy_stream) return 0;
  int st = fcuda_async_status(cudaStreamCreateWithFlags(&g_copy_stream, cudaStreamNonBlocking), "stream");
  if (st) { g_copy_stream = nullptr; return st; }
  st = fcuda_async_status(cudaEventCreateWithFlags(&g_copy_order, cudaEventDisableTiming), "order event");
  if (st) return st;
  for (int i = 0; i < FCUDA_ASYNC_SLOTS; i++) {
    st = fcuda_async_status(cudaEventCreateWithFlags(&g_async_slots[i].done, cudaEventDisableTiming), "slot event");
    if (st) return st;
  }
  return 0;
}

// fcuda_upload_async: 0 = issued (*slot/*gen name the fence), 1 = not issued (caller copies
// synchronously), >=1000 = CUDA failure (1000 + cudaError_t). An attempted
// copy may have touched the destination even when submission/recording failed.
// Drain that attempt before cleanup; a failed drain closes the shared terminal
// boundary, retaining both its pinned source and the native device allocation.
extern "C" int fcuda_upload_async(void *d, const void *h, size_t n, int *slot, uint64_t *gen) {
  if (!slot || !gen) return -1;
  *slot = -1;
  *gen = 0;
  if (g_cuda_cleanup_status != 0) return 1000 + g_cuda_cleanup_status;
  if (n == 0 || n > FCUDA_ASYNC_SLOT_CAP || g_capture_open) { g_async_fallback++; return 1; }
  int st = fcuda_async_init();
  if (st) return st;
  int pick = -1;
  for (int i = 0; i < FCUDA_ASYNC_SLOTS && pick < 0; i++) {
    fcuda_async_slot &s = g_async_slots[i];
    if (!s.inflight) { pick = i; break; }
    cudaError_t q = cudaEventQuery(s.done);
    if (q == cudaSuccess) { s.inflight = false; pick = i; }
    else if (q != cudaErrorNotReady) return fcuda_async_status(q, "slot query");
  }
  if (pick < 0) { g_async_fallback++; return 1; }
  fcuda_async_slot &s = g_async_slots[pick];
  if (s.cap < n) {
    size_t want = (n + ((size_t)1 << 20) - 1) & ~(((size_t)1 << 20) - 1);
    if (want > FCUDA_ASYNC_SLOT_CAP) want = FCUDA_ASYNC_SLOT_CAP;
    if (s.host) {
      cudaFreeHost(s.host);
      g_async_pinned_bytes -= s.cap;
      s.host = nullptr;
      s.cap = 0;
    }
    if (cudaHostAlloc(&s.host, want, cudaHostAllocPortable) != cudaSuccess) {
      cudaGetLastError();
      s.host = nullptr;
      g_async_fallback++;
      return 1; // pinned memory refused: synchronous copy, never unbounded growth
    }
    s.cap = want;
    g_async_pinned_bytes += want;
  }
  memcpy(s.host, h, n);
  st = fcuda_async_status(cudaEventRecord(g_copy_order, g_stream), "order record");
  if (!st) st = fcuda_async_status(cudaStreamWaitEvent(g_copy_stream, g_copy_order, 0), "order wait");
  fcuda_ts_untimed_transfer(); // copy-stream H2D: overlap is read from the ring fences, not a g_stream row
  bool copy_attempted = false;
  if (!st) {
    // An error from cudaMemcpyAsync does not prove that no DMA was queued.
    copy_attempted = true;
    st = fcuda_async_status(cudaMemcpyAsync(d, s.host, n, cudaMemcpyHostToDevice, g_copy_stream), "memcpy");
  }
  if (!st) st = fcuda_async_status(cudaEventRecord(s.done, g_copy_stream), "done record");
  if (st) {
    if (copy_attempted) {
      const cudaError_t drain = cudaStreamSynchronize(g_copy_stream);
      if (drain != cudaSuccess && !g_cuda_cleanup_status)
        g_cuda_cleanup_status = (int)drain;
    }
    // Keep the original issue error. The separate shared terminal retains a
    // failed drain and prevents Free, later slot reuse, and sync fallback.
    return st;
  }
  s.inflight = true;
  s.gen++;
  g_h2d_bytes += n;
  g_async_issued++;
  *slot = pick;
  *gen = s.gen;
  return 0;
}

// fcuda_upload_async_fence: 1 = landed, 0 = still in flight (block=0 only), >=1000 = failure.
extern "C" int fcuda_upload_async_fence(int slot, uint64_t gen, int block) {
  if (g_cuda_cleanup_status != 0) return 1000 + g_cuda_cleanup_status;
  if (slot < 0 || slot >= FCUDA_ASYNC_SLOTS) return 1;
  fcuda_async_slot &s = g_async_slots[slot];
  if (s.gen != gen || !s.inflight) return 1;
  cudaError_t e = block ? cudaEventSynchronize(s.done) : cudaEventQuery(s.done);
  if (e == cudaErrorNotReady) return 0;
  int st = fcuda_async_status(e, block ? "fence wait" : "fence query");
  if (st) return st;
  s.inflight = false;
  return 1;
}

extern "C" void fcuda_upload_async_stats(uint64_t *issued, uint64_t *fallback, size_t *pinned) {
  if (issued) *issued = g_async_issued;
  if (fallback) *fallback = g_async_fallback;
  if (pinned) *pinned = g_async_pinned_bytes;
}

// BEGIN checked expert-ring refill (#14199).
// Conceptual geometry-slot/completion adaptation of vLLM fa3b9d1377219247060d3c8b49eae199c9e7e154,
// offloader/prefetch.py (Apache-2.0); see third_party/vllm-offloader-refill.
// This eager arm does NOT use the legacy copy-stream/event slot protocol above.
// All calls hold cudaMu inside an already-owned complete execution domain.
#define FCUDA_OFFLOAD_REFILL_CAP ((size_t)64 << 20)
struct fcuda_offload_refill_hold {
  void *device = nullptr;
  void *host = nullptr;
  size_t bytes = 0;
};
static fcuda_offload_refill_hold g_offload_refill_hold;

static int fcuda_offload_refill_fail(int status) noexcept {
  if (!status) status = 46001;
  if (!g_cuda_cleanup_status) g_cuda_cleanup_status = status;
  return g_cuda_cleanup_status;
}

// Pure host bookkeeping query. In particular, do not consume CUDA last-error,
// allocate, record an event, mutate a ledger or clear another owner's terminal.
extern "C" int fcuda_offload_refill_probe(void *d, const void *h, size_t n) {
  if (!d || !h || !n || n > FCUDA_OFFLOAD_REFILL_CAP || !g_stream ||
      g_offload_refill_hold.device || g_capture_open || g_pool_graph_deep || g_ts_on ||
      v41_candidates_stream_uncertain() || v41c_pending_contains(d) ||
      n > (size_t)-1 - g_h2d_bytes) return 0;
  const uintptr_t dp = (uintptr_t)d, hp = (uintptr_t)h;
  if (dp > UINTPTR_MAX - n || hp > UINTPTR_MAX - n ||
      (dp < hp + n && hp < dp + n)) return 0;
  const auto live = g_live.find(d);
  if (live == g_live.end() || live->second != n ||
      g_managed_live.count(d) || g_panel_live.count(d)) return 0;
  fak_alloc_tags_guard guard;
  const auto tag = g_alloc_tags.find(d);
  return tag != g_alloc_tags.end() && tag->second.bytes == n &&
         tag->second.cls == FAK_ALLOC_WEIGHTS &&
         tag->second.kind == FAK_ALLOC_DEVICE_LOCAL;
}

// Go abnormal-exit bridge: no allocation, cleanup or exception. Closing the
// shared boundary prevents later Free/pool reuse even after the model releases
// its domain. A prior native attempt and its exact pinned source take precedence.
extern "C" void fcuda_offload_refill_stop(void *d) {
  if (!g_offload_refill_hold.device) g_offload_refill_hold.device = d;
  fcuda_offload_refill_fail(46002);
}

// Success alone certifies checked completion AND staging retirement. Every
// other outcome is terminal/retained, including pre-copy failures after model
// selection. This never frees the destination or retries a submitted transfer.
extern "C" int fcuda_offload_refill_copy(void *d, const void *h, size_t n) {
  try {
    if (!fcuda_offload_refill_probe(d, h, n)) {
      fcuda_offload_refill_stop(d);
      return fcuda_offload_refill_fail(46003);
    }
    g_offload_refill_hold.device = d;
    g_offload_refill_hold.bytes = n;
    int device = -1;
    cudaError_t e = cudaGetDevice(&device);
    if (e != cudaSuccess) return fcuda_offload_refill_fail(1000 + (int)e);
    if (device != 0) return fcuda_offload_refill_fail(46004);
    cudaPointerAttributes attr{};
    e = cudaPointerGetAttributes(&attr, d);
    if (e != cudaSuccess) return fcuda_offload_refill_fail(1000 + (int)e);
    if (attr.type != cudaMemoryTypeDevice || attr.device != 0 ||
        attr.devicePointer != d) return fcuda_offload_refill_fail(46005);
    e = cudaPeekAtLastError();
    if (e != cudaSuccess) return fcuda_offload_refill_fail(1000 + (int)e);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    e = cudaStreamIsCapturing(g_stream, &capture);
    if (e != cudaSuccess) return fcuda_offload_refill_fail(1000 + (int)e);
    if (capture != cudaStreamCaptureStatusNone) return fcuda_offload_refill_fail(46006);

    // Publish the native host root directly into the retained holder before
    // ledger registration or submission. No Go address survives this C call.
    e = cudaHostAlloc(&g_offload_refill_hold.host, n, cudaHostAllocPortable);
    if (e != cudaSuccess) return fcuda_offload_refill_fail(1000 + (int)e);
    if (!g_offload_refill_hold.host) return fcuda_offload_refill_fail(46007);
    fak_alloc_track(g_offload_refill_hold.host, FAK_ALLOC_WEIGHTS,
                    FAK_ALLOC_PINNED_HOST, FAK_ALLOC_SITE("expert-ring-refill-source"), n);
    memcpy(g_offload_refill_hold.host, h, n);
    // Canonical copy attribution, with profiling off admitted above. Its
    // disabled event path performs host dispatch bookkeeping only.
    FCUDA_TS("h2d_upload", 1,
      e = cudaMemcpyAsync(d, g_offload_refill_hold.host, n, cudaMemcpyHostToDevice, g_stream));
    if (e != cudaSuccess) return fcuda_offload_refill_fail(1000 + (int)e);
    g_h2d_bytes += n; // accepted bytes; the pure preflight checked size_t overflow
    e = cudaStreamSynchronize(g_stream);
    if (e != cudaSuccess) return fcuda_offload_refill_fail(1000 + (int)e);
    if (v41_candidates_stream_uncertain()) return fcuda_offload_refill_fail(46008);
    void *retired = g_offload_refill_hold.host;
    e = cudaFreeHost(retired);
    if (e != cudaSuccess) return fcuda_offload_refill_fail(1000 + (int)e);
    // Physical retirement precedes ledger retirement. Never invent a live host
    // root or retry a second free if bookkeeping fails/throws after this point.
    g_offload_refill_hold.host = nullptr;
    if (!fak_alloc_untrack(retired)) return fcuda_offload_refill_fail(46009);
    g_offload_refill_hold = {};
    return 0;
  } catch (...) {
    if (!g_offload_refill_hold.device) g_offload_refill_hold.device = d;
    return fcuda_offload_refill_fail(46010);
  }
}
// END checked expert-ring refill.
