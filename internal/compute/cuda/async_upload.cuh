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
// synchronously), >=1000 = CUDA failure (1000 + cudaError_t), and the destination is untouched.
extern "C" int fcuda_upload_async(void *d, const void *h, size_t n, int *slot, uint64_t *gen) {
  if (!slot || !gen) return -1;
  *slot = -1;
  *gen = 0;
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
  if (!st) st = fcuda_async_status(cudaMemcpyAsync(d, s.host, n, cudaMemcpyHostToDevice, g_copy_stream), "memcpy");
  if (!st) st = fcuda_async_status(cudaEventRecord(s.done, g_copy_stream), "done record");
  if (st) return st;
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
