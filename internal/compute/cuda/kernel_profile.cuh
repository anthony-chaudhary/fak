// Opt-in CUDA kernel timestamp profile (FAK_CUDA_TIMESTAMP_PROFILE=1), the CUDA peer of the
// Vulkan shim's FAK_VULKAN_TIMESTAMP_PROFILE. Every launch, cuBLAS call and async copy/fill on
// g_stream is bracketed by FCUDA_TS with its own pre-created event pair; the pairs resolve in
// one host wait at the token boundary (fcuda_ts_boundary, from Recycle), at a reader snapshot,
// or when the ring fills, never per launch. Rows fold per (kernel name, grid x, grid y), the
// same shape and stderr format as the Vulkan rows. Read once at load; with the env unset every
// FCUDA_TS site costs one predictable branch and no CUDA call. Contract: docs/cuda-dev.md.

#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <atomic>
#include <mutex>

// Always-on device-work counters behind the "cuda" DeviceWorkReader (cuda_device_work.go).
// Host clock only, no CUDA events: every FCUDA_TS launch bumps a dispatch counter and closes
// an open device-idle interval; every host wait on device completion in this translation unit
// (stream/device/event synchronize, synchronous H2D/D2H cudaMemcpy) books one (site, stage)
// cell with its blocking wall and D2H bytes. The interposing macros sit at the end of this file,
// so every source cuda_kernels.cu includes after it is covered. Stage is the last launch since
// the previous wait. Counters are atomics, so the reader takes no cudaMu.
enum : uint32_t { kDwSiteSubmitWait = 0, kDwSiteReadback = 1, kDwSiteUpload = 2, kDwSitePipelineDrain = 3, kDwSites = 4 };
// Site names, in enum order; Go's cudaSyncSiteByID must match (TestCUDADeviceWorkSiteTableMatchesShim).
static const char *const kDwSiteNames[kDwSites] = {"submit_wait", "readback", "upload", "pipeline_drain"};
static const uint32_t kDwStageNone = 0, kDwStageOther = 1, kDwStages = 64;
struct DwCell { std::atomic<uint64_t> syncs{0}, wait_ns{0}, d2h{0}; };
static DwCell g_dw_cells[kDwSites][kDwStages];
static std::atomic<const char *> g_dw_stage_names[kDwStages];
static std::atomic<uint32_t> g_dw_stage_count{2}; // ids 0 (none) and 1 (other) are reserved
static std::mutex g_dw_stage_mu;
static std::atomic<uint64_t> g_dw_dispatches{0}, g_dw_readback{0}, g_dw_unobserved{0};
// Idle clock: g_dw_idle_since is the host ns at which the device was last observed drained
// (0 = busy or never drained). g_dw_idle_seq is a seqlock so a reader never adds an interval
// twice: writers (serialized by the Go caller's cudaMu) make it odd while they move one.
static std::atomic<uint64_t> g_dw_idle_ns{0}, g_dw_idle_since{0}, g_dw_idle_seq{0};
static const char *g_dw_last_kernel = nullptr; // written under the caller's cudaMu

static inline uint64_t dw_now(void) {
  const uint64_t t = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  return t ? t : 1;
}

static void dw_busy(void) {
  const uint64_t since = g_dw_idle_since.load(std::memory_order_relaxed);
  if (since == 0) return;
  const uint64_t t = dw_now();
  g_dw_idle_seq.fetch_add(1, std::memory_order_acq_rel);
  if (t > since) g_dw_idle_ns.fetch_add(t - since, std::memory_order_relaxed);
  g_dw_idle_since.store(0, std::memory_order_relaxed);
  g_dw_idle_seq.fetch_add(1, std::memory_order_acq_rel);
}

// fcuda_dw_launch counts one launch, cuBLAS call or async copy/fill on g_stream. Captured work
// runs at graph replay, which the cudaGraphLaunch interposer books as unobserved instead.
static inline void fcuda_dw_launch(const char *name) {
  if (g_capture_open) return;
  g_dw_dispatches.fetch_add(1, std::memory_order_relaxed);
  g_dw_last_kernel = name;
  if (g_dw_idle_since.load(std::memory_order_relaxed) != 0) dw_busy();
}

static uint32_t dw_stage_find(const char *name, uint32_t n) {
  for (uint32_t i = 2; i < n; ++i) {
    const char *s = g_dw_stage_names[i].load(std::memory_order_relaxed);
    if (s == name || (s && strcmp(s, name) == 0)) return i;
  }
  return 0;
}

static uint32_t dw_stage(const char *name) {
  if (!name) return kDwStageNone;
  uint32_t id = dw_stage_find(name, g_dw_stage_count.load(std::memory_order_acquire));
  if (id) return id;
  std::lock_guard<std::mutex> lock(g_dw_stage_mu);
  const uint32_t n = g_dw_stage_count.load(std::memory_order_relaxed);
  if ((id = dw_stage_find(name, n)) != 0) return id;
  if (n >= kDwStages) return kDwStageOther;
  g_dw_stage_names[n].store(name, std::memory_order_relaxed);
  g_dw_stage_count.store(n + 1, std::memory_order_release);
  return n;
}

// dw_wait_done books one finished host wait. drained says the wait proved g_stream empty, which
// opens the idle clock unless it is already open.
static void dw_wait_done(uint32_t site, uint64_t t0, uint64_t d2h, bool drained) {
  const uint64_t t1 = dw_now();
  DwCell &c = g_dw_cells[site][dw_stage(g_dw_last_kernel)];
  g_dw_last_kernel = nullptr;
  c.syncs.fetch_add(1, std::memory_order_relaxed);
  c.wait_ns.fetch_add(t1 > t0 ? t1 - t0 : 0, std::memory_order_relaxed);
  if (d2h) {
    c.d2h.fetch_add(d2h, std::memory_order_relaxed);
    g_dw_readback.fetch_add(d2h, std::memory_order_relaxed);
  }
  if (drained && g_dw_idle_since.load(std::memory_order_relaxed) == 0) {
    g_dw_idle_seq.fetch_add(1, std::memory_order_acq_rel);
    g_dw_idle_since.store(t1, std::memory_order_relaxed);
    g_dw_idle_seq.fetch_add(1, std::memory_order_acq_rel);
  }
}

static const bool g_ts_on = [] {
  const char *v = getenv("FAK_CUDA_TIMESTAMP_PROFILE");
  return v && v[0] == '1' && v[1] == '\0';
}();
static const uint64_t g_ts_every = [] {
  const char *v = getenv("FAK_CUDA_TIMESTAMP_PROFILE_EVERY");
  long n = v ? atol(v) : 0;
  return (uint64_t)(n > 0 ? n : 32);
}();
static const size_t kTsPairs = 4096;
static const uint32_t kTsLost = 0xFFFFFFFFu; // a pending pair whose stop event failed

struct TsPending { uint32_t kid, gx, gy; };
struct TsAgg { uint64_t n = 0; double ns = 0; };
static bool g_ts_usable = false, g_ts_init_tried = false, g_ts_batch_open = false;
static std::vector<cudaEvent_t> g_ts_events; // 2*kTsPairs: start, stop per slot
static std::vector<TsPending> g_ts_pending;
static std::vector<const char *> g_ts_names;
static std::unordered_map<const void *, uint32_t> g_ts_name_ids; // literal address -> kid
static std::unordered_map<uint64_t, TsAgg> g_ts_agg;
static uint64_t g_ts_batches = 0, g_ts_ops = 0, g_ts_truncated = 0, g_ts_untimed_compute = 0, g_ts_untimed_transfer = 0;
static double g_ts_gpu_ns = 0, g_ts_wall_ns = 0;
static std::chrono::steady_clock::time_point g_ts_batch_start;

// A profile call clears only the CUDA last-error it raised itself, never an op's earlier one.
static void ts_clear_own_error(cudaError_t prior) {
  if (prior == cudaSuccess && cudaPeekAtLastError() != cudaSuccess) (void)cudaGetLastError();
}

static uint32_t ts_kid(const char *name) {
  auto it = g_ts_name_ids.find(name);
  if (it != g_ts_name_ids.end()) return it->second;
  uint32_t kid = 0;
  while (kid < g_ts_names.size() && strcmp(g_ts_names[kid], name) != 0) ++kid;
  if (kid == g_ts_names.size()) g_ts_names.push_back(name);
  g_ts_name_ids[name] = kid;
  return kid;
}

static uint64_t ts_key(uint32_t kid, uint32_t gx, uint32_t gy) {
  return ((uint64_t)kid << 48) | ((uint64_t)(gx & 0xFFFFFFF) << 20) | (uint64_t)(gy & 0xFFFFF);
}

static void ts_init(void) {
  g_ts_init_tried = true;
  fprintf(stderr, "fak-cuda ts-profile mode=per_launch_event_pairs timing=event_pairs coverage=g_stream_launches graphs=disabled\n");
  std::vector<cudaEvent_t> events(kTsPairs * 2, nullptr);
  const cudaError_t prior = cudaPeekAtLastError();
  for (size_t i = 0; i < events.size(); ++i) {
    cudaError_t e = cudaEventCreate(&events[i]);
    if (e != cudaSuccess) {
      for (size_t j = 0; j < i; ++j) cudaEventDestroy(events[j]);
      ts_clear_own_error(prior);
      fprintf(stderr, "fak-cuda ts-profile unavailable: cudaEventCreate failed: %s\n", cudaGetErrorString(e));
      return;
    }
  }
  g_ts_events.swap(events);
  g_ts_pending.reserve(kTsPairs);
  g_ts_usable = true;
}

// ts_drain folds every pending pair after one wait on the newest stop event. A pair whose
// interval cannot be read counts truncated, so rows and gpu_ns add the same intervals.
static void ts_drain(void) {
  if (g_ts_pending.empty() || g_capture_open) return;
  const cudaError_t prior = cudaPeekAtLastError();
  for (size_t i = g_ts_pending.size(); i-- > 0;) {
    if (g_ts_pending[i].kid != kTsLost) {
      (void)cudaEventSynchronize(g_ts_events[2 * i + 1]);
      break;
    }
  }
  for (size_t i = 0; i < g_ts_pending.size(); ++i) {
    const TsPending &p = g_ts_pending[i];
    float ms = 0;
    if (p.kid == kTsLost || cudaEventElapsedTime(&ms, g_ts_events[2 * i], g_ts_events[2 * i + 1]) != cudaSuccess || !(ms >= 0)) {
      ++g_ts_truncated;
      continue;
    }
    const double ns = (double)ms * 1e6;
    TsAgg &a = g_ts_agg[ts_key(p.kid, p.gx, p.gy)];
    a.n++;
    a.ns += ns;
    g_ts_gpu_ns += ns;
    ++g_ts_ops;
  }
  ts_clear_own_error(prior);
  g_ts_pending.clear();
}

// fcuda_ts_begin returns the slot whose start event now precedes the launch, or -1 when the
// launch is untimed. Captured work runs at replay, outside any pair: it counts untimed_compute.
static int ts_begin_slow(const char *name, uint32_t gx, uint32_t gy) {
  if (!g_ts_init_tried) ts_init();
  if (!g_ts_usable) return -1;
  if (g_capture_open) {
    ++g_ts_untimed_compute;
    return -1;
  }
  if (g_ts_pending.size() >= kTsPairs) ts_drain();
  if (!g_ts_batch_open) {
    g_ts_batch_open = true;
    g_ts_batch_start = std::chrono::steady_clock::now();
  }
  const size_t slot = g_ts_pending.size();
  const cudaError_t prior = cudaPeekAtLastError();
  if (cudaEventRecord(g_ts_events[2 * slot], g_stream) != cudaSuccess) {
    ts_clear_own_error(prior);
    ++g_ts_truncated;
    return -1;
  }
  g_ts_pending.push_back(TsPending{ts_kid(name), gx, gy});
  return (int)slot;
}

static inline int fcuda_ts_begin(const char *name, dim3 grid) {
  fcuda_dw_launch(name);
  if (!g_ts_on) return -1;
  return ts_begin_slow(name, grid.x, grid.y);
}

static inline void fcuda_ts_end(int slot) {
  if (slot < 0) return;
  const cudaError_t prior = cudaPeekAtLastError();
  if (cudaEventRecord(g_ts_events[2 * (size_t)slot + 1], g_stream) != cudaSuccess) {
    ts_clear_own_error(prior);
    g_ts_pending[(size_t)slot].kid = kTsLost;
  }
}

// Synchronous host<->device copies run on the legacy stream, outside every pair.
static inline void fcuda_ts_untimed_transfer(void) {
  if (g_ts_on) ++g_ts_untimed_transfer;
}

// FCUDA_TS brackets one g_stream launch, cuBLAS call or async copy/fill with an event pair.
// name is the kernel symbol without namespace or template arguments (or a pseudo-kernel name
// such as d2d_copy, fill or cublas_*); grid is the launch's grid (int or dim3).
#define FCUDA_TS(name, grid, ...)                                       \
  do {                                                                  \
    const int fcuda_ts_slot_ = fcuda_ts_begin((name), dim3(grid));      \
    __VA_ARGS__;                                                        \
    fcuda_ts_end(fcuda_ts_slot_);                                       \
  } while (0)

static const char *ts_name(uint32_t kid) { return kid < g_ts_names.size() ? g_ts_names[kid] : "unknown"; }

static void ts_report(void) {
  std::vector<std::pair<uint64_t, TsAgg>> rows(g_ts_agg.begin(), g_ts_agg.end());
  std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return a.second.ns > b.second.ns; });
  double perBatch = g_ts_batches ? 1.0 / (double)g_ts_batches : 0;
  fprintf(stderr, "fak-cuda ts-profile batches=%llu ops/batch=%.1f gpu_ms/batch=%.3f wall_ms/batch=%.3f truncated=%llu timing=event_pairs graphs=disabled\n",
      (unsigned long long)g_ts_batches, g_ts_ops * perBatch, g_ts_gpu_ns * perBatch / 1e6, g_ts_wall_ns * perBatch / 1e6,
      (unsigned long long)g_ts_truncated);
  size_t shown = 0;
  for (const auto &r : rows) {
    if (shown++ >= 40) break;
    uint32_t kid = (uint32_t)(r.first >> 48), gx = (uint32_t)((r.first >> 20) & 0xFFFFFFF), gy = (uint32_t)(r.first & 0xFFFFF);
    fprintf(stderr, "fak-cuda ts-stage kernel=%s groups=%ux%u calls/batch=%.2f ms/batch=%.3f us/call=%.2f share=%.1f%%\n",
        ts_name(kid), gx, gy, r.second.n * perBatch, r.second.ns * perBatch / 1e6,
        r.second.n ? r.second.ns / (double)r.second.n / 1e3 : 0, g_ts_gpu_ns > 0 ? 100.0 * r.second.ns / g_ts_gpu_ns : 0);
  }
}

// One token boundary closes the batch opened by its first timed launch.
extern "C" void fcuda_ts_boundary(void) {
  if (!g_ts_on || !g_ts_usable || g_capture_open) return;
  ts_drain();
  if (!g_ts_batch_open) return;
  g_ts_batch_open = false;
  g_ts_wall_ns += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - g_ts_batch_start).count();
  if ((++g_ts_batches % g_ts_every) == 0) ts_report();
}

extern "C" int fcuda_ts_enabled(void) { return g_ts_on ? 1 : 0; }

// fcuda_ts_init creates the event ring once the context exists; 0, or -1 when unusable.
extern "C" int fcuda_ts_init(void) {
  if (!g_ts_on) return -1;
  if (!g_ts_init_tried) ts_init();
  return g_ts_usable ? 0 : -1;
}

extern "C" const char *fcuda_ts_kernel_name(uint32_t kid) { return ts_name(kid); }

// The fold is plain state; the caller serializes this against every other shim call (Go holds
// cudaMu). It drains first, so the rows cover all work enqueued so far. Returns the row count,
// which may exceed cap (size with cap 0, then fill), or -1 while a capture is open.
extern "C" int fcuda_ts_snapshot(fak_cuda_ts_totals *tot, fak_cuda_ts_row *rows, int cap) {
  if (g_capture_open) return -1;
  if (g_ts_on && g_ts_usable) ts_drain();
  if (tot) {
    tot->enabled = g_ts_on ? 1 : 0;
    tot->usable = g_ts_usable ? 1 : 0;
    tot->init_tried = g_ts_init_tried ? 1 : 0;
    tot->pad = 0;
    tot->batches = g_ts_batches;
    tot->ops = g_ts_ops;
    tot->truncated = g_ts_truncated;
    tot->untimed_compute = g_ts_untimed_compute;
    tot->untimed_transfer = g_ts_untimed_transfer;
    tot->barriers_elided = 0;
    tot->gpu_ns = g_ts_gpu_ns;
    tot->wall_ns = g_ts_wall_ns;
  }
  int i = 0;
  if (rows) {
    for (const auto &r : g_ts_agg) {
      if (i >= cap) break;
      rows[i].kid = (uint32_t)(r.first >> 48);
      rows[i].gx = (uint32_t)((r.first >> 20) & 0xFFFFFFF);
      rows[i].gy = (uint32_t)(r.first & 0xFFFFF);
      rows[i].pad = 0;
      rows[i].calls = r.second.n;
      rows[i].ns = r.second.ns;
      ++i;
    }
  }
  return (int)g_ts_agg.size();
}

// fcuda_dw_counters fills out[0..n) with dispatches, readback bytes, device-idle ns (including
// an idle interval still open now) and unobserved launches, in that order (cudaDW* in Go).
extern "C" void fcuda_dw_counters(uint64_t *out, int n) {
  uint64_t idle = 0;
  for (;;) {
    const uint64_t seq = g_dw_idle_seq.load(std::memory_order_acquire);
    if (seq & 1) continue;
    idle = g_dw_idle_ns.load(std::memory_order_relaxed);
    const uint64_t since = g_dw_idle_since.load(std::memory_order_relaxed);
    if (g_dw_idle_seq.load(std::memory_order_acquire) != seq) continue;
    if (since) {
      const uint64_t t = dw_now();
      if (t > since) idle += t - since;
    }
    break;
  }
  const uint64_t v[4] = {g_dw_dispatches.load(std::memory_order_relaxed), g_dw_readback.load(std::memory_order_relaxed),
                         idle, g_dw_unobserved.load(std::memory_order_relaxed)};
  for (int i = 0; i < n && i < 4; ++i) out[i] = v[i];
}

// fcuda_dw_site_rows writes up to cap nonzero (site, stage) cells and returns how many exist,
// which may exceed cap (size with cap 0, then fill).
extern "C" int fcuda_dw_site_rows(uint32_t *site, uint32_t *stage, uint64_t *syncs, uint64_t *wait_ns, uint64_t *d2h, int cap) {
  const uint32_t stages = g_dw_stage_count.load(std::memory_order_acquire);
  int n = 0;
  for (uint32_t s = 0; s < kDwSites; ++s) {
    for (uint32_t g = 0; g < stages && g < kDwStages; ++g) {
      const DwCell &c = g_dw_cells[s][g];
      const uint64_t k = c.syncs.load(std::memory_order_relaxed);
      if (k == 0) continue;
      if (n < cap && site && stage && syncs && wait_ns && d2h) {
        site[n] = s;
        stage[n] = g;
        syncs[n] = k;
        wait_ns[n] = c.wait_ns.load(std::memory_order_relaxed);
        d2h[n] = c.d2h.load(std::memory_order_relaxed);
      }
      ++n;
    }
  }
  return n;
}

// fcuda_dw_stage_name names a kernel stage id; ids 0 and 1 are "none" and "other", and an id
// not yet assigned reads NULL.
extern "C" const char *fcuda_dw_stage_name(uint32_t id) {
  if (id == kDwStageNone) return "none";
  if (id == kDwStageOther) return "other";
  if (id >= kDwStages || id >= g_dw_stage_count.load(std::memory_order_acquire)) return nullptr;
  return g_dw_stage_names[id].load(std::memory_order_relaxed);
}

extern "C" const char *fcuda_dw_site_name(uint32_t id) { return id < kDwSites ? kDwSiteNames[id] : nullptr; }

// fcuda_dw_note_unobserved marks the counters incomplete for the process lifetime: a launch or
// host wait ran on a path these counters do not see (a CUDA graph replay, or a translation
// unit that does not include this file). The reader then reports SyncObserved=false.
extern "C" void fcuda_dw_note_unobserved(void) { g_dw_unobserved.fetch_add(1, std::memory_order_relaxed); }

// Host-wait interposers. Each calls the real CUDA API (the macros below are defined after them)
// and books the wait. A drained g_stream is proven only by a wait on g_stream, the legacy
// stream (it orders after every blocking stream) or the device, or by a synchronous D2H copy;
// a synchronous H2D copy from pageable memory may return before its DMA lands, and an event
// or other-stream wait proves nothing about g_stream, so those leave the idle clock closed.
static inline cudaError_t fcuda_dw_stream_synchronize(cudaStream_t s) {
  const uint64_t t0 = dw_now();
  const cudaError_t e = cudaStreamSynchronize(s);
  dw_wait_done(kDwSiteSubmitWait, t0, 0, e == cudaSuccess && (s == g_stream || s == 0 || s == cudaStreamLegacy));
  return e;
}

static inline cudaError_t fcuda_dw_device_synchronize(void) {
  const uint64_t t0 = dw_now();
  const cudaError_t e = cudaDeviceSynchronize();
  dw_wait_done(kDwSiteSubmitWait, t0, 0, e == cudaSuccess);
  return e;
}

static inline cudaError_t fcuda_dw_event_synchronize(cudaEvent_t ev) {
  const uint64_t t0 = dw_now();
  const cudaError_t e = cudaEventSynchronize(ev);
  dw_wait_done(kDwSitePipelineDrain, t0, 0, false);
  return e;
}

static inline cudaError_t fcuda_dw_memcpy(void *dst, const void *src, size_t n, cudaMemcpyKind kind) {
  if (kind != cudaMemcpyDeviceToHost && kind != cudaMemcpyHostToDevice) return cudaMemcpy(dst, src, n, kind);
  const uint64_t t0 = dw_now();
  const cudaError_t e = cudaMemcpy(dst, src, n, kind);
  const bool d2h = kind == cudaMemcpyDeviceToHost;
  dw_wait_done(d2h ? kDwSiteReadback : kDwSiteUpload, t0, d2h && e == cudaSuccess ? (uint64_t)n : 0, d2h && e == cudaSuccess);
  return e;
}

static inline cudaError_t fcuda_dw_graph_launch(cudaGraphExec_t exec, cudaStream_t s) {
  fcuda_dw_note_unobserved();
  dw_busy();
  return cudaGraphLaunch(exec, s);
}

#define cudaStreamSynchronize(s) fcuda_dw_stream_synchronize(s)
#define cudaDeviceSynchronize() fcuda_dw_device_synchronize()
#define cudaEventSynchronize(e) fcuda_dw_event_synchronize(e)
#define cudaMemcpy(d, s, n, k) fcuda_dw_memcpy((d), (s), (n), (k))
#define cudaGraphLaunch(g, s) fcuda_dw_graph_launch((g), (s))
