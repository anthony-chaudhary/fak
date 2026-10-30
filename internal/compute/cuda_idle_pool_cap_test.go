package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// Execute the production idle-pool alloc/free/cap bodies against a host allocator and
// ledger shim. This is an allocator-policy witness, not a CUDA or hardware qualification.
// fak-test:runtime medium est=2s lane=default
func TestCUDAIdlePoolCapBoundsReusesAndEvicts(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	memory := readCUDASource(t, "cuda/runtime_memory.cuh")
	program := cudaIdlePoolCapHarness +
		"\nstatic bool fcuda_pool_may_release(void) " + cudaFunctionBody(t, memory, "fcuda_pool_may_release") +
		"\nstatic void fcuda_pool_evict_to_cap(void) " + cudaFunctionBody(t, memory, "fcuda_pool_evict_to_cap") +
		"\nstatic void fcuda_pool_put(void *d, size_t bytes) " + cudaFunctionBody(t, memory, "fcuda_pool_put") +
		"\nextern \"C\" void fcuda_pool_set_cap(size_t cap_bytes) " + cudaFunctionBody(t, memory, "fcuda_pool_set_cap") +
		"\nextern \"C\" void *fcuda_malloc_tagged(size_t bytes, int cls, int site) " + cudaFunctionBody(t, memory, "fcuda_malloc_tagged") +
		"\nextern \"C\" void fcuda_free(void *d) " + cudaFunctionBody(t, memory, "fcuda_free") +
		cudaIdlePoolCapMain
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "idle_pool.cpp"), filepath.Join(dir, "idle_pool")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile production idle pool: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("production idle pool: %v\n%s", err, out)
	}
}

const cudaIdlePoolCapHarness = `
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <unordered_map>
#include <vector>
#define FAK_ALLOC_IDLE_POOL 7
#define FAK_ALLOC_DEVICE_LOCAL 0
#define FAK_ALLOC_SITE(name) 1
#define CK(value) do { (void)(value); } while (0)
using cudaError_t = int;
static constexpr int cudaSuccess = 0;
static const char *cudaGetErrorString(cudaError_t) { return "fake"; }
static bool g_capture_open = false, g_v41_router_uncertain = false, g_v41_mmid_uncertain = false, g_stream_uncertain = false;
static int g_cuda_cleanup_status = 0, free_status = 0;
static std::unordered_map<void *, size_t> g_live, g_managed_live, g_panel_live;
static std::unordered_map<size_t, std::vector<void *>> g_pool, g_panel_pool;
static size_t g_panel_live_bytes = 0, g_panel_cached_bytes = 0;
static size_t g_pool_bytes = 0, g_pool_peak_bytes = 0;
static size_t g_pool_cap_bytes = (size_t)512 << 20;
static unsigned long long g_pool_clock = 0, g_pool_hits = 0, g_pool_misses = 0, g_pool_evictions = 0;
static std::unordered_map<size_t, unsigned long long> g_pool_touch;
static bool g_pool_graph_deep = false;
struct LedgerRow { int cls; size_t bytes; };
static std::unordered_map<void *, LedgerRow> ledger;
static std::set<void *> device;
static std::vector<void *> freed;
static bool v41_candidates_stream_uncertain(void) { return g_stream_uncertain || g_cuda_cleanup_status != 0; }
static cudaError_t fcuda_alloc_status(cudaError_t e) { return e; }
static cudaError_t cudaMalloc(void **d, size_t bytes) {
  *d = std::malloc(bytes);
  device.insert(*d);
  return cudaSuccess;
}
static cudaError_t cudaFree(void *d) {
  if (free_status != 0) return free_status;
  device.erase(d);
  freed.push_back(d);
  std::free(d);
  return cudaSuccess;
}
static void fak_alloc_track(void *d, int cls, int, int, size_t bytes) { ledger[d] = {cls, bytes}; }
static void fak_alloc_untrack(void *d) { ledger.erase(d); }
static void fak_alloc_retag(void *d, int cls, int) { ledger[d].cls = cls; }
static void q4_coeff_cache_evict(const void *) {}
static void require(bool ok, const char *message) {
  if (!ok) { std::printf("FAIL: %s (pool=%zu cap=%zu)\n", message, g_pool_bytes, g_pool_cap_bytes); std::exit(1); }
}
static size_t idle_ledger_bytes() {
  size_t n = 0;
  for (auto &kv : ledger) if (kv.second.cls == FAK_ALLOC_IDLE_POOL) n += kv.second.bytes;
  return n;
}
static size_t bucket_bytes() {
  size_t n = 0;
  for (auto &kv : g_pool) n += kv.first * kv.second.size();
  return n;
}
static bool pooled(void *p) {
  for (auto &kv : g_pool) for (void *q : kv.second) if (q == p) return true;
  return false;
}
// Closed invariant: pooled bytes, bucket contents and the idle_pool ledger agree, within the cap.
static void bounded(const char *where) {
  require(g_pool_bytes == bucket_bytes(), where);
  require(idle_ledger_bytes() == g_pool_bytes, where);
  require(g_pool_bytes <= g_pool_cap_bytes, where);
}
static void reset(size_t cap) {
  for (auto &kv : g_pool) for (void *p : kv.second) { device.erase(p); std::free(p); }
  g_pool.clear(); g_pool_touch.clear(); ledger.clear(); freed.clear(); g_live.clear();
  g_pool_bytes = g_pool_peak_bytes = 0; g_pool_hits = g_pool_misses = g_pool_evictions = 0;
  g_pool_cap_bytes = cap; g_capture_open = g_pool_graph_deep = g_stream_uncertain = false;
  g_cuda_cleanup_status = free_status = 0;
}
`

const cudaIdlePoolCapMain = `
int main() {
  // LRU eviction: the least recently touched buckets leave first, oldest buffer first.
  reset(1000);
  void *a = fcuda_malloc_tagged(100, 3, 1), *b = fcuda_malloc_tagged(200, 3, 1);
  void *c = fcuda_malloc_tagged(300, 3, 1), *d = fcuda_malloc_tagged(400, 3, 1);
  fcuda_free(a); fcuda_free(b); fcuda_free(c); fcuda_free(d);
  bounded("fill to cap");
  require(g_pool_bytes == 1000 && g_pool_evictions == 0 && g_pool_misses == 4, "fill to cap evicted");
  void *again = fcuda_malloc_tagged(100, 3, 1);
  require(again == a && g_pool_hits == 1 && g_pool_bytes == 900 && ledger[a].cls == 3, "exact-size reuse missed");
  fcuda_free(again); // bucket 100 is now the most recently touched
  void *e = fcuda_malloc_tagged(250, 3, 1);
  fcuda_free(e);
  bounded("evict to cap");
  require(g_pool_evictions == 2 && freed.size() == 2 && freed[0] == b && freed[1] == c,
          "eviction did not take the least recently touched buckets in order");
  require(pooled(a) && pooled(d) && pooled(e) && !pooled(b) && !pooled(c), "wrong buffers retained");
  require(ledger.count(b) == 0 && ledger.count(c) == 0, "evicted buffers kept ledger rows");

  // A buffer larger than the whole cap is released at once; warm buckets survive.
  void *huge = fcuda_malloc_tagged(5000, 3, 1);
  fcuda_free(huge);
  bounded("oversized free");
  require(!pooled(huge) && pooled(a) && pooled(d) && pooled(e) && freed.back() == huge, "oversized buffer pooled");

  // Capture open: no cudaFree; set_cap after capture evicts back under the cap.
  void *f = fcuda_malloc_tagged(900, 3, 1);
  size_t freesBefore = freed.size();
  g_capture_open = true;
  fcuda_free(f);
  require(freed.size() == freesBefore && pooled(f) && g_pool_bytes > g_pool_cap_bytes, "evicted while capturing");
  g_capture_open = false;
  fcuda_pool_set_cap(1000);
  bounded("post-capture set_cap");

  // Graph prewarm deepened the pool on purpose: the cap never evicts its spares.
  reset(300);
  g_pool_graph_deep = true;
  void *g1 = fcuda_malloc_tagged(200, 3, 1), *g2 = fcuda_malloc_tagged(200, 3, 1);
  fcuda_free(g1); fcuda_free(g2);
  require(freed.empty() && g_pool_bytes == 400 && pooled(g1) && pooled(g2), "graph spares evicted");

  // A failed eviction free retains the entry and its ledger row and latches cleanup.
  reset(300);
  void *h1 = fcuda_malloc_tagged(200, 3, 1), *h2 = fcuda_malloc_tagged(200, 3, 1);
  fcuda_free(h1);
  free_status = 700;
  fcuda_free(h2);
  require(g_cuda_cleanup_status == 700 && pooled(h1) && pooled(h2) && ledger.count(h1) == 1 &&
          g_pool_bytes == bucket_bytes() && idle_ledger_bytes() == g_pool_bytes, "failed eviction lost an entry");
  free_status = 0;
  g_cuda_cleanup_status = 0;

  // Steady state: repeated same-shape requests reuse instead of accumulating.
  reset(1 << 20);
  const size_t sizes[3] = {64, 128, 4096};
  size_t peak = 0;
  for (int request = 0; request < 50; ++request) {
    void *held[6];
    for (int i = 0; i < 6; ++i) held[i] = fcuda_malloc_tagged(sizes[i % 3], 3, 1);
    for (int i = 0; i < 6; ++i) fcuda_free(held[i]);
    bounded("steady state");
    if (request == 0) peak = g_pool_bytes;
    require(g_pool_bytes == peak, "steady-state pool grew across requests");
  }
  require(g_pool_misses == 6 && g_pool_hits == 6 * 49 && g_pool_evictions == 0, "steady state did not reuse");

  // Unique per-request sizes (the old exact KV leak shape) stay bounded by the cap.
  reset(64 * 1024);
  for (int request = 0; request < 200; ++request) {
    void *kv = fcuda_malloc_tagged(4096 + 64 * (size_t)request, 3, 1);
    fcuda_free(kv);
    bounded("unique sizes");
  }
  require(g_pool_evictions > 0 && g_pool_peak_bytes <= g_pool_cap_bytes, "unique sizes not evicted");
  std::puts("ok");
  return 0;
}
`
