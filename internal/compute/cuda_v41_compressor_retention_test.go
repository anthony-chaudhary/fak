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

// Execute the production drain and release bodies with a host allocator shim.
// This is a lifetime regression witness, not a CUDA or hardware qualification.
// fak-test:runtime medium est=1s lane=default
func TestCUDAV41CompressorFailedDrainRetainsAllocations(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	memory := readCUDASource(t, "cuda/runtime_memory.cuh")
	services := readCUDASource(t, "cuda/runtime_services.cuh")
	compressor := readCUDASource(t, "cuda/v41_compressor.cuh")
	program := cudaV41CompressorRetentionHarness +
		"\nstatic int v41c_drain(int status) " + cudaFunctionBody(t, compressor, "v41c_drain") +
		"\nextern \"C\" void fcuda_free(void *d) " + cudaFunctionBody(t, memory, "fcuda_free") +
		"\nextern \"C\" void fcuda_free_on(int device, void *d) " + cudaFunctionBody(t, services, "fcuda_free_on") +
		"\nextern \"C\" int fcuda_discard_unsubmitted(void *d, size_t expected_bytes) " + cudaFunctionBody(t, memory, "fcuda_discard_unsubmitted") +
		cudaV41CompressorRetentionMain
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "retention.cpp"), filepath.Join(dir, "retention")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile production compressor retention: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("production compressor retention: %v\n%s", err, out)
	}
}

const cudaV41CompressorRetentionHarness = `
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>
#define FAK_CUDA_V41_COMPRESSOR
#define FAK_ALLOC_IDLE_POOL 1
#define FAK_ALLOC_SITE(name) 1
#define CK(value) do { (void)(value); } while (0)
using cudaError_t = int;
static constexpr int cudaSuccess = 0, cudaErrorInvalidValue = 1;
static bool g_v41_compressor_uncertain = false, g_v41_router_uncertain = false, g_capture_open = false;
static int g_stream = 1, sync_status = 0;
static int frees = 0, untracks = 0, retags = 0, evictions = 0, scopes = 0, syncs = 0;
static std::map<void *, size_t> g_live, g_managed_live, g_panel_live;
static std::map<size_t, std::vector<void *>> g_pool, g_panel_pool;
static size_t g_panel_live_bytes = 0, g_panel_cached_bytes = 0;
static cudaError_t cudaStreamSynchronize(int) { ++syncs; return sync_status; }
static cudaError_t cudaFree(void *) { ++frees; return cudaSuccess; }
static void fak_alloc_untrack(void *) { ++untracks; }
static void fak_alloc_retag(void *, int, int) { ++retags; }
static void q4_coeff_cache_evict(const void *) { ++evictions; }
struct fcuda_device_scope {
  explicit fcuda_device_scope(int) { ++scopes; }
  int status() const { return 0; }
};
static void require(bool ok, const char *message) {
  if (!ok) { std::puts(message); std::exit(1); }
}
`

const cudaV41CompressorRetentionMain = `
int main() {
  int storage[6] = {};
  void *pooled = &storage[0], *managed = &storage[1], *panel = &storage[2];
  void *direct = &storage[3], *on_device = &storage[4], *unsubmitted = &storage[5];
  auto allocate = [&] {
    g_live[pooled] = g_live[unsubmitted] = sizeof(int);
    g_managed_live[managed] = g_panel_live[panel] = sizeof(int);
    g_panel_live_bytes = sizeof(int);
  };
  auto release = [&] {
    fcuda_free(pooled);
    fcuda_free(managed);
    fcuda_free(panel);
    fcuda_free(direct);
    fcuda_free_on(0, on_device);
    return fcuda_discard_unsubmitted(unsubmitted, sizeof(int));
  };
  allocate();
  require(v41c_drain(30007) == 30007 && !g_v41_compressor_uncertain,
          "successful drain changed the launch error or latched uncertainty");
  require(release() == 0, "completed work refused ordinary cleanup");
  require(g_live.empty() && g_managed_live.empty() && g_panel_live.empty() &&
          g_pool[sizeof(int)].size() == 1 && g_panel_pool[sizeof(int)].size() == 1 &&
          g_panel_live_bytes == 0 && g_panel_cached_bytes == sizeof(int) &&
          frees == 4 && untracks == 4 && retags == 2 && evictions == 5 && scopes == 1,
          "successful compressor path changed normal release behavior");
  g_pool.clear(); g_panel_pool.clear(); g_panel_cached_bytes = 0;
  frees = untracks = retags = evictions = scopes = syncs = 0;
  allocate();
  sync_status = 700;
  require(v41c_drain(30007) == 50700 && g_v41_compressor_uncertain,
          "failed drain did not latch completion uncertainty");
  // A later successful fence must not rehabilitate the failed submission.
  sync_status = 0;
  require(v41c_drain(30008) == 30008 && g_v41_compressor_uncertain,
          "later drain erased sticky uncertainty");
  syncs = 0;
  for (int attempt = 0; attempt < 2; ++attempt) {
    require(release() == cudaErrorInvalidValue, "uncertain discard was admitted");
    require(g_live.size() == 2 && g_live.at(pooled) == sizeof(int) &&
            g_live.at(unsubmitted) == sizeof(int) &&
            g_managed_live.size() == 1 && g_managed_live.at(managed) == sizeof(int) &&
            g_panel_live.size() == 1 && g_panel_live.at(panel) == sizeof(int) &&
            g_pool.empty() && g_panel_pool.empty() &&
            g_panel_live_bytes == sizeof(int) && g_panel_cached_bytes == 0 &&
            frees == 0 && untracks == 0 && retags == 0 && evictions == 0 &&
            scopes == 0 && syncs == 0,
            "failed compressor drain allowed free, reuse, or allocator mutation");
  }
}
`
