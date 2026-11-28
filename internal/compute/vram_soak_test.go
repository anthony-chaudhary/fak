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

// fak-test:runtime medium est=2s lane=default
func TestVRAMSoakDirectSlabAllocator(t *testing.T) {
	const totalBytes = 1 << 20 // 1 MiB
	alloc, err := NewDirectSlabAllocator(DirectSlabConfig{
		TotalBytes: totalBytes,
		Alignment:  64,
	})
	if err != nil {
		t.Fatalf("NewDirectSlabAllocator: %v", err)
	}
	defer func() {
		if err := alloc.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()

	reqSizes := []int64{64, 512, 4096, 16384, 65536}
	for round := 0; round < 10; round++ {
		for req := 0; req < 32; req++ {
			held := make([]*SlabAllocation, 0, len(reqSizes))
			for i, sz := range reqSizes {
				// Vary order across requests to exercise coalescing.
				s := reqSizes[(i+req)%len(reqSizes)]
				_ = sz
				a, err := alloc.Allocate(s)
				if err != nil {
					t.Fatalf("round %d req %d alloc(%d): %v", round, req, s, err)
				}
				if used := alloc.Used(); used > totalBytes || used <= 0 {
					t.Fatalf("round %d req %d used=%d out of bounds (0, %d]", round, req, used, totalBytes)
				}
				held = append(held, a)
			}
			for _, a := range held {
				if err := a.Free(); err != nil {
					t.Fatalf("round %d req %d free: %v", round, req, err)
				}
			}
			if used := alloc.Used(); used != 0 {
				t.Fatalf("round %d req %d leaked %d bytes after free", round, req, used)
			}
			if avail := alloc.Available(); avail != totalBytes {
				t.Fatalf("round %d req %d available=%d want %d", round, req, avail, totalBytes)
			}
		}

		// Exercise Reset() with both clean state and unfreed allocations.
		if round%2 == 1 {
			if _, err := alloc.Allocate(4096); err != nil {
				t.Fatalf("round %d pre-reset alloc: %v", round, err)
			}
		}
		if err := alloc.Reset(); err != nil {
			t.Fatalf("round %d Reset: %v", round, err)
		}
		if used := alloc.Used(); used != 0 {
			t.Fatalf("round %d post-reset used=%d want 0", round, used)
		}
		if avail := alloc.Available(); avail != totalBytes {
			t.Fatalf("round %d post-reset available=%d want %d", round, avail, totalBytes)
		}
	}

	if err := alloc.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := alloc.Allocate(64); err == nil {
		t.Fatal("expected Allocate after Close to fail")
	}
}

// Execute sustained allocation/reuse/eviction/teardown cycles against the production
// CUDA idle-pool bodies using the host allocator shim from cuda_idle_pool_cap_test.go.
// fak-test:runtime medium est=2s lane=default
func TestVRAMSoakCUDAIdlePool(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	memory := readCUDASource(t, "cuda/runtime_memory.cuh")
	program := cudaIdlePoolCapHarness + "\n" + readCUDASource(t, "cuda/v41_compressor_pending.cuh") +
		"\nstatic bool fcuda_pool_may_release(void) " + cudaFunctionBody(t, memory, "fcuda_pool_may_release") +
		"\nstatic void fcuda_pool_evict_to_cap(void) " + cudaFunctionBody(t, memory, "fcuda_pool_evict_to_cap") +
		"\nstatic void fcuda_pool_put(void *d, size_t bytes) " + cudaFunctionBody(t, memory, "fcuda_pool_put") +
		"\nextern \"C\" void fcuda_pool_set_cap(size_t cap_bytes) " + cudaFunctionBody(t, memory, "fcuda_pool_set_cap") +
		"\nextern \"C\" void fcuda_pool_cap_stats(size_t *cap, size_t *bytes, size_t *peak, unsigned long long *hits, unsigned long long *misses, unsigned long long *evictions) " + cudaFunctionBody(t, memory, "fcuda_pool_cap_stats") +
		"\nextern \"C\" void *fcuda_malloc_tagged(size_t bytes, int cls, int site) " + cudaFunctionBody(t, memory, "fcuda_malloc_tagged") +
		"\nextern \"C\" void fcuda_free(void *d) " + cudaFunctionBody(t, memory, "fcuda_free") +
		vramSoakCUDAIdlePoolMain

	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "vram_soak.cpp"), filepath.Join(dir, "vram_soak")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile vram soak pool: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("vram soak pool: %v\n%s", err, out)
	}
}

const vramSoakCUDAIdlePoolMain = `
static size_t pooled_count() {
  size_t n = 0;
  for (auto &kv : g_pool) n += kv.second.size();
  return n;
}

int main() {
  for (int cycle = 0; cycle < 4; ++cycle) {
    // Phase 1: steady-state multi-request reuse stays flat and leak-free.
    reset(1 << 20);
    const size_t steady[6] = {256, 1024, 4096, 8192, 16384, 32768};
    size_t warmup_bytes = 0;
    for (int req = 0; req < 120; ++req) {
      void *ptrs[6];
      for (int i = 0; i < 6; ++i) {
        ptrs[i] = fcuda_malloc_tagged(steady[i], 3, 1);
        require(!pooled(ptrs[i]), "soak steady live ptr still pooled");
      }
      for (int i = 0; i < 6; ++i) fcuda_free(ptrs[i]);
      bounded("soak steady");
      require(g_live.empty(), "soak steady leaked live allocation");
      if (req == 0) warmup_bytes = g_pool_bytes;
      require(g_pool_bytes == warmup_bytes, "soak steady pool grew");
    }
    size_t cap = 0, bytes = 0, peak = 0;
    unsigned long long hits = 0, misses = 0, evictions = 0;
    fcuda_pool_cap_stats(&cap, &bytes, &peak, &hits, &misses, &evictions);
    require(hits == 6ULL * 119ULL && misses == 6ULL && evictions == 0, "soak steady stats mismatch");

    // Phase 2: eviction churn across unique + shared sizes stays bounded under cap.
    reset(64 * 1024);
    for (int req = 0; req < 250; ++req) {
      void *a = fcuda_malloc_tagged(4096 + (size_t)(req % 37) * 256, 3, 1);
      void *b = fcuda_malloc_tagged(8192 + (size_t)req * 64, 3, 1);
      fcuda_free(a);
      fcuda_free(b);
      bounded("soak churn");
      require(g_live.empty(), "soak churn leaked live allocation");
      require(ledger.size() == pooled_count(), "soak churn ledger count mismatch");
      require(device.size() == pooled_count(), "soak churn device count mismatch");
    }
    fcuda_pool_cap_stats(&cap, &bytes, &peak, &hits, &misses, &evictions);
    require(evictions > 0 && peak <= cap, "soak churn exceeded cap or never evicted");

    // Phase 3: cap-to-zero teardown releases every pooled allocation with zero leaks.
    fcuda_pool_set_cap(0);
    bounded("soak teardown");
    require(g_pool_bytes == 0 && idle_ledger_bytes() == 0, "soak teardown left pooled bytes");
    require(ledger.empty() && device.empty(), "soak teardown leaked device/ledger allocations");
  }
  std::puts("ok");
  return 0;
}
`
