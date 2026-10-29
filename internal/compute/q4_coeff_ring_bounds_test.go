package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This executes the production ring bookkeeping with a scaled, already-reserved
// fake device arena. The fake mint checks one contiguous write, not modulo bytes.
// CUDA allocation, kernels, graph capture, and device execution are not tested.
func TestQ4CoeffRingProductionContiguousBounds(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	data, err := os.ReadFile("cuda/quant_matmul.cuh")
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	begin := strings.Index(source, "constexpr size_t kQ4CoeffArenaBytes =")
	end := strings.Index(source, "// g_q4k_prefill_arm is the P>=128")
	if begin < 0 || end <= begin {
		t.Fatal("production Q4 coefficient ring boundaries missing")
	}
	body := source[begin:end]
	for _, want := range []string{"static void q4_coeff_cache_evict(", "static bool q4_coeff_arena_ready()", "static float2 *q4_coeffs_for("} {
		if !strings.Contains(body, want) {
			t.Fatalf("production ring function missing: %s", want)
		}
	}
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "ring.cpp"), filepath.Join(dir, "ring")
	if err := os.WriteFile(cpp, []byte(q4CoeffBoundsHarness+body+q4CoeffBoundsMain), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "alloc_ledger.h"), cudaAllocLedgerHeader, 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile production Q4 coefficient ring: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("production Q4 coefficient contiguous bounds: %v\n%s", err, out)
	}
}

const q4CoeffBoundsHarness = `
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <initializer_list>
#include "alloc_ledger.h"

struct float2 { float x, y; };
static_assert(sizeof(float2) == 8, "CUDA float2 geometry");
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
static bool g_capture_open = false;
alignas(256) static unsigned char arena[32768];
static uint8_t keys[1024];
static size_t configured_cap = 512;
static int allocations = 0, mints = 0;
static cudaError_t cudaGetDevice(int *device) { *device = 0; return cudaSuccess; }
static cudaError_t cudaMalloc(void **out, size_t) {
  ++allocations; *out = nullptr; return 2;
}
#define CK(call) do { if ((call) != cudaSuccess) std::abort(); } while (0)
static void simulated_mint(float2 *coeff, size_t bytes) {
  const uintptr_t base = reinterpret_cast<uintptr_t>(arena);
  const uintptr_t address = reinterpret_cast<uintptr_t>(coeff);
  const size_t offset = address - base;
  if (address < base || bytes > configured_cap || offset > configured_cap - bytes) {
    std::fprintf(stderr, "q4 coeff contiguous mint overflow: offset=%zu bytes=%zu cap=%zu\n",
                 offset, bytes, configured_cap);
    std::exit(23);
  }
  ++mints;
  std::memset(coeff, mints % 251 + 1, bytes);
}
// The exact production FCUDA_TS launch boundary is the fake hardware seam.
// Its kernel's table length is n contiguous float2 elements, not a wrapped write.
#define FCUDA_TS(name, grid, ...) simulated_mint(coeff, n * sizeof(float2))
`

const q4CoeffBoundsMain = `
#define CHECK(ok) do { if (!(ok)) { \
  std::fprintf(stderr, "ring invariant line=%d cursor=%zu span=%zu entries=%d mints=%d\n", \
    __LINE__, g_q4_coeff_cursor, g_q4_coeff_span, g_q4_coeff_n, mints); return 1; \
} } while (0)

static void reset(size_t cap = 512) {
  std::memset(arena, 0, sizeof(arena));
  std::memset(g_q4_coeff_cache, 0, sizeof(g_q4_coeff_cache));
  g_q4_coeff_head = g_q4_coeff_n = 0;
  g_q4_coeff_cursor = g_q4_coeff_span = 0;
  g_q4_coeff_arena = reinterpret_cast<float2 *>(arena);
  g_q4_coeff_cap = configured_cap = cap;
  allocations = mints = 0;
}
static float2 *mint(int key, size_t bytes) {
  return q4_coeffs_for(&keys[key], static_cast<int>(bytes / 64), 256);
}
static bool invariants() {
  if (g_q4_coeff_span > configured_cap || allocations != 0) return false;
  size_t total = 0;
  for (int k = 0; k < g_q4_coeff_n; ++k) {
    const auto &a = g_q4_coeff_cache[q4_coeff_slot(k)];
    const size_t start = reinterpret_cast<unsigned char *>(a.coeff) - arena;
    const size_t data_bytes = static_cast<size_t>(a.out) * (a.in / 256) * 64;
    if (start > configured_cap || a.len > configured_cap - start ||
        data_bytes > a.len || !a.len) return false;
    total += a.len;
    for (int j = 0; j < k; ++j) {
      const auto &b = g_q4_coeff_cache[q4_coeff_slot(j)];
      const size_t other = reinterpret_cast<unsigned char *>(b.coeff) - arena;
      if (start < other + b.len && other < start + a.len) return false;
    }
  }
  return total == g_q4_coeff_span;
}
int main() {
  // Causal witness: old production evicts A but writes B contiguously at384,
  // overrunning the512-byte logical arena by128. The fake kernel stops before
  // the write; this is not the old Go mirror's modulo-byte assumption.
  reset();
  CHECK(mint(0, 384) == reinterpret_cast<float2 *>(arena));
  CHECK(mint(1, 256) == reinterpret_cast<float2 *>(arena));
  CHECK(g_q4_coeff_n == 1 && g_q4_coeff_span == 256 && g_q4_coeff_cursor == 768);
  CHECK(invariants());
  CHECK(mint(1, 256) == reinterpret_cast<float2 *>(arena) && mints == 2);

  // Padding is reserved with the last region, even after its source is freed.
  // Earlier retained tables and their contents must survive the wrap.
  for (bool dead_last : {false, true}) {
    reset();
    CHECK(mint(0, 256) != nullptr);
    float2 *b = mint(1, 128);
    float2 *c = mint(2, 64);
    unsigned char saved_b[128], saved_c[64];
    std::memcpy(saved_b, b, sizeof(saved_b));
    std::memcpy(saved_c, c, sizeof(saved_c));
    if (dead_last) q4_coeff_cache_evict(&keys[2]);
    CHECK(mint(3, 128) == reinterpret_cast<float2 *>(arena));
    CHECK(g_q4_coeff_n == 3 && g_q4_coeff_span == 384);
    const auto &last_old = g_q4_coeff_cache[q4_coeff_slot(1)];
    CHECK(last_old.q == &keys[2] && last_old.len == 128 && last_old.alive == !dead_last);
    CHECK(std::memcmp(saved_b, b, sizeof(saved_b)) == 0);
    CHECK(std::memcmp(saved_c, c, sizeof(saved_c)) == 0);
    CHECK(mint(1, 128) == b && mints == 4);
    if (!dead_last) CHECK(mint(2, 64) == c && mints == 4);
    CHECK(invariants());
    CHECK(mint(4, 256) != nullptr && invariants());
    CHECK(mint(5, 128) != nullptr && invariants());
  }

  // Exact-end fit needs no padding; a cap-sized table after a nonzero cursor
  // evicts all earlier regions and starts atzero.
  reset();
  CHECK(mint(0, 384) != nullptr);
  CHECK(mint(1, 128) == reinterpret_cast<float2 *>(arena + 384));
  CHECK(g_q4_coeff_cursor == 512 && g_q4_coeff_span == 512 && g_q4_coeff_n == 2);
  CHECK(invariants());
  reset();
  CHECK(mint(0, 64) != nullptr);
  CHECK(mint(1, 512) == reinterpret_cast<float2 *>(arena));
  CHECK(g_q4_coeff_cursor == 1024 && g_q4_coeff_span == 512 && g_q4_coeff_n == 1);
  CHECK(invariants());

  // An empty window must not gain a phantom padding reservation.
  reset(); g_q4_coeff_cursor = 448;
  CHECK(mint(0, 128) == reinterpret_cast<float2 *>(arena));
  CHECK(g_q4_coeff_span == 128 && g_q4_coeff_n == 1 && invariants());

  // The real arena capacities are powers of two: unsigned cursor rollover
  // at the aligned boundary preserves the physical offset and reservations.
  reset(); g_q4_coeff_cursor = static_cast<size_t>(-1) - 63;
  CHECK(mint(0, 128) == reinterpret_cast<float2 *>(arena));
  CHECK(g_q4_coeff_cursor == 128 && g_q4_coeff_span == 128 && invariants());

  // Exercise repeated heterogeneous wraps with real contiguous fake writes.
  reset();
  for (int i = 0; i < 240; ++i) {
    const size_t bytes = static_cast<size_t>(i % 7 + 1) * 64;
    CHECK(mint(i, bytes) != nullptr && invariants());
    if (i % 3 == 0) q4_coeff_cache_evict(&keys[i]);
    CHECK(invariants());
  }

  // Slot-count eviction remains independent of byte capacity.
  reset(sizeof(arena));
  for (int i = 0; i < kQ4CoeffSlots; ++i) CHECK(mint(i, 64) != nullptr);
  CHECK(g_q4_coeff_n == kQ4CoeffSlots && invariants());
  CHECK(mint(kQ4CoeffSlots, 64) == reinterpret_cast<float2 *>(arena));
  CHECK(g_q4_coeff_n == kQ4CoeffSlots && g_q4_coeff_cache[g_q4_coeff_head].q == &keys[1]);
  CHECK(invariants());

  // No new fallback/allocation policy: an over-cap table is still refused
  // without minting or changing cursor/window state.
  reset();
  CHECK(mint(0, 576) == nullptr);
  CHECK(mints == 0 && g_q4_coeff_cursor == 0 && g_q4_coeff_span == 0 && invariants());
  return 0;
}
`
