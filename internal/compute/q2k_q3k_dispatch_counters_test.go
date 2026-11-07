package compute

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Ref #13860. The issued Q2_K/Q3_K scalar and tile dispatch counters are
// source-witnessed: the native profiler's classification is C++ inside
// vulkan_shim.cpp, so the red-before-green witness reads the exact branch
// text instead of driving a device. Scalar = the historical per-token
// scalar/batched path (q2k_matmul, q2k_matvec, routed id); tile = the V4.1
// token panel/pair kernels ("A missing module leaves the historical
// scalar/batched path alone", vulkan_v41_q2k_token_panel.go).
//
// fak-test:runtime fast est=5ms lane=default
func TestVulkanQ2KQ3KDispatchCounterClassificationSource(t *testing.T) {
	t.Parallel()
	read := func(path string) string {
		t.Helper()
		raw, err := os.ReadFile(filepath.FromSlash(path))
		if err != nil {
			t.Fatal(err)
		}
		return strings.ReplaceAll(string(raw), "\r\n", "\n")
	}
	shim := read("vulkan_shim.cpp")
	header := read("vulkan_backend.h")

	structBody := cFuncBody(t, shim, "struct DispatchProfileCounters {")
	for _, field := range []string{
		"std::atomic<uint64_t> q3k{0};",
		"std::atomic<uint64_t> q2kScalar{0};",
		"std::atomic<uint64_t> q2kTile{0};",
		"std::atomic<uint64_t> q3kScalar{0};",
		"std::atomic<uint64_t> q3kTile{0};",
	} {
		if !strings.Contains(structBody, field) {
			t.Errorf("DispatchProfileCounters lacks %q", field)
		}
	}

	for _, field := range []string{
		"uint64_t q3k_matmul_dispatches;",
		"uint64_t q2k_scalar_dispatches;",
		"uint64_t q2k_tile_dispatches;",
		"uint64_t q3k_scalar_dispatches;",
		"uint64_t q3k_tile_dispatches;",
	} {
		if !strings.Contains(header, field) {
			t.Errorf("fvk_dispatch_profile lacks %q", field)
		}
	}

	body := cFuncBody(t, shim, "static inline void dpDispatch(const Kernel& k) {")
	// Flag-only regression: classification only runs when the profiler is on.
	if gate := strings.Index(body, "if (!g_dp_on) return;"); gate < 0 {
		t.Fatal("dpDispatch lost the profiler gate")
	} else {
		for _, inc := range []string{"g_dp.q2kScalar", "g_dp.q2kTile", "g_dp.q3kScalar", "g_dp.q3kTile"} {
			if i := strings.Index(body, inc+".fetch_add"); i >= 0 && i < gate {
				t.Errorf("%s classification runs before the profiler gate", inc)
			}
		}
	}
	// Ordered classification: q4k, then the four Q2_K/Q3_K identity buckets,
	// then the catch-all other. Each guard names its exact kernel-id set.
	type branch struct {
		guard    string
		incremnt string
	}
	branches := []branch{
		{"id == K_Q4K_MATMUL || id == K_Q4K_MATMUL_WAVE32 || id == K_Q4K_MATMUL_COOPMAT", "g_dp.q4k.fetch_add"},
		{"id == K_Q2K_MATMUL || id == K_RMSNORM_Q2K_MATMUL2 || id == K_Q2K_MATVEC || id == K_V41_Q2K_MATVEC_ID", "g_dp.q2kScalar.fetch_add"},
		{"id == K_V41_Q2K_TOKEN_PANEL || id == K_V41_Q2K_TOKEN_PAIR", "g_dp.q2kTile.fetch_add"},
		{"id == K_Q3K_MATMUL || id == K_V41_Q3K_MATVEC_ID", "g_dp.q3kScalar.fetch_add"},
		{"id == K_V41_Q3K_TOKEN_PANEL", "g_dp.q3kTile.fetch_add"},
	}
	prevEnd := 0
	for i, b := range branches {
		guard := strings.Index(body, b.guard)
		if guard < 0 {
			t.Errorf("dpDispatch branch %d lost guard %q", i, b.guard)
			continue
		}
		if guard < prevEnd {
			t.Errorf("dpDispatch branch %d (%q) is not after branch %d", i, b.guard, i-1)
		}
		inc := strings.Index(body[guard:], b.incremnt)
		if inc < 0 {
			t.Errorf("dpDispatch branch %d (%q) does not increment %s", i, b.guard, b.incremnt)
			continue
		}
		prevEnd = guard + inc
	}
	// Each format total still counts its scalar and tile dispatches on the same
	// dispatch line (same branch), and the Q3_K scalar path keeps K_Q3K_MATMUL;
	// the catch-all must not double-count.
	for _, pair := range [][2]string{
		{"g_dp.q2kScalar.fetch_add", "g_dp.q2k.fetch_add"},
		{"g_dp.q2kTile.fetch_add", "g_dp.q2k.fetch_add"},
		{"g_dp.q3kScalar.fetch_add", "g_dp.q3k.fetch_add"},
		{"g_dp.q3kTile.fetch_add", "g_dp.q3k.fetch_add"},
	} {
		lines := shimLinesContaining(body, pair[0])
		if len(lines) != 1 || !strings.Contains(lines[0], pair[1]) {
			t.Errorf("%s does not fold into its format total %s on the same dispatch line", pair[0], pair[1])
		}
	}

	otherBody := cFuncBody(t, shim, "std::atomic<uint64_t>& dpOtherFamily(KId id) {")
	matmulCase := otherBody[strings.Index(otherBody, "case K_MATMUL:"):strings.Index(otherBody, "case K_RMSNORM:")]
	for _, dead := range []string{"K_Q3K_MATMUL", "K_V41_Q3K_MATVEC_ID", "K_V41_Q3K_TOKEN_PANEL"} {
		if strings.Contains(matmulCase, dead) {
			t.Errorf("dpOtherFamily otherMatmul still claims %s after it gained a dedicated bucket", dead)
		}
	}

	snapshot := cFuncBody(t, shim, `extern "C" void fvk_dispatch_profile_snapshot(fvk_dispatch_profile* out) {`)
	for _, line := range []string{
		"out->q3k_matmul_dispatches = g_dp.q3k.load();",
		"out->q2k_scalar_dispatches = g_dp.q2kScalar.load();",
		"out->q2k_tile_dispatches = g_dp.q2kTile.load();",
		"out->q3k_scalar_dispatches = g_dp.q3kScalar.load();",
		"out->q3k_tile_dispatches = g_dp.q3kTile.load();",
	} {
		if !strings.Contains(snapshot, line) {
			t.Errorf("fvk_dispatch_profile_snapshot lacks %q", line)
		}
	}

	reset := cFuncBody(t, shim, `extern "C" void fvk_dispatch_profile_reset(void) {`)
	for _, line := range []string{
		"g_dp.q3k.store(0);",
		"g_dp.q2kScalar.store(0);",
		"g_dp.q2kTile.store(0);",
		"g_dp.q3kScalar.store(0);",
		"g_dp.q3kTile.store(0);",
	} {
		if !strings.Contains(reset, line) {
			t.Errorf("fvk_dispatch_profile_reset lacks %q", line)
		}
	}

	periodic := strings.Join(shimLinesContaining(shim, "fak-vulkan profile"), "\n")
	for _, label := range []string{"q2k=", "q3k=", "q2k_scalar=", "q2k_tile=", "q3k_scalar=", "q3k_tile="} {
		if !strings.Contains(periodic, label) {
			t.Errorf("periodic profile line lacks %q", label)
		}
	}
}

func shimLinesContaining(src, needle string) []string {
	var lines []string
	for _, line := range strings.Split(src, "\n") {
		if strings.Contains(line, needle) {
			lines = append(lines, line)
		}
	}
	return lines
}

// The explicit aggregate scope of the issued kernel-identity counters: all-zero
// counters (no-issue and profile-off reads) are in scope, consistent folds are
// in scope, and any counter that escapes the non-Q4_K compute remainder or
// breaks its format fold is refused. fak-test:runtime fast est=2ms lane=default
func TestBackendDispatchCounterScopeValidation(t *testing.T) {
	t.Parallel()
	zero := BackendCounterSnapshot{}
	if err := validateBackendDispatchCounterScope(zero); err != nil {
		t.Fatalf("all-zero counters refused: %v", err)
	}
	consistent := BackendCounterSnapshot{
		ComputeDispatches: 20, Q4KMatmulDispatches: 8, OtherDispatches: 12,
		Q2KMatmulDispatches: 5, Q2KScalarDispatches: 3, Q2KTileDispatches: 2,
		Q3KMatmulDispatches: 4, Q3KScalarDispatches: 3, Q3KTileDispatches: 1,
	}
	if err := validateBackendDispatchCounterScope(consistent); err != nil {
		t.Fatalf("consistent counters refused: %v", err)
	}
	for _, tc := range []struct {
		name     string
		counters BackendCounterSnapshot
	}{
		{
			name:     "q2k fold broken",
			counters: BackendCounterSnapshot{ComputeDispatches: 10, OtherDispatches: 10, Q2KMatmulDispatches: 5, Q2KScalarDispatches: 2, Q2KTileDispatches: 2},
		},
		{
			name:     "q3k fold broken",
			counters: BackendCounterSnapshot{ComputeDispatches: 10, OtherDispatches: 10, Q3KMatmulDispatches: 4, Q3KScalarDispatches: 3, Q3KTileDispatches: 2},
		},
		{
			name:     "format totals exceed the non-Q4_K compute remainder",
			counters: BackendCounterSnapshot{ComputeDispatches: 10, Q4KMatmulDispatches: 8, OtherDispatches: 2, Q2KMatmulDispatches: 3, Q2KScalarDispatches: 3},
		},
		{
			name:     "q3k exceeds the remainder left by q2k",
			counters: BackendCounterSnapshot{ComputeDispatches: 10, OtherDispatches: 5, Q2KMatmulDispatches: 3, Q2KScalarDispatches: 3, Q3KMatmulDispatches: 3, Q3KScalarDispatches: 3},
		},
	} {
		if err := validateBackendDispatchCounterScope(tc.counters); err == nil {
			t.Errorf("%s: out-of-scope counters accepted", tc.name)
		}
	}
}

// The observation delta carries the issued Q2_K/Q3_K scalar and tile counters
// field by field, and a reset or wrap in any of them is refused.
// fak-test:runtime fast est=2ms lane=default
func TestBackendExecutionDeltaCarriesQ2KQ3KDispatchCounters(t *testing.T) {
	t.Parallel()
	identity := BackendRuntimeIdentity{Backend: "fixture", Device: "d", Driver: "dr", Runtime: "rt"}
	before := BackendExecutionSnapshot{Identity: identity, TransferCountersObserved: true}
	after := BackendExecutionSnapshot{
		Identity: identity, TransferCountersObserved: true,
		Counters: BackendCounterSnapshot{
			ComputeDispatches: 20, Q4KMatmulDispatches: 8, OtherDispatches: 12, DispatchSubmits: 20,
			Q2KMatmulDispatches: 5, Q2KScalarDispatches: 3, Q2KTileDispatches: 2,
			Q3KMatmulDispatches: 4, Q3KScalarDispatches: 3, Q3KTileDispatches: 1,
			H2DCount: 1, H2DBytes: 16,
		},
	}
	delta, err := BackendExecutionDelta(before, after)
	if err != nil {
		t.Fatalf("delta: %v", err)
	}
	want := BackendCounterSnapshot{
		ComputeDispatches: 20, Q4KMatmulDispatches: 8, OtherDispatches: 12, DispatchSubmits: 20,
		Q2KMatmulDispatches: 5, Q2KScalarDispatches: 3, Q2KTileDispatches: 2,
		Q3KMatmulDispatches: 4, Q3KScalarDispatches: 3, Q3KTileDispatches: 1,
		H2DCount: 1, H2DBytes: 16,
	}
	if delta.Counters != want {
		t.Fatalf("delta counters = %+v, want %+v", delta.Counters, want)
	}
	wrapped := after
	wrapped.Counters.Q3KTileDispatches = 0
	if _, err := BackendExecutionDelta(after, wrapped); err == nil {
		t.Fatal("q3k tile counter wrap accepted")
	}
}

// AddBackendCounters carries the issued Q2_K/Q3_K scalar and tile counters.
// fak-test:runtime fast est=2ms lane=default
func TestAddBackendCountersCarriesQ2KQ3KDispatchCounters(t *testing.T) {
	t.Parallel()
	a := BackendCounterSnapshot{Q2KMatmulDispatches: 2, Q2KScalarDispatches: 1, Q2KTileDispatches: 1, Q3KMatmulDispatches: 3, Q3KScalarDispatches: 2, Q3KTileDispatches: 1}
	b := BackendCounterSnapshot{Q2KMatmulDispatches: 1, Q2KScalarDispatches: 1, Q3KMatmulDispatches: 1, Q3KTileDispatches: 1}
	got := AddBackendCounters(a, b)
	want := BackendCounterSnapshot{Q2KMatmulDispatches: 3, Q2KScalarDispatches: 2, Q2KTileDispatches: 1, Q3KMatmulDispatches: 4, Q3KScalarDispatches: 2, Q3KTileDispatches: 2}
	if got != want {
		t.Fatalf("AddBackendCounters = %+v, want %+v", got, want)
	}
}
