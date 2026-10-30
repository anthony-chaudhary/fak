//go:build cuda && cgo

package model

import (
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_ring_async_cuda_test.go — the real-CUDA arm of the owned/classed ring routing: inside a
// live execution scope the expert ring issues its routed-expert page-ins on the CUDA copy stream,
// books overlap, leaves nothing in flight after release, and the MoE output is bit-equal to the
// synchronous (unscoped) arm on the same backend.

func requiredCUDARingBackend(t *testing.T) compute.Backend {
	t.Helper()
	be, ok := compute.Lookup("cuda")
	if !ok || be == nil || be.Name() != "cuda" {
		if os.Getenv("FAK_CUDA_REQUIRE_DEVICE") == "1" {
			t.Fatal("FAK_CUDA_REQUIRE_DEVICE=1 but the cuda backend is not registered")
		}
		t.Skip("cuda backend not registered")
	}
	return be
}

// fak-test:runtime fast est=3s lane=default
func TestCUDAExpertRingScopedPrefetchGoesAsyncAndStaysBitEqual(t *testing.T) {
	be := requiredCUDARingBackend(t)
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)
	in := expertRingTestInput(H)
	newSession := func() *Session {
		return &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}, ExpertRingBytes: perWeight * 3 * K}
	}

	syncS := newSession()
	want := moeFFN{}.apply(m, 0, in, sessionQ4KKernel{s: syncS})
	if st := syncS.ExpertRing(); st.AsyncOverlapped+st.AsyncWaited != 0 {
		t.Fatalf("unscoped ring went async on CUDA: overlapped=%d waited=%d", st.AsyncOverlapped, st.AsyncWaited)
	}
	syncS.Close()

	asyncS := newSession()
	var got []float32
	scope := asyncS.beginBackendExecution()
	func() {
		defer scope.finish()
		got = moeFFN{}.apply(m, 0, in, sessionQ4KKernel{s: asyncS})
	}()
	st := asyncS.ExpertRing()
	t.Logf("cuda ring: prefetched=%d pageIn=%d asyncOverlapped=%d asyncWaited=%d fraction=%.3f",
		st.Prefetched, st.PageIns, st.AsyncOverlapped, st.AsyncWaited, st.AsyncOverlapFraction())
	if st.AsyncOverlapped+st.AsyncWaited == 0 {
		t.Fatal("scoped ring issued no copy-stream transfer on CUDA")
	}
	if r := asyncS.expertRing; r != nil && len(r.pending) != 0 {
		t.Fatalf("%d transfer(s) still pending after the scope released", len(r.pending))
	}
	if len(got) != len(want) {
		t.Fatalf("async produced %d values, sync %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("element %d: async=%v sync=%v", i, got[i], want[i])
		}
	}
	asyncS.Close()
}
