//go:build darwin && arm64 && cgo

package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestMetalQ2KDispatch tests that resident Q2_K matrices route through Metal GEMV/GEMM
// when MetalQ4K is enabled, verifying handle caching, handle reuse across decode/prefill,
// parity with the CPU reference, and deterministic teardown/isolation via releaseModelQ4KHandles.
func TestMetalQ2KDispatch(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ2K()

	const out, in = 64, 512
	raw1 := randomQ2KTensor(out, in, 119081)
	qt1 := &kQuantTensor{out: out, in: in, nblk: in / qkK, kind: kindQ2K, raw: raw1}
	name1 := "layer.0.q2k.weight"

	m1 := &Model{
		kqw: map[string]*kQuantTensor{
			name1: qt1,
		},
	}
	s1 := &Session{M: m1, MetalQ4K: true}

	// 1. Verify handle is not cached before first dispatch.
	metalQ4KMu.Lock()
	if h := metalQ2KW[m1][name1]; h != nil {
		metalQ4KMu.Unlock()
		t.Fatalf("expected nil handle before dispatch, got %v", h)
	}
	metalQ4KMu.Unlock()

	// 2. Decode GEMV (P=1) via kQuantMatRowsIntoDispatch.
	xf := randomVecF(in, 119082)
	gotY1 := make([]float32, out)
	s1.kQuantMatRowsIntoDispatch(name1, qt1, xf, gotY1)

	// Verify handle was cached and has valid ID.
	metalQ4KMu.Lock()
	cachedHandle := metalQ2KW[m1][name1]
	metalQ4KMu.Unlock()
	if cachedHandle == nil {
		t.Fatal("expected cached Q2_K handle after kQuantMatRowsIntoDispatch, got nil")
	}
	if cachedHandle.ID() < 0 {
		t.Fatalf("expected valid handle ID >= 0, got %d", cachedHandle.ID())
	}

	// Verify GEMV results match CPU reference within tolerance.
	refY1 := make([]float32, out)
	kQuantMatRowsInto(qt1, xf, refY1)
	cos1, maxRel1 := cosineAndMaxRel(refY1, gotY1)
	if cos1 < 0.9999 || maxRel1 > 5e-3 {
		t.Fatalf("Q2_K GEMV (P=1): cosine=%.6f maxRel=%.4g (want cos>=0.9999 maxRel<=5e-3)\n  ref[:4]=%v\n  got[:4]=%v",
			cos1, maxRel1, refY1[:4], gotY1[:4])
	}

	// 3. Batched GEMM (P>1) via kQuantGemmDispatch.
	const P = 4
	Xf := randomVecF(P*in, 119083)
	gotYP := s1.kQuantGemmDispatch(name1, qt1, Xf, P)

	// Verify handle was REUSED (same pointer).
	metalQ4KMu.Lock()
	reusedHandle := metalQ2KW[m1][name1]
	metalQ4KMu.Unlock()
	if reusedHandle != cachedHandle {
		t.Fatalf("expected handle to be reused, got %p vs %p", reusedHandle, cachedHandle)
	}

	// Verify GEMM results match CPU reference within tolerance.
	refYP := make([]float32, P*out)
	kQuantMatRowsIntoBatch(qt1, Xf, P, refYP)
	cosP, maxRelP := cosineAndMaxRel(refYP, gotYP)
	if cosP < 0.9999 || maxRelP > 5e-3 {
		t.Fatalf("Q2_K GEMM (P=%d): cosine=%.6f maxRel=%.4g (want cos>=0.9999 maxRel<=5e-3)\n  ref[:4]=%v\n  got[:4]=%v",
			P, cosP, maxRelP, refYP[:4], gotYP[:4])
	}

	// 4. Isolation across models: separate Model m2 gets distinct handle.
	raw2 := randomQ2KTensor(out, in, 119084)
	qt2 := &kQuantTensor{out: out, in: in, nblk: in / qkK, kind: kindQ2K, raw: raw2}
	m2 := &Model{
		kqw: map[string]*kQuantTensor{
			name1: qt2,
		},
	}
	s2 := &Session{M: m2, MetalQ4K: true}
	gotY2 := make([]float32, out)
	s2.kQuantMatRowsIntoDispatch(name1, qt2, xf, gotY2)

	metalQ4KMu.Lock()
	handle2 := metalQ2KW[m2][name1]
	metalQ4KMu.Unlock()
	if handle2 == nil {
		t.Fatal("expected cached Q2_K handle for m2, got nil")
	}
	if handle2 == cachedHandle {
		t.Fatalf("expected distinct handle for m2, got same pointer %p", handle2)
	}
	if handle2.ID() == cachedHandle.ID() {
		t.Fatalf("expected distinct handle ID for m2, got %d", handle2.ID())
	}

	// 5. releaseModelQ4KHandles(m1) cleans up m1 handles and preserves m2.
	releaseModelQ4KHandles(m1)

	metalQ4KMu.Lock()
	tbl1 := metalQ2KW[m1]
	metalQ4KMu.Unlock()
	if len(tbl1) != 0 {
		t.Fatalf("expected metalQ2KW[m1] to be deleted, got %v", tbl1)
	}
	if cachedHandle.ID() != -1 {
		t.Fatalf("expected m1 cachedHandle to be invalidated (ID=-1), got %d", cachedHandle.ID())
	}

	// m2 handle must still be intact and valid.
	if handle2.ID() < 0 {
		t.Fatalf("expected m2 handle to remain valid, got ID %d", handle2.ID())
	}
	metalQ4KMu.Lock()
	tbl2 := metalQ2KW[m2]
	metalQ4KMu.Unlock()
	if tbl2[name1] != handle2 {
		t.Fatalf("expected m2 handle to still be present in metalQ2KW[m2]")
	}

	// Clean up m2 as well.
	releaseModelQ4KHandles(m2)
	metalQ4KMu.Lock()
	tbl2After := metalQ2KW[m2]
	metalQ4KMu.Unlock()
	if len(tbl2After) != 0 {
		t.Fatalf("expected metalQ2KW[m2] to be deleted, got %v", tbl2After)
	}
	if handle2.ID() != -1 {
		t.Fatalf("expected m2 handle to be invalidated (ID=-1), got %d", handle2.ID())
	}
}
