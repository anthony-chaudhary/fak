package model

import (
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// paging_test.go — the gate for pagedKernel (paging.go), the native-753B Pillar-4
// upload→compute→free primitive. With no device hardware (cpu-ref) it pins the primitive's
// contract: a paged GEMM is bit-equal to the resident GEMM, each op pages in afresh (the kernel
// caches nothing — the paging, not residency, property), and a paged weight leaves no resident
// footprint in a session's halW.

func pagingRandVec(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}

// TestPagedKernelMatMulBitEqualResident pins the paged GEMM byte-for-byte against the resident
// GEMM, and proves the page-in counter: one op pages in once, a second op pages in AGAIN (so the
// weight is re-uploaded each use, not cached — the defining paging behavior).
func TestPagedKernelMatMulBitEqualResident(t *testing.T) {
	be := compute.Default()
	const out, in = 8, 16
	rng := rand.New(rand.NewSource(7))
	w := pagingRandVec(rng, out*in)
	x := pagingRandVec(rng, in)
	xt := be.Upload(compute.NewF32(be, []int{in}, x), compute.F32)

	// resident: upload the weight once and keep it, then MatMul (the weightHAL behavior).
	wRes := be.Upload(compute.NewF32(be, []int{out, in}, w), compute.F32)
	yRes := be.Read(be.MatMul(wRes, xt))

	pk := newPagedKernel(be)
	yPaged := pk.matMul([]int{out, in}, w, xt)
	if len(yPaged) != len(yRes) {
		t.Fatalf("paged len %d, want %d", len(yPaged), len(yRes))
	}
	for i := range yRes {
		if yPaged[i] != yRes[i] {
			t.Fatalf("paged GEMM not bit-equal to resident at %d: %v != %v", i, yPaged[i], yRes[i])
		}
	}
	if pk.pageIn != 1 {
		t.Fatalf("pageIn = %d after one op, want 1", pk.pageIn)
	}

	// A second op pages in AGAIN (nothing cached) — the property that distinguishes paging from a
	// resident weight cache (which would stay at pageIn==1). Still bit-equal.
	yPaged2 := pk.matMul([]int{out, in}, w, xt)
	if pk.pageIn != 2 {
		t.Fatalf("pageIn = %d after two ops, want 2 (each use pages in afresh)", pk.pageIn)
	}
	for i := range yRes {
		if yPaged2[i] != yRes[i] {
			t.Fatalf("second paged GEMM not bit-equal to resident at %d", i)
		}
	}
}

// TestPagedKernelLeavesNoResidentFootprint proves a paged weight is absent from a session's halW
// after the op — the page-OUT that lets VRAM hold only the active weight. It contrasts the paged
// path (halW untouched) with weightHAL (which caches the weight resident), so the difference is the
// residency the Pillar-4 streaming step exploits.
func TestPagedKernelLeavesNoResidentFootprint(t *testing.T) {
	be := compute.Default()
	m := NewSynthetic(tpFwdBaseCfg())
	s := m.NewBackendSession(be)
	defer s.Close()

	const out, in = 8, 16
	rng := rand.New(rand.NewSource(11))
	w := pagingRandVec(rng, out*in)
	x := pagingRandVec(rng, in)
	xt := be.Upload(compute.NewF32(be, []int{in}, x), compute.F32)

	before := len(s.halW)
	pk := newPagedKernel(be)
	_ = pk.matMul([]int{out, in}, w, xt)
	if len(s.halW) != before {
		t.Fatalf("paged matMul changed halW (%d -> %d): a paged weight must leave no resident footprint", before, len(s.halW))
	}

	// Contrast: a resident weightHAL DOES populate halW (the behavior paging deliberately avoids).
	name := layerName(0, "self_attn.q_proj.weight")
	if !m.has(name) {
		t.Skipf("fixture has no %s", name)
	}
	_ = s.weightHAL(name)
	if len(s.halW) <= before {
		t.Fatalf("weightHAL did not populate halW (%d), expected a resident weight — the paging contrast is meaningless", len(s.halW))
	}
}
