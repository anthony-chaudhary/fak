//go:build darwin && arm64 && cgo

package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func testQ8TensorWithIn(in int) *q8Tensor {
	return &q8Tensor{out: 1, in: in, nblk: in / qBlk, q: make([]int8, in), d: make([]float32, in/qBlk)}
}

func testQ8Tensor() *q8Tensor {
	qt := testQ8TensorWithIn(qBlk)
	qt.d[0] = 1
	return qt
}

func cleanupMetalProfileModel(t *testing.T, m *Model) {
	t.Helper()
	t.Cleanup(func() {
		metalQ4KMu.Lock()
		delete(metalQ4KW, m)
		delete(metalQ6KW, m)
		delete(metalQ8Budget, m)
		delete(metalQ8KW, m)
		metalQ4KMu.Unlock()
		metalgemm.ResetQ4K()
		metalgemm.ResetQ8()
	})
}

func forceQ8UploadBudget(t *testing.T, m *Model, allowed bool) {
	t.Helper()
	metalQ4KMu.Lock()
	metalQ8Budget[m] = allowed
	metalQ4KMu.Unlock()
	cleanupMetalProfileModel(t, m)
}

func requireFallbackRoutes(t *testing.T, profiler *PhaseProfiler, want ...MetalFallbackRoute) {
	t.Helper()
	receipt, err := profiler.MetalFallbackReceipt()
	if err != nil {
		t.Fatal(err)
	}
	if len(receipt.Events) != len(want) {
		t.Fatalf("fallback events = %+v, want routes %v", receipt.Events, want)
	}
	for i, route := range want {
		if receipt.Events[i].Route != route {
			t.Fatalf("fallback event[%d] = %+v, want route %q", i, receipt.Events[i], route)
		}
	}
}

func TestQ8GemmDispatchRecordsDeclinedMetalUpload(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	qt := testQ8Tensor()
	m := &Model{q8w: map[string]*q8Tensor{"q8": qt}}
	forceQ8UploadBudget(t, m, false)
	profiler := NewPhaseProfiler()
	s := &Session{M: m, MetalQ4K: true, PhaseProfiler: profiler}
	panel := &q8Panel{q: make([]int8, qBlk), d: []float32{1}, P: 1, in: qBlk, nblk: 1}

	if got := s.q8GemmDispatch("q8", qt, panel); len(got) != qt.out {
		t.Fatalf("q8GemmDispatch len = %d, want %d", len(got), qt.out)
	}
	if got := profiler.MetalFallbackCount(); got != 1 {
		t.Fatalf("fallback count = %d, want 1", got)
	}
	requireFallbackRoutes(t, profiler, MetalFallbackQ8GEMMCPU)
}

func TestQ8MatRowsDispatchRecordsDeclinedMetalUpload(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	qt := testQ8Tensor()
	m := &Model{q8w: map[string]*q8Tensor{"q8": qt}}
	forceQ8UploadBudget(t, m, false)
	profiler := NewPhaseProfiler()
	s := &Session{M: m, MetalQ4K: true, PhaseProfiler: profiler}

	if got := s.q8MatRowsDispatch("q8", qt, make([]float32, qBlk)); len(got) != qt.out {
		t.Fatalf("q8MatRowsDispatch len = %d, want %d", len(got), qt.out)
	}
	if got := profiler.MetalFallbackCount(); got != 1 {
		t.Fatalf("fallback count = %d, want 1", got)
	}
	requireFallbackRoutes(t, profiler, MetalFallbackQ8GEMVCPU)
}

func TestSessionQ4KKernelSingletonQ6RecordsCPUFallback(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	qt := randomQ6KTensor(1, qkK, 9020)
	m := &Model{kqw: map[string]*kQuantTensor{"q6": qt}}
	metalQ4KMu.Lock()
	metalQ6KW[m] = map[string]*metalgemm.Q6KWeight{"q6": nil}
	metalQ4KMu.Unlock()
	cleanupMetalProfileModel(t, m)
	profiler := NewPhaseProfiler()
	s := &Session{M: m, MetalQ4K: true, PhaseProfiler: profiler}

	got := (sessionQ4KKernel{s: s}).mul("q6", make([]float32, qkK), qt.out, qt.in)
	if len(got) != qt.out {
		t.Fatalf("singleton Q6 result len = %d, want %d", len(got), qt.out)
	}
	requireFallbackRoutes(t, profiler, MetalFallbackQ6KGEMVCPU)
}

func TestQ4KGroupDispatchRecordsQ4GEMVGroupDecline(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	first := &q4kTensor{out: 1, in: qkK, nblk: 1, raw: make([]byte, q4kBlockBytes)}
	second := &q4kTensor{out: 1, in: 2 * qkK, nblk: 2, raw: make([]byte, 2*q4kBlockBytes)}
	m := &Model{q4kw: map[string]*q4kTensor{"first": first, "second": second}, kqw: map[string]*kQuantTensor{}, q8w: map[string]*q8Tensor{}}
	w1 := metalgemm.UploadQ4K(first.raw, first.out, first.in)
	w2 := metalgemm.UploadQ4K(second.raw, second.out, second.in)
	if w1 == nil || w2 == nil {
		t.Fatal("Q4_K test upload returned nil")
	}
	metalQ4KMu.Lock()
	metalQ4KW[m] = map[string]*metalgemm.Q4KWeight{"first": w1, "second": w2}
	metalQ4KMu.Unlock()
	cleanupMetalProfileModel(t, m)
	profiler := NewPhaseProfiler()
	s := &Session{M: m, MetalQ4K: true, PhaseProfiler: profiler}

	if got := s.q4kGroupDispatch([]string{"first", "second"}, make([]float32, 2*qkK), []int{1, 1}); got != nil {
		t.Fatalf("mismatched Q4_K group = %#v, want nil", got)
	}
	requireFallbackRoutes(t, profiler, MetalFallbackQ4KGEMVGroupDispatch)
}

func TestQ4KGroupDispatchRecordsQ8GEMVGroupDecline(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	first, second := testQ8TensorWithIn(qBlk), testQ8TensorWithIn(2*qBlk)
	first.d[0], second.d[0], second.d[1] = 1, 1, 1
	m := &Model{q4kw: map[string]*q4kTensor{}, kqw: map[string]*kQuantTensor{}, q8w: map[string]*q8Tensor{"first": first, "second": second}}
	forceQ8UploadBudget(t, m, true)
	w1 := metalgemm.UploadQ8(first.q, first.d, first.out, first.in)
	w2 := metalgemm.UploadQ8(second.q, second.d, second.out, second.in)
	if w1 == nil || w2 == nil {
		t.Fatal("Q8 test upload returned nil")
	}
	metalQ4KMu.Lock()
	metalQ8KW[m] = map[string]*metalgemm.Q8Weight{"first": w1, "second": w2}
	metalQ4KMu.Unlock()
	profiler := NewPhaseProfiler()
	s := &Session{M: m, MetalQ4K: true, PhaseProfiler: profiler}

	if got := s.q4kGroupDispatch([]string{"first", "second"}, make([]float32, 2*qBlk), []int{1, 1}); got != nil {
		t.Fatalf("mismatched Q8 group = %#v, want nil", got)
	}
	requireFallbackRoutes(t, profiler, MetalFallbackQ8GEMVGroupDispatch)
}

func TestQ4KGroupDispatchRecordsUnroutedQ8CPUFill(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	first, fallback := testQ8Tensor(), testQ8Tensor()
	first.q[0], fallback.q[0] = 1, 2
	m := &Model{q4kw: map[string]*q4kTensor{}, kqw: map[string]*kQuantTensor{}, q8w: map[string]*q8Tensor{"routed": first, "fallback": fallback}}
	forceQ8UploadBudget(t, m, true)
	w := metalgemm.UploadQ8(first.q, first.d, first.out, first.in)
	if w == nil {
		t.Fatal("UploadQ8 returned nil")
	}
	metalQ4KMu.Lock()
	metalQ8KW[m] = map[string]*metalgemm.Q8Weight{"routed": w, "fallback": nil}
	metalQ4KMu.Unlock()
	profiler := NewPhaseProfiler()
	s := &Session{M: m, MetalQ4K: true, PhaseProfiler: profiler}

	got := s.q4kGroupDispatch([]string{"routed", "fallback"}, make([]float32, qBlk), []int{1, 1})
	if len(got) != 2 || len(got[0]) != 1 || len(got[1]) != 1 {
		t.Fatalf("group result shape = %#v", got)
	}
	if count := profiler.MetalFallbackCount(); count != 1 {
		t.Fatalf("fallback count = %d, want 1", count)
	}
	requireFallbackRoutes(t, profiler, MetalFallbackQ4KGroupQ8CPU)
}
