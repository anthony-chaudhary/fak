//go:build darwin && arm64 && cgo

package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestQ4KGemmGroupDispatchDeclinesOnTypedStallClass locks the classification dependency the wired
// decline branches use: a receipt the native bounded-wait timeout/fault emits must be recognized by
// metalgemm.IsMetalCommandBufferStall for the exact operation strings the grouped, single-weight and
// fused-MLP Q4_K dispatches emit, so a wedged command buffer is never mistaken for a healthy result.
func TestQ4KGemmGroupDispatchDeclinesOnTypedStallClass(t *testing.T) {
	for _, op := range []string{"q4k gemv", "q4k gemv group", "q4k gemm group", "q4k mlp"} {
		receipt := metalgemm.MetalCommandBufferStallError{
			Operation:          op,
			WaitedMilliseconds: metalgemm.DefaultCommandBufferWaitLimit,
			LimitMilliseconds:  metalgemm.DefaultCommandBufferWaitLimit,
		}
		if !metalgemm.IsMetalCommandBufferStall(receipt) {
			t.Fatalf("IsMetalCommandBufferStall(%q receipt) = false, want true", op)
		}
		var typed metalgemm.MetalCommandBufferStallError
		if !errors.As(receipt, &typed) || typed.Operation != op {
			t.Fatalf("errors.As(%q receipt) did not recover the typed stall", op)
		}
	}
	if metalgemm.IsMetalCommandBufferStall(nil) {
		t.Fatal("IsMetalCommandBufferStall(nil) = true, want false")
	}
}

// TestQ4KGemmGroupDispatchDeclinesAndRecordsRoute proves the decline contract the typed-stall
// branches trigger: when the grouped Metal call yields no result, q4kGemmGroupDispatch returns nil
// (so the caller loops per-weight and the turn completes) and records the P-specific fallback route
// - MetalFallbackQ4KGEMVGroupDispatch for the decode (P==1) group, MetalFallbackQ4KGEMMGroupDispatch
// for the prefill (P>1) group.
func TestQ4KGemmGroupDispatchDeclinesAndRecordsRoute(t *testing.T) {
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

	names := []string{"first", "second"}

	prefillProfiler := NewPhaseProfiler()
	prefill := &Session{M: m, MetalQ4K: true, PhaseProfiler: prefillProfiler}
	if got := prefill.q4kGemmGroupDispatch(names, make([]float32, 2*qkK), 2); got != nil {
		t.Fatalf("mismatched prefill group = %#v, want nil", got)
	}
	requireFallbackRoutes(t, prefillProfiler, MetalFallbackQ4KGEMMGroupDispatch)

	decodeProfiler := NewPhaseProfiler()
	decode := &Session{M: m, MetalQ4K: true, PhaseProfiler: decodeProfiler}
	if got := decode.q4kGemmGroupDispatch(names, make([]float32, 2*qkK), 1); got != nil {
		t.Fatalf("mismatched decode group = %#v, want nil", got)
	}
	requireFallbackRoutes(t, decodeProfiler, MetalFallbackQ4KGEMVGroupDispatch)
}
