package model

import (
	"math/rand"
	"reflect"
	"testing"
	"time"
)

func TestSharedExpertOverlapWitness(t *testing.T) {
	// First witness requirements (#9963):
	// 1. Instrumented fixture proves identical outputs (bit-exact parity).
	// 2. Declared dependency order verified in receipt.
	// 3. Positive overlap nanoseconds (saved duration > 0).
	// 4. Lower end-to-end layer latency than the serialized arm.

	hiddenDim := 64
	rng := rand.New(rand.NewSource(9963))

	routedVal := make([]float32, hiddenDim)
	sharedVal := make([]float32, hiddenDim)
	for i := range routedVal {
		routedVal[i] = rng.Float32()*2 - 1
		sharedVal[i] = rng.Float32()*2 - 1
	}

	// Work simulation functions with measurable execution durations
	routedWork := func() ([]float32, error) {
		time.Sleep(10 * time.Millisecond) // simulates EP network dispatch + remote GEMM + combine
		return append([]float32(nil), routedVal...), nil
	}

	sharedWork := func() ([]float32, error) {
		time.Sleep(8 * time.Millisecond) // simulates local shared expert SwiGLU computation
		return append([]float32(nil), sharedVal...), nil
	}

	// 1. Serialized execution baseline
	serialStart := time.Now()
	rOut, err := routedWork()
	if err != nil {
		t.Fatal(err)
	}
	sOut, err := sharedWork()
	if err != nil {
		t.Fatal(err)
	}
	serialDelta := make([]float32, hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		serialDelta[i] = rOut[i] + sOut[i]
	}
	serialDuration := time.Since(serialStart)

	// 2. Overlapped side-stream execution
	overlapDelta, receipt, err := OverlappedMoeExecution(routedWork, sharedWork, hiddenDim)
	if err != nil {
		t.Fatalf("OverlappedMoeExecution failed: %v", err)
	}

	// 1. Verify bit-exact output parity
	if !reflect.DeepEqual(overlapDelta, serialDelta) {
		t.Fatalf("output mismatch between overlapped and serial arms")
	}

	// 2. Verify declared dependency order
	expectedOrder := "fork(shared_side_stream)-exec(routed_main_stream)-join(side_stream)-reduce"
	if receipt.DependencyOrder != expectedOrder {
		t.Fatalf("dependency order = %q, want %q", receipt.DependencyOrder, expectedOrder)
	}

	// 3 & 4. Verify positive overlap nanoseconds and latency reduction
	if receipt.SavedDurationNs <= 0 {
		t.Fatalf("expected positive overlap nanoseconds, got %d", receipt.SavedDurationNs)
	}
	if receipt.TotalElapsedNs >= serialDuration.Nanoseconds() {
		t.Fatalf("overlapped elapsed %d ns >= serialized duration %d ns", receipt.TotalElapsedNs, serialDuration.Nanoseconds())
	}
	if !receipt.IdenticalOutput {
		t.Fatal("IdenticalOutput is false")
	}
}

func TestSharedExpertOverlapFailClosed(t *testing.T) {
	if _, _, err := OverlappedMoeExecution(nil, nil, 64); err == nil {
		t.Fatal("expected error on nil functions")
	}
	dummy := func() ([]float32, error) { return make([]float32, 64), nil }
	if _, _, err := OverlappedMoeExecution(dummy, dummy, 0); err == nil {
		t.Fatal("expected error on non-positive hiddenDim")
	}
}
