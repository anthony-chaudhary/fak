//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"testing"
)

// TestQ8GEMVHealthyWaitIsNotStall drives the REAL Q8 GEMV native path and asserts a healthy
// completion is not classified as a stall, and that the sibling Err wrapper returns no error.
func TestQ8GEMVHealthyWaitIsNotStall(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ8()

	const in, out = 32, 1
	codes := make([]int8, in)
	xq := make([]int8, in)
	for i := range codes {
		codes[i] = 1
		xq[i] = 1
	}
	w := UploadQ8(codes, []float32{1}, out, in)
	if w == nil {
		t.Fatal("UploadQ8 returned nil")
	}
	defer w.Release()

	observation := NewExecutionObservation(ExecutionQ8GEMV)
	y := []float32{0}
	if err := w.GEMVWithEventsErr(xq, []float32{1}, y, observation); err != nil {
		t.Fatalf("healthy GEMV err = %v, want nil", err)
	}
	if y[0] != in {
		t.Fatalf("Q8 result = %v, want %d", y[0], in)
	}
	snapshot, err := observation.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Events) != 1 || !snapshot.Events[0].Committed || !snapshot.Events[0].CompletedWait {
		t.Fatalf("healthy GEMV observation = %+v, want committed+completed", snapshot.Events)
	}
	// The semaphore path must complete strictly inside the shared bound: a healthy wait at/over
	// DefaultCommandBufferWaitLimit would mean the native MG_Q8_WAIT_LIMIT_MS did not fire and is
	// not in parity with the Go classification budget.
	if wait := snapshot.Events[0].WaitMilliseconds; wait < 0 || wait >= DefaultCommandBufferWaitLimit {
		t.Fatalf("healthy GEMV wait = %.3fms, want [0, %.0f)", wait, DefaultCommandBufferWaitLimit)
	}
	if q8StallFromReceipt(true, 0.5, 4, 0, "q8 gemv") != nil {
		t.Fatal("healthy completed receipt classified as stall")
	}
}

// TestQ8GEMVGroupHealthyWaitIsNotStall drives the REAL grouped Q8 GEMV native path (the live GDN
// in_proj quad shape) so the second bounded-wait site - mg_q8_gemv_group - is exercised, not only
// the single-weight one. A healthy group completion must not classify as a stall.
func TestQ8GEMVGroupHealthyWaitIsNotStall(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ8()

	const in, out = 32, 1
	codes := make([]int8, in)
	xq := make([]int8, in)
	for i := range codes {
		codes[i] = 1
		xq[i] = 1
	}
	a := UploadQ8(codes, []float32{1}, out, in)
	b := UploadQ8(codes, []float32{2}, out, in)
	if a == nil || b == nil {
		t.Fatal("UploadQ8 returned nil")
	}
	defer a.Release()
	defer b.Release()

	observation := NewExecutionObservation(ExecutionQ8GEMV)
	got, err := GEMVGroupQ8WithEventsErr([]*Q8Weight{a, b}, xq, []float32{1}, observation)
	if err != nil {
		t.Fatalf("healthy group GEMV err = %v, want nil", err)
	}
	if len(got) != 2 || got[0][0] != in || got[1][0] != 2*in {
		t.Fatalf("group results = %v, want rows [[%d] [%d]]", got, in, 2*in)
	}
	snapshot, err := observation.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Events) != 1 || !snapshot.Events[0].Committed || !snapshot.Events[0].CompletedWait {
		t.Fatalf("healthy group observation = %+v, want committed+completed", snapshot.Events)
	}
	if wait := snapshot.Events[0].WaitMilliseconds; wait < 0 || wait >= DefaultCommandBufferWaitLimit {
		t.Fatalf("healthy group wait = %.3fms, want [0, %.0f)", wait, DefaultCommandBufferWaitLimit)
	}
}

// TestQ8GEMVTimeoutReceiptIsStall drives the classification seam with the EXACT receipt the native
// timeout branch emits: completed_wait forced to 0 and the elapsed wait recorded at the shared
// bound (MG_Q8_WAIT_LIMIT_MS == DefaultCommandBufferWaitLimit), with status -1 / errno ETIMEDOUT.
// It must classify as the package's typed stall for both single and grouped operations.
func TestQ8GEMVTimeoutReceiptIsStall(t *testing.T) {
	for _, op := range []string{"q8 gemv", "q8 gemv group"} {
		err := q8StallFromReceipt(false, DefaultCommandBufferWaitLimit, -1, 60, op)
		if err == nil {
			t.Fatalf("%s timeout receipt: want typed stall, got nil", op)
		}
		if !IsMetalCommandBufferStall(err) {
			t.Fatalf("%s timeout receipt = %T %v, want MetalCommandBufferStallError", op, err, err)
		}
		var stall MetalCommandBufferStallError
		if !errors.As(err, &stall) {
			t.Fatalf("errors.As(%v, MetalCommandBufferStallError) = false, want true", err)
		}
		if stall.Operation != op {
			t.Errorf("Operation = %q, want %q", stall.Operation, op)
		}
		if stall.LimitMilliseconds != DefaultCommandBufferWaitLimit {
			t.Errorf("LimitMilliseconds = %.0f, want %.0f", stall.LimitMilliseconds, DefaultCommandBufferWaitLimit)
		}
	}
}

// TestQ8GEMVNonCompletedBufferIsStall drives the classification seam with the native receipt a
// non-completed (device-fault) buffer produces: completedWait=false, a terminal status other than
// Completed. It must return the package's typed stall.
func TestQ8GEMVNonCompletedBufferIsStall(t *testing.T) {
	err := q8StallFromReceipt(false, 12_500, 3, 7, "q8 gemv group")
	if err == nil {
		t.Fatal("non-completed observation: want typed stall error, got nil")
	}
	if !IsMetalCommandBufferStall(err) {
		t.Fatalf("q8StallFromReceipt = %T %v, want MetalCommandBufferStallError", err, err)
	}
	var stall MetalCommandBufferStallError
	if !errors.As(err, &stall) {
		t.Fatalf("errors.As(%v, MetalCommandBufferStallError) = false, want true", err)
	}
	if stall.Operation != "q8 gemv group" {
		t.Errorf("Operation = %q, want %q", stall.Operation, "q8 gemv group")
	}
}

// TestQ8GEMVNonCompletedUnderLimitIsStillStall covers the device fault that is NOT a timeout: the
// wait was under the limit yet the buffer never reached Completed, so it must still classify as a
// typed stall.
func TestQ8GEMVNonCompletedUnderLimitIsStillStall(t *testing.T) {
	if err := q8StallFromReceipt(false, 5, 3, 7, "q8 gemv"); !IsMetalCommandBufferStall(err) {
		t.Fatalf("under-limit non-completed buffer = %T %v, want typed stall", err, err)
	}
}
