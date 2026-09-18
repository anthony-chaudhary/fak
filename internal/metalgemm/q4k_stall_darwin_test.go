//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"testing"
)

// q4kZeroPayload builds nblk all-zero q4_k super-blocks (144 B each). A zero super-block dequants to
// zero, so the GEMV/GEMM result is a defined 0 while still exercising the real Metal dispatch/wait.
func q4kZeroPayload(nblk int) []byte { return make([]byte, nblk*144) }

// TestQ4KGEMVHealthyWaitIsNotStall drives the REAL Q4_K GEMV native path and asserts a healthy
// completion is not classified as a stall, and that the Err wrapper returns no error.
func TestQ4KGEMVHealthyWaitIsNotStall(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()

	const in, out = 256, 1
	w := UploadQ4K(q4kZeroPayload(in/256), out, in)
	if w == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer w.Release()

	observation := NewExecutionObservation(ExecutionQ4KGEMV)
	x := make([]float32, in)
	y := make([]float32, out)
	if err := w.GEMVWithEventsErr(x, y, observation); err != nil {
		t.Fatalf("healthy GEMV err = %v, want nil", err)
	}
	if y[0] != 0 {
		t.Fatalf("Q4_K zero-payload result = %v, want 0", y[0])
	}
	snapshot, err := observation.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Events) != 1 || !snapshot.Events[0].Committed || !snapshot.Events[0].CompletedWait {
		t.Fatalf("healthy GEMV observation = %+v, want committed+completed", snapshot.Events)
	}
	// The semaphore path must complete strictly inside the shared bound: a healthy wait at/over
	// DefaultCommandBufferWaitLimit would mean the native MG_Q4K_WAIT_LIMIT_MS did not fire and is
	// not in parity with the Go classification budget.
	if wait := snapshot.Events[0].WaitMilliseconds; wait < 0 || wait >= DefaultCommandBufferWaitLimit {
		t.Fatalf("healthy GEMV wait = %.3fms, want [0, %.0f)", wait, DefaultCommandBufferWaitLimit)
	}
	if q4kStallFromReceipt(true, 0.5, 0, 0, "q4k gemv") != nil {
		t.Fatal("healthy completed receipt classified as stall")
	}
}

// TestQ4KGEMVGroupHealthyWaitIsNotStall drives the REAL grouped Q4_K GEMV native path (the live
// decode q/k/v group shape) so the second bounded-wait site - mg_q4k_gemv_group - is exercised. A
// healthy group completion must not classify as a stall.
func TestQ4KGEMVGroupHealthyWaitIsNotStall(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()

	const in, out = 256, 1
	a := UploadQ4K(q4kZeroPayload(in/256), out, in)
	b := UploadQ4K(q4kZeroPayload(in/256), out, in)
	if a == nil || b == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer a.Release()
	defer b.Release()

	observation := NewExecutionObservation(ExecutionQ4KGEMVGroup)
	x := make([]float32, in)
	got, err := GEMVGroupWithEventsErr([]*Q4KWeight{a, b}, x, observation)
	if err != nil {
		t.Fatalf("healthy group GEMV err = %v, want nil", err)
	}
	if len(got) != 2 {
		t.Fatalf("group results = %v, want 2 rows", got)
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

// TestQ4KGEMMHealthyWaitIsNotStall drives the REAL Q4_K prefill GEMM native path (P=1, the serve
// GEMM entrypoint) and asserts a healthy completion is not classified as a stall.
func TestQ4KGEMMHealthyWaitIsNotStall(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()

	const in, out, P = 256, 1, 1
	w := UploadQ4K(q4kZeroPayload(in/256), out, in)
	if w == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer w.Release()

	observation := NewExecutionObservation(ExecutionQ4KGEMM)
	X := make([]float32, P*in)
	Y := make([]float32, P*out)
	_, err := w.GEMMWithEventsModeErr(X, P, Y, observation, Q4KGEMMModeScalar)
	if err != nil {
		t.Fatalf("healthy GEMM err = %v, want nil", err)
	}
	snapshot, err := observation.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Events) != 1 || !snapshot.Events[0].Committed || !snapshot.Events[0].CompletedWait {
		t.Fatalf("healthy GEMM observation = %+v, want committed+completed", snapshot.Events)
	}
}

// TestQ4KGEMVTimeoutReceiptIsStall drives the classification seam with the EXACT receipt the native
// timeout branch emits: completed_wait forced to 0 and the elapsed wait recorded at the shared bound
// (MG_Q4K_WAIT_LIMIT_MS == DefaultCommandBufferWaitLimit). It must classify as the package's typed
// stall for both single and grouped operations.
func TestQ4KGEMVTimeoutReceiptIsStall(t *testing.T) {
	for _, op := range []string{"q4k gemv", "q4k gemv group", "q4k gemm", "q4k mlp"} {
		err := q4kStallFromReceipt(false, DefaultCommandBufferWaitLimit, -1, 60, op)
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

// TestQ4KGEMVNonCompletedBufferIsStall drives the classification seam with the native receipt a
// non-completed (device-fault) buffer produces: completedWait=false, a terminal status other than
// Completed. It must return the package's typed stall.
func TestQ4KGEMVNonCompletedBufferIsStall(t *testing.T) {
	err := q4kStallFromReceipt(false, 12_500, 3, 7, "q4k gemv group")
	if err == nil {
		t.Fatal("non-completed observation: want typed stall error, got nil")
	}
	if !IsMetalCommandBufferStall(err) {
		t.Fatalf("q4kStallFromReceipt = %T %v, want MetalCommandBufferStallError", err, err)
	}
	var stall MetalCommandBufferStallError
	if !errors.As(err, &stall) {
		t.Fatalf("errors.As(%v, MetalCommandBufferStallError) = false, want true", err)
	}
	if stall.Operation != "q4k gemv group" {
		t.Errorf("Operation = %q, want %q", stall.Operation, "q4k gemv group")
	}
}

// TestQ4KGEMVNonCompletedUnderLimitIsStillStall covers the device fault that is NOT a timeout: the
// wait was under the limit yet the buffer never reached Completed, so it must still classify as a
// typed stall.
func TestQ4KGEMVNonCompletedUnderLimitIsStillStall(t *testing.T) {
	if err := q4kStallFromReceipt(false, 5, 3, 7, "q4k gemv"); !IsMetalCommandBufferStall(err) {
		t.Fatalf("under-limit non-completed buffer = %T %v, want typed stall", err, err)
	}
}
