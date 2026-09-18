//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"testing"
)

// graphStallFixture builds the smallest committable Q4_K graph so Finish can
// take the real native commit/wait path without a full parity packet.
func graphStallFixture(t *testing.T) *ProjectionGraph {
	t.Helper()
	const P, in, out = 1, 256, 32
	xf := q4kTestVector(P*in, 2064)
	q4 := UploadQ4K(q4kTestRaw(out, in, 2064), out, in)
	if q4 == nil {
		t.Fatal("q4 upload")
	}
	t.Cleanup(ResetQ4K)
	g, err := BeginProjectionGraph(xf, nil, nil, P, in)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.EncodeQ4K(q4); err != nil {
		t.Fatal(err)
	}
	return g
}

// TestProjectionGraphFinishClassifiesNonCompletedBufferAsStall is the witness
// for the unbounded `waitUntilCompleted` defect: a command buffer whose terminal
// status is not Completed must surface the package's typed stall, not a silent
// success. A real GPU hang cannot be forced here, so the classification seam
// Finish() uses is driven directly with the native receipt a non-completed
// buffer produces (completed_wait=0, a status other than Completed).
func TestProjectionGraphFinishClassifiesNonCompletedBufferAsStall(t *testing.T) {
	err := commandBufferStallError(12_500, 1, 7, "", "graph finish")
	if err == nil {
		t.Fatal("non-completed command buffer: want typed stall error, got nil")
	}
	if !IsMetalCommandBufferStall(err) {
		t.Fatalf("commandBufferStallError = %T %v, want MetalCommandBufferStallError", err, err)
	}
	var stall MetalCommandBufferStallError
	if !errors.As(err, &stall) {
		t.Fatalf("errors.As(%v, MetalCommandBufferStallError) = false, want true", err)
	}
	if stall.Operation != "graph finish" {
		t.Errorf("Operation = %q, want %q", stall.Operation, "graph finish")
	}
}

// TestProjectionGraphFinishNonCompletedUnderLimitIsStillStall covers the device
// fault that is NOT a timeout: the wait was under the limit yet the buffer never
// reached Completed, so it must still classify as a typed stall.
func TestProjectionGraphFinishNonCompletedUnderLimitIsStillStall(t *testing.T) {
	err := commandBufferStallError(5, 1, 5, "", "graph finish")
	if !IsMetalCommandBufferStall(err) {
		t.Fatalf("under-limit non-completed buffer = %T %v, want typed stall", err, err)
	}
}

// TestProjectionGraphFinishDeviceFaultReturnsStall drives the REAL Finish path
// with a committed buffer whose terminal status never reaches Completed (the
// inject=2 device-fault seam, which does not block) and asserts Finish returns
// the package's typed stall. This is the routing witness: the classification is
// reached from Finish, not only from the helper.
func TestProjectionGraphFinishDeviceFaultReturnsStall(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	g := graphStallFixture(t)
	defer g.Free()
	g.InjectDeviceFaultForTest()
	receipt, err := g.Finish()
	if err == nil {
		t.Fatal("device fault: want typed stall error, got nil")
	}
	if !IsMetalCommandBufferStall(err) {
		t.Fatalf("Finish under device fault = %T %v, want MetalCommandBufferStallError", err, err)
	}
	if !receipt.Committed || receipt.CompletedWait {
		t.Fatalf("receipt=%+v, want committed+not-completed", receipt)
	}
}

// TestProjectionGraphFinishRealCompletionIsNotStall pins the healthy path: a real
// device commit/wait that reaches Completed returns success with no typed stall.
func TestProjectionGraphFinishRealCompletionIsNotStall(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	g := graphStallFixture(t)
	defer g.Free()
	receipt, err := g.Finish()
	if err != nil {
		t.Fatalf("Finish() = %v, want nil", err)
	}
	if !receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("receipt=%+v, want committed+completed", receipt)
	}
	if IsMetalCommandBufferStall(err) {
		t.Fatalf("healthy Finish reported a stall: %v", err)
	}
}

// TestProjectionGraphFinishInjectedFailureStaysPostSubmitError preserves the
// existing inject seam: an injected post-submit failure is NOT reclassified as a
// device stall, so the lifecycle witnesses keep their GraphPostSubmitError.
func TestProjectionGraphFinishInjectedFailureStaysPostSubmitError(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	g := graphStallFixture(t)
	defer g.Free()
	g.InjectPostSubmitFailureForTest()
	receipt, err := g.Finish()
	if err == nil {
		t.Fatal("injected failure: want error, got nil")
	}
	if IsMetalCommandBufferStall(err) {
		t.Fatalf("injected failure misclassified as device stall: %v", err)
	}
	var post *GraphPostSubmitError
	if !errors.As(err, &post) {
		t.Fatalf("injected failure = %T %v, want *GraphPostSubmitError", err, err)
	}
	if !receipt.Committed {
		t.Fatalf("receipt=%+v, want committed", receipt)
	}
}
