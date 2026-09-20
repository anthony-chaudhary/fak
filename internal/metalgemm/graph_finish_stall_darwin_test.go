//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"testing"
	"time"
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

func waitForGraphOwnerCount(t *testing.T, want int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for graphLiveOwnerCount() != want && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if got := graphLiveOwnerCount(); got != want {
		t.Fatalf("native graph owner count=%d want %d", got, want)
	}
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

// TestProjectionGraphTerminalGateTestSeam proves the deterministic #12958
// reproducer itself uses a real Metal terminal fence and always has a cleanup
// path that opens the fence before freeing native graph ownership.
func TestProjectionGraphTerminalGateTestSeam(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	t.Cleanup(ResetQ4K)
	owners, buffers := graphLiveOwnerCount(), graphLiveBufferCount()
	g := graphStallFixture(t)
	t.Cleanup(func() {
		g.releaseTerminalForTest()
		if g.finished {
			g.awaitTerminalForTest()
		}
		g.Free()
		waitForGraphOwnerCount(t, owners)
	})
	if err := g.holdTerminalForTest(5 * time.Millisecond); err != nil {
		t.Fatal(err)
	}
	receipt, err := g.Finish()
	if !IsMetalCommandBufferStall(err) || !receipt.Committed || receipt.CompletedWait {
		t.Fatalf("held terminal receipt=%+v err=%T %v, want committed typed stall", receipt, err, err)
	}
	if got := graphLiveOwnerCount(); got != owners+1 {
		t.Fatalf("held graph owner count=%d want %d", got, owners+1)
	}
	if got := graphLiveBufferCount(); got <= buffers {
		t.Fatalf("held graph buffer count=%d want >%d", got, buffers)
	}
	g.releaseTerminalForTest()
	if g.ptr != nil && !g.awaitTerminalForTest() {
		t.Fatal("released terminal fence did not complete command buffer")
	}
	g.Free()
	waitForGraphOwnerCount(t, owners)
	if got := graphLiveBufferCount(); got != buffers {
		t.Fatalf("released graph buffer count=%d want %d", got, buffers)
	}
}

// TestProjectionGraphFinishTimeoutQuarantinesOwnersUntilRealTerminalCompletion
// reproduces #12958 with a real command-buffer fence held past Finish's bounded
// wait. The graph and its GDN owner must remain live while Metal can still touch
// them; caller Free is idempotent and cannot release either owner early.
func TestProjectionGraphFinishTimeoutQuarantinesOwnersUntilRealTerminalCompletion(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	t.Cleanup(ResetQ4K)
	ownerBaseline, graphBufferBaseline := graphLiveOwnerCount(), graphLiveBufferCount()
	gdnBufferBaseline := GDNLiveBufferCount()
	g, state, _, _ := newProjectionGraphGDNLeaseFixture(t)
	t.Cleanup(state.Close)
	t.Cleanup(func() {
		g.releaseTerminalForTest()
		if g.ptr != nil && g.finished {
			g.awaitTerminalForTest()
		}
		g.Free()
		waitForGraphOwnerCount(t, ownerBaseline)
	})
	if err := g.holdTerminalForTest(5 * time.Millisecond); err != nil {
		t.Fatal(err)
	}

	receipt, err := g.Finish()
	if !IsMetalCommandBufferStall(err) || !receipt.Committed || receipt.CompletedWait {
		t.Fatalf("held terminal receipt=%+v err=%T %v, want committed typed stall", receipt, err, err)
	}
	state.mu.Lock()
	busy := state.graphDone != nil
	state.mu.Unlock()
	if !busy {
		t.Fatal("timed-out graph released its GDN lease before real terminal completion")
	}
	if got := graphLiveOwnerCount(); got != ownerBaseline+1 {
		t.Fatalf("quarantined graph owner count=%d want %d", got, ownerBaseline+1)
	}
	if got := graphLiveBufferCount(); got <= graphBufferBaseline {
		t.Fatalf("quarantined graph buffer count=%d want >%d", got, graphBufferBaseline)
	}
	if _, err := BeginProjectionGraph(q4kTestVector(256, 12958), nil, nil, 1, 256); err == nil {
		t.Fatal("new graph admitted while a timed-out command buffer was quarantined")
	}

	resetDone := make(chan error, 1)
	go func() { resetDone <- state.Reset() }()
	waitForGDNGraphWaiters(t, state, 1)
	g.Free()
	g.Free()
	select {
	case err := <-resetDone:
		t.Fatalf("GDN owner escaped quarantine before terminal completion: %v", err)
	default:
	}

	g.releaseTerminalForTest()
	select {
	case err := <-resetDone:
		if err != nil {
			t.Fatalf("GDN owner remained unusable after terminal cleanup: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("GDN owner did not release after terminal completion")
	}
	waitForGraphOwnerCount(t, ownerBaseline)
	if got := graphLiveBufferCount(); got != graphBufferBaseline {
		t.Fatalf("quarantine leaked native graph buffers: got %d want %d", got, graphBufferBaseline)
	}
	var probe *ProjectionGraph
	deadline := time.Now().Add(5 * time.Second)
	for {
		probe, err = BeginProjectionGraph(q4kTestVector(256, 12959), nil, nil, 1, 256)
		if err == nil {
			break
		}
		if err.Error() != "metalgemm: projection graph unavailable while a timed-out command buffer remains quarantined" {
			t.Fatalf("unexpected post-terminal graph admission error: %v", err)
		}
		if time.Now().After(deadline) {
			t.Fatalf("graph admission remained quarantined after terminal cleanup: %v", err)
		}
		time.Sleep(time.Millisecond)
	}
	probe.Free()
	state.Close()
	if got := GDNLiveBufferCount(); got != gdnBufferBaseline {
		t.Fatalf("quarantine leaked GDN buffers: got %d want %d", got, gdnBufferBaseline)
	}
}
