package main

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	computestrix "github.com/anthony-chaudhary/fak/internal/compute/strix"
)

// strixAttachBackend is a fake compute backend that records whether it received
// the Strix Halo UMA/MALL subsystem handoff.
type strixAttachBackend struct {
	compute.Backend
	gotUMA  *computestrix.UMAPointerManager
	gotMALL *computestrix.MALLTiler
	err     error
	calls   int
}

func (b *strixAttachBackend) AttachStrixSubsystems(uma *computestrix.UMAPointerManager, mall *computestrix.MALLTiler) error {
	b.calls++
	b.gotUMA = uma
	b.gotMALL = mall
	return b.err
}

func TestServeStages_StrixSubsystemsRetained(t *testing.T) {
	uma := computestrix.NewUMAPointerManager()
	mall := computestrix.NewMALLTiler()
	be := &strixAttachBackend{}
	rt := &serveRuntime{
		chatBackend: be,
		strixPreflight: ServeStrixHaloPreflightResult{
			Detected:          true,
			UMAPointerManager: uma,
			MALLTiler:         mall,
			DeviceName:        computestrix.CanonicalDeviceNameStrixHalo,
		},
	}

	rt.attachStrixSubsystems()

	if !rt.strixSubsystems.Attached {
		t.Fatal("expected preflight subsystems retained (Attached=true)")
	}
	if rt.strixSubsystems.UMAPointerManager != uma {
		t.Fatal("runtime did not retain the preflight UMAPointerManager identity")
	}
	if rt.strixSubsystems.MALLTiler != mall {
		t.Fatal("runtime did not retain the preflight MALLTiler identity")
	}
	if be.calls != 1 {
		t.Fatalf("expected backend seam called once, got %d", be.calls)
	}
	if be.gotUMA != uma || be.gotMALL != mall {
		t.Fatal("backend seam did not receive the retained subsystem pointers")
	}

	// Detected=false with nil subsystems must be a no-op (no panic, no attach).
	noopBe := &strixAttachBackend{}
	noop := &serveRuntime{
		chatBackend:    noopBe,
		strixPreflight: ServeStrixHaloPreflightResult{Detected: false},
	}
	noop.attachStrixSubsystems()
	if noop.strixSubsystems.Attached {
		t.Fatal("undetected preflight must not report Attached=true")
	}
	if noopBe.calls != 0 {
		t.Fatalf("undetected preflight must not call the backend seam, got %d calls", noopBe.calls)
	}
}

func TestServeStages_StrixAttachErrorFallsBack(t *testing.T) {
	be := &strixAttachBackend{err: errors.New("simulated attach failure")}
	rt := &serveRuntime{
		chatBackend: be,
		strixPreflight: ServeStrixHaloPreflightResult{
			Detected:          true,
			UMAPointerManager: computestrix.NewUMAPointerManager(),
			MALLTiler:         computestrix.NewMALLTiler(),
		},
	}

	rt.attachStrixSubsystems()

	if rt.strixSubsystems.Attached {
		t.Fatal("a failed attach must clear Attached (quarantined fallback)")
	}
}

// TestServeStages_StrixKVAllocationConsumesSubsystems proves the retained Strix
// Halo UMA subsystems are CONSUMED by a real serving allocation, not merely
// recorded: the pinned buffer is zero-copy (host == device pointer), the manager
// counter reflects it, and every non-Strix/unattached path is a strict no-op.
func TestServeStages_StrixKVAllocationConsumesSubsystems(t *testing.T) {
	// Case A: attached subsystem with a real manager -> a real zero-copy buffer.
	uma := computestrix.NewUMAPointerManager()
	rt := &serveRuntime{}
	rt.strixSubsystems = serveStrixSubsystemBinding{
		Attached:          true,
		UMAPointerManager: uma,
		MALLTiler:         computestrix.NewMALLTiler(),
	}

	before := uma.ActiveAllocations()
	alloc := rt.allocateStrixKVBuffers(strixKVProbeBytes, computestrix.UMACacheLineAlignment)
	if alloc.Err != nil {
		t.Fatalf("expected no allocation error, got %v", alloc.Err)
	}
	if alloc.AllocatedBytes <= 0 {
		t.Fatalf("expected AllocatedBytes > 0, got %d", alloc.AllocatedBytes)
	}
	if len(alloc.Buffers) != 1 {
		t.Fatalf("expected exactly one pinned buffer, got %d", len(alloc.Buffers))
	}
	buf := alloc.Buffers[0]
	if !buf.IsZeroCopy() {
		t.Fatal("pinned KV buffer must be zero-copy (host == device pointer)")
	}
	if buf.HostPtr() != buf.DevPtr() {
		t.Fatalf("host/device pointer identity broken: host=%#x dev=%#x", buf.HostPtr(), buf.DevPtr())
	}
	if got := uma.ActiveAllocations(); got != before+1 {
		t.Fatalf("manager ActiveAllocations should reflect the pin: want %d got %d", before+1, got)
	}
	if err := uma.FreeUMABuffer(buf); err != nil {
		t.Fatalf("free pinned buffer: %v", err)
	}

	// Case B: Attached=false is a strict no-op — zero-value record, no panic.
	noop := &serveRuntime{}
	zero := noop.allocateStrixKVBuffers(strixKVProbeBytes, computestrix.UMACacheLineAlignment)
	if zero.AllocatedBytes != 0 || len(zero.Buffers) != 0 || zero.Err != nil {
		t.Fatalf("unattached runtime must be a strict no-op, got %+v", zero)
	}

	// Case C: attached flag but nil manager is also a no-op (not-detected runtime).
	nilMgr := &serveRuntime{}
	nilMgr.strixSubsystems = serveStrixSubsystemBinding{Attached: true}
	zeroC := nilMgr.allocateStrixKVBuffers(strixKVProbeBytes, computestrix.UMACacheLineAlignment)
	if zeroC.AllocatedBytes != 0 || len(zeroC.Buffers) != 0 || zeroC.Err != nil {
		t.Fatalf("nil manager must be a strict no-op, got %+v", zeroC)
	}
}

// TestServeStages_StrixKVReleaseFreesBuffers proves the startup KV probe buffer
// is released at shutdown: releaseStrixKVBuffers returns the manager's active
// allocation count to its prior value and zeroes the allocation record, and is
// safe on a runtime that never pinned anything.
func TestServeStages_StrixKVReleaseFreesBuffers(t *testing.T) {
	uma := computestrix.NewUMAPointerManager()
	rt := &serveRuntime{}
	rt.strixSubsystems = serveStrixSubsystemBinding{
		Attached:          true,
		UMAPointerManager: uma,
		MALLTiler:         computestrix.NewMALLTiler(),
	}

	before := uma.ActiveAllocations()
	rt.strixKVAllocation = rt.allocateStrixKVBuffers(strixKVProbeBytes, computestrix.UMACacheLineAlignment)
	if rt.strixKVAllocation.Err != nil {
		t.Fatalf("expected no allocation error, got %v", rt.strixKVAllocation.Err)
	}
	if got := uma.ActiveAllocations(); got != before+1 {
		t.Fatalf("pin should register one active allocation: want %d got %d", before+1, got)
	}

	rt.releaseStrixKVBuffers()

	if got := uma.ActiveAllocations(); got != before {
		t.Fatalf("release should return ActiveAllocations to prior value: want %d got %d", before, got)
	}
	if rt.strixKVAllocation.AllocatedBytes != 0 || len(rt.strixKVAllocation.Buffers) != 0 || rt.strixKVAllocation.Err != nil {
		t.Fatalf("release should zero the allocation record, got %+v", rt.strixKVAllocation)
	}

	// A never-pinned runtime (zero value) and a nil manager must both be safe.
	(&serveRuntime{}).releaseStrixKVBuffers()
	orphan := &serveRuntime{}
	orphan.strixKVAllocation = serveStrixKVAllocation{
		AllocatedBytes: strixKVProbeBytes,
		Buffers:        []*computestrix.UMABuffer{nil},
	}
	orphan.releaseStrixKVBuffers()
	if orphan.strixKVAllocation.AllocatedBytes != 0 || len(orphan.strixKVAllocation.Buffers) != 0 {
		t.Fatalf("release with nil manager must zero the record, got %+v", orphan.strixKVAllocation)
	}
}
