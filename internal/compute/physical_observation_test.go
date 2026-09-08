package compute

import (
	"errors"
	"strings"
	"testing"
)

type observationTestBackend struct {
	Backend
	name      string
	snapshots []BackendExecutionSnapshot
	err       error
	calls     int
}

func (b *observationTestBackend) Name() string            { return b.name }
func (b *observationTestBackend) Tier() string            { return "test" }
func (b *observationTestBackend) Class() CorrectnessClass { return Approx }
func (b *observationTestBackend) Caps() Caps              { return Caps{DeviceMemory: true} }
func (b *observationTestBackend) BackendExecutionSnapshot() (BackendExecutionSnapshot, error) {
	b.calls++
	if b.err != nil {
		return BackendExecutionSnapshot{}, b.err
	}
	return b.snapshots[b.calls-1], nil
}

type unsupportedObservationBackend struct {
	Backend
	name string
}

func (b unsupportedObservationBackend) Name() string            { return b.name }
func (b unsupportedObservationBackend) Tier() string            { return "test" }
func (b unsupportedObservationBackend) Class() CorrectnessClass { return Approx }
func (b unsupportedObservationBackend) Caps() Caps              { return Caps{} }

func TestPhysicalObservationDeltaExcludesPriorCumulativeCounters(t *testing.T) {
	identity := BackendRuntimeIdentity{Backend: "vulkan", Device: "observed-device", Driver: "observed-driver", Runtime: "vulkan-1.3.0"}
	before := BackendExecutionSnapshot{
		Identity: identity,
		Counters: BackendCounterSnapshot{
			ComputeDispatches: 100, Q4KMatmulDispatches: 80, OtherDispatches: 20, DispatchSubmits: 50,
			H2DBytes: 1000, D2HBytes: 100, D2DCopies: 7, Q4KStageCalls: 4, Q4KStageBytes: 400,
			Fallbacks: 2, TensorHomeHits: 10, TensorHomeAdmissions: 3, TensorHomeBypasses: 1, TensorHomeCopiedBytes: 300,
		},
		TensorHomeEntries: 3, TensorHomeResidentBytes: 300,
		DeviceMemoryTotalBytes: 10000, DeviceMemoryFreeBytes: 7000, DeviceMemoryObserved: true,
	}
	after := before
	after.Counters.ComputeDispatches += 9
	after.Counters.Q4KMatmulDispatches += 7
	after.Counters.OtherDispatches += 2
	after.Counters.DispatchSubmits += 3
	after.Counters.H2DBytes += 64
	after.Counters.D2HBytes += 8
	after.Counters.D2DCopies += 1
	after.Counters.Q4KStageCalls += 2
	after.Counters.Q4KStageBytes += 32
	after.Counters.TensorHomeHits += 4
	after.Counters.TensorHomeAdmissions += 1
	after.Counters.TensorHomeCopiedBytes += 16
	after.TensorHomeEntries = 4
	after.TensorHomeResidentBytes = 316
	after.DeviceMemoryFreeBytes = 6800

	got, err := BackendExecutionDelta(before, after)
	if err != nil {
		t.Fatal(err)
	}
	if got.Counters.ComputeDispatches != 9 || got.Counters.Q4KMatmulDispatches != 7 || got.Counters.H2DBytes != 64 || got.Counters.TensorHomeHits != 4 {
		t.Fatalf("delta includes cumulative history or lost execution values: %+v", got.Counters)
	}
	if got.Counters.Fallbacks != 0 || got.TensorHomeEntries != 4 || got.TensorHomeResidentBytes != 316 || got.DeviceMemoryFreeBytes != 6800 {
		t.Fatalf("closing gauges/fallback delta mismatch: %+v", got)
	}
}

func TestPhysicalObservationRejectsResetIdentityDriftAndMissingIdentity(t *testing.T) {
	identity := BackendRuntimeIdentity{Backend: "vulkan", Device: "device", Driver: "driver", Runtime: "vulkan-1.3.0"}
	before := BackendExecutionSnapshot{Identity: identity, Counters: BackendCounterSnapshot{ComputeDispatches: 2}}

	t.Run("counter reset", func(t *testing.T) {
		after := before
		after.Counters.ComputeDispatches = 1
		if got, err := BackendExecutionDelta(before, after); err == nil || got != (BackendExecutionObservation{}) || !strings.Contains(err.Error(), "reset") {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	t.Run("identity drift", func(t *testing.T) {
		after := before
		after.Identity.Device = "other"
		if got, err := BackendExecutionDelta(before, after); err == nil || got != (BackendExecutionObservation{}) || !strings.Contains(err.Error(), "changed") {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	t.Run("missing identity", func(t *testing.T) {
		backend := &observationTestBackend{name: "vulkan", snapshots: []BackendExecutionSnapshot{{Identity: BackendRuntimeIdentity{Backend: "vulkan"}}}}
		if got, available, err := CaptureBackendExecutionSnapshot(backend); err == nil || available || got != (BackendExecutionSnapshot{}) {
			t.Fatalf("got=%+v available=%v err=%v", got, available, err)
		}
	})
}

func TestPhysicalObservationUnsupportedAndObserverErrorStayUnavailable(t *testing.T) {
	unsupported := unsupportedObservationBackend{name: "cpu-ref"}
	if got, available, err := CaptureBackendExecutionSnapshot(unsupported); err != nil || available || got != (BackendExecutionSnapshot{}) {
		t.Fatalf("unsupported got=%+v available=%v err=%v", got, available, err)
	}

	backend := &observationTestBackend{name: "vulkan", err: errors.New("snapshot failed")}
	if got, available, err := CaptureBackendExecutionSnapshot(backend); err == nil || available || got != (BackendExecutionSnapshot{}) {
		t.Fatalf("failed observer got=%+v available=%v err=%v", got, available, err)
	}
}
