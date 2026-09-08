package compute

import (
	"errors"
	"fmt"
	"strings"
)

// BackendRuntimeIdentity is reported by the backend instance that performed
// execution. Request/configuration labels are deliberately absent.
type BackendRuntimeIdentity struct {
	Backend string
	Device  string
	Driver  string
	Runtime string
}

// BackendCounterSnapshot contains cumulative backend-owned counters. Consumers
// must use BackendExecutionDelta rather than publishing these process totals.
type BackendCounterSnapshot struct {
	ComputeDispatches     uint64
	Q4KMatmulDispatches   uint64
	OtherDispatches       uint64
	DispatchSubmits       uint64
	H2DBytes              uint64
	D2HBytes              uint64
	D2DCopies             uint64
	Q4KStageCalls         uint64
	Q4KStageBytes         uint64
	Fallbacks             uint64
	TensorHomeHits        uint64
	TensorHomeAdmissions  uint64
	TensorHomeBypasses    uint64
	TensorHomeCopiedBytes uint64
}

// BackendExecutionSnapshot is one backend-owned cumulative observation plus
// current gauges. It is not a receipt and must not be emitted directly.
type BackendExecutionSnapshot struct {
	Identity                BackendRuntimeIdentity
	Counters                BackendCounterSnapshot
	TensorHomeEntries       uint64
	TensorHomeResidentBytes uint64
	DeviceMemoryTotalBytes  uint64
	DeviceMemoryFreeBytes   uint64
	DeviceMemoryObserved    bool
}

// BackendExecutionObservation contains only the counter delta across one
// execution window. Gauge values are the actually observed closing snapshot.
type BackendExecutionObservation struct {
	Identity                BackendRuntimeIdentity
	Counters                BackendCounterSnapshot
	TensorHomeEntries       uint64
	TensorHomeResidentBytes uint64
	DeviceMemoryTotalBytes  uint64
	DeviceMemoryFreeBytes   uint64
	DeviceMemoryObserved    bool
}

type backendExecutionSnapshotter interface {
	BackendExecutionSnapshot() (BackendExecutionSnapshot, error)
}

// CaptureBackendExecutionSnapshot asks the actual selected backend for an
// observation. Unsupported backends return available=false and a zero value.
func CaptureBackendExecutionSnapshot(backend Backend) (snapshot BackendExecutionSnapshot, available bool, err error) {
	if backend == nil {
		return BackendExecutionSnapshot{}, false, nil
	}
	observer, ok := backend.(backendExecutionSnapshotter)
	if !ok {
		return BackendExecutionSnapshot{}, false, nil
	}
	snapshot, err = observer.BackendExecutionSnapshot()
	if err != nil {
		return BackendExecutionSnapshot{}, false, err
	}
	if err := validateBackendRuntimeIdentity(snapshot.Identity, backend.Name()); err != nil {
		return BackendExecutionSnapshot{}, false, err
	}
	return snapshot, true, nil
}

// BackendExecutionDelta validates identity stability and subtracts cumulative
// counters. A reset, wrap, or identity change returns a zero observation.
func BackendExecutionDelta(before, after BackendExecutionSnapshot) (BackendExecutionObservation, error) {
	if err := validateBackendRuntimeIdentity(before.Identity, before.Identity.Backend); err != nil {
		return BackendExecutionObservation{}, err
	}
	if before.Identity != after.Identity {
		return BackendExecutionObservation{}, errors.New("compute: backend execution identity changed during observation")
	}
	if before.DeviceMemoryObserved != after.DeviceMemoryObserved {
		return BackendExecutionObservation{}, errors.New("compute: backend device-memory availability changed during observation")
	}
	delta, err := subtractBackendCounters(before.Counters, after.Counters)
	if err != nil {
		return BackendExecutionObservation{}, err
	}
	if after.DeviceMemoryObserved && (after.DeviceMemoryTotalBytes == 0 || after.DeviceMemoryFreeBytes > after.DeviceMemoryTotalBytes) {
		return BackendExecutionObservation{}, errors.New("compute: backend reported invalid device-memory observation")
	}
	return BackendExecutionObservation{
		Identity:                after.Identity,
		Counters:                delta,
		TensorHomeEntries:       after.TensorHomeEntries,
		TensorHomeResidentBytes: after.TensorHomeResidentBytes,
		DeviceMemoryTotalBytes:  after.DeviceMemoryTotalBytes,
		DeviceMemoryFreeBytes:   after.DeviceMemoryFreeBytes,
		DeviceMemoryObserved:    after.DeviceMemoryObserved,
	}, nil
}

func validateBackendRuntimeIdentity(identity BackendRuntimeIdentity, backendName string) error {
	if strings.TrimSpace(identity.Backend) == "" || identity.Backend != backendName {
		return fmt.Errorf("compute: observed backend identity %q does not match selected backend %q", identity.Backend, backendName)
	}
	if strings.TrimSpace(identity.Device) == "" || strings.TrimSpace(identity.Driver) == "" || strings.TrimSpace(identity.Runtime) == "" {
		return errors.New("compute: backend execution identity requires device, driver, and runtime")
	}
	return nil
}

func subtractBackendCounters(before, after BackendCounterSnapshot) (BackendCounterSnapshot, error) {
	b := []uint64{
		before.ComputeDispatches, before.Q4KMatmulDispatches, before.OtherDispatches, before.DispatchSubmits,
		before.H2DBytes, before.D2HBytes, before.D2DCopies, before.Q4KStageCalls, before.Q4KStageBytes,
		before.Fallbacks, before.TensorHomeHits, before.TensorHomeAdmissions, before.TensorHomeBypasses,
		before.TensorHomeCopiedBytes,
	}
	a := []uint64{
		after.ComputeDispatches, after.Q4KMatmulDispatches, after.OtherDispatches, after.DispatchSubmits,
		after.H2DBytes, after.D2HBytes, after.D2DCopies, after.Q4KStageCalls, after.Q4KStageBytes,
		after.Fallbacks, after.TensorHomeHits, after.TensorHomeAdmissions, after.TensorHomeBypasses,
		after.TensorHomeCopiedBytes,
	}
	for i := range b {
		if a[i] < b[i] {
			return BackendCounterSnapshot{}, errors.New("compute: backend execution counter reset or wrapped during observation")
		}
		a[i] -= b[i]
	}
	return BackendCounterSnapshot{
		ComputeDispatches: a[0], Q4KMatmulDispatches: a[1], OtherDispatches: a[2], DispatchSubmits: a[3],
		H2DBytes: a[4], D2HBytes: a[5], D2DCopies: a[6], Q4KStageCalls: a[7], Q4KStageBytes: a[8],
		Fallbacks: a[9], TensorHomeHits: a[10], TensorHomeAdmissions: a[11], TensorHomeBypasses: a[12],
		TensorHomeCopiedBytes: a[13],
	}, nil
}
