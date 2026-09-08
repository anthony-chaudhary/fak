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
	H2DCount              uint64
	D2HBytes              uint64
	D2HCount              uint64
	D2DCopies             uint64
	D2DBytes              uint64
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
	Identity                  BackendRuntimeIdentity
	Counters                  BackendCounterSnapshot
	TensorHomeEntries         uint64
	TensorHomeResidentBytes   uint64
	DeviceMemoryTotalBytes    uint64
	DeviceMemoryFreeBytes     uint64
	DeviceMemoryObserved      bool
	TransferCountersObserved  bool
	DeviceAllocationLiveBytes uint64
	DeviceAllocationObserved  bool
}

// BackendExecutionObservation contains only the counter delta across one
// execution window. Gauge values are the actually observed closing snapshot.
type BackendExecutionObservation struct {
	Identity                  BackendRuntimeIdentity
	Counters                  BackendCounterSnapshot
	TensorHomeEntries         uint64
	TensorHomeResidentBytes   uint64
	DeviceMemoryTotalBytes    uint64
	DeviceMemoryFreeBytes     uint64
	DeviceMemoryObserved      bool
	TransferCountersObserved  bool
	DeviceAllocationLiveBytes uint64
	DeviceAllocationPeakBytes uint64
	DeviceAllocationObserved  bool
}

type backendExecutionSnapshotter interface {
	BackendExecutionSnapshot() (BackendExecutionSnapshot, error)
}

// BackendExecutionWindow is an optional backend-owned observation bracket.
// End is single-use: stale or repeated completion must return a zero observation
// and an error rather than replaying evidence from an earlier execution.
type BackendExecutionWindow interface {
	End() (BackendExecutionObservation, error)
}

type backendExecutionWindower interface {
	BeginBackendExecutionWindow() (BackendExecutionWindow, error)
}

// BeginBackendExecutionObservation asks the selected backend to create one
// execution-scoped observation window. Unsupported backends return
// available=false and no partial observation.
func BeginBackendExecutionObservation(backend Backend) (window BackendExecutionWindow, available bool, err error) {
	if backend == nil {
		return nil, false, nil
	}
	provider, ok := backend.(backendExecutionWindower)
	if !ok {
		return nil, false, nil
	}
	window, err = provider.BeginBackendExecutionWindow()
	if err != nil {
		return nil, false, err
	}
	if window == nil {
		return nil, false, errors.New("compute: backend returned a nil execution observation window")
	}
	return window, true, nil
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
	if before.TransferCountersObserved != after.TransferCountersObserved {
		return BackendExecutionObservation{}, errors.New("compute: backend transfer-counter availability changed during observation")
	}
	if before.DeviceAllocationObserved != after.DeviceAllocationObserved {
		return BackendExecutionObservation{}, errors.New("compute: backend device-allocation availability changed during observation")
	}
	if before.TransferCountersObserved {
		if err := validateBackendTransferCounterPairs(before.Counters); err != nil {
			return BackendExecutionObservation{}, fmt.Errorf("compute: invalid opening backend transfer counters: %w", err)
		}
		if err := validateBackendTransferCounterPairs(after.Counters); err != nil {
			return BackendExecutionObservation{}, fmt.Errorf("compute: invalid closing backend transfer counters: %w", err)
		}
	}
	delta, err := subtractBackendCounters(before.Counters, after.Counters)
	if err != nil {
		return BackendExecutionObservation{}, err
	}
	if before.TransferCountersObserved {
		if err := validateBackendTransferCounterPairs(delta); err != nil {
			return BackendExecutionObservation{}, fmt.Errorf("compute: invalid backend transfer delta: %w", err)
		}
	}
	if after.DeviceMemoryObserved && (after.DeviceMemoryTotalBytes == 0 || after.DeviceMemoryFreeBytes > after.DeviceMemoryTotalBytes) {
		return BackendExecutionObservation{}, errors.New("compute: backend reported invalid device-memory observation")
	}
	return BackendExecutionObservation{
		Identity:                 after.Identity,
		Counters:                 delta,
		TensorHomeEntries:        after.TensorHomeEntries,
		TensorHomeResidentBytes:  after.TensorHomeResidentBytes,
		DeviceMemoryTotalBytes:   after.DeviceMemoryTotalBytes,
		DeviceMemoryFreeBytes:    after.DeviceMemoryFreeBytes,
		DeviceMemoryObserved:     after.DeviceMemoryObserved,
		TransferCountersObserved: after.TransferCountersObserved,
	}, nil
}

func validateBackendTransferCounterPairs(counters BackendCounterSnapshot) error {
	for _, direction := range []struct {
		name         string
		count, bytes uint64
	}{
		{name: "H2D", count: counters.H2DCount, bytes: counters.H2DBytes},
		{name: "D2H", count: counters.D2HCount, bytes: counters.D2HBytes},
		{name: "D2D", count: counters.D2DCopies, bytes: counters.D2DBytes},
	} {
		if (direction.count == 0) != (direction.bytes == 0) {
			return fmt.Errorf("%s count and bytes availability disagree", direction.name)
		}
	}
	return nil
}

// BackendExecutionWindowDelta completes a snapshot delta with the allocation
// live/peak pair returned by the same backend-owned window. The closing value
// must match the after snapshot and the peak may not be below either endpoint.
// Plain BackendExecutionDelta deliberately leaves allocation evidence
// unavailable because two closing gauges cannot reconstruct a peak.
func BackendExecutionWindowDelta(before, after BackendExecutionSnapshot, allocationLive, allocationPeak uint64) (BackendExecutionObservation, error) {
	if !before.TransferCountersObserved || !after.TransferCountersObserved ||
		!before.DeviceAllocationObserved || !after.DeviceAllocationObserved {
		return BackendExecutionObservation{}, errors.New("compute: backend execution observation window is incomplete")
	}
	if allocationLive != after.DeviceAllocationLiveBytes ||
		allocationPeak < before.DeviceAllocationLiveBytes || allocationPeak < after.DeviceAllocationLiveBytes {
		return BackendExecutionObservation{}, errors.New("compute: backend device-allocation window is inconsistent")
	}
	observation, err := BackendExecutionDelta(before, after)
	if err != nil {
		return BackendExecutionObservation{}, err
	}
	observation.DeviceAllocationLiveBytes = allocationLive
	observation.DeviceAllocationPeakBytes = allocationPeak
	observation.DeviceAllocationObserved = true
	return observation, nil
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
		before.H2DBytes, before.H2DCount, before.D2HBytes, before.D2HCount, before.D2DCopies, before.D2DBytes,
		before.Q4KStageCalls, before.Q4KStageBytes,
		before.Fallbacks, before.TensorHomeHits, before.TensorHomeAdmissions, before.TensorHomeBypasses,
		before.TensorHomeCopiedBytes,
	}
	a := []uint64{
		after.ComputeDispatches, after.Q4KMatmulDispatches, after.OtherDispatches, after.DispatchSubmits,
		after.H2DBytes, after.H2DCount, after.D2HBytes, after.D2HCount, after.D2DCopies, after.D2DBytes,
		after.Q4KStageCalls, after.Q4KStageBytes,
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
		H2DBytes: a[4], H2DCount: a[5], D2HBytes: a[6], D2HCount: a[7], D2DCopies: a[8], D2DBytes: a[9],
		Q4KStageCalls: a[10], Q4KStageBytes: a[11], Fallbacks: a[12], TensorHomeHits: a[13],
		TensorHomeAdmissions: a[14], TensorHomeBypasses: a[15], TensorHomeCopiedBytes: a[16],
	}, nil
}
