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
//
// The Q2_K/Q3_K kernel-identity counters (Ref #13860) are the issued dispatch
// counts of each format: Q2KMatmulDispatches/Q3KMatmulDispatches are the format
// totals, and each folds exactly into its scalar (historical per-token
// scalar/batched path) and tile (V4.1 token panel/pair) sub-counters. Every one
// is a sub-scope of OtherDispatches (the non-Q4_K compute remainder);
// validateBackendDispatchCounterScope enforces that aggregate scope. Backends
// that do not track kernel identity report zeros.
type BackendCounterSnapshot struct {
	ComputeDispatches     uint64
	Q4KMatmulDispatches   uint64
	Q2KMatmulDispatches   uint64
	Q3KMatmulDispatches   uint64
	Q2KScalarDispatches   uint64
	Q2KTileDispatches     uint64
	Q3KScalarDispatches   uint64
	Q3KTileDispatches     uint64
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

func validateBackendH2DAttribution(s BackendExecutionSnapshot) error {
	return s.H2DAttribution.validate(s.Counters.H2DBytes, s.H2DAttributionObserved, s.TransferCountersObserved)
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
	H2DAttribution            BackendH2DAttribution
	H2DAttributionObserved    bool
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
	H2DAttribution            BackendH2DAttribution
	H2DAttributionObserved    bool
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
	if before.H2DAttributionObserved != after.H2DAttributionObserved {
		return BackendExecutionObservation{}, errors.New("compute: backend H2D attribution availability changed during observation")
	}
	if err := validateBackendH2DAttribution(before); err != nil {
		return BackendExecutionObservation{}, fmt.Errorf("compute: invalid opening H2D attribution: %w", err)
	}
	if err := validateBackendH2DAttribution(after); err != nil {
		return BackendExecutionObservation{}, fmt.Errorf("compute: invalid closing H2D attribution: %w", err)
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
	if err := validateBackendDispatchCounterScope(delta); err != nil {
		return BackendExecutionObservation{}, fmt.Errorf("compute: invalid backend dispatch counter delta: %w", err)
	}
	attribution, err := subtractBackendH2DAttribution(before.H2DAttribution, after.H2DAttribution)
	if err != nil {
		return BackendExecutionObservation{}, err
	}
	if err := validateBackendH2DAttribution(BackendExecutionSnapshot{
		Counters: delta, TransferCountersObserved: after.TransferCountersObserved,
		H2DAttribution: attribution, H2DAttributionObserved: after.H2DAttributionObserved,
	}); err != nil {
		return BackendExecutionObservation{}, fmt.Errorf("compute: invalid H2D attribution delta: %w", err)
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
		H2DAttribution:           attribution, H2DAttributionObserved: after.H2DAttributionObserved,
	}, nil
}

// validateBackendDispatchCounterScope checks the explicit aggregate scope of
// the issued Q2_K/Q3_K kernel-identity counters (Ref #13860): each format's
// scalar and tile sub-counters fold exactly into its format total, and both
// format totals fit inside OtherDispatches, the non-Q4_K compute remainder.
// All-zero counters (a backend that does not track kernel identity, or a
// profile-off read) are in scope: zeros never mean no work without the
// backend's own live gate.
func validateBackendDispatchCounterScope(counters BackendCounterSnapshot) error {
	for _, scope := range []struct {
		name   string
		total  uint64
		scalar uint64
		tile   uint64
	}{
		{name: "Q2_K", total: counters.Q2KMatmulDispatches, scalar: counters.Q2KScalarDispatches, tile: counters.Q2KTileDispatches},
		{name: "Q3_K", total: counters.Q3KMatmulDispatches, scalar: counters.Q3KScalarDispatches, tile: counters.Q3KTileDispatches},
	} {
		// Overflow-safe fold: scalar <= total and tile == total - scalar.
		if scope.scalar > scope.total || scope.tile != scope.total-scope.scalar {
			return fmt.Errorf("compute: %s scalar (%d) and tile (%d) dispatch counters do not fold into the %s total (%d)",
				scope.name, scope.scalar, scope.tile, scope.name, scope.total)
		}
	}
	// Overflow-safe sub-scope: Q2_K + Q3_K <= OtherDispatches.
	if counters.Q2KMatmulDispatches > counters.OtherDispatches ||
		counters.Q3KMatmulDispatches > counters.OtherDispatches-counters.Q2KMatmulDispatches {
		return fmt.Errorf("compute: Q2_K (%d) and Q3_K (%d) dispatch counters exceed the non-Q4_K compute total (%d)",
			counters.Q2KMatmulDispatches, counters.Q3KMatmulDispatches, counters.OtherDispatches)
	}
	return nil
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
		before.ComputeDispatches, before.Q4KMatmulDispatches, before.Q2KMatmulDispatches, before.Q3KMatmulDispatches,
		before.Q2KScalarDispatches, before.Q2KTileDispatches, before.Q3KScalarDispatches, before.Q3KTileDispatches,
		before.OtherDispatches, before.DispatchSubmits,
		before.H2DBytes, before.H2DCount, before.D2HBytes, before.D2HCount, before.D2DCopies, before.D2DBytes,
		before.Q4KStageCalls, before.Q4KStageBytes,
		before.Fallbacks, before.TensorHomeHits, before.TensorHomeAdmissions, before.TensorHomeBypasses,
		before.TensorHomeCopiedBytes,
	}
	a := []uint64{
		after.ComputeDispatches, after.Q4KMatmulDispatches, after.Q2KMatmulDispatches, after.Q3KMatmulDispatches,
		after.Q2KScalarDispatches, after.Q2KTileDispatches, after.Q3KScalarDispatches, after.Q3KTileDispatches,
		after.OtherDispatches, after.DispatchSubmits,
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
		ComputeDispatches: a[0], Q4KMatmulDispatches: a[1], Q2KMatmulDispatches: a[2], Q3KMatmulDispatches: a[3],
		Q2KScalarDispatches: a[4], Q2KTileDispatches: a[5], Q3KScalarDispatches: a[6], Q3KTileDispatches: a[7],
		OtherDispatches: a[8], DispatchSubmits: a[9],
		H2DBytes: a[10], H2DCount: a[11], D2HBytes: a[12], D2HCount: a[13], D2DCopies: a[14], D2DBytes: a[15],
		Q4KStageCalls: a[16], Q4KStageBytes: a[17], Fallbacks: a[18], TensorHomeHits: a[19],
		TensorHomeAdmissions: a[20], TensorHomeBypasses: a[21], TensorHomeCopiedBytes: a[22],
	}, nil
}

// PhasePerformanceCounterScope mirrors VkPerformanceCounterScopeKHR without
// importing cgo into the untagged observation path.
type PhasePerformanceCounterScope int

const (
	PhaseCounterScopeUnspecified PhasePerformanceCounterScope = iota
	PhaseCounterScopeCommand
	PhaseCounterScopeRenderPass
	PhaseCounterScopeCompute
)

// PhasePerformanceCounterUnit mirrors VkPerformanceCounterUnitKHR.
type PhasePerformanceCounterUnit string

const (
	PhaseCounterUnitUnknown     PhasePerformanceCounterUnit = "unknown"
	PhaseCounterUnitGeneric     PhasePerformanceCounterUnit = "generic"
	PhaseCounterUnitPercentage  PhasePerformanceCounterUnit = "percentage"
	PhaseCounterUnitNanoseconds PhasePerformanceCounterUnit = "nanoseconds"
	PhaseCounterUnitBytes       PhasePerformanceCounterUnit = "bytes"
	PhaseCounterUnitBytesPerSec PhasePerformanceCounterUnit = "bytes_per_second"
	PhaseCounterUnitCycles      PhasePerformanceCounterUnit = "cycles"
)

// PhasePerformanceCounterDescriptor is one device-declared counter. It is
// reported only when a live device enumerates it; a counter that the device
// does not declare is never substituted with a constant or a zero reading.
type PhasePerformanceCounterDescriptor struct {
	Index int
	Name  string
	Unit  PhasePerformanceCounterUnit
	Scope PhasePerformanceCounterScope
}

// PhasePerformancePhaseReading is one sampled counter for one timestamped
// phase. Available=false means the counter was NOT observed; Value is
// meaningful only when Available is true. A typed-unavailable reading is the
// fail-closed substitute for a fabricated constant.
type PhasePerformancePhaseReading struct {
	Phase     string
	Counter   string
	Unit      PhasePerformanceCounterUnit
	Value     uint64
	Available bool
}

// PhasePerformanceQueryObservation is the fail-closed result of probing the
// RADV/Vulkan performance-query surface. When Supported is false the entire
// observation is typed-unavailable: Descriptors is empty and every phase
// reading is Available=false. No field is ever defaulted to a plausible
// constant.
type PhasePerformanceQueryObservation struct {
	Supported   bool
	Reason      string
	Descriptors []PhasePerformanceCounterDescriptor
	Readings    []PhasePerformancePhaseReading
}

// phasePerformanceQueryProbe is the backend-owned availability seam. The
// Vulkan backend implements it with cgo; other backends simply do not.
type phasePerformanceQueryProbe interface {
	PhasePerformanceQuerySupported() (bool, string)
	PhasePerformanceCounterDescriptors() ([]PhasePerformanceCounterDescriptor, bool)
}

// ObservePhasePerformanceQuery probes the selected backend for a live
// performance-query surface. Unsupported backends, a nil probe, or an
// unavailable extension all return Supported=false with a typed reason; the
// caller must publish that typed-unavailable state rather than constants.
func ObservePhasePerformanceQuery(backend Backend) PhasePerformanceQueryObservation {
	if backend == nil {
		return PhasePerformanceQueryObservation{Reason: "no backend selected"}
	}
	probe, ok := backend.(phasePerformanceQueryProbe)
	if !ok {
		return PhasePerformanceQueryObservation{Reason: "backend does not expose a phase performance-query surface"}
	}
	supported, reason := probe.PhasePerformanceQuerySupported()
	if !supported {
		if reason == "" {
			reason = "device reported no RADV/Vulkan performance-query support"
		}
		return PhasePerformanceQueryObservation{Reason: reason}
	}
	descriptors, observed := probe.PhasePerformanceCounterDescriptors()
	if !observed {
		return PhasePerformanceQueryObservation{Reason: "performance-query counters could not be enumerated on this device"}
	}
	return PhasePerformanceQueryObservation{Supported: true, Descriptors: descriptors}
}

// PhasePerformanceReadingsForPhase attaches per-counter readings for one
// timestamped phase. A counter not present in the observation's descriptors is
// emitted as a typed-unavailable reading, never as a zero value that could be
// mistaken for a measured constant.
func (o PhasePerformanceQueryObservation) ReadingsForPhase(phase string, samples map[string]uint64) []PhasePerformancePhaseReading {
	readings := make([]PhasePerformancePhaseReading, 0, len(o.Descriptors))
	for _, descriptor := range o.Descriptors {
		value, ok := samples[descriptor.Name]
		readings = append(readings, PhasePerformancePhaseReading{
			Phase:     phase,
			Counter:   descriptor.Name,
			Unit:      descriptor.Unit,
			Value:     value,
			Available: ok,
		})
	}
	return readings
}
