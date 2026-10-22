package compute

import "errors"

// backendCounterReader is the optional cheap seam behind ReadBackendCounters:
// the cumulative Counters of BackendExecutionSnapshot without identity,
// memory-budget or upload-attribution work.
type backendCounterReader interface {
	BackendCounters() (BackendCounterSnapshot, error)
}

// backendDispatchCountGate is implemented by a backend that keeps dispatch
// and submit counts only under a startup profile switch.
type backendDispatchCountGate interface {
	DispatchCountsLive() bool
}

// backendSubmitCounter is implemented by a backend that counts every queue
// submit whatever its profile switch, readable without the device lock.
type backendSubmitCounter interface {
	DeviceSubmits() uint64
}

// backendDeviceWorkCounter is implemented by a backend that reads its
// DeviceWork scalars without building the host-sync cells.
type backendDeviceWorkCounter interface {
	DeviceWorkCounters() DeviceWork
}

// BackendCounterRead is one read of a backend's cumulative counters plus its
// always-on submit and device-work counters.
type BackendCounterRead struct {
	Counters BackendCounterSnapshot
	// DispatchCountsLive is false when ComputeDispatches, Q4KMatmulDispatches,
	// OtherDispatches and DispatchSubmits stay zero because the backend's
	// startup profile switch is off; a zero then does not mean no work.
	DispatchCountsLive bool
	// Submits is the backend's always-on cumulative queue-submit count,
	// meaningful only when SubmitsObserved; it does not wait on a profile.
	Submits         uint64
	SubmitsObserved bool
	// Work carries only DeviceWork's Dispatches, ReadbackBytes, SubmitWaitNS
	// and SubmitWaitObserved; host-sync cells and the idle clock stay zero.
	Work         DeviceWork
	WorkObserved bool
}

// ReadBackendCounters reads the selected backend's cumulative counters, with
// the field semantics of BackendExecutionSnapshot.Counters, its always-on
// submit count when it keeps one, and the scalars of its registered
// DeviceWork reader. A backend without the cheap seam falls back to a full
// snapshot. Unsupported backends return available=false and a zero value.
// The cheap seam and scalar DeviceWork paths do not allocate.
func ReadBackendCounters(backend Backend) (read BackendCounterRead, available bool, err error) {
	if backend == nil {
		return BackendCounterRead{}, false, nil
	}
	if reader, ok := backend.(backendCounterReader); ok {
		read.Counters, err = reader.BackendCounters()
		if err != nil {
			return BackendCounterRead{}, false, err
		}
	} else {
		snapshot, available, err := CaptureBackendExecutionSnapshot(backend)
		if err != nil || !available {
			return BackendCounterRead{}, false, err
		}
		read.Counters = snapshot.Counters
	}
	read.DispatchCountsLive = true
	if gate, ok := backend.(backendDispatchCountGate); ok {
		read.DispatchCountsLive = gate.DispatchCountsLive()
	}
	if counter, ok := backend.(backendSubmitCounter); ok {
		read.Submits, read.SubmitsObserved = counter.DeviceSubmits(), true
	}
	read.Work, read.WorkObserved = readDeviceWorkScalars(backend)
	return read, true, nil
}

// readDeviceWorkScalars reads a registered backend's DeviceWork scalars,
// through the backend's own scalar seam when it has one.
func readDeviceWorkScalars(backend Backend) (DeviceWork, bool) {
	name := backend.Name()
	read := registeredDeviceWorkReader(name)
	if read == nil {
		return DeviceWork{}, false
	}
	var work DeviceWork
	if counter, ok := backend.(backendDeviceWorkCounter); ok {
		work = counter.DeviceWorkCounters()
	} else {
		work = read()
	}
	return DeviceWork{Backend: name, Dispatches: work.Dispatches, ReadbackBytes: work.ReadbackBytes,
		SubmitWaitNS: work.SubmitWaitNS, SubmitWaitObserved: work.SubmitWaitObserved}, true
}

// ReadDeviceWork reads one registered backend's always-on device-work
// counters without the slice and sort of DeviceWorkSnapshot.
func ReadDeviceWork(backend string) (DeviceWork, bool) {
	read := registeredDeviceWorkReader(backend)
	if read == nil {
		return DeviceWork{}, false
	}
	work := read()
	work.Backend = backend
	return work, true
}

func registeredDeviceWorkReader(backend string) func() DeviceWork {
	deviceWorkReaders.mu.RLock()
	defer deviceWorkReaders.mu.RUnlock()
	return deviceWorkReaders.byBackend[backend]
}

// ErrBackendCounterWrap reports a cumulative counter that went backwards
// between two reads of one backend.
var ErrBackendCounterWrap = errors.New("compute: backend counter reset or wrapped")

// SubBackendCounters returns after-before field by field, refusing a reset or
// wrap with ErrBackendCounterWrap.
func SubBackendCounters(after, before BackendCounterSnapshot) (BackendCounterSnapshot, error) {
	delta, err := subtractBackendCounters(before, after)
	if err != nil {
		return BackendCounterSnapshot{}, ErrBackendCounterWrap
	}
	return delta, nil
}

// AddBackendCounters returns a+b field by field.
func AddBackendCounters(a, b BackendCounterSnapshot) BackendCounterSnapshot {
	return BackendCounterSnapshot{
		ComputeDispatches:     a.ComputeDispatches + b.ComputeDispatches,
		Q4KMatmulDispatches:   a.Q4KMatmulDispatches + b.Q4KMatmulDispatches,
		OtherDispatches:       a.OtherDispatches + b.OtherDispatches,
		DispatchSubmits:       a.DispatchSubmits + b.DispatchSubmits,
		H2DBytes:              a.H2DBytes + b.H2DBytes,
		H2DCount:              a.H2DCount + b.H2DCount,
		D2HBytes:              a.D2HBytes + b.D2HBytes,
		D2HCount:              a.D2HCount + b.D2HCount,
		D2DCopies:             a.D2DCopies + b.D2DCopies,
		D2DBytes:              a.D2DBytes + b.D2DBytes,
		Q4KStageCalls:         a.Q4KStageCalls + b.Q4KStageCalls,
		Q4KStageBytes:         a.Q4KStageBytes + b.Q4KStageBytes,
		Fallbacks:             a.Fallbacks + b.Fallbacks,
		TensorHomeHits:        a.TensorHomeHits + b.TensorHomeHits,
		TensorHomeAdmissions:  a.TensorHomeAdmissions + b.TensorHomeAdmissions,
		TensorHomeBypasses:    a.TensorHomeBypasses + b.TensorHomeBypasses,
		TensorHomeCopiedBytes: a.TensorHomeCopiedBytes + b.TensorHomeCopiedBytes,
	}
}
