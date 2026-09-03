package compute

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// HardwareEpochPresence records which hardware and runtime identity attributes
// were successfully witnessed rather than assumed or defaulted.
type HardwareEpochPresence struct {
	DriverVersion  bool `json:"driver_version"`
	RuntimeVersion bool `json:"runtime_version"`
	DeviceUUID     bool `json:"device_uuid"`
	DevicePCI      bool `json:"device_pci"`
	VisibleDevices bool `json:"visible_devices"`
	MIGPartition   bool `json:"mig_partition"`
	BuildIdentity  bool `json:"build_identity"`
}

// HardwareEpochFingerprint captures scrubbed, stable hardware and runtime
// identity facts for a backend.
type HardwareEpochFingerprint struct {
	BackendName    string                `json:"backend_name"`
	BuildIdentity  string                `json:"build_identity"`
	DriverVersion  string                `json:"driver_version"`
	RuntimeVersion string                `json:"runtime_version"`
	DeviceName     string                `json:"device_name"`
	DeviceUUID     string                `json:"device_uuid"`
	PCIBusID       string                `json:"pci_bus_id"`
	VisibleDevices string                `json:"visible_devices"`
	MIGConfig      string                `json:"mig_config"`
	Presence       HardwareEpochPresence `json:"presence"`
}

// Digest computes a deterministic sha256 hex string over the fingerprint.
func (f HardwareEpochFingerprint) Digest() string {
	h := sha256.New()
	fmt.Fprintf(h, "backend_name=%s\n", f.BackendName)
	fmt.Fprintf(h, "build_identity=%s\n", f.BuildIdentity)
	fmt.Fprintf(h, "driver_version=%s\n", f.DriverVersion)
	fmt.Fprintf(h, "runtime_version=%s\n", f.RuntimeVersion)
	fmt.Fprintf(h, "device_name=%s\n", f.DeviceName)
	fmt.Fprintf(h, "device_uuid=%s\n", f.DeviceUUID)
	fmt.Fprintf(h, "pci_bus_id=%s\n", f.PCIBusID)
	fmt.Fprintf(h, "visible_devices=%s\n", f.VisibleDevices)
	fmt.Fprintf(h, "mig_config=%s\n", f.MIGConfig)
	fmt.Fprintf(h, "presence.driver_version=%t\n", f.Presence.DriverVersion)
	fmt.Fprintf(h, "presence.runtime_version=%t\n", f.Presence.RuntimeVersion)
	fmt.Fprintf(h, "presence.device_uuid=%t\n", f.Presence.DeviceUUID)
	fmt.Fprintf(h, "presence.device_pci=%t\n", f.Presence.DevicePCI)
	fmt.Fprintf(h, "presence.visible_devices=%t\n", f.Presence.VisibleDevices)
	fmt.Fprintf(h, "presence.mig_partition=%t\n", f.Presence.MIGPartition)
	fmt.Fprintf(h, "presence.build_identity=%t\n", f.Presence.BuildIdentity)
	return hex.EncodeToString(h.Sum(nil))
}

// HardwareEpoch represents a witnessed generation of backend hardware/runtime state.
type HardwareEpoch struct {
	ID          uint64                   `json:"id"`
	Fingerprint HardwareEpochFingerprint `json:"fingerprint"`
	Digest      string                   `json:"digest"`
	ObservedAt  time.Time                `json:"observed_at"`
}

// NewHardwareEpoch constructs a HardwareEpoch with an evaluated digest.
func NewHardwareEpoch(id uint64, fp HardwareEpochFingerprint, observedAt time.Time) HardwareEpoch {
	if observedAt.IsZero() {
		observedAt = time.Now().UTC()
	}
	return HardwareEpoch{
		ID:          id,
		Fingerprint: fp,
		Digest:      fp.Digest(),
		ObservedAt:  observedAt,
	}
}

// HardwareEpochChurnReason names the cause of an epoch invalidation or transition.
type HardwareEpochChurnReason string

const (
	EpochChurnDriverUpgrade HardwareEpochChurnReason = "driver_upgrade"
	EpochChurnRuntimeChange HardwareEpochChurnReason = "runtime_change"
	EpochChurnDeviceReset   HardwareEpochChurnReason = "device_reset"
	EpochChurnMIGReconfig   HardwareEpochChurnReason = "mig_reconfig"
	EpochChurnDeviceLost    HardwareEpochChurnReason = "device_lost"
	EpochChurnInitialProbe  HardwareEpochChurnReason = "initial_probe"
)

// String returns the string representation of the churn reason.
func (r HardwareEpochChurnReason) String() string {
	return string(r)
}

// EpochMismatchError records a refusal caused by an unexpected or changed hardware epoch.
type EpochMismatchError struct {
	Backend       string                   `json:"backend"`
	ExpectedEpoch string                   `json:"expected_epoch"`
	CurrentEpoch  string                   `json:"current_epoch"`
	Reason        HardwareEpochChurnReason `json:"reason"`
}

func (e *EpochMismatchError) Error() string {
	return fmt.Sprintf("compute: epoch mismatch on backend %q: expected epoch %s, current epoch %s (reason: %s)",
		e.Backend, e.ExpectedEpoch, e.CurrentEpoch, e.Reason)
}

// BackendEpochReporter is the optional interface a backend implements to report
// and manage hardware epoch bindings.
type BackendEpochReporter interface {
	CurrentHardwareEpoch() HardwareEpoch
	InvalidateHardwareEpoch(reason HardwareEpochChurnReason) (HardwareEpoch, error)
	ValidateEpoch(expectedDigest string) error
}

// BackendCurrentEpoch returns the current hardware epoch reported by backend b, if supported.
func BackendCurrentEpoch(b Backend) (HardwareEpoch, bool) {
	if b == nil {
		return HardwareEpoch{}, false
	}
	rep, ok := b.(BackendEpochReporter)
	if !ok {
		return HardwareEpoch{}, false
	}
	return rep.CurrentHardwareEpoch(), true
}

// ValidateBackendEpoch verifies that backend b matches the expected epoch digest.
// If expectedDigest is empty, validation passes unconditionally.
func ValidateBackendEpoch(b Backend, expectedDigest string) error {
	if expectedDigest == "" {
		return nil
	}
	if b == nil {
		return &EpochMismatchError{
			Backend:       "nil",
			ExpectedEpoch: expectedDigest,
			CurrentEpoch:  "",
			Reason:        EpochChurnDeviceLost,
		}
	}
	if rep, ok := b.(BackendEpochReporter); ok {
		return rep.ValidateEpoch(expectedDigest)
	}
	return &EpochMismatchError{
		Backend:       b.Name(),
		ExpectedEpoch: expectedDigest,
		CurrentEpoch:  "unsupported",
		Reason:        EpochChurnRuntimeChange,
	}
}

// EpochTracker is a thread-safe helper for backends to manage epochs,
// invalidations, callbacks, and cached state invalidation.
type EpochTracker struct {
	mu              sync.RWMutex
	backendName     string
	current         HardwareEpoch
	lastReason      HardwareEpochChurnReason
	probeFn         func() (HardwareEpochFingerprint, error)
	callbacks       []func(oldEpoch, newEpoch HardwareEpoch, reason HardwareEpochChurnReason)
	statePurgeHooks []func()
}

// NewEpochTracker initializes an EpochTracker with an initial fingerprint.
func NewEpochTracker(backendName string, initial HardwareEpochFingerprint) *EpochTracker {
	digest := initial.Digest()
	return &EpochTracker{
		backendName: backendName,
		current: HardwareEpoch{
			ID:          1,
			Fingerprint: initial,
			Digest:      digest,
			ObservedAt:  time.Now().UTC(),
		},
		lastReason: EpochChurnInitialProbe,
	}
}

// BackendName returns the name of the tracked backend.
func (t *EpochTracker) BackendName() string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.backendName
}

// CurrentHardwareEpoch returns the current epoch snapshot.
func (t *EpochTracker) CurrentHardwareEpoch() HardwareEpoch {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.current
}

// ValidateEpoch checks whether the expected digest matches the current epoch digest.
func (t *EpochTracker) ValidateEpoch(expectedDigest string) error {
	if expectedDigest == "" {
		return nil
	}
	t.mu.RLock()
	cur := t.current
	backend := t.backendName
	reason := t.lastReason
	t.mu.RUnlock()

	if cur.Digest == expectedDigest {
		return nil
	}
	return &EpochMismatchError{
		Backend:       backend,
		ExpectedEpoch: expectedDigest,
		CurrentEpoch:  cur.Digest,
		Reason:        reason,
	}
}

// SetProbeFunc registers a probe function to re-evaluate the hardware fingerprint
// during an invalidation.
func (t *EpochTracker) SetProbeFunc(fn func() (HardwareEpochFingerprint, error)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.probeFn = fn
}

// RegisterCallback registers a hook invoked whenever the epoch changes.
func (t *EpochTracker) RegisterCallback(cb func(oldEpoch, newEpoch HardwareEpoch, reason HardwareEpochChurnReason)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.callbacks = append(t.callbacks, cb)
}

// RegisterPurgeHook registers a hook to discard cached backend state on invalidation.
func (t *EpochTracker) RegisterPurgeHook(fn func()) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.statePurgeHooks = append(t.statePurgeHooks, fn)
}

// PurgeStaleState explicitly executes all registered state purge hooks.
func (t *EpochTracker) PurgeStaleState() {
	t.mu.RLock()
	invs := make([]func(), len(t.statePurgeHooks))
	copy(invs, t.statePurgeHooks)
	t.mu.RUnlock()
	for _, fn := range invs {
		fn()
	}
}

// InvalidateHardwareEpoch advances the epoch generation and invalidates cached state.
// If a probe function is configured, it is called to re-probe the fingerprint.
func (t *EpochTracker) InvalidateHardwareEpoch(reason HardwareEpochChurnReason) (HardwareEpoch, error) {
	t.mu.Lock()
	oldEpoch := t.current
	newFp := oldEpoch.Fingerprint
	if t.probeFn != nil {
		fp, err := t.probeFn()
		if err != nil {
			t.lastReason = reason
			t.mu.Unlock()
			return HardwareEpoch{}, err
		}
		newFp = fp
	}

	t.current.ID++
	t.current.Fingerprint = newFp
	t.current.Digest = newFp.Digest()
	t.current.ObservedAt = time.Now().UTC()
	t.lastReason = reason
	newEpoch := t.current

	cbs := make([]func(HardwareEpoch, HardwareEpoch, HardwareEpochChurnReason), len(t.callbacks))
	copy(cbs, t.callbacks)
	invs := make([]func(), len(t.statePurgeHooks))
	copy(invs, t.statePurgeHooks)
	t.mu.Unlock()

	for _, inv := range invs {
		inv()
	}
	for _, cb := range cbs {
		cb(oldEpoch, newEpoch, reason)
	}
	return newEpoch, nil
}

// UpdateFingerprint mutates the active fingerprint directly, advancing the epoch.
func (t *EpochTracker) UpdateFingerprint(fp HardwareEpochFingerprint, reason HardwareEpochChurnReason) HardwareEpoch {
	t.mu.Lock()
	oldEpoch := t.current
	t.current.ID++
	t.current.Fingerprint = fp
	t.current.Digest = fp.Digest()
	t.current.ObservedAt = time.Now().UTC()
	t.lastReason = reason
	newEpoch := t.current

	cbs := make([]func(HardwareEpoch, HardwareEpoch, HardwareEpochChurnReason), len(t.callbacks))
	copy(cbs, t.callbacks)
	invs := make([]func(), len(t.statePurgeHooks))
	copy(invs, t.statePurgeHooks)
	t.mu.Unlock()

	for _, inv := range invs {
		inv()
	}
	for _, cb := range cbs {
		cb(oldEpoch, newEpoch, reason)
	}
	return newEpoch
}
