package compute

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

type epochTestMockBackend struct {
	Backend
	name    string
	tracker *EpochTracker
}

func (m *epochTestMockBackend) Name() string {
	if m.name != "" {
		return m.name
	}
	return "mock-device"
}

func (m *epochTestMockBackend) CurrentHardwareEpoch() HardwareEpoch {
	return m.tracker.CurrentHardwareEpoch()
}

func (m *epochTestMockBackend) InvalidateHardwareEpoch(reason HardwareEpochChurnReason) (HardwareEpoch, error) {
	return m.tracker.InvalidateHardwareEpoch(reason)
}

func (m *epochTestMockBackend) ValidateEpoch(expectedDigest string) error {
	return m.tracker.ValidateEpoch(expectedDigest)
}

type mockCapacityBackend struct {
	*epochTestMockBackend
	totalMem int64
	freeMem  int64
	hasProbe bool
}

func (m *mockCapacityBackend) DeviceMemory() (total, free int64, known bool) {
	return m.totalMem, m.freeMem, true
}

func (m *mockCapacityBackend) Caps() Caps {
	return Caps{CapacityProbe: m.hasProbe}
}

func sampleFingerprint(backend string, driverVer string) HardwareEpochFingerprint {
	return HardwareEpochFingerprint{
		BackendName:    backend,
		BuildIdentity:  "fak-v0.1.0-test",
		DriverVersion:  driverVer,
		RuntimeVersion: "12.4",
		DeviceName:     "Mock-Accelerator-100",
		DeviceUUID:     "UUID-1234-5678-ABCD",
		PCIBusID:       "0000:01:00.0",
		VisibleDevices: "0",
		MIGConfig:      "none",
		Presence: HardwareEpochPresence{
			DriverVersion:  true,
			RuntimeVersion: true,
			DeviceUUID:     true,
			DevicePCI:      true,
			VisibleDevices: true,
			MIGPartition:   false,
			BuildIdentity:  true,
		},
	}
}

func TestHardwareEpochFingerprintDeterminism(t *testing.T) {
	fp1 := sampleFingerprint("cuda", "550.54.14")
	fp2 := sampleFingerprint("cuda", "550.54.14")

	// 1. SHA256 digest is deterministic.
	d1 := fp1.Digest()
	d2 := fp2.Digest()

	if d1 == "" {
		t.Fatal("expected non-empty digest")
	}
	if len(d1) != 64 {
		t.Fatalf("expected 64-character sha256 hex digest, got len=%d: %q", len(d1), d1)
	}
	if d1 != d2 {
		t.Fatalf("identical fingerprints produced differing digests: %q vs %q", d1, d2)
	}

	for i := 0; i < 5; i++ {
		if got := fp1.Digest(); got != d1 {
			t.Fatalf("repeated invocation %d produced differing digest: %q vs %q", i, got, d1)
		}
	}

	// 2. Modifying any individual field changes the digest.
	fieldChecks := []struct {
		name   string
		mutate func(f *HardwareEpochFingerprint)
	}{
		{"BackendName", func(f *HardwareEpochFingerprint) { f.BackendName = "vulkan" }},
		{"BuildIdentity", func(f *HardwareEpochFingerprint) { f.BuildIdentity = "fak-v0.2.0-modified" }},
		{"DriverVersion", func(f *HardwareEpochFingerprint) { f.DriverVersion = "550.54.15" }},
		{"RuntimeVersion", func(f *HardwareEpochFingerprint) { f.RuntimeVersion = "12.5" }},
		{"DeviceName", func(f *HardwareEpochFingerprint) { f.DeviceName = "Mock-Accelerator-200" }},
		{"DeviceUUID", func(f *HardwareEpochFingerprint) { f.DeviceUUID = "UUID-DIFFERENT" }},
		{"PCIBusID", func(f *HardwareEpochFingerprint) { f.PCIBusID = "0000:02:00.0" }},
		{"VisibleDevices", func(f *HardwareEpochFingerprint) { f.VisibleDevices = "1" }},
		{"MIGConfig", func(f *HardwareEpochFingerprint) { f.MIGConfig = "3g.20gb" }},
	}

	for _, tc := range fieldChecks {
		t.Run("Field_"+tc.name, func(t *testing.T) {
			mutated := fp1
			tc.mutate(&mutated)
			if mutated.Digest() == d1 {
				t.Fatalf("modifying field %s did not change digest", tc.name)
			}
		})
	}

	// 3. Presence flags faithfully reflect set fields without fabricated defaults.
	var emptyFP HardwareEpochFingerprint
	if emptyFP.Presence.DriverVersion || emptyFP.Presence.RuntimeVersion ||
		emptyFP.Presence.DeviceUUID || emptyFP.Presence.DevicePCI ||
		emptyFP.Presence.VisibleDevices || emptyFP.Presence.MIGPartition ||
		emptyFP.Presence.BuildIdentity {
		t.Fatalf("uninitialized fingerprint must not fabricate true presence flags: %+v", emptyFP.Presence)
	}

	emptyDigest := emptyFP.Digest()
	if emptyDigest == "" {
		t.Fatal("empty fingerprint should produce a valid SHA256 digest")
	}

	// Setting string field values must NOT fabricate presence flags.
	fpWithValueOnly := emptyFP
	fpWithValueOnly.DriverVersion = "550.54.14"
	fpWithValueOnly.DeviceUUID = "UUID-1234"
	if fpWithValueOnly.Presence.DriverVersion || fpWithValueOnly.Presence.DeviceUUID {
		t.Fatal("presence flags must not be fabricated when only string fields are populated")
	}

	// Fingerprint with presence=false must differ in digest from presence=true even with identical text.
	fpWithValueAndPresence := fpWithValueOnly
	fpWithValueAndPresence.Presence.DriverVersion = true
	fpWithValueAndPresence.Presence.DeviceUUID = true
	if fpWithValueOnly.Digest() == fpWithValueAndPresence.Digest() {
		t.Fatal("fingerprint with presence=false must have different digest from presence=true")
	}

	// Toggling each presence flag independently changes the digest.
	presenceChecks := []struct {
		name   string
		mutate func(p *HardwareEpochPresence)
	}{
		{"DriverVersion", func(p *HardwareEpochPresence) { p.DriverVersion = !p.DriverVersion }},
		{"RuntimeVersion", func(p *HardwareEpochPresence) { p.RuntimeVersion = !p.RuntimeVersion }},
		{"DeviceUUID", func(p *HardwareEpochPresence) { p.DeviceUUID = !p.DeviceUUID }},
		{"DevicePCI", func(p *HardwareEpochPresence) { p.DevicePCI = !p.DevicePCI }},
		{"VisibleDevices", func(p *HardwareEpochPresence) { p.VisibleDevices = !p.VisibleDevices }},
		{"MIGPartition", func(p *HardwareEpochPresence) { p.MIGPartition = !p.MIGPartition }},
		{"BuildIdentity", func(p *HardwareEpochPresence) { p.BuildIdentity = !p.BuildIdentity }},
	}

	for _, tc := range presenceChecks {
		t.Run("Presence_"+tc.name, func(t *testing.T) {
			mutated := fp1
			tc.mutate(&mutated.Presence)
			if mutated.Digest() == d1 {
				t.Fatalf("toggling presence flag %s did not change digest", tc.name)
			}
		})
	}
}

func TestHardwareEpochConstruction(t *testing.T) {
	fp := sampleFingerprint("vulkan", "1.3.280")
	now := time.Now().UTC()
	epoch := NewHardwareEpoch(1, fp, now)

	if epoch.ID != 1 {
		t.Fatalf("epoch ID = %d, want 1", epoch.ID)
	}
	if epoch.Digest != fp.Digest() {
		t.Fatalf("epoch Digest = %q, want %q", epoch.Digest, fp.Digest())
	}
	if epoch.ObservedAt != now {
		t.Fatalf("epoch ObservedAt = %v, want %v", epoch.ObservedAt, now)
	}
}

func TestHardwareEpochChurnReasons(t *testing.T) {
	reasons := []HardwareEpochChurnReason{
		EpochChurnDriverUpgrade,
		EpochChurnRuntimeChange,
		EpochChurnDeviceReset,
		EpochChurnMIGReconfig,
		EpochChurnDeviceLost,
		EpochChurnInitialProbe,
	}

	for _, r := range reasons {
		if r.String() == "" {
			t.Fatalf("expected non-empty string for reason %v", r)
		}
	}
}

func TestEpochMismatchErrorFormat(t *testing.T) {
	err := &EpochMismatchError{
		Backend:       "cuda",
		ExpectedEpoch: "hash-expected-1234",
		CurrentEpoch:  "hash-current-5678",
		Reason:        EpochChurnDriverUpgrade,
	}

	msg := err.Error()
	for _, expectedSub := range []string{"cuda", "hash-expected-1234", "hash-current-5678", "driver_upgrade"} {
		if !strings.Contains(msg, expectedSub) {
			t.Fatalf("error message %q missing %q", msg, expectedSub)
		}
	}
}

func TestEpochTrackerLifecycleAndInvalidation(t *testing.T) {
	initialFp := sampleFingerprint("cuda", "550.54.14")
	tracker := NewEpochTracker("cuda", initialFp)

	// 1. Initial probe verification
	if tracker.BackendName() != "cuda" {
		t.Fatalf("backend name = %q, want cuda", tracker.BackendName())
	}

	epoch1 := tracker.CurrentHardwareEpoch()
	if epoch1.ID != 1 {
		t.Fatalf("initial epoch ID = %d, want 1", epoch1.ID)
	}
	if epoch1.Digest != initialFp.Digest() {
		t.Fatalf("initial epoch digest = %q, want %q", epoch1.Digest, initialFp.Digest())
	}
	if epoch1.Fingerprint.DriverVersion != initialFp.DriverVersion {
		t.Fatalf("initial driver version = %q, want %q", epoch1.Fingerprint.DriverVersion, initialFp.DriverVersion)
	}
	if epoch1.ObservedAt.IsZero() {
		t.Fatal("initial epoch ObservedAt is zero")
	}
	if time.Since(epoch1.ObservedAt) > 10*time.Second {
		t.Fatalf("initial epoch ObservedAt too far in past: %v", epoch1.ObservedAt)
	}

	// ValidateEpoch behavior on initial epoch
	if err := tracker.ValidateEpoch(epoch1.Digest); err != nil {
		t.Fatalf("ValidateEpoch failed for initial digest: %v", err)
	}
	if err := tracker.ValidateEpoch(""); err != nil {
		t.Fatalf("ValidateEpoch with empty digest should pass unconditionally, got %v", err)
	}
	staleErr := tracker.ValidateEpoch("stale-initial-digest")
	if staleErr == nil {
		t.Fatal("ValidateEpoch with mismatched digest must return error, got nil")
	}
	var initMismatch *EpochMismatchError
	if !errors.As(staleErr, &initMismatch) {
		t.Fatalf("want *EpochMismatchError, got %T (%v)", staleErr, staleErr)
	}
	if initMismatch.Reason != EpochChurnInitialProbe {
		t.Fatalf("initial mismatch reason = %v, want %v", initMismatch.Reason, EpochChurnInitialProbe)
	}

	// 2. Monotonic ID increment and callback firing to flush cached buffers
	type transitionEvent struct {
		oldEpoch HardwareEpoch
		newEpoch HardwareEpoch
		reason   HardwareEpochChurnReason
	}
	var transitions []transitionEvent
	tracker.RegisterCallback(func(oldEpoch, newEpoch HardwareEpoch, reason HardwareEpochChurnReason) {
		transitions = append(transitions, transitionEvent{
			oldEpoch: oldEpoch,
			newEpoch: newEpoch,
			reason:   reason,
		})
	})

	cachedBuffers := map[string][]byte{
		"gemm_workspace": make([]byte, 4096),
		"kv_staging":     make([]byte, 8192),
	}
	var cacheFlushes int
	tracker.RegisterPurgeHook(func() {
		cacheFlushes++
		for k := range cachedBuffers {
			delete(cachedBuffers, k)
		}
	})

	// Invalidation 1: DeviceReset
	epoch2, err := tracker.InvalidateHardwareEpoch(EpochChurnDeviceReset)
	if err != nil {
		t.Fatalf("InvalidateHardwareEpoch failed: %v", err)
	}
	if epoch2.ID != 2 {
		t.Fatalf("advanced epoch ID = %d, want 2", epoch2.ID)
	}
	if cacheFlushes != 1 {
		t.Fatalf("cacheFlushes = %d, want 1", cacheFlushes)
	}
	if len(cachedBuffers) != 0 {
		t.Fatalf("cached buffers not flushed on invalidation, len = %d", len(cachedBuffers))
	}
	if len(transitions) != 1 {
		t.Fatalf("len(transitions) = %d, want 1", len(transitions))
	}
	if transitions[0].oldEpoch.ID != 1 || transitions[0].newEpoch.ID != 2 {
		t.Fatalf("unexpected transition IDs: %d -> %d", transitions[0].oldEpoch.ID, transitions[0].newEpoch.ID)
	}
	if transitions[0].reason != EpochChurnDeviceReset {
		t.Fatalf("transition reason = %v, want %v", transitions[0].reason, EpochChurnDeviceReset)
	}

	// ValidateEpoch behavior after invalidation without probe func (digest retained, reason updated)
	if err := tracker.ValidateEpoch(epoch2.Digest); err != nil {
		t.Fatalf("ValidateEpoch for epoch2 digest failed: %v", err)
	}
	errMismatched := tracker.ValidateEpoch("unmatched-epoch-digest")
	if errMismatched == nil {
		t.Fatal("ValidateEpoch with unmatched digest on epoch2 tracker should fail, got nil")
	}
	var mismatchReset *EpochMismatchError
	if !errors.As(errMismatched, &mismatchReset) {
		t.Fatalf("want *EpochMismatchError, got %T (%v)", errMismatched, errMismatched)
	}
	if mismatchReset.Reason != EpochChurnDeviceReset {
		t.Fatalf("mismatch reason = %v, want %v", mismatchReset.Reason, EpochChurnDeviceReset)
	}

	// Invalidation 2: Driver upgrade with probe func mutating the fingerprint
	upgradedFp := sampleFingerprint("cuda", "550.54.15")
	tracker.SetProbeFunc(func() (HardwareEpochFingerprint, error) {
		return upgradedFp, nil
	})
	cachedBuffers["new_buf"] = make([]byte, 1024)

	epoch3, err := tracker.InvalidateHardwareEpoch(EpochChurnDriverUpgrade)
	if err != nil {
		t.Fatalf("second invalidation failed: %v", err)
	}
	if epoch3.ID != 3 {
		t.Fatalf("epoch3 ID = %d, want 3", epoch3.ID)
	}
	if epoch3.Digest != upgradedFp.Digest() {
		t.Fatalf("epoch3 digest = %q, want %q", epoch3.Digest, upgradedFp.Digest())
	}
	if cacheFlushes != 2 || len(cachedBuffers) != 0 {
		t.Fatalf("cacheFlushes = %d, len(cachedBuffers) = %d", cacheFlushes, len(cachedBuffers))
	}

	// ValidateEpoch behavior after digest-mutating invalidation:
	// current digest succeeds; old epoch1 digest fails closed.
	if err := tracker.ValidateEpoch(epoch3.Digest); err != nil {
		t.Fatalf("ValidateEpoch for epoch3 digest failed: %v", err)
	}
	errOld := tracker.ValidateEpoch(epoch1.Digest)
	if errOld == nil {
		t.Fatal("ValidateEpoch with epoch1 digest on epoch3 tracker should fail, got nil")
	}
	var mismatch *EpochMismatchError
	if !errors.As(errOld, &mismatch) {
		t.Fatalf("want *EpochMismatchError, got %T (%v)", errOld, errOld)
	}
	if mismatch.ExpectedEpoch != epoch1.Digest {
		t.Fatalf("mismatch expected epoch = %q, want %q", mismatch.ExpectedEpoch, epoch1.Digest)
	}
	if mismatch.CurrentEpoch != epoch3.Digest {
		t.Fatalf("mismatch current epoch = %q, want %q", mismatch.CurrentEpoch, epoch3.Digest)
	}
	if mismatch.Reason != EpochChurnDriverUpgrade {
		t.Fatalf("mismatch reason = %v, want %v", mismatch.Reason, EpochChurnDriverUpgrade)
	}

	// Invalidation 3: UpdateFingerprint for MIG reconfig
	migFp := upgradedFp
	migFp.MIGConfig = "3g.20gb"
	migFp.Presence.MIGPartition = true
	epoch4 := tracker.UpdateFingerprint(migFp, EpochChurnMIGReconfig)
	if epoch4.ID != 4 {
		t.Fatalf("epoch4 ID = %d, want 4", epoch4.ID)
	}
	if cacheFlushes != 3 {
		t.Fatalf("cacheFlushes after UpdateFingerprint = %d, want 3", cacheFlushes)
	}

	// Explicit PurgeStaleState flushes buffers without advancing epoch ID
	cachedBuffers["temp_alloc"] = make([]byte, 512)
	tracker.PurgeStaleState()
	if cacheFlushes != 4 {
		t.Fatalf("cacheFlushes after PurgeStaleState = %d, want 4", cacheFlushes)
	}
	if len(cachedBuffers) != 0 {
		t.Fatalf("cached buffers not flushed by PurgeStaleState")
	}
	if cur := tracker.CurrentHardwareEpoch(); cur.ID != 4 {
		t.Fatalf("epoch ID changed on PurgeStaleState: %d, want 4", cur.ID)
	}
}

func TestEpochTrackerProbeFunc(t *testing.T) {
	initialFp := sampleFingerprint("cuda", "550.00")
	tracker := NewEpochTracker("cuda", initialFp)

	probeErr := errors.New("nvml query failed")
	tracker.SetProbeFunc(func() (HardwareEpochFingerprint, error) {
		return HardwareEpochFingerprint{}, probeErr
	})

	_, err := tracker.InvalidateHardwareEpoch(EpochChurnDeviceReset)
	if !errors.Is(err, probeErr) {
		t.Fatalf("expected probe error %v, got %v", probeErr, err)
	}

	upgradedFp := sampleFingerprint("cuda", "550.54")
	tracker.SetProbeFunc(func() (HardwareEpochFingerprint, error) {
		return upgradedFp, nil
	})

	nextEpoch, err := tracker.InvalidateHardwareEpoch(EpochChurnDriverUpgrade)
	if err != nil {
		t.Fatalf("unexpected invalidation error: %v", err)
	}
	if nextEpoch.Fingerprint.DriverVersion != "550.54" {
		t.Fatalf("driver version = %q, want 550.54", nextEpoch.Fingerprint.DriverVersion)
	}
	if nextEpoch.Digest != upgradedFp.Digest() {
		t.Fatalf("next digest = %q, want %q", nextEpoch.Digest, upgradedFp.Digest())
	}
}

func TestEpochTrackerUpdateFingerprint(t *testing.T) {
	initialFp := sampleFingerprint("vulkan", "1.3.100")
	tracker := NewEpochTracker("vulkan", initialFp)

	migFp := initialFp
	migFp.MIGConfig = "3g.20gb"
	migFp.Presence.MIGPartition = true

	updated := tracker.UpdateFingerprint(migFp, EpochChurnMIGReconfig)
	if updated.ID != 2 {
		t.Fatalf("epoch ID = %d, want 2", updated.ID)
	}
	if updated.Fingerprint.MIGConfig != "3g.20gb" {
		t.Fatalf("MIGConfig = %q, want 3g.20gb", updated.Fingerprint.MIGConfig)
	}
	if err := tracker.ValidateEpoch(updated.Digest); err != nil {
		t.Fatalf("validation failed for current digest: %v", err)
	}
}

func TestValidateBackendEpochAndHelpers(t *testing.T) {
	fp := sampleFingerprint("cuda", "550.54.14")
	tracker := NewEpochTracker("cuda", fp)
	mock := &epochTestMockBackend{tracker: tracker}

	epoch, ok := BackendCurrentEpoch(mock)
	if !ok {
		t.Fatal("expected BackendCurrentEpoch to return true for reporter")
	}
	if epoch.Digest != fp.Digest() {
		t.Fatalf("epoch digest = %q, want %q", epoch.Digest, fp.Digest())
	}

	emptyEpoch, ok := BackendCurrentEpoch(nil)
	if ok || emptyEpoch.Digest != "" {
		t.Fatalf("nil backend returned ok=%v epoch=%v", ok, emptyEpoch)
	}

	if err := ValidateBackendEpoch(mock, ""); err != nil {
		t.Fatalf("empty expected digest should succeed, got: %v", err)
	}
	if err := ValidateBackendEpoch(mock, fp.Digest()); err != nil {
		t.Fatalf("matching digest should succeed, got: %v", err)
	}

	err := ValidateBackendEpoch(mock, "stale-digest")
	if err == nil {
		t.Fatal("expected error on mismatched digest, got nil")
	}
	var mismatch *EpochMismatchError
	if !errors.As(err, &mismatch) {
		t.Fatalf("want *EpochMismatchError, got %T (%v)", err, err)
	}
	if mismatch.ExpectedEpoch != "stale-digest" {
		t.Fatalf("expected epoch = %q, want stale-digest", mismatch.ExpectedEpoch)
	}

	if err := ValidateBackendEpoch(nil, "digest"); err == nil {
		t.Fatal("expected error for nil backend with digest")
	}
}

// fakeMutatingBackend simulates an accelerator backend whose hardware/driver
// environment mutates mid-lifecycle.
type fakeMutatingBackend struct {
	Backend
	name          string
	tracker       *EpochTracker
	cachedEntries map[string]cachedItem
	mu            sync.Mutex
}

type cachedItem struct {
	epochDigest string
	payload     string
}

func newFakeMutatingBackend(name string, initialFp HardwareEpochFingerprint) *fakeMutatingBackend {
	tracker := NewEpochTracker(name, initialFp)
	b := &fakeMutatingBackend{
		name:          name,
		tracker:       tracker,
		cachedEntries: make(map[string]cachedItem),
	}
	tracker.RegisterPurgeHook(func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		b.cachedEntries = make(map[string]cachedItem)
	})
	return b
}

func (b *fakeMutatingBackend) Name() string {
	return b.name
}

func (b *fakeMutatingBackend) CurrentHardwareEpoch() HardwareEpoch {
	return b.tracker.CurrentHardwareEpoch()
}

func (b *fakeMutatingBackend) InvalidateHardwareEpoch(reason HardwareEpochChurnReason) (HardwareEpoch, error) {
	return b.tracker.InvalidateHardwareEpoch(reason)
}

func (b *fakeMutatingBackend) ValidateEpoch(expectedDigest string) error {
	return b.tracker.ValidateEpoch(expectedDigest)
}

func (b *fakeMutatingBackend) StoreCachedEntry(key string, payload string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.cachedEntries[key] = cachedItem{
		epochDigest: b.tracker.CurrentHardwareEpoch().Digest,
		payload:     payload,
	}
}

func (b *fakeMutatingBackend) ExecuteOperation(boundEpochDigest string, opName string) (string, error) {
	if err := ValidateBackendEpoch(b, boundEpochDigest); err != nil {
		return "", err
	}
	return fmt.Sprintf("executed %s on %s at epoch %s", opName, b.name, boundEpochDigest), nil
}

func (b *fakeMutatingBackend) AccessCachedEntry(key string, expectedEpochDigest string) (string, error) {
	if err := ValidateBackendEpoch(b, expectedEpochDigest); err != nil {
		return "", err
	}
	b.mu.Lock()
	item, ok := b.cachedEntries[key]
	b.mu.Unlock()
	if !ok {
		return "", fmt.Errorf("entry %q not found in cache", key)
	}
	return item.payload, nil
}

func testFirstCheckableStepMutatingEpochFailsClosed(t *testing.T) {
	// Step 1: Initialize backend under Epoch 1.
	fp1 := sampleFingerprint("cuda", "550.54.14")
	backend := newFakeMutatingBackend("cuda", fp1)

	epoch1 := backend.CurrentHardwareEpoch()
	if epoch1.ID != 1 {
		t.Fatalf("epoch1 ID = %d, want 1", epoch1.ID)
	}
	if epoch1.Digest != fp1.Digest() {
		t.Fatalf("epoch1 digest = %q, want %q", epoch1.Digest, fp1.Digest())
	}

	// Create operations and cached entries bound to Epoch 1.
	backend.StoreCachedEntry("kv_kernel_opt", "compiled_ptx_v1")
	op1EpochDigest := epoch1.Digest

	// Operations and cached entries created under Epoch 1 succeed.
	res1, err := backend.ExecuteOperation(op1EpochDigest, "flash_attn_fwd")
	if err != nil {
		t.Fatalf("operation under epoch 1 failed: %v", err)
	}
	if !strings.Contains(res1, "flash_attn_fwd") {
		t.Fatalf("unexpected op result: %q", res1)
	}

	cachedVal, err := backend.AccessCachedEntry("kv_kernel_opt", op1EpochDigest)
	if err != nil {
		t.Fatalf("cache lookup under epoch 1 failed: %v", err)
	}
	if cachedVal != "compiled_ptx_v1" {
		t.Fatalf("cachedVal = %q, want compiled_ptx_v1", cachedVal)
	}

	if err := ValidateBackendEpoch(backend, epoch1.Digest); err != nil {
		t.Fatalf("ValidateBackendEpoch failed for epoch 1: %v", err)
	}

	// Step 2: Mid-lifecycle hardware/driver mutation.
	// Synthetic driver upgrade occurs mid-process.
	fp2 := sampleFingerprint("cuda", "550.54.15")
	backend.tracker.SetProbeFunc(func() (HardwareEpochFingerprint, error) {
		return fp2, nil
	})

	epoch2, err := backend.InvalidateHardwareEpoch(EpochChurnDriverUpgrade)
	if err != nil {
		t.Fatalf("InvalidateHardwareEpoch failed: %v", err)
	}
	if epoch2.ID != 2 {
		t.Fatalf("epoch2 ID = %d, want 2", epoch2.ID)
	}
	if epoch2.Digest == epoch1.Digest {
		t.Fatal("mutated epoch digest must differ from epoch 1 digest")
	}

	// Step 3: Assert operations or cached entries created under Epoch 1 fail closed
	// with *EpochMismatchError when presented to Epoch 2.
	_, errOpStale := backend.ExecuteOperation(op1EpochDigest, "flash_attn_fwd")
	if errOpStale == nil {
		t.Fatal("expected stale epoch operation to fail closed, got nil")
	}
	var mismatchOp *EpochMismatchError
	if !errors.As(errOpStale, &mismatchOp) {
		t.Fatalf("want *EpochMismatchError for stale operation, got %T (%v)", errOpStale, errOpStale)
	}
	if mismatchOp.Backend != "cuda" {
		t.Fatalf("mismatch backend = %q, want cuda", mismatchOp.Backend)
	}
	if mismatchOp.ExpectedEpoch != epoch1.Digest {
		t.Fatalf("mismatch expected epoch = %q, want %q", mismatchOp.ExpectedEpoch, epoch1.Digest)
	}
	if mismatchOp.CurrentEpoch != epoch2.Digest {
		t.Fatalf("mismatch current epoch = %q, want %q", mismatchOp.CurrentEpoch, epoch2.Digest)
	}
	if mismatchOp.Reason != EpochChurnDriverUpgrade {
		t.Fatalf("mismatch reason = %v, want %v", mismatchOp.Reason, EpochChurnDriverUpgrade)
	}

	// Presenting cached entry from Epoch 1 fails closed with *EpochMismatchError
	_, errCacheStale := backend.AccessCachedEntry("kv_kernel_opt", op1EpochDigest)
	if errCacheStale == nil {
		t.Fatal("expected stale cached entry access to fail closed, got nil")
	}
	var mismatchCache *EpochMismatchError
	if !errors.As(errCacheStale, &mismatchCache) {
		t.Fatalf("want *EpochMismatchError for stale cache access, got %T (%v)", errCacheStale, errCacheStale)
	}

	// Direct backend validation for Epoch 1 fails closed
	errValidateStale := ValidateBackendEpoch(backend, epoch1.Digest)
	if errValidateStale == nil {
		t.Fatal("expected ValidateBackendEpoch to fail closed for epoch 1, got nil")
	}
	var mismatchVal *EpochMismatchError
	if !errors.As(errValidateStale, &mismatchVal) {
		t.Fatalf("want *EpochMismatchError, got %T (%v)", errValidateStale, errValidateStale)
	}

	// Step 4: After explicit re-probe, Epoch 2 is accepted.
	currentEpoch := backend.CurrentHardwareEpoch()
	if currentEpoch.ID != 2 || currentEpoch.Digest != epoch2.Digest {
		t.Fatalf("expected backend current epoch to be epoch 2, got %+v", currentEpoch)
	}

	// Operations bound to the re-probed Epoch 2 succeed.
	op2EpochDigest := currentEpoch.Digest
	res2, err := backend.ExecuteOperation(op2EpochDigest, "flash_attn_fwd")
	if err != nil {
		t.Fatalf("operation under re-probed epoch 2 failed: %v", err)
	}
	if !strings.Contains(res2, "flash_attn_fwd") {
		t.Fatalf("unexpected op2 result: %q", res2)
	}

	// New cache entries bound to Epoch 2 succeed.
	backend.StoreCachedEntry("kv_kernel_opt_v2", "compiled_ptx_v2")
	cachedVal2, err := backend.AccessCachedEntry("kv_kernel_opt_v2", op2EpochDigest)
	if err != nil {
		t.Fatalf("cache lookup under epoch 2 failed: %v", err)
	}
	if cachedVal2 != "compiled_ptx_v2" {
		t.Fatalf("cachedVal2 = %q, want compiled_ptx_v2", cachedVal2)
	}

	if err := ValidateBackendEpoch(backend, epoch2.Digest); err != nil {
		t.Fatalf("ValidateBackendEpoch failed for epoch 2: %v", err)
	}
}

func TestFirstCheckableStep_MutatingEpochFailsClosed(t *testing.T) {
	testFirstCheckableStepMutatingEpochFailsClosed(t)
}

func TestEpochFirstCheckableStep_MutatingEpochFailsClosed(t *testing.T) {
	testFirstCheckableStepMutatingEpochFailsClosed(t)
}

func TestMemoryPlanEpochBinding(t *testing.T) {
	fp := sampleFingerprint("cuda", "550.54.14")
	tracker := NewEpochTracker("cuda", fp)

	be := &mockCapacityBackend{
		epochTestMockBackend: &epochTestMockBackend{
			tracker: tracker,
			name:    "cuda",
		},
		totalMem: 1024,
		freeMem:  1024,
		hasProbe: true,
	}

	// 1. Unbound MemoryPlan has empty EpochDigest.
	var emptyPlan MemoryPlan
	if emptyPlan.EpochDigest() != "" {
		t.Fatalf("emptyPlan EpochDigest = %q, want empty", emptyPlan.EpochDigest())
	}
	if emptyPlan.WithEpochDigest("foo") != nil {
		t.Fatal("WithEpochDigest on nil plan must return nil")
	}

	plan := MemoryPlan{
		{Class: MemoryWeights, Bytes: 2048, Detail: "weights"},
		{Class: MemoryKVCache, Bytes: 1024, Detail: "kv"},
	}
	if plan.EpochDigest() != "" {
		t.Fatalf("expected empty EpochDigest on raw plan, got %q", plan.EpochDigest())
	}

	// 2. WithEpochDigest binds epoch digest across all demands and returns a copy.
	planBound := plan.WithEpochDigest(fp.Digest())
	if planBound.EpochDigest() != fp.Digest() {
		t.Fatalf("planBound EpochDigest = %q, want %q", planBound.EpochDigest(), fp.Digest())
	}
	for i, d := range planBound {
		if d.EpochDigest != fp.Digest() {
			t.Fatalf("demand %d EpochDigest = %q, want %q", i, d.EpochDigest, fp.Digest())
		}
	}
	// Original plan remains unmodified.
	if plan.EpochDigest() != "" {
		t.Fatalf("original plan was mutated: %q", plan.EpochDigest())
	}

	// 3. FitVerdict EpochDigest method and string representations.
	for _, v := range []FitVerdict{FitOK, FitTooBig, FitUnknown} {
		if v.EpochDigest() != "" {
			t.Fatalf("FitVerdict(%s).EpochDigest = %q, want empty", v, v.EpochDigest())
		}
	}
	for v, wantStr := range map[FitVerdict]string{FitOK: "ok", FitTooBig: "too_big", FitUnknown: "unknown"} {
		if v.String() != wantStr {
			t.Fatalf("FitVerdict(%d).String() = %q, want %q", v, v.String(), wantStr)
		}
	}

	// 4. RefuseMemoryPlanIfTooBig propagates bound plan epoch digest into FitError.
	refuseErr := RefuseMemoryPlanIfTooBig(be, planBound, 0)
	if refuseErr == nil {
		t.Fatal("expected FitTooBig error for plan exceeding budget, got nil")
	}
	var fitErr *FitError
	if !errors.As(refuseErr, &fitErr) {
		t.Fatalf("want *FitError, got %T (%v)", refuseErr, refuseErr)
	}
	if fitErr.Verdict != FitTooBig {
		t.Fatalf("fitErr.Verdict = %v, want %v", fitErr.Verdict, FitTooBig)
	}
	if fitErr.EpochDigest != fp.Digest() {
		t.Fatalf("fitErr.EpochDigest = %q, want %q", fitErr.EpochDigest, fp.Digest())
	}
	if fitErr.Scope != MemoryScopeDevice {
		t.Fatalf("fitErr.Scope = %v, want %v", fitErr.Scope, MemoryScopeDevice)
	}
	if len(fitErr.Demands) != len(planBound) {
		t.Fatalf("len(fitErr.Demands) = %d, want %d", len(fitErr.Demands), len(planBound))
	}
	for i, d := range fitErr.Demands {
		if d.EpochDigest != fp.Digest() {
			t.Fatalf("cloned demand %d EpochDigest = %q, want %q", i, d.EpochDigest, fp.Digest())
		}
	}

	// 5. If plan has no EpochDigest, RefuseMemoryPlanIfTooBig propagates backend's current epoch.
	unboundExceedingPlan := MemoryPlan{
		{Class: MemoryWeights, Bytes: 4096, Detail: "weights"},
	}
	refuseUnboundErr := RefuseMemoryPlanIfTooBig(be, unboundExceedingPlan, 0)
	if refuseUnboundErr == nil {
		t.Fatal("expected error for unbound exceeding plan")
	}
	var fitUnboundErr *FitError
	if !errors.As(refuseUnboundErr, &fitUnboundErr) {
		t.Fatalf("want *FitError, got %T", refuseUnboundErr)
	}
	if fitUnboundErr.EpochDigest != fp.Digest() {
		t.Fatalf("unbound plan fit error EpochDigest = %q, want backend epoch %q", fitUnboundErr.EpochDigest, fp.Digest())
	}

	// 6. RefuseIfTooBig propagates backend current epoch.
	refuseSingleErr := RefuseIfTooBig(be, 4096, 0)
	if refuseSingleErr == nil {
		t.Fatal("expected RefuseIfTooBig error")
	}
	var fitSingleErr *FitError
	if !errors.As(refuseSingleErr, &fitSingleErr) {
		t.Fatalf("want *FitError, got %T", refuseSingleErr)
	}
	if fitSingleErr.EpochDigest != fp.Digest() {
		t.Fatalf("single fit error EpochDigest = %q, want %q", fitSingleErr.EpochDigest, fp.Digest())
	}

	// 7. Host memory scope plan propagation.
	hostPlan := MemoryPlan{
		{Class: MemoryWeights, Bytes: 4096, Scope: MemoryScopeHost, Detail: "offload"},
	}.WithEpochDigest("host-epoch-digest-xyz")
	refuseHostErr := refuseMemoryPlanForHostMem(hostPlan, 1024, 1024, true, 0)
	if refuseHostErr == nil {
		t.Fatal("expected error for host memory plan exceeding host RAM")
	}
	var fitHostErr *FitError
	if !errors.As(refuseHostErr, &fitHostErr) {
		t.Fatalf("want *FitError, got %T", refuseHostErr)
	}
	if fitHostErr.EpochDigest != "host-epoch-digest-xyz" {
		t.Fatalf("fitHostErr.EpochDigest = %q, want host-epoch-digest-xyz", fitHostErr.EpochDigest)
	}

	// 8. Non-reporting backend fails open (nil error).
	beNoCap := &mockCapacityBackend{
		epochTestMockBackend: &epochTestMockBackend{
			tracker: tracker,
			name:    "cuda",
		},
		hasProbe: false,
	}
	if err := RefuseMemoryPlanIfTooBig(beNoCap, planBound, 0); err != nil {
		t.Fatalf("non-reporting backend must fail open, got: %v", err)
	}
	if err := RefuseIfTooBig(beNoCap, 4096, 0); err != nil {
		t.Fatalf("non-reporting backend must fail open for RefuseIfTooBig, got: %v", err)
	}
}

func TestCapacityEpochDigestBinding(t *testing.T) {
	TestMemoryPlanEpochBinding(t)
}

func TestEpochTrackerThreadSafety(t *testing.T) {
	fp := sampleFingerprint("cuda", "550.00")
	tracker := NewEpochTracker("cuda", fp)

	var wg sync.WaitGroup
	start := make(chan struct{})

	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			<-start
			for j := 0; j < 50; j++ {
				_ = tracker.CurrentHardwareEpoch()
				_ = tracker.ValidateEpoch(fp.Digest())
				if j%10 == 0 {
					_, _ = tracker.InvalidateHardwareEpoch(EpochChurnDeviceReset)
				}
				if j%15 == 0 {
					tracker.PurgeStaleState()
				}
			}
		}(i)
	}

	close(start)
	wg.Wait()

	finalEpoch := tracker.CurrentHardwareEpoch()
	if finalEpoch.ID <= 1 {
		t.Fatalf("expected advanced epoch ID after concurrent churn, got %d", finalEpoch.ID)
	}
}
