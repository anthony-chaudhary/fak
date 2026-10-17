package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func mtpHostDraftFallbackCount() uint64 {
	want := hostFallbackLabel(HostStageMTPDraftHostForward)
	for _, c := range compute.HostFallbackCounts() {
		if c.Stage == want {
			return c.Count
		}
	}
	return 0
}

// fak-test:runtime fast est=1ms lane=default
func TestSpecVerifyDeviceMTPHostDraftCountsBesideDeviceAndRefusesDeviceOnly(t *testing.T) {
	unregister := compute.RegisterDeviceWorkReader("fake-device", func() compute.DeviceWork { return compute.DeviceWork{} })
	t.Cleanup(unregister)

	portable := &Qwen35MTPForward{draft: &Session{M: &Model{}}}
	before := mtpHostDraftFallbackCount()
	if err := portable.admitHostDraftPosition(); err != nil {
		t.Fatalf("portable host draft refused: %v", err)
	}
	if got := mtpHostDraftFallbackCount(); got != before+1 {
		t.Fatalf("portable host draft beside a device counted %d, want %d", got, before+1)
	}

	strict := &Qwen35MTPForward{draft: &Session{M: &Model{}}}
	strict.draft.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	err := strict.admitHostDraftPosition()
	if !errors.Is(err, ErrDeviceOnlyHostStage) {
		t.Fatalf("device-only err=%v, want ErrDeviceOnlyHostStage", err)
	}
	var typed *BackendForwardOperationError
	if !errors.As(err, &typed) || typed.Stage != string(HostStageMTPDraftHostForward) {
		t.Fatalf("device-only err=%#v, want BackendForwardOperationError at the MTP host stage", err)
	}
	if !strict.draft.HostFallbackObserved() {
		t.Fatal("device-only refusal did not record the host fallback")
	}
	if got := mtpHostDraftFallbackCount(); got != before+2 {
		t.Fatalf("device-only refusal counted %d, want %d", got, before+2)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSpecVerifyDeviceMTPHostDraftOnCPUOnlyHostIsNotCounted(t *testing.T) {
	if compute.DeviceBackendRegistered() {
		t.Skip("a device backend is registered in this process")
	}
	f := &Qwen35MTPForward{draft: &Session{M: &Model{}}}
	before := mtpHostDraftFallbackCount()
	if err := f.admitHostDraftPosition(); err != nil {
		t.Fatalf("CPU-only host draft refused: %v", err)
	}
	if got := mtpHostDraftFallbackCount(); got != before {
		t.Fatalf("CPU-only host draft counted as a fallback (%d -> %d)", before, got)
	}
}
