package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v4LiveSharedAddBackend advertises device memory so the live expert binds the
// V4.1 shared-add device seam, and counts or faults its F32 adds.
type v4LiveSharedAddBackend struct {
	compute.Backend
	adds     int
	faultAdd bool
}

func (b *v4LiveSharedAddBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	return c
}

func (b *v4LiveSharedAddBackend) AddInPlace(dst, src compute.Tensor) {
	b.adds++
	if b.faultAdd {
		panic(&compute.BackendError{Backend: "live-shared-add-test", Site: "AddInPlace", Err: errors.New("injected add fault")})
	}
	b.Backend.AddInPlace(dst, src)
}

func v4LiveSharedAddRun(t *testing.T, be compute.Backend) ([]float32, error, *Session) {
	t.Helper()
	cfg, m := v4SharedExpertTestFixture(t, true)
	s := m.NewSession()
	s.Backend = be
	s.v4Expert = &sharedExpertFakeRouted{hidden: cfg.HiddenSize, out: sharedRoutedProbe(cfg, 0.25)}
	normalized := s.uploadHostF32([]int{cfg.HiddenSize}, sharedTestProbe(cfg), compute.MemoryActivation, "v4-test-live-shared-add")
	out, err := s.v4ExpertForward(sharedTestLayer, 1, normalized)
	if err != nil {
		return nil, err, s
	}
	got := append([]float32(nil), be.Read(out)...)
	return got, nil, s
}

// fak-test:runtime fast est=2s lane=default
func TestV4LiveExpertSharedAddRunsOnDeviceSeamWithHostBits(t *testing.T) {
	host, err, hs := v4LiveSharedAddRun(t, compute.Default())
	if err != nil {
		t.Fatalf("host live expert: %v", err)
	}
	hs.Close()
	be := &v4LiveSharedAddBackend{Backend: compute.Default()}
	got, err, s := v4LiveSharedAddRun(t, be)
	if err != nil {
		t.Fatalf("device live expert: %v", err)
	}
	defer s.Close()
	if be.adds != 1 {
		t.Fatalf("live shared add dispatched %d device adds, want 1", be.adds)
	}
	if !v41SameBits(got, host) {
		t.Fatalf("device shared add differs from host bits: got[0:3]=%v want[0:3]=%v", got[:3], host[:3])
	}
}

// fak-test:runtime fast est=2s lane=default
func TestV4LiveExpertSharedAddDeviceFaultIsTypedWithoutHostRetry(t *testing.T) {
	be := &v4LiveSharedAddBackend{Backend: compute.Default(), faultAdd: true}
	got, err, s := v4LiveSharedAddRun(t, be)
	var closed *BackendForwardOperationError
	var backend *compute.BackendError
	if got != nil || !errors.Is(err, ErrV4SharedExpert) || !errors.As(err, &closed) || closed.Path != "v41-shared-add" || !errors.As(err, &backend) {
		t.Fatalf("device shared-add fault: got=%v err=%v", got != nil, err)
	}
	if !s.BackendSessionClosed() {
		t.Fatal("device shared-add fault left the backend session open")
	}
	if be.adds != 1 {
		t.Fatalf("device shared-add fault replayed: %d adds, want 1", be.adds)
	}
}
