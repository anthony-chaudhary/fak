package model

import (
	"errors"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func recordUnsanctionedHostStages(t *testing.T) *[]HostStage {
	t.Helper()
	var seen []HostStage
	prev := observeUnsanctionedHostStage
	observeUnsanctionedHostStage = func(stage HostStage) { seen = append(seen, stage) }
	t.Cleanup(func() { observeUnsanctionedHostStage = prev })
	return &seen
}

func deviceOnlyRefusal(t *testing.T, invoke func()) (err error) {
	t.Helper()
	defer func() {
		got := recover()
		if got == nil {
			t.Fatal("device-only host stage returned instead of refusing")
		}
		var ok bool
		if err, ok = got.(error); !ok {
			t.Fatalf("panic=%T %v, want error", got, got)
		}
	}()
	invoke()
	return nil
}

// fak-test:runtime fast est=2ms lane=default
func TestDeviceOnlyUnlistedHostStageRefusesBySentinel(t *testing.T) {
	for _, tc := range []struct {
		stage  HostStage
		invoke func(*Session)
	}{
		{HostStageHostMatKernel, func(s *Session) { _ = s.hostMatKernel() }},
		{HostStageHostMatmul, func(s *Session) {
			_ = sessionResidentKernel{s}.mul("model.layers.0.mlp.up_proj.weight", []float32{1}, 1, 1)
		}},
		{HostStageHostMatmul, func(s *Session) { _ = sessionQ8Kernel{s}.mul("model.layers.0.mlp.up_proj.weight", q8Vec{}, 1, 1) }},
	} {
		seen := recordUnsanctionedHostStages(t)
		s := &Session{M: &Model{}, Backend: compute.Default()}
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		err := deviceOnlyRefusal(t, func() { tc.invoke(s) })
		if !errors.Is(err, ErrDeviceOnlyHostStage) {
			t.Fatalf("%s: err=%v, want ErrDeviceOnlyHostStage", tc.stage, err)
		}
		var typed *BackendForwardOperationError
		if !errors.As(err, &typed) || typed.Stage != string(tc.stage) {
			t.Fatalf("%s: err=%#v, want BackendForwardOperationError at that stage", tc.stage, err)
		}
		if !s.HostFallbackObserved() || !s.BackendSessionClosed() {
			t.Fatalf("%s: refusal did not record and close the session", tc.stage)
		}
		if !slices.Equal(*seen, []HostStage{tc.stage}) {
			t.Fatalf("%s: counted %v, want exactly that stage", tc.stage, *seen)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDeviceOnlyLegacyRefusalCarriesSentinel(t *testing.T) {
	s := &Session{M: &Model{}, Backend: compute.Default()}
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	err := deviceOnlyRefusal(t, func() { _ = expertEngineForWeight(s, "missing.expert.weight") })
	if !errors.Is(err, ErrDeviceOnlyHostStage) {
		t.Fatalf("err=%v, want ErrDeviceOnlyHostStage", err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDeviceOnlyAllowListAdmitsOnlyMeasuredHostStages(t *testing.T) {
	want := []HostStage{HostStageEmbeddingRowUpload, HostStageLogitsReadback}
	if got := DeviceOnlyAllowedHostStages(); !slices.Equal(got, want) {
		t.Fatalf("allow-list=%v, want %v", got, want)
	}
	seen := recordUnsanctionedHostStages(t)
	s := &Session{M: &Model{}, Backend: compute.Default()}
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	for _, stage := range want {
		s.admitHostStage(stage)
	}
	if s.HostFallbackObserved() || s.BackendSessionClosed() || len(*seen) != 0 {
		t.Fatalf("sanctioned stages refused or counted: observed=%t closed=%t counted=%v", s.HostFallbackObserved(), s.BackendSessionClosed(), *seen)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDeviceOnlyBypassSessionCountsHostStagesAndReleasesOnClose(t *testing.T) {
	seen := recordUnsanctionedHostStages(t)
	portable := &Session{M: &Model{}}
	_ = portable.hostMatKernel()
	if len(*seen) != 0 {
		t.Fatalf("portable session without a bypass counted %v", *seen)
	}

	releases := 0
	s := &Session{M: &Model{}}
	s.SetHostPathBypass("device-only-session", func() { releases++ })
	_ = s.hostMatKernel()
	if s.HostFallbackObserved() || !slices.Equal(*seen, []HostStage{HostStageHostMatKernel}) {
		t.Fatalf("bypass session: observed=%t counted=%v, want counted and admitted", s.HostFallbackObserved(), *seen)
	}
	if s.HostPathBypass() != "device-only-session" {
		t.Fatalf("bypass token=%q", s.HostPathBypass())
	}
	s.Close()
	s.Close()
	if releases != 1 {
		t.Fatalf("bypass released %d times, want 1", releases)
	}
}
