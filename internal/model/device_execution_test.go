package model

import (
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type hostReadRecordingBackend struct {
	compute.Backend
	reads int
}

func (b *hostReadRecordingBackend) Read(t compute.Tensor) []float32 {
	b.reads++
	return b.Backend.Read(t)
}

func (b *hostReadRecordingBackend) Caps() compute.Caps {
	caps := b.Backend.Caps()
	caps.DeviceMemory = true
	return caps
}

type partialRoPEOnlyBackend struct{ *hostReadRecordingBackend }

func (b *partialRoPEOnlyBackend) PartialRoPEQK(q, k compute.Tensor, _, _, _, _, _ int, _ float64) (compute.Tensor, compute.Tensor) {
	return q, k
}

type qwenDeviceOnlyBackend struct{ *partialRoPEOnlyBackend }

func (*qwenDeviceOnlyBackend) SigmoidMulInPlace(_, _ compute.Tensor) {}

// fak-test:runtime fast est=2ms lane=default
func TestDeviceOnlySessionRejectsDSAHostFallbackAndCloses(t *testing.T) {
	s := &Session{M: &Model{}, Backend: compute.Default()}
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)

	assertDeviceOnlyFallback(t, s, func() {
		_, _ = (backendKernel{s: s}).indexSelect(
			[]float32{1}, []float32{1}, []float32{1},
			1, 1, 1, 0, 1, 1,
		)
	})
}

// fak-test:runtime fast est=2ms lane=default
func TestDeviceOnlySessionRejectsExpertHostFallbackAndCloses(t *testing.T) {
	s := &Session{M: &Model{}, Backend: compute.Default()}
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)

	assertDeviceOnlyFallback(t, s, func() {
		_ = expertEngineForWeight(s, "missing.expert.weight")
	})
}

// fak-test:runtime fast est=2ms lane=default
func TestPortableSessionKeepsCPUReferenceFallbacks(t *testing.T) {
	s := &Session{M: &Model{}, Backend: compute.Default()}

	selected, ok := (backendKernel{s: s}).indexSelect(
		[]float32{1}, []float32{1}, []float32{1},
		1, 1, 1, 0, 1, 1,
	)
	if !ok || len(selected) != 1 || selected[0] != 0 {
		t.Fatalf("portable CPU reference DSA = (%v,%t), want direct equivalence result", selected, ok)
	}
	if got := expertEngineForWeight(s, "missing.expert.weight"); got != expertEngineHost {
		t.Fatalf("portable missing expert engine=%v, want host", got)
	}
	if s.HostFallbackObserved() {
		t.Fatal("portable equivalence-validator fallback must not be reported as a production hidden fallback")
	}
	if s.BackendSessionClosed() {
		t.Fatal("portable equivalence-validator session was closed")
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestDeviceOnlyQwenPartialRoPERefusesBeforeHostRead(t *testing.T) {
	backend := &hostReadRecordingBackend{Backend: compute.Default()}
	s := deviceOnlyQwenSession(backend, false)

	assertDeviceOnlyFallback(t, s, func() { s.Prefill([]int{1}) })
	if backend.reads != 0 {
		t.Fatalf("device-only partial RoPE performed %d host reads before refusal", backend.reads)
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestDeviceOnlyQwenSigmoidGateRefusesBeforeHostRead(t *testing.T) {
	reads := &hostReadRecordingBackend{Backend: compute.Default()}
	backend := &partialRoPEOnlyBackend{hostReadRecordingBackend: reads}
	s := deviceOnlyQwenSession(backend, true)

	assertDeviceOnlyFallback(t, s, func() { s.PrefillNoLogits([]int{1}) })
	if reads.reads != 0 {
		t.Fatalf("device-only sigmoid gate performed %d host reads before refusal", reads.reads)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDeviceOnlyQwenDeviceKernelsPassAdmission(t *testing.T) {
	reads := &hostReadRecordingBackend{Backend: compute.Default()}
	backend := &qwenDeviceOnlyBackend{partialRoPEOnlyBackend: &partialRoPEOnlyBackend{hostReadRecordingBackend: reads}}
	s := deviceOnlyQwenSession(backend, true)

	s.validateDeviceOnlyExecution("test admission")
	if reads.reads != 0 || s.HostFallbackObserved() || s.BackendSessionClosed() {
		t.Fatalf("device-capable admission reads=%d fallback=%t closed=%t", reads.reads, s.HostFallbackObserved(), s.BackendSessionClosed())
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestDeviceOnlyExecutionPolicySurvivesPrefixCloneAndRestore(t *testing.T) {
	s := &Session{M: &Model{}, Cache: NewKVCache(Config{})}
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	snapshot, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatalf("PrefixSnapshot: %v", err)
	}
	defer snapshot.Close()
	clone, err := snapshot.Clone()
	if err != nil {
		t.Fatalf("Clone: %v", err)
	}
	defer clone.Close()

	target := &Session{M: &Model{}, Cache: NewKVCache(Config{})}
	if err := clone.Restore(target); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if target.executionPolicy != ExecutionPolicyDeviceOnly {
		t.Fatalf("restored policy=%v, want device-only", target.executionPolicy)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestDeviceOnlyEntryRefusalClosesWithoutCacheGeometryDeadlock(t *testing.T) {
	s := &Session{M: &Model{}, Cache: NewKVCache(Config{})}
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	done := make(chan any, 1)
	go func() {
		defer func() { done <- recover() }()
		s.Step(1)
	}()

	select {
	case got := <-done:
		var typed *BackendForwardOperationError
		err, ok := got.(error)
		if got == nil || !ok || !errors.As(err, &typed) {
			t.Fatalf("panic=%T %v, want *BackendForwardOperationError", got, got)
		}
		if !s.BackendSessionClosed() {
			t.Fatal("nil-backend device-only refusal did not retire the session")
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("device-only refusal deadlocked while closing around cache geometry")
	}
}

func deviceOnlyQwenSession(backend compute.Backend, outputGate bool) *Session {
	cfg := qwen35HybridTestCfg()
	cfg.PartialRotaryFactor = 0.5
	cfg.AttnOutputGate = outputGate
	s := &Session{M: &Model{Cfg: cfg}, Cache: NewKVCache(cfg), Backend: backend}
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	return s
}

func assertDeviceOnlyFallback(t *testing.T, s *Session, invoke func()) {
	t.Helper()
	defer func() {
		got := recover()
		if got == nil {
			t.Fatal("device-only fallback returned instead of refusing")
		}
		var typed *BackendForwardOperationError
		switch value := got.(type) {
		case error:
			if !errors.As(value, &typed) {
				t.Fatalf("panic=%T %v, want *BackendForwardOperationError", got, got)
			}
		default:
			t.Fatalf("panic=%T %v, want error", got, got)
		}
		if !s.HostFallbackObserved() {
			t.Fatal("device-only refusal did not record attempted host fallback")
		}
		if !s.BackendSessionClosed() {
			t.Fatal("device-only refusal did not close the backend session")
		}
	}()
	invoke()
}
