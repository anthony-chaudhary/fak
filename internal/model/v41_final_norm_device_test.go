package model

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41FinalNormTestBackend struct {
	*v41DenseTestBackend
	norms, readsAfterNorm, opsAtFailure int
	failNorm                            error
	normalized                          map[compute.Buffer]bool
}

func (b *v41FinalNormTestBackend) RMSNorm(x, weight compute.Tensor, eps float32) compute.Tensor {
	b.norms++
	if b.failNorm != nil {
		b.opsAtFailure = len(b.ops)
		panic(b.failNorm)
	}
	y := b.Backend.RMSNorm(x, weight, eps)
	b.live[y.Buf()] = true
	b.normalized[y.Buf()] = true
	return y
}

func (b *v41FinalNormTestBackend) Read(x compute.Tensor) []float32 {
	if b.normalized[x.Buf()] {
		b.readsAfterNorm++
	}
	return b.v41DenseTestBackend.Read(x)
}

func (b *v41FinalNormTestBackend) Free(x compute.Tensor) {
	delete(b.normalized, x.Buf())
	b.v41DenseTestBackend.Free(x)
}

// The recorder delegates to cpu-ref. This witnesses dispatch and final-norm
// semantics only; physical device execution and performance remain unqualified.
// fak-test:runtime fast est=500ms lane=default
func TestV41FinalNormDeviceDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("final-norm dispatch fixture requires cpu-ref")
	}
	m := v41IncrementalPlainModel(t, 1)
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	newSession := func(t *testing.T) (*Session, *v41FinalNormTestBackend) {
		b := &v41FinalNormTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), normalized: map[compute.Buffer]bool{}}
		s := v41DenseTestSession(t, m, b)
		if s.v41State().finalNorm == nil {
			t.Fatal("device session did not bind final RMSNorm")
		}
		return s, b
	}
	t.Run("prefill-decode-and-strict-refusal", func(t *testing.T) {
		s, b := newSession(t)
		v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-5)
		if b.norms != 2 || b.readsAfterNorm != 2 {
			t.Fatalf("prefill norms/readbacks=%d/%d, want 2/2", b.norms, b.readsAfterNorm)
		}
		if !s.v41IncrementalEligible() {
			t.Fatal("fixture did not seed incremental decode")
		}
		v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-5)
		if b.norms != 3 || b.readsAfterNorm != 3 || len(b.normalized) != 0 {
			t.Fatalf("decode norms/readbacks/live results=%d/%d/%d", b.norms, b.readsAfterNorm, len(b.normalized))
		}
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		err := recoverError(func() { s.Step(4) })
		var refused *BackendForwardOperationError
		if !errors.As(err, &refused) || refused.Stage != "decode: architecture uses host model compute" || b.norms != 3 {
			t.Fatalf("strict architecture guard changed: norms=%d err=%v", b.norms, err)
		}
	})
	for _, tc := range []struct {
		name         string
		err          error
		unclassified bool
	}{
		{"vulkan", &compute.BackendError{Backend: "cpu-ref", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}, false},
		{"cuda", &compute.CUDAOpError{Op: "RMSNorm", Err: ErrV41ForwardStage}, false},
		{"wrapped-cuda", fmt.Errorf("norm dispatch: %w", &compute.CUDAOpError{Op: "RMSNorm", Err: ErrV41ForwardStage}), false},
		{"allocation", &compute.DeviceAllocError{Bytes: 64, Site: "rmsnorm", Class: compute.MemoryScratchpad}, false},
		{"fault", &compute.DeviceFaultError{Backend: "cuda", Site: "rmsnorm"}, false},
		{"unclassified", fmt.Errorf("rocm: rmsnorm: native call failed"), true},
	} {
		t.Run("selected-failure-"+tc.name, func(t *testing.T) {
			s, b := newSession(t)
			s.Prefill([]int{1, 2})
			before := captureV41ForwardSnapshot(s.v41Forward)
			b.live = map[compute.Buffer]bool{}
			b.failNorm = tc.err
			err := recoverError(func() { s.Step(3) })
			var operation *V41ProjectionOperationError
			var closed *BackendForwardOperationError
			if tc.unclassified {
				if err != tc.err {
					t.Fatalf("unclassified panic identity changed: %v", err)
				}
				if !errors.As(s.halFailure, &closed) {
					t.Fatalf("unclassified failure left no closed-session error: %v", s.halFailure)
				}
			} else if !errors.As(err, &operation) || operation.Stage != string(v41StageFinalNorm) || !errors.As(err, &closed) {
				t.Fatalf("selected final-norm failure lost identity: %v", err)
			}
			if closed == nil || closed.Path != "v41-final-norm" || closed.Stage != "rmsnorm" || !errors.Is(closed, tc.err) {
				t.Fatalf("session did not retain selected failure: %v", closed)
			}
			if b.norms != 3 || len(b.ops) != b.opsAtFailure || !s.BackendSessionClosed() || len(b.live) != 0 {
				t.Fatalf("failed norm retried, projected, or leaked: norms=%d ops=%d/%d closed=%t live=%d", b.norms, len(b.ops), b.opsAtFailure, s.BackendSessionClosed(), len(b.live))
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
				t.Fatal("selected final-norm failure advanced continuation state")
			}
			if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || b.norms != 3 {
				t.Fatalf("closed session retried final norm: norms=%d err=%v", b.norms, retry)
			}
		})
	}
}
