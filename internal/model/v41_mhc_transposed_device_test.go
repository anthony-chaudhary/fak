package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41MHCTransposeBackend struct {
	*v41MHCProjBackend
	packed []float32
	poison bool
}

func (b *v41MHCTransposeBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	if dt == compute.F32 && len(x.Shape) == 2 && x.Shape[0] == v41MHCMixWidth && x.Shape[1] == b.in {
		b.packed = append([]float32(nil), b.Backend.Read(x)...)
	}
	return b.v41MHCProjBackend.Upload(x, dt)
}

func (b *v41MHCTransposeBackend) Free(x compute.Tensor) {
	if _, output := b.outputs[x.Buf()]; output && b.poison {
		values := b.Backend.Read(x)
		for i := range values {
			values[i] = float32(math.NaN())
		}
	}
	b.v41MHCProjBackend.Free(x)
}

// The fake device delegates to cpu-ref; this test is a dispatch/lifetime
// contract, not a physical-device qualification or a performance receipt.
// fak-test:runtime medium est=4s lane=default
func TestV41MHCTransposedDeviceDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("transposed mHC recorder requires cpu-ref")
	}
	m := v41MHCProjVariant(t, "F32", true, false, 1)
	b := &v41MHCTransposeBackend{v41MHCProjBackend: newV41MHCProjBackend(4*m.Cfg.HiddenSize, false), poison: true}
	s := v41EngProjSession(t, m, b)
	name := layerName(0, "mhc.mixes.weight")
	stored := append([]float32(nil), m.tensor(name)...)
	for i, ids := range [][]int{{1, 2}, {3}} {
		var got []float32
		if i == 0 {
			got = s.Prefill(ids)
		} else {
			if !s.v41IncrementalEligible() {
				t.Fatal("transposed fixture did not seed incremental decode")
			}
			got = s.Step(ids[0])
		}
		prefix := []int{1, 2}
		if i != 0 {
			prefix = append(prefix, 3)
		}
		v41GroupedParity(t, got, lastLogits(m.Forward(prefix)), 1e-4)
	}
	if b.stages != 1 || b.attempts != 3 || len(b.operations) != 3 || len(b.packed) != len(stored) {
		t.Fatalf("transposed immutable staging/dispatch=%d/%d/%d packed=%d", b.stages, b.attempts, len(b.operations), len(b.packed))
	}
	for out := 0; out < v41MHCMixWidth; out++ {
		for in := 0; in < b.in; in++ {
			if b.packed[out*b.in+in] != stored[in*v41MHCMixWidth+out] {
				t.Fatal("device weight was not reoriented from input-major storage")
			}
		}
	}
	if !reflect.DeepEqual(m.tensor(name), stored) {
		t.Fatal("reorientation mutated the model's weight storage")
	}
	for _, op := range b.operations {
		if op.upload != 4*b.in || op.read != 4*v41MHCMixWidth {
			t.Fatalf("transposed projection transfers=%d/%d", op.upload, op.read)
		}
		for out, value := range op.result {
			var want float64
			for in, activation := range op.activation {
				want += float64(stored[in*v41MHCMixWidth+out]) * float64(activation)
			}
			if math.Abs(float64(value)-want) > 1e-4*math.Max(1, math.Abs(want)) {
				t.Fatalf("raw transposed dot[%d]=%g want=%g", out, value, want)
			}
		}
	}
	before := b.attempts
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	var refused *BackendForwardOperationError
	if err := recoverError(func() { s.Step(4) }); !errors.As(err, &refused) || refused.Stage != "decode: architecture uses host model compute" || b.attempts != before {
		t.Fatalf("whole-architecture refusal changed: attempts=%d/%d err=%v", b.attempts, before, err)
	}
}

// fak-test:runtime medium est=3s lane=default
func TestV41MHCTransposedSelectedFailure(t *testing.T) {
	for _, fault := range []error{
		&compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage},
		&compute.CUDAOpError{Op: "MatMul", Err: ErrV41ForwardStage},
	} {
		m := v41MHCProjVariant(t, "F32", true, false, 2)
		b := newV41MHCProjBackend(4*m.Cfg.HiddenSize, false)
		s := v41EngProjSession(t, m, b)
		s.Prefill([]int{1, 2})
		before := captureV41ForwardSnapshot(s.v41Forward)
		b.live = map[compute.Buffer]bool{}
		b.fault, b.site, b.cause = true, "matmul", fault
		err := recoverError(func() { s.Step(3) })
		var closed *BackendForwardOperationError
		if !errors.As(err, &closed) || closed.Path != "v41-mhc-projection" || closed.Layer != 1 || closed.Stage != "matmul" || !errors.Is(err, fault) {
			t.Fatalf("selected transposed failure lost its identity: %v", err)
		}
		if b.faultAttempts != 2 || !s.BackendSessionClosed() || len(b.live) != 0 || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
			t.Fatalf("selected transposed failure retried, leaked, or advanced state: attempts=%d closed=%t live=%d", b.faultAttempts, s.BackendSessionClosed(), len(b.live))
		}
		if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || b.faultAttempts != 2 {
			t.Fatalf("closed transposed session retried: %v", retry)
		}
	}
}
