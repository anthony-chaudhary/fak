package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41QueryNormTestBackend struct {
	*v41DenseTestBackend
	queryWeight func() compute.Tensor
	outputs     map[compute.Buffer]bool
	norms       int
	readbacks   int
	uploadBytes int
	readBytes   int
	fault       error
	badRead     bool
	opsAtFault  int
	input, gain []float32
}

func (b *v41QueryNormTestBackend) RMSNorm(x, weight compute.Tensor, eps float32) compute.Tensor {
	query := weight.Buf() == b.queryWeight().Buf()
	if query {
		b.norms++
		b.uploadBytes += b.uploads[x.Buf()]
		b.input = append([]float32(nil), b.Backend.Read(x)...)
		b.gain = append([]float32(nil), b.Backend.Read(weight)...)
		if b.fault != nil {
			b.opsAtFault = len(b.ops)
			panic(b.fault)
		}
	}
	y := b.Backend.RMSNorm(x, weight, eps)
	b.live[y.Buf()] = true
	if query {
		b.outputs[y.Buf()] = true
	}
	return y
}

func (b *v41QueryNormTestBackend) Read(x compute.Tensor) []float32 {
	values := b.v41DenseTestBackend.Read(x)
	if b.outputs[x.Buf()] {
		b.readbacks++
		b.readBytes += 4 * len(values)
		if b.badRead {
			b.opsAtFault = len(b.ops)
			return values[:len(values)-1]
		}
	}
	return values
}

func (b *v41QueryNormTestBackend) Free(x compute.Tensor) {
	if b.outputs[x.Buf()] {
		// Poison backend-owned storage: the production readback must be retained
		// before Free rather than returning a slice into the released allocation.
		values := b.Backend.Read(x)
		for i := range values {
			values[i] = float32(math.NaN())
		}
		delete(b.outputs, x.Buf())
	}
	b.v41DenseTestBackend.Free(x)
}

// This recorder uses cpu-ref arithmetic. It establishes dispatch, operands,
// transfers, and cleanup only; it is not a hardware or throughput witness.
// fak-test:runtime medium est=5s lane=default
func TestV41QueryNormDeviceDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("query-norm dispatch fixture requires cpu-ref")
	}
	newSession := func(t *testing.T, m *Model) (*Session, *v41QueryNormTestBackend) {
		b := &v41QueryNormTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), outputs: map[compute.Buffer]bool{}}
		s := v41DenseTestSession(t, m, b)
		b.queryWeight = func() compute.Tensor { return s.halW[layerName(0, "attn.wq_a_norm.weight")] }
		if s.v41State().queryNorm == nil {
			t.Fatal("device session did not bind query RMSNorm")
		}
		return s, b
	}
	for _, role := range []bool{false, true} {
		name := "plain"
		if role {
			name = "compressed-role"
		}
		t.Run(name, func(t *testing.T) {
			var m *Model
			if role {
				m = v41IncrementalExpertFixture(t, true, false)
			} else {
				m = v41FullStepPatchedNorms(t)
			}
			s, b := newSession(t, m)
			v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-4)
			if !s.v41IncrementalEligible() || b.norms != 2 || b.readbacks != 2 {
				t.Fatalf("prefill query calls/readbacks=%d/%d incremental=%t", b.norms, b.readbacks, s.v41IncrementalEligible())
			}
			v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-4)
			if b.norms != 3 || b.readbacks != 3 || len(b.outputs) != 0 || b.uploadBytes != 3*4*m.Cfg.QLoraRank || b.readBytes != b.uploadBytes {
				t.Fatalf("query dispatch/transfer/cleanup mismatch: norms=%d reads=%d outputs=%d upload=%d read=%d", b.norms, b.readbacks, len(b.outputs), b.uploadBytes, b.readBytes)
			}
			if !reflect.DeepEqual(b.gain, m.tensor(layerName(0, "attn.wq_a_norm.weight"))) {
				t.Fatal("query norm did not consume the learned gain")
			}
			// Direct output parity is non-vacuous even when later attention hides
			// a normalization error. Free poisoning also exercises owned readback.
			input := append([]float32(nil), b.input...)
			got, err := s.v41State().queryNorm(0, input)
			if err != nil {
				t.Fatal(err)
			}
			v41GroupedParity(t, got, cpuOracleRMSNorm(input, b.gain, float32(m.Cfg.RMSNormEps)), 1e-5)
			if !reflect.DeepEqual(input, b.input) {
				t.Fatal("query norm mutated its host input")
			}
			beforeNorms := b.norms
			s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
			var refused *BackendForwardOperationError
			if err := recoverError(func() { s.Step(4) }); !errors.As(err, &refused) || refused.Stage != "decode: architecture uses host model compute" || b.norms != beforeNorms {
				t.Fatalf("strict architecture guard changed: calls=%d/%d err=%v", b.norms, beforeNorms, err)
			}
		})
	}
	for _, tc := range []struct {
		name    string
		fault   error
		badRead bool
	}{
		{"selected-vulkan", &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}, false},
		{"selected-cuda", &compute.CUDAOpError{Op: "RMSNorm", Err: ErrV41ForwardStage}, false},
		{"invalid-readback", nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := v41FullStepPatchedNorms(t)
			s, b := newSession(t, m)
			s.Prefill([]int{1, 2})
			before := captureV41ForwardSnapshot(s.v41Forward)
			b.live = map[compute.Buffer]bool{}
			b.fault, b.badRead = tc.fault, tc.badRead
			err := recoverError(func() { s.Step(3) })
			var selected *V41ProjectionOperationError
			var closed *BackendForwardOperationError
			stage := "rmsnorm"
			if tc.badRead {
				stage = "readback"
			}
			if !errors.As(err, &selected) || selected.Leaf != "attn.wq_a_norm.weight" || selected.Stage != string(v41StageAttention) || !errors.As(err, &closed) || closed.Path != "v41-query-norm" || closed.Layer != 0 || closed.Stage != stage {
				t.Fatalf("selected query-norm failure lost identity: %v", err)
			}
			if tc.fault != nil && !errors.Is(err, tc.fault) {
				t.Fatalf("query-norm failure lost its cause: %v", err)
			}
			if b.norms != 3 || len(b.ops) != b.opsAtFault || !s.BackendSessionClosed() || len(b.live) != 0 || len(b.outputs) != 0 {
				t.Fatalf("query failure retried, projected, or leaked: norms=%d ops=%d/%d closed=%t live=%d outputs=%d", b.norms, len(b.ops), b.opsAtFault, s.BackendSessionClosed(), len(b.live), len(b.outputs))
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
				t.Fatal("failed query norm advanced continuation state")
			}
			if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || b.norms != 3 {
				t.Fatalf("closed session retried query norm: norms=%d err=%v", b.norms, retry)
			}
		})
	}
}
