package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41SharedActivationRecorder struct {
	*v41DenseTestBackend
	outputs                                           map[compute.Buffer]bool
	calls, limited, readbacks, uploadBytes, readBytes int
	opsAtFault                                        int
	fault                                             error
	badRead                                           bool
}

func newV41SharedActivationRecorder() *v41SharedActivationRecorder {
	return &v41SharedActivationRecorder{v41DenseTestBackend: newV41DenseTestBackend(), outputs: map[compute.Buffer]bool{}}
}

func (b *v41SharedActivationRecorder) activate(g, u compute.Tensor, limit float32) compute.Tensor {
	// Shared activation receives two uploaded host rows. Routed activations
	// receive MatMul outputs; those calls must not inflate this witness.
	shared := b.uploads[g.Buf()] > 0 && b.uploads[u.Buf()] > 0
	if shared {
		b.calls++
		b.uploadBytes += b.uploads[g.Buf()] + b.uploads[u.Buf()]
		if limit > 0 {
			b.limited++
		}
		if b.fault != nil {
			b.opsAtFault = len(b.ops)
			panic(b.fault)
		}
	}
	gate, up := b.Backend.Read(g), b.Backend.Read(u)
	values := make([]float32, len(gate))
	for i := range values {
		a, c := gate[i], up[i]
		if limit > 0 {
			a = min(a, limit)
			c = max(-limit, min(c, limit))
		}
		values[i] = a / (1 + float32(math.Exp(float64(-a)))) * c
	}
	y := compute.NewF32(b, append([]int(nil), g.Shape...), values)
	b.live[y.Buf()] = true
	if shared {
		b.outputs[y.Buf()] = true
	}
	return y
}

func (b *v41SharedActivationRecorder) SwiGLU(g, u compute.Tensor) compute.Tensor {
	return b.activate(g, u, 0)
}

func (b *v41SharedActivationRecorder) Read(x compute.Tensor) []float32 {
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

func (b *v41SharedActivationRecorder) Free(x compute.Tensor) {
	if b.outputs[x.Buf()] {
		values := b.Backend.Read(x)
		for i := range values {
			values[i] = float32(math.NaN())
		}
		delete(b.outputs, x.Buf())
	}
	b.v41DenseTestBackend.Free(x)
}

type v41SharedLimitedRecorder struct{ *v41SharedActivationRecorder }

func (b *v41SharedLimitedRecorder) SwiGLUWithLimit(g, u compute.Tensor, limit float32) compute.Tensor {
	return b.activate(g, u, limit)
}

// Software dispatch, semantic and lifetime assertions only. The recorder uses
// scalar host arithmetic, and cannot establish hardware execution or net benefit.
// fak-test:runtime medium est=10s lane=default
func TestV41SharedActivationDeviceDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("shared activation fixture requires cpu-ref")
	}
	for _, role := range []bool{false, true} {
		for _, scenario := range []struct {
			capable bool
			limit   float64
		}{{false, .01}, {true, .01}, {false, 0}} {
			capable := scenario.capable
			name := "plain"
			if role {
				name = "role"
			}
			if scenario.limit == 0 {
				name += "/zero-limit-device"
			} else if capable {
				name += "/device"
			} else {
				name += "/declined"
			}
			t.Run(name, func(t *testing.T) {
				m := v41IncrementalExpertFixture(t, role, false)
				m.Cfg.SwigluLimit = scenario.limit
				b := newV41SharedActivationRecorder()
				var backend compute.Backend = b
				if capable {
					backend = &v41SharedLimitedRecorder{b}
				}
				s := v41DenseTestSession(t, m, backend)
				v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-4)
				if !s.v41IncrementalEligible() {
					t.Fatal("shared activation fixture did not seed incremental state")
				}
				v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-4)
				want := 0
				if capable || scenario.limit == 0 {
					want = 3 * m.Cfg.NumLayers
				}
				wantLimited := want
				if scenario.limit == 0 {
					wantLimited = 0
				}
				if b.calls != want || b.limited != wantLimited || b.readbacks != want || b.uploadBytes != 8*want*m.Cfg.MoEIntermediateSize || b.readBytes != 4*want*m.Cfg.MoEIntermediateSize || len(b.outputs) != 0 {
					t.Fatalf("shared activation calls/limited/readbacks=%d/%d/%d want=%d upload/read=%d/%d outputs=%d", b.calls, b.limited, b.readbacks, want, b.uploadBytes, b.readBytes, len(b.outputs))
				}
				s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
				var refused *BackendForwardOperationError
				if err := recoverError(func() { s.Step(4) }); !errors.As(err, &refused) || refused.Stage != "decode: architecture uses host model compute" || b.calls != want {
					t.Fatalf("strict architecture refusal changed: calls=%d err=%v", b.calls, err)
				}
			})
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41SharedActivationClampAndOwnership(t *testing.T) {
	for _, limit := range []float32{0, -2, 2} {
		b := &v41SharedLimitedRecorder{newV41SharedActivationRecorder()}
		s := &Session{M: &Model{Cfg: Config{MoEIntermediateSize: 4}}, Backend: b, halW: map[string]compute.Tensor{}}
		gate, up := []float32{4, -4, 1, -1}, []float32{-4, 4, 3, -3}
		gateBefore, upBefore := append([]float32(nil), gate...), append([]float32(nil), up...)
		got, outcome, err := s.v41SharedActivationFunc()(0, gate, up, limit)
		if err != nil || outcome != v41ProjectionHandled || len(got) != 4 {
			t.Fatalf("shared activation outcome=%v err=%v", outcome, err)
		}
		for i := range gate {
			g, u := gate[i], up[i]
			if limit > 0 {
				g, u = min(g, limit), max(-limit, min(u, limit))
			}
			want := g / (1 + float32(math.Exp(float64(-g)))) * u
			if !finite32(got[i]) || math.Abs(float64(got[i]-want)) > 1e-6 {
				t.Fatalf("shared activation[%d]=%g want=%g", i, got[i], want)
			}
		}
		if !reflect.DeepEqual(gate, gateBefore) || !reflect.DeepEqual(up, upBefore) || len(b.live) != 0 {
			t.Fatal("shared activation mutated operands or leaked transient tensors")
		}
		s.Close()
	}
}

// fak-test:runtime medium est=6s lane=default
func TestV41SharedActivationSelectedFailure(t *testing.T) {
	for _, fault := range []error{
		&compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage},
		&compute.CUDAOpError{Op: "SwiGLU", Err: ErrV41ForwardStage},
		nil,
	} {
		m := v41IncrementalExpertFixture(t, true, false)
		m.Cfg.SwigluLimit = .01
		b := &v41SharedLimitedRecorder{newV41SharedActivationRecorder()}
		s := v41DenseTestSession(t, m, b)
		s.Prefill([]int{1, 2})
		before := captureV41ForwardSnapshot(s.v41Forward)
		b.live = map[compute.Buffer]bool{}
		calls := b.calls
		b.fault, b.badRead = fault, fault == nil
		err := recoverError(func() { s.Step(3) })
		var operation *V41ProjectionOperationError
		var closed *BackendForwardOperationError
		stage := "limited swiglu"
		if fault == nil {
			stage = "readback"
		}
		if !errors.As(err, &operation) || operation.Stage != string(v41StageMoE) || !errors.As(err, &closed) || closed.Path != "v41-shared-activation" || closed.Stage != stage {
			t.Fatalf("selected shared activation lost failure identity: %v", err)
		}
		if fault != nil && !errors.Is(err, fault) {
			t.Fatal("selected shared activation lost backend cause")
		}
		if b.calls != calls+1 || len(b.ops) != b.opsAtFault || !s.BackendSessionClosed() || len(b.live) != 0 || len(b.outputs) != 0 || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
			t.Fatalf("shared failure retried, projected down, leaked, or advanced state: calls=%d/%d ops=%d/%d live=%d", b.calls, calls, len(b.ops), b.opsAtFault, len(b.live))
		}
		if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || b.calls != calls+1 {
			t.Fatalf("closed shared-activation session retried: %v", retry)
		}
	}
}
