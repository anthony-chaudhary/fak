package model

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41InverseRowsCall struct {
	x, table                 []float32
	rows, headDim, rotaryDim int
}

// This recorder rotates through the independent scalar oracle over cpu-ref
// storage. Its DeviceMemory promise tests routing only, never silicon.
type v41InverseRowsTestBackend struct {
	*v41DenseTestBackend
	calls    []v41InverseRowsCall
	outputs  map[compute.Buffer]bool
	fault    string
	failCall int
	cause    error
}

func newV41InverseRowsTestBackend() *v41InverseRowsTestBackend {
	return &v41InverseRowsTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), outputs: map[compute.Buffer]bool{}}
}

func (b *v41InverseRowsTestBackend) V41TailRoPERowsAvailable() bool { return true }

func (b *v41InverseRowsTestBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if !strings.HasPrefix(site, "V4.1 inverse-RoPE ") {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if class != compute.MemoryActivation || dt != compute.F32 {
		panic("inverse-RoPE upload is not a classed F32 activation")
	}
	return compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
}

func (b *v41InverseRowsTestBackend) V41TailRoPERows(x, table compute.Tensor, rows, headDim, rotaryDim int) (compute.Tensor, error) {
	if x.Backend() != b || table.Backend() != b || !reflect.DeepEqual(x.Shape, []int{rows, headDim}) || !reflect.DeepEqual(table.Shape, []int{rotaryDim / 2, 2}) {
		return compute.Tensor{}, errors.New("invalid inverse-RoPE rows contract")
	}
	call := v41InverseRowsCall{x: append([]float32(nil), b.Backend.Read(x)...), table: append([]float32(nil), b.Backend.Read(table)...), rows: rows, headDim: headDim, rotaryDim: rotaryDim}
	b.calls = append(b.calls, call)
	if len(b.calls) == b.failCall && b.fault == "dispatch" {
		return compute.Tensor{}, b.cause
	}
	out := compute.NewF32(b, []int{rows, headDim}, v41TailRoPEExactOracle(call.x, call.table, headDim, rotaryDim))
	b.outputs[out.Buf()] = true
	if len(b.calls) == b.failCall && b.fault == "prefix" {
		values := b.Backend.Read(out)
		values[0] = math.Float32frombits(math.Float32bits(values[0]) ^ 1)
	}
	return out, nil
}

func (b *v41InverseRowsTestBackend) Free(x compute.Tensor) {
	delete(b.outputs, x.Buf())
	b.v41DenseTestBackend.Free(x)
}

func v41InverseOutputOnly(s *Session) {
	st := s.v41State()
	st.expertGateUp, st.expertDown = nil, nil
	st.denseProjection, st.groupedOutput, st.engramProjection = nil, nil, nil
	st.mhcProjection, st.mhcFFNProjection, st.finalNorm, st.queryNorm, st.kvNorm = nil, nil, nil, nil, nil
	st.ffnNorm, st.sharedActivation = nil, nil
}

// Refs #13668. The inverse output rotation leaves the host: the selected seam
// sends the conjugate of the exact query table, matches the host inverse bit for
// bit in both geometries, and fails closed without a host retry.
// fak-test:runtime medium est=5s lane=default
func TestV41InverseOutputDeviceDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("inverse-output device witness requires cpu-ref")
	}

	t.Run("site-parity", func(t *testing.T) {
		for _, full := range []bool{false, true} {
			cfg := Config{HeadDim: 8, NumHeads: 3, QKRopeHeadDim: 4, RopeTheta: 256}
			heads, hd := 3, 8
			if full {
				cfg = Config{HeadDim: v41KVLoraRank, NumHeads: 2, QKRopeHeadDim: 64, RopeTheta: 256}
				heads, hd = 2, v41KVLoraRank
			}
			b := newV41InverseRowsTestBackend()
			s := &Session{M: &Model{}, Backend: b}
			t.Cleanup(s.Close)
			ops := s.v41DeviceGatherOpsFunc()
			if ops.inverseOutputFunc() == nil {
				t.Fatal("available rows operation was not bound")
			}
			out := make([]float32, heads*hd)
			for i := range out {
				out[i] = float32((i%23)-11)*0.0625 + float32(i)*0.001
			}
			out[0] = math.Float32frombits(0x80000000)
			before := append([]float32(nil), out...)
			want, err := v41AttentionOutputForProjection(cfg, 1, 5, append([]float32(nil), out...), heads, hd)
			if err != nil {
				t.Fatal(err)
			}
			got, err := v41RotatedOutputForProjection(cfg, 1, 5, out, heads, hd, ops)
			if err != nil {
				t.Fatal(err)
			}
			v41TailRoPECheckBits(t, "device inverse", got, want)
			if len(b.calls) != 1 || b.calls[0].rows != heads || b.calls[0].headDim != hd || b.calls[0].rotaryDim != cfg.QKRopeHeadDim {
				t.Fatalf("calls=%+v", b.calls)
			}
			cos, sin := v41RopeTableForLayer(cfg, 1, 5)
			conj := make([]float32, 2*len(cos))
			for j := range cos {
				conj[2*j], conj[2*j+1] = -sin[j], cos[j]
			}
			v41TailRoPECheckBits(t, "conjugate table", b.calls[0].table, conj)
			if full {
				v41TailRoPECheckBits(t, "unchanged full input", out, before)
				for _, v := range b.calls[0].x {
					if math.Float32bits(v)&0xffff != 0 {
						t.Fatal("full inverse operand was not BF16-staged")
					}
				}
			}
			if len(b.outputs) != 0 {
				t.Fatal("selected inverse leaked its device output")
			}
		}
	})

	t.Run("forward-routes-every-layer-position", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 2)
		b := newV41InverseRowsTestBackend()
		s := v41DenseTestSession(t, m, b)
		v41InverseOutputOnly(s)
		if s.v41State().gatherOps.inverseOutputFunc() == nil {
			t.Fatal("session did not bind the inverse output seam")
		}
		v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-4)
		if len(b.calls) != 2*m.Cfg.NumLayers {
			t.Fatalf("cold prefill inverse calls=%d want %d", len(b.calls), 2*m.Cfg.NumLayers)
		}
		v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-4)
		if len(b.calls) != 3*m.Cfg.NumLayers {
			t.Fatalf("step inverse calls=%d want %d", len(b.calls), 3*m.Cfg.NumLayers)
		}
	})

	for _, fault := range []string{"dispatch", "prefix"} {
		t.Run("selected-failure-"+fault, func(t *testing.T) {
			m := v41IncrementalPlainModel(t, 2)
			b := newV41InverseRowsTestBackend()
			s := v41DenseTestSession(t, m, b)
			v41InverseOutputOnly(s)
			s.Prefill([]int{1, 2})
			b.fault, b.failCall = fault, len(b.calls)+1
			b.cause = &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			err := recoverError(func() { s.Step(3) })
			var named *V41TailRoPEOperationError
			var closed *BackendForwardOperationError
			if !errors.As(err, &named) || !errors.As(err, &closed) || closed.Path != "v41-inverse-output-rope" || !s.BackendSessionClosed() {
				t.Fatalf("selected inverse failure lost identity: %v", err)
			}
			if fault == "dispatch" && !errors.Is(err, b.cause) {
				t.Fatalf("selected inverse failure lost its cause: %v", err)
			}
			if len(b.calls) != b.failCall || len(b.outputs) != 0 {
				t.Fatal("selected inverse failure replayed or leaked")
			}
			if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || len(b.calls) != b.failCall {
				t.Fatalf("closed session retried the inverse rotation: %v", retry)
			}
		})
	}

	t.Run("decline-keeps-host-and-device-only-refuses", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		s := v41DenseTestSession(t, m, newV41DenseTestBackend())
		ops := s.v41State().gatherOps
		if ops == nil || ops.inverseOutput != nil || ops.inverseOutputFunc() != nil {
			t.Fatal("a backend without the rows operation selected a device inverse")
		}
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		refuse := ops.inverseOutputFunc()
		if refuse == nil {
			t.Fatal("device-only session kept the host inverse arm")
		}
		cfg := Config{HeadDim: 8, NumHeads: 1, QKRopeHeadDim: 4, RopeTheta: 256}
		out := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		got, err := v41RotatedOutputForProjection(cfg, 0, 1, out, 1, 8, ops)
		var closed *BackendForwardOperationError
		if got != nil || !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &closed) || closed.Path != "device-only" || !s.BackendSessionClosed() {
			t.Fatalf("device-only missing inverse did not fail closed: %v", err)
		}
		if !reflect.DeepEqual(out, []float32{1, 2, 3, 4, 5, 6, 7, 8}) {
			t.Fatal("device-only refusal computed on the host")
		}
	})
}
