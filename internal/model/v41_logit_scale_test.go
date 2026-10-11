package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CPU storage witnesses only routing, exact scalar operation order, and
// ownership. The recorder's DeviceMemory claim is not a physical-device test.
type v41LogitScaleRecorder struct {
	*v41DenseTestBackend
	scaleLive                    map[compute.Buffer]bool
	calls, uploads, reads, frees int
	lastInput, borrowed          []float32
	lastScale                    float32
	fault                        error
	panicFault, closePanic       any
	closeCalls                   int
	cleanup                      any
	faultAt                      int
	opsAtFailure                 int
}

func newV41LogitScaleRecorder() *v41LogitScaleRecorder {
	return &v41LogitScaleRecorder{v41DenseTestBackend: newV41DenseTestBackend(), scaleLive: map[compute.Buffer]bool{}}
}
func (b *v41LogitScaleRecorder) V41LogitScaleAvailable() bool { return true }
func (b *v41LogitScaleRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site != "V4.1 logit scale input" {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if dt != compute.F32 || class != compute.MemoryActivation {
		panic("logit scale upload lost activation class")
	}
	b.uploads++
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	b.scaleLive[out.Buf()] = true
	return out
}
func (b *v41LogitScaleRecorder) V41LogitScale(input compute.Tensor, scale float32) (compute.Tensor, error) {
	b.calls++
	if b.panicFault != nil && b.calls == b.faultAt {
		b.opsAtFailure = len(b.ops)
		panic(b.panicFault)
	}
	if input.Backend() != b || len(input.Shape) != 1 {
		panic("logit scale operand owner or rank changed")
	}
	b.lastInput = append([]float32(nil), b.Backend.Read(input)...)
	b.lastScale = scale
	values := append([]float32(nil), b.lastInput...)
	for i := range values {
		values[i] = float32(values[i] * scale)
	}
	out := compute.NewF32(b, []int{len(values)}, values)
	b.scaleLive[out.Buf()] = true
	if b.fault != nil && b.calls == b.faultAt {
		b.opsAtFailure = len(b.ops)
		return out, b.fault // The selected owner must also release this partial result.
	}
	return out, nil
}
func (b *v41LogitScaleRecorder) Read(x compute.Tensor) []float32 {
	if b.scaleLive[x.Buf()] {
		b.reads++
		b.borrowed = append(b.borrowed[:0], b.Backend.Read(x)...)
		return b.borrowed
	}
	return b.v41DenseTestBackend.Read(x)
}
func (b *v41LogitScaleRecorder) Free(x compute.Tensor) {
	if b.scaleLive[x.Buf()] {
		b.frees++
		delete(b.scaleLive, x.Buf())
		b.Backend.Free(x)
		for i := range b.borrowed {
			b.borrowed[i] = 99
		}
		if b.cleanup != nil {
			panic(b.cleanup)
		}
		return
	}
	b.v41DenseTestBackend.Free(x)
}

func (b *v41LogitScaleRecorder) Recycle() {
	b.closeCalls++
	if b.closePanic != nil {
		panic(b.closePanic)
	}
}

func v41LogitScaleCheckBits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("width=%d want %d", len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("lane %d bits=%08x want %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

// Uses only existing Session/Config APIs and fixtures, so the unchanged test
// compiles on its parent: the missing dispatch, not a new symbol, makes it red.
// fak-test:runtime medium est=2s lane=default
func TestV41LogitScaleSessionDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("logit scale software witness requires cpu-ref")
	}
	t.Run("host-bit-contract", func(t *testing.T) {
		for _, tc := range []struct{ input, factor, want uint32 }{
			{0x80000000, 0x3f000000, 0x80000000},
			{0, 0xbf000000, 0x80000000},
			{0x80000000, 0xbf000000, 0},
			{1, 0x3f000000, 0},
			{0x80000001, 0x3f000000, 0x80000000},
			{3, 0x3f000000, 2},
			{0x00800000, 0x3f000000, 0x00400000},
			{0x3f800001, 0x3f800001, 0x3f800002},
			{0x7f7fffff, 0x40000000, 0x7f800000},
		} {
			values := []float32{math.Float32frombits(tc.input)}
			logitScaleInPlace(values, Config{LogitScale: float64(math.Float32frombits(tc.factor))})
			if got := math.Float32bits(values[0]); got != tc.want {
				t.Fatalf("%08x * %08x = %08x want %08x", tc.input, tc.factor, got, tc.want)
			}
		}
		// A nonzero float64 config may round to -0 in the existing F32 cast.
		values := []float32{1, -1}
		logitScaleInPlace(values, Config{LogitScale: -math.SmallestNonzeroFloat64})
		v41LogitScaleCheckBits(t, values, []float32{math.Float32frombits(0x80000000), 0})
	})
	t.Run("selected-scalar-boundaries", func(t *testing.T) {
		for _, tc := range []struct {
			name        string
			factor      float64
			input, want []uint32
		}{
			{"tiny-config-to-negative-zero", -math.SmallestNonzeroFloat64, []uint32{0x3f800000, 0xbf800000, 0, 0x80000000}, []uint32{0x80000000, 0, 0x80000000, 0}},
			{"subnormal-ties", 0.5, []uint32{1, 0x80000001, 3, 0x00800000}, []uint32{0, 0x80000000, 2, 0x00400000}},
			{"finite-overflow", 2, []uint32{0x7f7fffff, 0xff7fffff, 0x80000000, 0}, []uint32{0x7f800000, 0xff800000, 0x80000000, 0}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				m := v41IncrementalPlainModel(t, 1)
				m.Cfg.LogitScale = tc.factor
				m.Cfg.LogitSoftcap, m.Cfg.SuppressTokens = 0, nil
				b := newV41LogitScaleRecorder()
				s := v41DenseTestSession(t, m, b)
				state := s.v41State()
				original := state.denseProjection
				state.denseProjection = func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
					if layer == -1 && leaf == m.headName() {
						if out < len(tc.input) || rows != 1 {
							t.Fatal("fixture head geometry changed")
						}
						values := make([]float32, out)
						for i, bits := range tc.input {
							values[i] = math.Float32frombits(bits)
						}
						return values, v41ProjectionHandled, nil
					}
					if original != nil {
						return original(layer, leaf, panel, out, in, rows)
					}
					return nil, v41ProjectionDeclined, nil
				}
				check := func(got []float32, calls int) {
					t.Helper()
					if b.calls != calls || math.Float32bits(b.lastScale) != math.Float32bits(float32(tc.factor)) || len(b.scaleLive) != 0 {
						t.Fatal("scalar boundary bypassed selected callback or leaked output")
					}
					for i, bits := range tc.want {
						if math.Float32bits(got[i]) != bits {
							t.Fatalf("lane %d bits=%08x want %08x", i, math.Float32bits(got[i]), bits)
						}
					}
				}
				check(s.Prefill([]int{1}), 1)
				check(s.Step(2), 2)
			})
		}
	})
	t.Run("prefill-decode-restored-owner", func(t *testing.T) {
		for _, full := range []bool{false, true} {
			m := v41IncrementalPlainModel(t, 2)
			if full {
				m = v41FullStepPatchedNorms(t)
			}
			m.Cfg.LogitScale = 0.0625
			// The selected multiply must precede the unmodified host softcap
			// and suppression; neither operation is claimed device-resident.
			m.Cfg.LogitSoftcap, m.Cfg.SuppressTokens = 0.75, []int{1, -1, m.Cfg.VocabSize}
			b := newV41LogitScaleRecorder()
			s := v41DenseTestSession(t, m, b)
			check := func(got []float32, calls int) {
				t.Helper()
				if b.calls != calls || b.lastScale != 0.0625 || len(b.scaleLive) != 0 {
					t.Fatalf("full=%t calls=%d want %d scale=%g live=%d", full, b.calls, calls, b.lastScale, len(b.scaleLive))
				}
				want := append([]float32(nil), b.lastInput...)
				logitScaleInPlace(want, m.Cfg)
				v41LogitScaleCheckBits(t, got, want)
			}
			check(s.Prefill([]int{1, 2}), 2)
			check(s.Step(3), 3)
			if b.uploads != 3 || b.reads != 3 || b.frees != 6 || !s.v41IncrementalEligible() {
				t.Fatal("scale copies changed the incremental path or leaked ownership")
			}
			snap, err := s.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			// Snapshots require the same physical backend identity. A distinct
			// destination session still must replace the closed source owner.
			target := v41DenseTestSession(t, m, b)
			if err := snap.Restore(target); err != nil {
				t.Fatal(err)
			}
			snap.Close()
			s.Close()
			got := target.Prefill([]int{4})
			if b.calls != 4 || target.v41State().callbackOwner != target {
				t.Fatal("restored suffix retained the source scale owner")
			}
			want := append([]float32(nil), b.lastInput...)
			logitScaleInPlace(want, m.Cfg)
			v41LogitScaleCheckBits(t, got, want)
		}
	})
	t.Run("nonfinite-head-keeps-host-postprocessing", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		m.Cfg.LogitScale, m.Cfg.LogitSoftcap, m.Cfg.SuppressTokens = 0.5, 1, []int{2}
		b := newV41LogitScaleRecorder()
		s := v41DenseTestSession(t, m, b)
		state := s.v41State()
		original := state.denseProjection
		var head []float32
		state.denseProjection = func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
			if layer == -1 && leaf == m.headName() {
				if out < 4 || rows != 1 {
					t.Fatal("fixture head geometry changed")
				}
				head = make([]float32, out)
				head[0], head[1], head[2], head[3] = float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), 0.25
				return append([]float32(nil), head...), v41ProjectionHandled, nil
			}
			if original != nil {
				return original(layer, leaf, panel, out, in, rows)
			}
			return nil, v41ProjectionDeclined, nil
		}
		got := s.Prefill([]int{1})
		if b.calls != 0 || s.BackendSessionClosed() {
			t.Fatal("out-of-domain row selected a finite-only scale operation")
		}
		want := append([]float32(nil), head...)
		logitScaleInPlace(want, m.Cfg)
		v41LogitScaleCheckBits(t, got, want)
		if got[0] != 1 || got[1] != -1 || !math.IsInf(float64(got[2]), -1) {
			t.Fatal("nonfinite raw head no longer uses original softcap/suppression semantics")
		}
	})
	t.Run("identity-keeps-host-postprocessing", func(t *testing.T) {
		for _, factor := range []float64{0, 1, math.Nextafter(1, 2)} {
			m := v41IncrementalPlainModel(t, 1)
			m.Cfg.LogitScale, m.Cfg.LogitSoftcap, m.Cfg.SuppressTokens = factor, 0.75, []int{1}
			b := newV41LogitScaleRecorder()
			s := v41DenseTestSession(t, m, b)
			got := s.Prefill([]int{1})
			if b.calls != 0 || !math.IsInf(float64(got[1]), -1) {
				t.Fatalf("identity scalar=%g dispatched or lost suppression", factor)
			}
		}
	})
	t.Run("typed-forward-panic-retires-current-owner", func(t *testing.T) {
		for _, suffix := range []bool{false, true} {
			m := v41IncrementalPlainModel(t, 2)
			m.Cfg.LogitScale = 0.0625
			b := newV41LogitScaleRecorder()
			s := v41DenseTestSession(t, m, b)
			// Disarm only at teardown, before the fixture's registered Close:
			// a missing parent dispatch must fail this subtest without a cleanup panic.
			t.Cleanup(func() { b.closePanic = nil })
			s.Prefill([]int{1, 2})
			before := captureV41ForwardSnapshot(s.v41Forward)
			primary := &BackendForwardOperationError{Backend: "scale-recorder", Forward: ForwardPathKind("deepseek41"), Path: "wrapped-scale-operation", Layer: -1, Stage: "multiply", Cause: ErrV41ForwardStage}
			b.panicFault, b.faultAt = primary, 3
			b.cleanup = errors.New("secondary scalar free failure")
			b.closePanic = errors.New("secondary session close failure")
			err := recoverError(func() {
				if suffix {
					s.Prefill([]int{3})
				} else {
					s.Step(3)
				}
			})
			if err != primary || s.halFailure != primary || !s.BackendSessionClosed() || b.closeCalls != 1 || b.calls != 3 || len(b.ops) != b.opsAtFailure || len(b.scaleLive) != 0 || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatalf("typed panic lost identity, replayed or left current owner open: suffix=%t err=%v closed=%v calls=%d close_calls=%d", suffix, err, s.BackendSessionClosed(), b.calls, b.closeCalls)
			}
			if retry := recoverError(func() { s.Step(3) }); retry != primary || b.calls != 3 {
				t.Fatal("retired scalar owner retried")
			}
		}
	})
	t.Run("selected-error-with-partial-output-and-cleanup", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 2)
		m.Cfg.LogitScale = 0.0625
		b := newV41LogitScaleRecorder()
		s := v41DenseTestSession(t, m, b)
		s.Prefill([]int{1, 2})
		before := captureV41ForwardSnapshot(s.v41Forward)
		primary := &compute.BackendError{Backend: "scale-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
		b.fault, b.faultAt = primary, 3
		b.cleanup = errors.New("secondary scale cleanup failure")
		err := recoverError(func() { s.Step(3) })
		var closed *BackendForwardOperationError
		if !errors.Is(err, primary) || !errors.As(err, &closed) || closed.Path != "v41-logit-scale" {
			t.Fatalf("selected scale failure lost primary identity: %v", err)
		}
		if b.calls != 3 || len(b.ops) != b.opsAtFailure || len(b.scaleLive) != 0 || !s.BackendSessionClosed() || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
			t.Fatal("selected scale failure replayed, leaked, or advanced state")
		}
		if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || b.calls != 3 {
			t.Fatal("closed scale session retried")
		}
	})
}
