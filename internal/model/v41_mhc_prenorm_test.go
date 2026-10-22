package model

import (
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func (b *v41MHCTestBackend) V41MHCPrenormAvailable() bool { return b.prenorm }

func (b *v41MHCTestBackend) SupportsV41MHCPrenormWeight(w compute.Tensor, hidden int) bool {
	return b.prenorm && (w.Dtype == compute.F32 || w.Dtype == compute.Q2_K) && slices.Equal(w.Shape, []int{compute.V41MHCCoeffWidth, compute.V41MHCStreams * hidden})
}

// V41MHCPrenormSplit composes the cpu-ref projection the MatMul path would run
// with the rms-split transcription, so routing parity is exact. The device
// order is witnessed by TestVulkanV41MHCPrenorm* in internal/compute.
func (b *v41MHCTestBackend) V41MHCPrenormSplit(weight, streams, params compute.Tensor, rows, hidden, iters int, eps, rmsEps float32) (compute.Tensor, error) {
	const op = "prenorm-split"
	if b.failOp == op && b.calls[op]+1 == b.failCall {
		b.calls[op]++
		return compute.Tensor{}, b.cause
	}
	x := streams
	var raw compute.Tensor
	if rows == 1 {
		x.Shape = []int{compute.V41MHCStreams * hidden}
		raw = b.Backend.MatMul(weight, x)
	} else {
		x.Shape = []int{rows, compute.V41MHCStreams * hidden}
		raw = b.Backend.BatchedMatMul(weight, x, rows)
	}
	raw.Shape = []int{rows, compute.V41MHCCoeffWidth}
	out, err := b.V41MHCRMSSplit(raw, streams, params, rows, hidden, iters, eps, rmsEps)
	if b.calls["rms-split"]--; b.calls["rms-split"] == 0 {
		delete(b.calls, "rms-split")
	}
	b.calls[op]++
	if out.Buf() != nil {
		b.outputs[out.Buf()] = op
	}
	return out, err
}

// v41MHCMixProjections counts the recorded MatMul/BatchedMatMul ops that are a
// [24, 4H] mHC mix projection.
func v41MHCMixProjections(b *v41MHCTestBackend, hidden int) int {
	n := 0
	for _, op := range b.ops {
		if op.out == compute.V41MHCCoeffWidth && op.in == compute.V41MHCStreams*hidden {
			n++
		}
	}
	return n
}

// fak-test:runtime medium est=2s lane=default
func TestV41MHCPrenormReplacesMixMatMulAndRMSSplit(t *testing.T) {
	t.Setenv(v41MHCPrenormEnv, "") // default selection, whatever the caller's shell sets
	run := func(t *testing.T, prenorm bool) (*v41MHCTestBackend, []float32, []float32) {
		m := v41FullStepPatchedNorms(t)
		b := newV41MHCTestBackend(true)
		b.prenorm = prenorm
		s := v41DenseTestSession(t, m, b)
		prefill := s.Prefill([]int{1, 2})
		step := s.Step(3)
		return b, prefill, step
	}
	m := v41FullStepPatchedNorms(t)
	L, H := m.Cfg.NumLayers, m.Cfg.HiddenSize
	host := newV41MHCTestBackend(false)
	hs := v41DenseTestSession(t, v41FullStepPatchedNorms(t), host)
	wantPrefill, wantStep := hs.Prefill([]int{1, 2}), hs.Step(3)

	fused, gotPrefill, gotStep := run(t, true)
	if !reflect.DeepEqual(gotPrefill, wantPrefill) || !reflect.DeepEqual(gotStep, wantStep) {
		t.Fatal("prenorm mHC logits differ from the host mHC reference")
	}
	// Two prompt rows, then one step row: every split is a prenorm-split and no
	// mix projection reaches MatMul.
	want := map[string]int{"initial": 2, "prenorm-split": 4 * L, "pre": 5*L + 1, "post": 4 * L}
	if !reflect.DeepEqual(fused.calls, want) || len(fused.outputs) != 0 {
		t.Fatalf("prenorm dispatches=%v live outputs=%d want=%v", fused.calls, len(fused.outputs), want)
	}
	if n := v41MHCMixProjections(fused, H); n != 0 {
		t.Fatalf("prenorm path still ran %d mix MatMuls", n)
	}

	t.Setenv(v41MHCPrenormEnv, "0")
	off, offPrefill, offStep := run(t, true)
	if !reflect.DeepEqual(offPrefill, gotPrefill) || !reflect.DeepEqual(offStep, gotStep) {
		t.Fatal("FAK_V41_MHC_PRENORM=0 logits differ from the prenorm path")
	}
	want = map[string]int{"initial": 2, "rms-split": 4 * L, "pre": 5*L + 1, "post": 4 * L}
	if !reflect.DeepEqual(off.calls, want) || v41MHCMixProjections(off, H) != 4*L {
		t.Fatalf("disabled prenorm dispatches=%v mix MatMuls=%d want=%v and %d", off.calls, v41MHCMixProjections(off, H), want, 4*L)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestV41MHCPrenormSelectionKeepsPanelQ2KOnMatMul(t *testing.T) {
	const hidden = 64
	b := newV41MHCTestBackend(true)
	b.prenorm = true
	r := &v41MHCResident{d: &v41MHCDevice{s: &Session{Backend: b}, op: b}}
	q2k := compute.NewQ2K(compute.Default(), []int{compute.V41MHCCoeffWidth, 4 * hidden}, make([]byte, compute.V41MHCCoeffWidth*84))
	f32 := compute.NewF32(compute.Default(), []int{compute.V41MHCCoeffWidth, 4 * hidden}, make([]float32, compute.V41MHCCoeffWidth*4*hidden))
	for _, tc := range []struct {
		name   string
		weight compute.Tensor
		rows   int
		off    bool
		want   bool
	}{
		{"q2k decode row", q2k, 1, false, true},
		{"q2k panel", q2k, 2, false, false},
		{"f32 panel", f32, 3, false, true},
		{"disabled", f32, 1, true, false},
	} {
		r.prenormOff = tc.off
		if got := r.prenorm(tc.weight, tc.rows, hidden) != nil; got != tc.want {
			t.Fatalf("%s: prenorm selected=%t want %t", tc.name, got, tc.want)
		}
	}
	b.prenorm = false
	if r.prenorm(f32, 1, hidden) != nil {
		t.Fatal("unavailable prenorm was selected")
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41MHCPrenormFailureClosesSessionWithoutReplay(t *testing.T) {
	t.Setenv(v41MHCPrenormEnv, "")
	m := v41FullStepPatchedNorms(t)
	b := newV41MHCTestBackend(true)
	b.prenorm = true
	s := v41DenseTestSession(t, m, b)
	s.Prefill([]int{1, 2})
	const op = "prenorm-split"
	b.failOp, b.failCall = op, b.calls[op]+1
	b.cause = &compute.BackendError{Backend: "cpu-ref-mhc", Class: compute.VulkanClassExecutionFailed, Err: compute.ErrVulkanExecutionFailed}
	before := append([]int(nil), s.v41State().history...)
	func() {
		defer func() {
			r := recover()
			err, ok := r.(error)
			var selected *V41MHCOperationError
			if !ok || !errors.As(err, &selected) || !errors.Is(err, compute.ErrVulkanExecutionFailed) || errors.Is(err, ErrV41ForwardStage) {
				t.Fatalf("step failure = %v, want a selected mHC operation error and no stage sentinel", r)
			}
		}()
		s.Step(3)
	}()
	if !s.BackendSessionClosed() || !reflect.DeepEqual(s.v41Forward.history, before) {
		t.Fatalf("failed step left session open=%t or advanced history %v -> %v", !s.BackendSessionClosed(), before, s.v41Forward.history)
	}
	if b.calls[op] != b.failCall || b.calls["rms-split"] != 0 {
		t.Fatalf("after the failure: %s=%d rms-split=%d, want %d and no MatMul replay", op, b.calls[op], b.calls["rms-split"], b.failCall)
	}
}
