package model

import (
	"errors"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"math"
	"testing"
)

// CPU-only wiring/ownership recorder. Its host oracle is not a GPU witness.
type v41FusedIndexScaleRecorder struct {
	*v41IndexWeightScaleRecorder
	enabled              bool
	probes, fused        int
	mode                 string
	primary, freeFailure error
	frees                map[compute.Buffer]int
}

func newV41FusedIndexScaleRecorder() *v41FusedIndexScaleRecorder {
	b := &v41FusedIndexScaleRecorder{v41IndexWeightScaleRecorder: newV41IndexWeightScaleRecorder(), enabled: true, frees: map[compute.Buffer]int{}}
	b.self = b
	b.v41DeferredLedger = newV41DeferredLedger(b.owner)
	return b
}
func (b *v41FusedIndexScaleRecorder) track(x compute.Tensor) compute.Tensor {
	if x.Buf() != nil {
		if _, seen := b.frees[x.Buf()]; !seen {
			b.frees[x.Buf()] = 0
		}
	}
	return x
}
func (b *v41FusedIndexScaleRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site == "V4.1 index weight scale input" {
		return b.track(compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...)))
	}
	return b.v41BF16PublishRecorder.UploadClass(x, dt, class, site)
}
func (b *v41FusedIndexScaleRecorder) Free(x compute.Tensor) {
	if _, tracked := b.frees[x.Buf()]; tracked {
		b.frees[x.Buf()]++
		if b.v41DeferredLedger != nil {
			delete(b.deferredPending, x.Buf())
		}
		if b.freeFailure != nil {
			e := b.freeFailure
			b.freeFailure = nil
			panic(e)
		}
	}
	b.Backend.Free(x)
}
func (b *v41FusedIndexScaleRecorder) V41ScaleExact(x compute.Tensor, scale float32) (compute.Tensor, error) {
	b.scales = append(b.scales, scale)
	values := append([]float32(nil), b.Backend.Read(x)...)
	for i := range values {
		values[i] = float32(values[i] * scale)
	}
	return b.track(compute.NewF32(b, []int{len(values)}, values)), nil
}
func (b *v41FusedIndexScaleRecorder) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	out, st, e := b.v41DeferredLedger.V41BF16PublishDeferred(x, site)
	b.track(out)
	b.track(st.Record)
	return out, st, e
}
func (b *v41FusedIndexScaleRecorder) SupportsV41IndexScaleBF16(x compute.Tensor, scale float32) bool {
	b.probes++
	if x.Backend() != b || len(x.Shape) != 1 || x.Shape[0] <= 0 || !finite32(scale) {
		panic("fused capability operands")
	}
	return b.enabled
}
func (b *v41FusedIndexScaleRecorder) V41IndexScaleBF16Deferred(x compute.Tensor, scale float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.fused++
	values := append([]float32(nil), b.Backend.Read(x)...)
	var verdict error
	for i, value := range values {
		bits := math.Float32bits(float32(value * scale))
		rounded := v41BF16BoundaryOracle(bits)
		if verdict == nil && (bits&0x7f800000 == 0x7f800000 || rounded&0x7f800000 == 0x7f800000) {
			verdict = &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishIndexerScaledWeight, Index: i}
		}
		values[i] = math.Float32frombits(rounded)
	}
	var out, record compute.Tensor
	if b.mode == "error-status-only" || b.mode == "error-empty" {
		out = compute.Tensor{}
	} else if b.mode == "output-input" {
		out = x
	} else {
		out = b.track(compute.NewF32(b, []int{len(values)}, values))
	}
	switch b.mode {
	case "error-output-only", "error-empty":
		record = compute.Tensor{}
	case "status-input":
		record = x
	case "status-output":
		record = out
	default:
		record = b.track(compute.NewF32(b, []int{4}, make([]float32, 4)))
		b.deferredPending[record.Buf()] = verdict
	}
	st := compute.V41DeferredStatus{Kind: compute.V41DeferredBF16Publish, Count: len(values), Site: compute.V41BF16PublishIndexerScaledWeight, Record: record}
	switch b.mode {
	case "output-shape":
		out.Shape = []int{len(values) + 1}
	case "status-shape":
		st.Record.Shape = []int{3}
	case "status-site":
		st.Site = compute.V41BF16PublishRoutedDown
	case "status-kind":
		st.Kind = compute.V41DeferredRoutedSwiGLU
	case "status-count":
		st.Count++
	case "status-records":
		st.Records = 2
	}
	return out, st, b.primary
}
func (b *v41FusedIndexScaleRecorder) checkReleased(t *testing.T) {
	t.Helper()
	for _, n := range b.frees {
		if n != 1 {
			t.Fatalf("release attempts=%d, want exactly one", n)
		}
	}
	if b.v41DeferredLedger != nil && len(b.deferredPending) != 0 {
		t.Fatal("unconsumed status ownership")
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestV41IndexWeightScaleFusedParityAndCounts(t *testing.T) {
	v41IndexWeightScaleRequireCPU(t)
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, enabled := range []bool{false, true} {
		m := v41IndexerTestFixture(t)
		m.Cfg.IndexNHeads = 4
		b := newV41FusedIndexScaleRecorder()
		b.enabled = enabled
		s := v41DenseTestSession(t, m, b)
		fn := s.v41IndexWeightScaleFunc()
		if fn == nil {
			t.Fatal("missing callback")
		}
		values := []float32{math.Float32frombits(0x80000000), math.Float32frombits(0x00010000), 1.5, -3}
		scale := float32(0.25)
		for layer := 0; layer < 8; layer++ {
			got, e := fn(layer, values, scale)
			if e != nil {
				t.Fatal(e)
			}
			for i, value := range values {
				want := v41BF16BoundaryOracle(math.Float32bits(float32(value * scale)))
				if math.Float32bits(got[i]) != want {
					t.Fatalf("layer %d lane %d", layer, i)
				}
			}
		}
		wantFused, wantScalar, wantAllocs := 0, 8, 32
		if enabled {
			wantFused, wantScalar, wantAllocs = 8, 0, 24
		}
		if b.probes != 8 || b.fused != wantFused || len(b.scales) != wantScalar || b.deferredPublishes != wantScalar || b.deferredFences != 8 || len(b.frees) != wantAllocs {
			t.Fatalf("counts probes=%d fused=%d scalar=%d publish=%d reads=%d allocations=%d", b.probes, b.fused, len(b.scales), b.deferredPublishes, b.deferredFences, len(b.frees))
		}
		b.checkReleased(t)
		if s.BackendSessionClosed() {
			t.Fatal("healthy session closed")
		}
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestV41IndexWeightScaleFusedDeferredRefusal(t *testing.T) {
	v41IndexWeightScaleRequireCPU(t)
	for _, missing := range []bool{false, true} {
		t.Setenv(v41DeferredBF16PublishEnv, "0")
		m := v41IndexerTestFixture(t)
		m.Cfg.IndexNHeads = 2
		b := newV41FusedIndexScaleRecorder()
		if missing {
			b.v41DeferredLedger = nil
			t.Setenv(v41DeferredBF16PublishEnv, "1")
		}
		s := v41DenseTestSession(t, m, b)
		fn := s.v41IndexWeightScaleFunc()
		got, e := fn(0, []float32{1, -2}, 0.5)
		if e != nil || len(got) != 2 || b.probes != 0 || b.fused != 0 || len(b.scales) != 1 || b.calls != 1 {
			t.Fatalf("deferred refusal %v", e)
		}
		b.checkReleased(t)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestV41IndexWeightScaleFusedFailureOwnership(t *testing.T) {
	v41IndexWeightScaleRequireCPU(t)
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, mode := range []string{"error-output-only", "error-status-only", "error-empty", "partial-error", "partial-error-free-panic", "output-input", "status-input", "status-output", "output-shape", "status-shape", "status-site", "status-kind", "status-count", "status-records", "numeric", "transport"} {
		t.Run(mode, func(t *testing.T) {
			m := v41IndexerTestFixture(t)
			m.Cfg.IndexNHeads = 2
			b := newV41FusedIndexScaleRecorder()
			b.mode = mode
			primary := errors.New("selected fused failure")
			weights := []float32{1, -2}
			scale := float32(0.5)
			if mode == "error-output-only" || mode == "error-status-only" || mode == "error-empty" || mode == "partial-error" || mode == "partial-error-free-panic" {
				b.primary = primary
			}
			if mode == "partial-error-free-panic" {
				b.freeFailure = errors.New("secondary Free failure")
			}
			if mode == "numeric" {
				weights = []float32{1, math.Float32frombits(0x7f7f0000)}
				scale = 2
			}
			if mode == "transport" {
				b.deferredReadErr = primary
			}
			s := v41DenseTestSession(t, m, b)
			fn := s.v41IndexWeightScaleFunc()
			got, e := fn(3, weights, scale)
			var closed *BackendForwardOperationError
			if got != nil || !errors.Is(e, ErrV41IndexWeightScaleOperation) || !errors.As(e, &closed) || !s.BackendSessionClosed() || s.halFailure != e || b.fused != 1 || len(b.scales) != 0 || b.deferredPublishes != 0 {
				t.Fatalf("selected failure %v", e)
			}
			if (b.primary != nil || mode == "transport") && !errors.Is(e, primary) {
				t.Fatalf("primary replaced: %v", e)
			}
			if b.primary != nil && closed.Stage != "exact multiply and scaled weight publication" {
				t.Fatalf("combined transport stage %q", closed.Stage)
			}
			if mode == "transport" && closed.Stage != "scaled weight readback" {
				t.Fatalf("read stage %q", closed.Stage)
			}
			if mode == "numeric" {
				var fault *compute.V41BF16CheckedArithmeticError
				if !errors.As(e, &fault) || fault.Site != compute.V41BF16PublishIndexerScaledWeight || fault.Index != 1 || closed.Stage != "scaled weight publication" {
					t.Fatalf("numeric order/site %v", e)
				}
			}
			b.checkReleased(t)
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41IndexWeightScaleFusedRejectsBeforeProbe(t *testing.T) {
	v41IndexWeightScaleRequireCPU(t)
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, tc := range []struct {
		weights []float32
		scale   float32
	}{
		{[]float32{1}, 1},
		{[]float32{1, math.Float32frombits(0x3f808000)}, 1},
		{[]float32{1, 2}, float32(math.Inf(1))},
	} {
		m := v41IndexerTestFixture(t)
		m.Cfg.IndexNHeads = 2
		b := newV41FusedIndexScaleRecorder()
		s := v41DenseTestSession(t, m, b)
		fn := s.v41IndexWeightScaleFunc()
		got, e := fn(0, tc.weights, tc.scale)
		if got != nil || e == nil || b.probes != 0 || b.fused != 0 || len(b.scales) != 0 || len(b.frees) != 0 || !s.BackendSessionClosed() {
			t.Fatalf("pre-dispatch refusal %v", e)
		}
	}
}
