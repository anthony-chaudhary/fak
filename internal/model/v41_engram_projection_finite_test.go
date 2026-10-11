package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// A borrowed recorder readback can be finite F32 yet overflow when consumed as
// BF16. This drives the real session callback's checked-readback/close path.
type v41EngramBF16OverflowBackend struct{ *v41EngProjBackend }

func (b *v41EngramBF16OverflowBackend) Read(x compute.Tensor) []float32 {
	values := b.v41EngProjBackend.Read(x)
	if _, ok := b.outputBuffers[x.Buf()]; ok {
		values[len(values)-1] = math.Float32frombits(0x7f7f8000)
	}
	return values
}

// Projection results must be valid in their consumed BF16 representation before
// ANY stream at the current position is changed. This does not assert batch-wide
// atomicity: later-position failures retain the existing session rollback rules.
// fak-test:runtime medium est=1s lane=default
func TestV41EngramProjectionFiniteBoundary(t *testing.T) {
	t.Parallel()
	m, layout := v41FullEngramModel(t)
	H := m.Cfg.HiddenSize
	stage := m.v41EngramStageFor()
	in, out := stage.columns*stage.headDim, (stage.hc+1)*H
	// Every packed row decodes to [1, 0, ...]. Only weight column 0 is
	// nonzero, so the real host contraction returns the literal desired output.
	packed := make([]byte, int(layout.Rows[0])*V41EngramPackedRowBytes)
	for i := 0; i < int(layout.Rows[0]); i++ {
		packed[i*V41EngramPackedRowBytes] = 0x38    // E4M3 1
		packed[i*V41EngramPackedRowBytes+256] = 127 // E8M0 1
	}
	if err := m.wireV41Engram(layout, []V41EngramRowSource{&v41EngramMemorySource{packed: packed, rows: int(layout.Rows[0])}}, int64(V41EngramPackedRowBytes)*4); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name  string
		bits  uint32
		valid bool
	}{
		{"nan", 0x7fc00000, false},
		{"positive_inf", 0x7f800000, false},
		{"negative_inf", 0xff800000, false},
		// Independent RNE threshold literals: the retained maximum finite
		// mantissa is odd, so the half-way tie increments it to infinity.
		{"positive_bf16_overflow_tie", 0x7f7f8000, false},
		{"negative_bf16_overflow_tie", 0xff7f8000, false},
		{"positive_below_bf16_overflow", 0x7f7f7fff, true},
		{"negative_below_bf16_overflow", 0xff7f7fff, true},
		{"ordinary_finite", 0x3f800000, true},
	}
	for _, route := range []string{"host", "handled"} {
		for _, leaf := range []string{"key", "value"} {
			for _, tc := range cases {
				t.Run(route+"/"+leaf+"/"+tc.name, func(t *testing.T) {
					projected := make([]float32, out)
					for i := range projected {
						projected[i] = 1
					}
					index := out - 1
					if leaf == "key" {
						index = stage.hc*H - 1 // last key: no earlier stream may move
					}
					projected[index] = math.Float32frombits(tc.bits)
					weights := make([]float32, out*in)
					for i, v := range projected {
						weights[i*in] = v
					}
					v41WriteTensorF32(t, m, layerName(0, "engram_kv.weight"), weights)
					var project v41EngramProjectionFunc
					calls := 0
					if route == "handled" {
						project = func(layer int, row []float32, gotOut, gotIn int) ([]float32, v41DenseProjectionOutcome, error) {
							calls++
							if layer != 0 || gotOut != out || gotIn != in || len(row) != in || row[0] != 1 {
								t.Fatal("projection fixture lost exact geometry or unit input")
							}
							return projected, v41ProjectionHandled, nil
						}
					}
					x, streams := v41FullEngramAsymmetricStreams([]int{1}, H)
					before := sysFlatten(streams[0])
					err := m.v41EngramInjectWithProjection(0, x, streams, true, []int{1}, float32(m.Cfg.RMSNormEps), project)
					if route == "handled" && calls != 1 {
						t.Fatalf("handled calls=%d, want 1", calls)
					}
					if tc.valid {
						if err != nil || reflect.DeepEqual(before, sysFlatten(streams[0])) {
							t.Fatalf("finite control did not publish: %v", err)
						}
						for _, v := range sysFlatten(streams[0]) {
							if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
								t.Fatal("finite control published a non-finite residual")
							}
						}
						return
					}
					var attributed *V41ForwardError
					var selected *V41ProjectionOperationError
					if !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &attributed) || attributed.Layer != 0 || attributed.Stage != v41StageEngram {
						t.Errorf("invalid consumed projection error=%v, want typed Engram failure", err)
					}
					if got := errors.As(err, &selected); got != (route == "handled") {
						t.Errorf("selected-operation attribution=%v for %s", got, route)
					}
					if !reflect.DeepEqual(before, sysFlatten(streams[0])) {
						t.Error("invalid projection partially published the current position")
					}
				})
			}
		}
	}
	t.Run("healthy_decline", func(t *testing.T) {
		weights := make([]float32, out*in)
		for i := 0; i < out; i++ {
			weights[i*in] = 1
		}
		v41WriteTensorF32(t, m, layerName(0, "engram_kv.weight"), weights)
		x, streams := v41FullEngramAsymmetricStreams([]int{1}, H)
		calls := 0
		err := m.v41EngramInjectWithProjection(0, x, streams, true, []int{1}, float32(m.Cfg.RMSNormEps), func(int, []float32, int, int) ([]float32, v41DenseProjectionOutcome, error) {
			calls++
			return nil, v41ProjectionDeclined, nil
		})
		if err != nil || calls != 1 {
			t.Fatalf("healthy host decline changed: err=%v calls=%d", err, calls)
		}
	})
	t.Run("actual_session_readback_overflow_closes", func(t *testing.T) {
		if compute.Default().Name() != "cpu-ref" {
			t.Fatal("the software recorder requires the cpu-ref backend")
		}
		b := &v41EngramBF16OverflowBackend{newV41EngProjBackend(m)}
		s := v41EngProjSession(t, m, b)
		x, streams := v41FullEngramAsymmetricStreams([]int{1}, H)
		before := sysFlatten(streams[0])
		err := m.v41EngramInjectWithProjection(0, x, streams, true, []int{1}, float32(m.Cfg.RMSNormEps), s.v41EngramProjectionFunc())
		var selected *V41ProjectionOperationError
		var closed *BackendForwardOperationError
		if !errors.As(err, &selected) || !errors.As(err, &closed) || !errors.Is(err, ErrV41ForwardStage) || !s.BackendSessionClosed() {
			t.Errorf("BF16-overflow readback did not close selected session: %v", err)
		}
		if b.attempts != 1 || len(b.live) != 0 || !reflect.DeepEqual(before, sysFlatten(streams[0])) {
			t.Errorf("overflow retries/leaks/publication: attempts=%d live=%d", b.attempts, len(b.live))
		}
	})
}
