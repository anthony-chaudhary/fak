package model

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CPU storage witnesses selection lifetime and allocations, not GPU throughput.
type v41SelectionScratchBackend struct {
	compute.Backend
	hook func() error
}

func (b *v41SelectionScratchBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	return c
}
func (b *v41SelectionScratchBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.F32
}
func (b *v41SelectionScratchBackend) SupportsV41SharedAttention() bool { return true }
func (b *v41SelectionScratchBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	return compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
}
func (b *v41SelectionScratchBackend) V41SharedAttention(q, kv, sink compute.Tensor, rows, heads, dim int, scale float32, mode compute.V41SharedAttentionMode, hasSink bool) (compute.Tensor, error) {
	if hook := b.hook; hook != nil {
		b.hook = nil
		if err := hook(); err != nil {
			return compute.Tensor{}, err
		}
		return compute.Tensor{}, &compute.V41SharedAttentionArithmeticError{
			Stage: compute.V41SharedAttentionStageScore, Head: 0, SelectedSlot: 0, Element: -1, ValueBits: math.Float32bits(float32(math.Inf(1))),
		}
	}
	return compute.NewF32(b, []int{heads, dim}, make([]float32, heads*dim)), nil
}

// fak-test:runtime fast est=100ms lane=default
func TestV41SharedAttentionSelectionScratch(t *testing.T) {
	plain := func(ids []int32, length []int32) v41SharedAttentionRequest {
		return v41SharedAttentionRequest{
			mode: compute.V41SharedAttentionPlain, q: []float32{0.25, -0.5}, kv: []float32{2, -4, 6, 8, -1, 3}, sink: []float32{0}, idx: ids,
			plain: V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: 2, N: 3, TopK: len(ids), TopKLength: length, Softmax: 0.7},
		}
	}
	compressed := func(ids []int32, offset int) v41SharedAttentionRequest {
		r := plain(ids, nil)
		r.mode, r.kv = compute.V41SharedAttentionCompressed, nil
		r.values = [][]float32{{2, -4}, {6, 8}, {-1, 3}}
		r.compressed = V41AttentionSharedKVOptions{Layer: 4, Ratio: 2, QueryOffset: offset, Groups: 3, Heads: 1, HeadDim: 2, Softmax: 0.7, IndexTopK: len(ids)}
		return r
	}
	cases := []struct {
		request v41SharedAttentionRequest
		want    []int
	}{
		{plain([]int32{2, 0, 2, -1}, nil), []int{2, 0, 2}},
		{plain([]int32{1, -1, 2}, []int32{2}), []int{1}},
		{plain([]int32{-1, -1}, nil), nil},
		{plain([]int32{0, 1, 2, 1, 0, 2, 1, 0, 2}, nil), []int{0, 1, 2, 1, 0, 2, 1, 0, 2}},
		{compressed([]int32{2, 0, 2, -1, 1}, 1), []int{0}},
		{compressed(nil, 5), []int{0, 1, 2}},
	}

	t.Run("order-growth-shrink-empty-and-independent-payloads", func(t *testing.T) {
		var scratch []int
		for i, tc := range cases {
			prior := scratch
			p, err := v41PrepareSharedAttentionWithRows(tc.request, scratch)
			if err != nil {
				t.Fatal(err)
			}
			if len(p.sourceRows) != len(tc.want) {
				t.Fatalf("case %d selected %v, want %v", i, p.sourceRows, tc.want)
			}
			for slot, row := range tc.want {
				if p.sourceRows[slot] != row {
					t.Fatalf("case %d slot %d = %d, want %d", i, slot, p.sourceRows[slot], row)
				}
			}
			if cap(prior) > 0 && cap(prior) >= len(tc.want) && &prior[:cap(prior)][0] != &p.sourceRows[:cap(p.sourceRows)][0] {
				t.Fatalf("case %d discarded sufficient scratch capacity", i)
			}
			fresh, err := v41PrepareSharedAttention(tc.request)
			if err != nil || !reflect.DeepEqual(p.q, fresh.q) || !reflect.DeepEqual(p.kv, fresh.kv) || !reflect.DeepEqual(p.sink, fresh.sink) {
				t.Fatalf("case %d payload changed: %v", i, err)
			}
			if len(p.sourceRows) != 0 {
				p.sourceRows[0] = -99
				if fresh.sourceRows[0] != tc.want[0] {
					t.Fatal("standalone payload borrowed reusable descriptors")
				}
			}
			scratch = p.sourceRows
		}
		bad := plain([]int32{0, 99}, []int32{1})
		bad.q[0] = float32(math.NaN())
		before := append([]int(nil), scratch...)
		_, err := v41PrepareSharedAttentionWithRows(bad, scratch)
		if err == nil || !strings.Contains(err.Error(), "non-finite query element 0") || !reflect.DeepEqual(scratch, before) {
			t.Fatalf("validation precedence or untouched scratch changed: %v", err)
		}
		bad.q[0] = 0
		_, err = v41PrepareSharedAttentionWithRows(bad, scratch)
		if err == nil || !strings.Contains(err.Error(), "original slot 1") {
			t.Fatalf("masked invalid index was hidden: %v", err)
		}
	})

	t.Run("selected-callback-exact-output-and-retention", func(t *testing.T) {
		b := newV41SharedAttentionTestBackend()
		s := &Session{M: &Model{}, Backend: b}
		t.Cleanup(s.Close)
		attend := s.v41SharedAttentionFunc()
		var retained, expected [][]float32
		for _, tc := range cases {
			p, err := v41PrepareSharedAttention(tc.request)
			if err != nil {
				t.Fatal(err)
			}
			want, err := v41SharedAttentionRoundedControl(v41SharedAttentionTestCall{q: p.q, kv: p.kv, sink: p.sink, rows: len(p.sourceRows), heads: p.heads, dim: p.headDim, scale: p.scale, mode: p.mode, hasSink: p.sink != nil})
			if err != nil {
				t.Fatal(err)
			}
			got, err := attend(4, tc.request)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(want) {
				t.Fatalf("selected output width %d, want %d", len(got), len(want))
			}
			retained, expected = append(retained, got), append(expected, want)
			for row := range retained {
				for col, value := range retained[row] {
					if math.Float32bits(value) != math.Float32bits(expected[row][col]) {
						t.Fatalf("retained row %d column %d changed: %g want %g", row, col, value, expected[row][col])
					}
				}
			}
		}
		if len(b.allocations) != 0 {
			t.Fatal("selected callback leaked device tensors")
		}
	})

	t.Run("reentrant-fault-keeps-outer-source-row", func(t *testing.T) {
		b := &v41SelectionScratchBackend{Backend: compute.Default()}
		s := &Session{M: &Model{}, Backend: b}
		t.Cleanup(s.Close)
		attend := s.v41SharedAttentionFunc()
		b.hook = func() error {
			_, err := attend(9, compressed([]int32{1}, 5))
			return err
		}
		_, err := attend(4, compressed([]int32{2, 0, 2}, 5))
		var selected *BackendForwardOperationError
		var arithmetic *compute.V41SharedAttentionArithmeticError
		if !errors.As(err, &selected) || !errors.As(err, &arithmetic) || selected.Layer != 4 || !strings.Contains(err.Error(), "group=2") {
			t.Fatalf("outer fault attribution corrupted by reentrant selection: %v", err)
		}
	})

	t.Run("warm-allocation-reduction", func(t *testing.T) {
		r := plain(make([]int32, 2048), nil)
		r.resident = &v41ResidentKVView{} // Payload preparation does not gather the mirror.
		measure := func(reuse bool) float64 {
			var scratch []int
			return testing.AllocsPerRun(50, func() {
				p, err := v41PrepareSharedAttentionWithRows(r, scratch)
				if err != nil {
					panic(err)
				}
				if reuse {
					scratch = p.sourceRows
				}
			})
		}
		fresh, warm := measure(false), measure(true)
		b := &v41SelectionScratchBackend{Backend: compute.Default()}
		s := &Session{M: &Model{}, Backend: b}
		t.Cleanup(s.Close)
		attend := s.v41SharedAttentionFunc()
		r.resident = nil
		measureCallback := func(reuse bool) float64 {
			return testing.AllocsPerRun(50, func() {
				call := attend
				if !reuse {
					call = s.v41SharedAttentionFunc()
				}
				if _, err := call(4, r); err != nil {
					panic(err)
				}
			})
		}
		freshCall, warmCall := measureCallback(false), measureCallback(true)
		t.Logf("2048 selected slots allocs/op: payload fresh=%g warm=%g; selected callback fresh=%g warm=%g", fresh, warm, freshCall, warmCall)
		if fresh-warm < 8 || freshCall-warmCall < 8 {
			t.Fatalf("selection scratch allocation reduction missing: payload %g -> %g, callback %g -> %g", fresh, warm, freshCall, warmCall)
		}
	})
}
