package model

import (
	"errors"
	"math"
	"slices"
	"strings"
	"testing"
)

// TestV41SparseSinkNonFiniteGuard covers the canonical contraction: finite
// inputs must not publish NaN/Inf or silently turn an overflow into a zero row.
// The success controls retain masking, tied scores, and valid exp(-Inf)=0.
// fak-test:runtime fast est=1ms lane=default
func TestV41SparseSinkNonFiniteGuard(t *testing.T) {
	t.Parallel()
	max := float32(math.MaxFloat32)
	var priorTail []float32
	for _, tc := range []struct {
		name    string
		q, kv   []float32
		sink    []float32
		idx     []int32
		scale   float32
		heads   int
		inverse func(int, []float32) error
		stage   string
		detail  string
	}{
		{
			name: "product", q: []float32{max}, kv: []float32{0, 2}, sink: []float32{0},
			idx: []int32{1}, scale: 1, stage: "score accumulate", detail: "kvRow=1",
		},
		{
			name: "negative_product", q: []float32{max}, kv: []float32{-2},
			idx: []int32{0}, scale: 1, stage: "score accumulate", detail: "kvRow=0",
		},
		{
			name: "dot_sum", q: []float32{max, max}, kv: []float32{0.75, 0.75}, sink: []float32{0},
			idx: []int32{0}, scale: 1, stage: "score accumulate", detail: "kvRow=0",
		},
		{
			name: "scaled_score", q: []float32{max}, kv: []float32{0.5}, sink: []float32{0},
			idx: []int32{0}, scale: 4, stage: "score accumulate", detail: "kvRow=0",
		},
		{
			// Ten tied scores give float32 weight 0.1. Rounding their
			// weighted MaxFloat32 values overflows the final addition.
			name: "weighted_sum", q: []float32{0}, kv: []float32{max},
			idx: make([]int32, 10), scale: 1, stage: "weighted value accumulate", detail: "element=0",
		},
		{
			name: "inverse_output", q: []float32{0, 0}, kv: []float32{max, max},
			idx: []int32{0}, scale: 1, stage: "inverse rotation", detail: "element=0",
			inverse: func(_ int, tail []float32) error { tail[0] += tail[1]; return nil },
		},
		{
			name: "inverse_retained_tail", q: []float32{0, 0, 0, 0}, kv: []float32{1, 1},
			idx: []int32{0}, scale: 1, heads: 2, stage: "inverse rotation", detail: "element=0",
			inverse: func(_ int, tail []float32) error {
				if priorTail == nil {
					priorTail = tail
				} else {
					priorTail[0] = float32(math.Inf(1))
				}
				return nil
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			heads := tc.heads
			if heads == 0 {
				heads = 1
			}
			hd := len(tc.q) / heads
			opt := V41SparseAttentionSinkOptions{
				B: 1, M: 1, Heads: heads, HeadDim: hd, TopK: len(tc.idx), N: len(tc.kv) / hd,
				Softmax: tc.scale, Inverse: tc.inverse, RopeDim: hd,
			}
			q, kv, sink, idx := slices.Clone(tc.q), slices.Clone(tc.kv), slices.Clone(tc.sink), slices.Clone(tc.idx)
			out, err := V41SparseAttentionSink(tc.q, tc.kv, tc.sink, tc.idx, opt)
			if out != nil || !errors.Is(err, ErrV41SparseSinkNonFinite) {
				t.Fatalf("out=%v err=%v, want nil output and ErrV41SparseSinkNonFinite", out, err)
			}
			for _, token := range []string{"sparse sink " + tc.stage, "b=0", "m=0", "h=0", tc.detail, "value="} {
				if !strings.Contains(err.Error(), token) {
					t.Fatalf("error %q is missing %q", err, token)
				}
			}
			if !slices.Equal(tc.q, q) || !slices.Equal(tc.kv, kv) || !slices.Equal(tc.sink, sink) || !slices.Equal(tc.idx, idx) {
				t.Fatal("refusal changed caller-owned inputs")
			}
		})
	}

	for _, tc := range []struct {
		name         string
		q            float32
		kv, sink     []float32
		idx, topKLen []int32
		want         float32
	}{
		{name: "empty_slot", q: max, kv: []float32{2}, idx: []int32{-1}},
		{name: "padding", q: max, kv: []float32{0, 2}, sink: []float32{0}, idx: []int32{0, 1}, topKLen: []int32{1}},
		{name: "empty_length", q: max, kv: []float32{2}, idx: []int32{0}, topKLen: []int32{0}},
		{name: "sink_tie", kv: []float32{4, 8}, sink: []float32{0}, idx: []int32{0, 1}, want: 4},
		{name: "nil_sink_tie", kv: []float32{4, 8}, idx: []int32{0, 1}, want: 6},
		{name: "negative_infinite_gap", q: 1, kv: []float32{-max, max}, sink: []float32{-max}, idx: []int32{0, 1}, want: max},
		{name: "sink_dominates", q: 1, kv: []float32{-max}, sink: []float32{max}, idx: []int32{0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := V41SparseAttentionSink([]float32{tc.q}, tc.kv, tc.sink, tc.idx, V41SparseAttentionSinkOptions{
				B: 1, M: 1, Heads: 1, HeadDim: 1, TopK: len(tc.idx), N: len(tc.kv), Softmax: 1, TopKLength: tc.topKLen,
			})
			if err != nil || len(out) != 1 || !finite32(out[0]) || out[0] != tc.want {
				t.Fatalf("out=%v err=%v, want finite [%g]", out, err, tc.want)
			}
		})
	}

	t.Run("inverse_error_identity", func(t *testing.T) {
		want := errors.New("inverse failed")
		out, err := V41SparseAttentionSink([]float32{0, 0}, []float32{1, 1}, nil, []int32{0}, V41SparseAttentionSinkOptions{
			B: 1, M: 1, Heads: 1, HeadDim: 2, TopK: 1, N: 1, Softmax: 1, RopeDim: 2,
			Inverse: func(_ int, tail []float32) error { tail[0] = float32(math.NaN()); return want },
		})
		if out != nil || err != want {
			t.Fatalf("out=%v err=%v, want nil output and unchanged inverse error %v", out, err, want)
		}
	})
}
