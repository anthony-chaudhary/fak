package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"
)

func compressedVisibilityBitsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

func compressedVisibilityFixture(seq, heads int, indexed bool) ([]float32, [][]float32, V41AttentionSharedKVOptions) {
	const groups, dim = 32, 8
	q := make([]float32, seq*heads*dim)
	for i := range q {
		q[i] = float32(i%11-5) / 16
	}
	values := make([][]float32, groups)
	for g := range values {
		values[g] = make([]float32, dim)
		for d := range values[g] {
			values[g][d] = float32((g+d)%9-4) / 8
		}
	}
	opt := V41AttentionSharedKVOptions{Ratio: 4, QueryOffset: 127, Groups: groups, Heads: heads, HeadDim: dim, Softmax: 0.25}
	if indexed {
		opt.IndexTopK = 5
		opt.Idx = make([]int32, seq*5)
		for i := range opt.Idx {
			opt.Idx[i] = []int32{7, -1, 2, 7, 31}[i%5]
		}
	}
	return q, values, opt
}

// fak-test:runtime fast est=100ms
func TestV41CompressedVisibilityReuse(t *testing.T) {
	for _, indexed := range []bool{false, true} {
		for _, seq := range []int{1, 4} {
			for _, heads := range []int{1, 8} {
				t.Run(fmt.Sprintf("indexed%t_seq%d_heads%d", indexed, seq, heads), func(t *testing.T) {
					q, v, opt := compressedVisibilityFixture(seq, heads, indexed)
					for _, offset := range []int{0, 7, 127} {
						opt.QueryOffset = offset
						for _, sink := range []bool{false, true} {
							opt.Sink = nil
							if sink {
								opt.Sink = make([]float32, heads)
							}
							var traces [2][][]float32
							var outputs [2][]float32
							for arm, fn := range []func([]float32, [][]float32, V41AttentionSharedKVOptions) ([]float32, error){v41CompressedVisibilityBaseline, V41AttentionCompressedForward} {
								o := opt
								o.RopeDim = 2
								o.Inverse = func(_ int, row []float32) error {
									traces[arm] = append(traces[arm], append([]float32(nil), row...))
									row[0] = -row[0]
									return nil
								}
								var err error
								outputs[arm], err = fn(q, v, o)
								if err != nil {
									t.Fatal(err)
								}
							}
							if !compressedVisibilityBitsEqual(outputs[0], outputs[1]) || !reflect.DeepEqual(traces[0], traces[1]) {
								t.Fatalf("offset%d sink%t numerical/callback mismatch", offset, sink)
							}
							before := append([]float32(nil), q...)
							outputs[1][0]++
							if !reflect.DeepEqual(before, q) {
								t.Fatal("output aliases queries")
							}
						}
					}
					opt.QueryOffset = 127
					opt.Sink = nil
					opt.Inverse = nil
					old := testing.AllocsPerRun(10, func() {
						if _, err := v41CompressedVisibilityBaseline(q, v, opt); err != nil {
							panic(err)
						}
					})
					got := testing.AllocsPerRun(10, func() {
						if _, err := V41AttentionCompressedForward(q, v, opt); err != nil {
							panic(err)
						}
					})
					t.Logf("baseline=%g candidate=%g", old, got)
					if heads > 1 && got >= old {
						t.Fatalf("visibility allocations not reduced: baseline=%g candidate=%g", old, got)
					}
					if heads == 1 && got != old {
						t.Fatalf("single-head allocations changed: baseline=%g candidate=%g", old, got)
					}
				})
			}
		}
	}
}

// fak-test:runtime fast est=20ms
func TestV41CompressedVisibilityRefusalParity(t *testing.T) {
	t.Run("inverse_failure_order", func(t *testing.T) {
		q, v, o := compressedVisibilityFixture(2, 8, true)
		sentinel := errors.New("inverse witness")
		var traces [2][][]float32
		var messages [2]string
		for arm, fn := range []func([]float32, [][]float32, V41AttentionSharedKVOptions) ([]float32, error){v41CompressedVisibilityBaseline, V41AttentionCompressedForward} {
			opt := o
			opt.RopeDim = 2
			opt.Inverse = func(_ int, row []float32) error {
				traces[arm] = append(traces[arm], append([]float32(nil), row...))
				if len(traces[arm]) == 3 {
					return sentinel
				}
				row[0] = -row[0]
				return nil
			}
			out, err := fn(q, v, opt)
			if out != nil || !errors.Is(err, sentinel) || len(traces[arm]) != 3 {
				t.Fatalf("arm%d output=%v error=%v calls=%d", arm, out, err, len(traces[arm]))
			}
			messages[arm] = err.Error()
		}
		if messages[0] != messages[1] || !reflect.DeepEqual(traces[0], traces[1]) {
			t.Fatal("inverse failure order differs")
		}
	})
	cases := []struct {
		name   string
		mutate func(*[]float32, [][]float32, *V41AttentionSharedKVOptions)
	}{
		{"ratio_before_heads", func(_ *[]float32, _ [][]float32, o *V41AttentionSharedKVOptions) { o.Ratio = 0; o.Heads = 0 }},
		{"width_before_query_nan", func(q *[]float32, v [][]float32, _ *V41AttentionSharedKVOptions) {
			v[0] = nil
			(*q)[0] = float32(math.NaN())
		}},
		{"query_nan_before_index", func(q *[]float32, _ [][]float32, o *V41AttentionSharedKVOptions) {
			(*q)[0] = float32(math.NaN())
			o.Idx[0] = 99
		}},
		{"sink_before_index", func(_ *[]float32, _ [][]float32, o *V41AttentionSharedKVOptions) {
			o.Sink = []float32{1}
			o.Idx[0] = 99
		}},
		{"index_out_of_range", func(_ *[]float32, _ [][]float32, o *V41AttentionSharedKVOptions) { o.Idx[0] = 99 }},
		{"score_overflow_later_head", func(q *[]float32, v [][]float32, _ *V41AttentionSharedKVOptions) {
			(*q)[8] = math.MaxFloat32
			v[7][0] = math.MaxFloat32
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			q, v, o := compressedVisibilityFixture(1, 8, true)
			tc.mutate(&q, v, &o)
			callbacks := 0
			o.RopeDim = 2
			o.Inverse = func(_ int, _ []float32) error { callbacks++; return nil }
			want, we := v41CompressedVisibilityBaseline(q, v, o)
			got, ge := V41AttentionCompressedForward(q, v, o)
			if we == nil || ge == nil || we.Error() != ge.Error() || want != nil || got != nil || callbacks != 0 {
				t.Fatalf("refusal mismatch: baseline=%v candidate=%v callbacks=%d", we, ge, callbacks)
			}
			if tc.name != "score_overflow_later_head" && !raceEnabled() {
				old := testing.AllocsPerRun(10, func() { _, _ = v41CompressedVisibilityBaseline(q, v, o) })
				fresh := testing.AllocsPerRun(10, func() { _, _ = V41AttentionCompressedForward(q, v, o) })
				if old != fresh {
					t.Fatalf("validation-refusal allocations changed: %g -> %g", old, fresh)
				}
			}
		})
	}
}

func BenchmarkV41CompressedVisibility(b *testing.B) {
	for _, seq := range []int{1, 16} {
		for _, heads := range []int{1, 8, 32} {
			for _, indexed := range []bool{false, true} {
				q, v, o := compressedVisibilityFixture(seq, heads, indexed)
				for _, arm := range []struct {
					name string
					fn   func([]float32, [][]float32, V41AttentionSharedKVOptions) ([]float32, error)
				}{{"baseline", v41CompressedVisibilityBaseline}, {"candidate", V41AttentionCompressedForward}} {
					b.Run(fmt.Sprintf("seq%d_heads%d_indexed%t/%s", seq, heads, indexed, arm.name), func(b *testing.B) {
						b.ReportAllocs()
						for i := 0; i < b.N; i++ {
							if _, err := arm.fn(q, v, o); err != nil {
								b.Fatal(err)
							}
						}
					})
				}
			}
		}
	}
}

// Captured unchanged contraction oracle from public41126964; only function name differs.
func v41CompressedVisibilityBaseline(q []float32, values [][]float32, opt V41AttentionSharedKVOptions) ([]float32, error) {
	if opt.Ratio < 1 {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention needs ratio >= 1, got %d", ErrV41ForwardStage, opt.Ratio))
	}
	if opt.Heads <= 0 || opt.HeadDim <= 0 {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention geometry heads=%d dim=%d", ErrV41ForwardStage, opt.Heads, opt.HeadDim))
	}
	queryWidth, queryOK := checkedMulInt(opt.Heads, opt.HeadDim)
	if !queryOK {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention query geometry overflows heads=%d dim=%d", ErrV41ForwardStage, opt.Heads, opt.HeadDim))
	}
	if opt.TopK < 0 || opt.Groups < 0 {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention topk=%d groups=%d", ErrV41ForwardStage, opt.TopK, opt.Groups))
	}
	if !finite32(opt.Softmax) || opt.Softmax == 0 {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention softmax scale must be finite and non-zero, got %g", ErrV41ForwardStage, opt.Softmax))
	}
	if opt.Inverse != nil && (opt.RopeDim <= 0 || opt.RopeDim > opt.HeadDim || opt.RopeDim%2 != 0) {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention inverse rotation needs an even rope dim in (0,%d], got %d", ErrV41ForwardStage, opt.HeadDim, opt.RopeDim))
	}
	if opt.HeadDim != v41KVHeadDim(opt) {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention head dim %d does not match the compressed row width", ErrV41ForwardStage, opt.HeadDim))
	}
	// Groups describes the supplied completed stream, not spare capacity. Check
	// it before allocating a mask or indexing a causally visible source row.
	if len(values) != opt.Groups {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed stream has %d rows, want %d", ErrV41ForwardStage, len(values), opt.Groups))
	}
	for g, row := range values {
		if len(row) != opt.HeadDim {
			return nil, v41StageErr(v41StageAttention, opt.Layer,
				fmt.Errorf("%w: compressed row %d width %d, want %d", ErrV41ForwardStage, g, len(row), opt.HeadDim))
		}
		for i, v := range row {
			if !finite32(v) {
				return nil, v41StageErr(v41StageAttention, opt.Layer,
					fmt.Errorf("%w: compressed row %d element %d is non-finite", ErrV41ForwardStage, g, i))
			}
		}
	}
	if opt.SourceRows != nil && len(opt.SourceRows) != opt.Groups {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: shared KV stream has %d rows, want %d", ErrV41ForwardStage, len(opt.SourceRows), opt.Groups))
	}
	if len(q)%queryWidth != 0 {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention query length %d not a multiple of heads*dim %d", ErrV41ForwardStage, len(q), queryWidth))
	}
	seq := len(q) / queryWidth
	for i, v := range q {
		if !finite32(v) {
			return nil, v41StageErr(v41StageAttention, opt.Layer,
				fmt.Errorf("%w: compressed attention query element %d is non-finite", ErrV41ForwardStage, i))
		}
	}
	if opt.Sink != nil {
		if len(opt.Sink) != opt.Heads {
			return nil, v41StageErr(v41StageAttention, opt.Layer,
				fmt.Errorf("%w: compressed attention sink length %d, want %d", ErrV41ForwardStage, len(opt.Sink), opt.Heads))
		}
		for i, v := range opt.Sink {
			if !finite32(v) {
				return nil, v41StageErr(v41StageAttention, opt.Layer,
					fmt.Errorf("%w: compressed attention sink element %d is non-finite", ErrV41ForwardStage, i))
			}
		}
	}

	if opt.QueryOffset < 0 || opt.QueryOffset > math.MaxInt-seq {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention query offset %d overflows %d rows", ErrV41ForwardStage, opt.QueryOffset, seq))
	}
	if opt.Idx != nil {
		if opt.IndexTopK <= 0 {
			return nil, v41StageErr(v41StageAttention, opt.Layer,
				fmt.Errorf("%w: compressed attention index list supplied with non-positive top-k %d", ErrV41ForwardStage, opt.IndexTopK))
		}
		indexCount, indexOK := checkedMulInt(seq, opt.IndexTopK)
		if !indexOK || len(opt.Idx) != indexCount {
			return nil, v41StageErr(v41StageAttention, opt.Layer,
				fmt.Errorf("%w: compressed attention index list length %d does not match seq=%d top-k=%d", ErrV41ForwardStage, len(opt.Idx), seq, opt.IndexTopK))
		}
		for i, row := range opt.Idx {
			if row < -1 || int(row) >= opt.Groups {
				return nil, v41StageErr(v41StageAttention, opt.Layer,
					fmt.Errorf("%w: compressed attention row index %d out of range at %d", ErrV41ForwardStage, row, i))
			}
		}
	}

	if _, maskOK := checkedMulInt(seq, opt.Groups); !maskOK {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention mask geometry overflows seq=%d groups=%d", ErrV41ForwardStage, seq, opt.Groups))
	}
	mask := v41CompressedCausalMaskAt(seq, opt.Ratio, opt.Groups, opt.QueryOffset)
	out := make([]float32, len(q))
	for t := 0; t < seq; t++ {
		qBase := t * opt.Heads * opt.HeadDim
		for h := 0; h < opt.Heads; h++ {
			hBase := qBase + h*opt.HeadDim
			// Resolve this position's visible groups. When the caller supplied
			// an index list, a group is visible only if it is both selected (-1
			// means empty) and causally complete; otherwise the causal mask is
			// the visibility authority.
			var visible []int
			if opt.Idx != nil {
				idxBase := t * opt.IndexTopK
				for i := 0; i < opt.IndexTopK; i++ {
					row := int(opt.Idx[idxBase+i])
					if row < 0 || !mask[t][row] {
						continue
					}
					visible = append(visible, row)
				}
			} else {
				for g := 0; g < opt.Groups; g++ {
					if mask[t][g] {
						visible = append(visible, g)
					}
				}
			}

			maxScore := float32(negInf32)
			if opt.Sink != nil {
				maxScore = opt.Sink[h]
			}
			for _, g := range visible {
				var dot float32
				row := values[g]
				for d := 0; d < opt.HeadDim; d++ {
					dot += float32(q[hBase+d] * row[d])
				}
				dot *= opt.Softmax
				if !finite32(dot) {
					return nil, v41CompressedNonFinite(opt.Layer, v41CompressedStageScore, t, h, g, -1, dot)
				}
				if dot > maxScore {
					maxScore = dot
				}
			}
			if maxScore == float32(negInf32) {
				continue
			}
			var sum float32
			if opt.Sink != nil {
				sum = exp32(opt.Sink[h] - maxScore)
			}
			for _, g := range visible {
				var dot float32
				row := values[g]
				for d := 0; d < opt.HeadDim; d++ {
					dot += float32(q[hBase+d] * row[d])
				}
				term := exp32(dot*opt.Softmax - maxScore)
				if !finite32(term) {
					return nil, v41CompressedNonFinite(opt.Layer, v41CompressedStageSoftmax, t, h, g, -1, term)
				}
				sum += term
			}
			if !finite32(sum) {
				return nil, v41CompressedNonFinite(opt.Layer, v41CompressedStageSoftmax, t, h, -1, -1, sum)
			}
			if sum == 0 {
				continue
			}
			for _, g := range visible {
				var dot float32
				row := values[g]
				for d := 0; d < opt.HeadDim; d++ {
					dot += float32(q[hBase+d] * row[d])
				}
				weight := exp32(dot*opt.Softmax-maxScore) / sum
				if !finite32(weight) {
					return nil, v41CompressedNonFinite(opt.Layer, v41CompressedStageValue, t, h, g, -1, weight)
				}
				for d := 0; d < opt.HeadDim; d++ {
					out[hBase+d] += float32(weight * row[d])
					if !finite32(out[hBase+d]) {
						return nil, v41CompressedNonFinite(opt.Layer, v41CompressedStageValue, t, h, g, d, out[hBase+d])
					}
				}
			}
		}
	}
	if opt.Inverse != nil {
		for t := 0; t < seq; t++ {
			for h := 0; h < opt.Heads; h++ {
				headBase := (t*opt.Heads + h) * opt.HeadDim
				tail := out[headBase+opt.HeadDim-opt.RopeDim : headBase+opt.HeadDim]
				if err := opt.Inverse(opt.RopeDim, tail); err != nil {
					return nil, v41StageErr(v41StageAttention, opt.Layer, err)
				}
			}
		}
	}
	return out, nil
}
