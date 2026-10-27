package model

import (
	"math"
	"slices"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
// Source estimate: three tiny contractions, each with 12 heads and five slots.
func TestV41SparseSinkScoreReuseMatchesRecomputation(t *testing.T) {
	t.Parallel()
	opt := V41SparseAttentionSinkOptions{
		B: 2, M: 3, Heads: 2, HeadDim: 4, TopK: 5, N: 3, Softmax: .75,
		TopKLength: []int32{5, 2, 0, 3, 5, 1},
	}
	q := make([]float32, opt.B*opt.M*opt.Heads*opt.HeadDim)
	kv := make([]float32, opt.B*opt.N*opt.HeadDim)
	for i := range q {
		q[i] = float32(i%13-6) / 11
	}
	for i := range kv {
		kv[i] = float32(i%7-3) / 7
	}
	// Finite scores whose gap from the negative extreme sink rounds to -Inf.
	q[0], q[4], kv[0] = math.MaxFloat32/4, math.MaxFloat32/4, 1
	idx := []int32{
		2, -1, 0, 2, 1, // duplicate rows remain distinct contributions
		-1, 1, 0, 2, 0, // trailing valid IDs are masked by TopKLength
		0, 1, 2, 1, 0, // no valid slots, after a populated scratch row
		1, 0, -1, 2, 1,
		-1, -1, -1, -1, -1,
		2, 1, 0, 1, 2,
	}
	qBefore, kvBefore, idxBefore := slices.Clone(q), slices.Clone(kv), slices.Clone(idx)
	for _, sink := range [][]float32{nil, {.25, -.75}, {math.MaxFloat32, -math.MaxFloat32}} {
		want := v41SparseSinkRecomputedScoreReference(q, kv, sink, idx, opt)
		// The normal host-dispatch seam must preserve inverse callback order
		// and contents, not merely the final numerical result.
		calls := 0
		withInverse := opt
		withInverse.RopeDim = 2
		withInverse.Inverse = func(_ int, tail []float32) error {
			base := calls*opt.HeadDim + opt.HeadDim - 2
			for d := range tail {
				if math.Float32bits(tail[d]) != math.Float32bits(want[base+d]) {
					t.Fatalf("inverse head %d element %d changed", calls, d)
				}
			}
			tail[0] = -tail[0]
			want[base] = -want[base]
			calls++
			return nil
		}
		got, err := v41SparseAttentionSinkResident(0, q, kv, sink, idx, withInverse, nil, nil)
		if err != nil || len(got) != len(want) || calls != opt.B*opt.M*opt.Heads {
			t.Fatalf("host contraction: len=%d callbacks=%d err=%v", len(got), calls, err)
		}
		for i := range got {
			if !finite32(got[i]) || math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("element %d: got %g (%08x), want %g (%08x)", i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}
	if !slices.Equal(q, qBefore) || !slices.Equal(kv, kvBefore) || !slices.Equal(idx, idxBefore) {
		t.Fatal("contraction changed caller-owned inputs")
	}
}

// The pre-optimization F32 arithmetic: independently recompute the dot in each
// pass and the exponential in both latter passes. Valid fixtures only; public
// admission and non-finite error behavior retain their existing focused tests.
func v41SparseSinkRecomputedScoreReference(q, kv, sink []float32, idx []int32, opt V41SparseAttentionSinkOptions) []float32 {
	out := make([]float32, len(q))
	for b := 0; b < opt.B; b++ {
		for m := 0; m < opt.M; m++ {
			count := opt.TopK
			if opt.TopKLength != nil {
				count = int(opt.TopKLength[b*opt.M+m])
			}
			for h := 0; h < opt.Heads; h++ {
				base := ((b*opt.M+m)*opt.Heads + h) * opt.HeadDim
				score := func(row int32) float32 {
					var dot float32
					for d := 0; d < opt.HeadDim; d++ {
						dot += float32(q[base+d] * kv[(b*opt.N+int(row))*opt.HeadDim+d])
					}
					return float32(dot * opt.Softmax)
				}
				slots := idx[(b*opt.M+m)*opt.TopK:][:count]
				best := float32(math.Inf(-1))
				if sink != nil {
					best = sink[h]
				}
				for _, row := range slots {
					if row >= 0 {
						if s := score(row); s > best {
							best = s
						}
					}
				}
				if math.IsInf(float64(best), -1) {
					continue
				}
				var sum float32
				if sink != nil {
					sum = float32(math.Exp(float64(sink[h] - best)))
				}
				for _, row := range slots {
					if row >= 0 {
						sum += float32(math.Exp(float64(score(row) - best)))
					}
				}
				if sum == 0 {
					continue
				}
				for _, row := range slots {
					if row >= 0 {
						weight := float32(math.Exp(float64(score(row)-best))) / sum
						for d := 0; d < opt.HeadDim; d++ {
							out[base+d] += float32(weight * kv[(b*opt.N+int(row))*opt.HeadDim+d])
						}
					}
				}
			}
		}
	}
	return out
}
