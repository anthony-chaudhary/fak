package model

import (
	"math"
	"slices"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41SparseSinkParallelHeads(t *testing.T) {
	old := NumWorkers()
	defer SetWorkers(old)
	opt := V41SparseAttentionSinkOptions{B: 2, M: 2, Heads: 8, HeadDim: 64, TopK: 257, N: 257, Softmax: 0.125}
	q, kv := make([]float32, opt.B*opt.M*opt.Heads*opt.HeadDim), make([]float32, opt.B*opt.N*opt.HeadDim)
	idx, sink := make([]int32, opt.B*opt.M*opt.TopK), make([]float32, opt.Heads)
	for i := range q {
		q[i] = float32(i%23-11) / 32
	}
	for i := range kv {
		kv[i] = float32(i%31-15) / 64
	}
	for i := range idx {
		idx[i] = int32(i % 101)
		if i%13 == 0 {
			idx[i] = -1
		}
	}
	for i := range sink {
		sink[i] = float32(i-4) / 8
	}
	q0, kv0, idx0, sink0 := slices.Clone(q), slices.Clone(kv), slices.Clone(idx), slices.Clone(sink)
	for _, masked := range []bool{false, true} {
		opt.TopKLength = nil
		if masked {
			opt.TopKLength = []int32{257, 0, 67, 129}
		}
		want, err := v41SparseSinkSerialOracle(q, sink, idx, opt, v41SparseSinkRows{flat: kv})
		if err != nil {
			t.Fatal(err)
		}
		for _, workers := range []int{1, 4} {
			if err := SetWorkers(workers); err != nil {
				t.Fatal(err)
			}
			got, err := V41SparseAttentionSink(q, kv, sink, idx, opt)
			if err != nil || len(got) != len(want) {
				t.Fatalf("workers=%d: %v", workers, err)
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("workers=%d element=%d", workers, i)
				}
			}
		}
	}
	if !slices.Equal(q, q0) || !slices.Equal(kv, kv0) || !slices.Equal(idx, idx0) || !slices.Equal(sink, sink0) {
		t.Fatal("input mutation")
	}
	// Competing failures must still report the earliest head, and no callback
	// may run when any contraction fails. All inputs remain finite.
	opt.TopKLength = nil
	for i := range kv {
		kv[i] = 2
	}
	for _, h := range []int{1, 6} {
		for d := 0; d < opt.HeadDim; d++ {
			q[h*opt.HeadDim+d] = math.MaxFloat32
		}
	}
	called := false
	opt.RopeDim = 2
	opt.Inverse = func(_ int, _ []float32) error { called = true; return nil }
	for _, workers := range []int{1, 4} {
		if err := SetWorkers(workers); err != nil {
			t.Fatal(err)
		}
		got, err := V41SparseAttentionSink(q, kv, sink, idx, opt)
		if got != nil || err == nil || !strings.Contains(err.Error(), "b=0 m=0 h=1") || called {
			t.Fatalf("workers=%d result=%v callback=%v", workers, err, called)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41SparseSinkParallelInverseOrderAndFinalValidation(t *testing.T) {
	old := NumWorkers()
	defer SetWorkers(old)
	if err := SetWorkers(4); err != nil {
		t.Fatal(err)
	}
	opt := V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 8, HeadDim: 64, TopK: 257, N: 1, Softmax: 1, RopeDim: 2}
	q, kv, idx := make([]float32, 8*64), make([]float32, 64), make([]int32, 257)
	var tails [][]float32
	opt.Inverse = func(_ int, tail []float32) error {
		tails = append(tails, tail)
		tail[0] = float32(len(tails))
		return nil
	}
	got, err := V41SparseAttentionSink(q, kv, nil, idx, opt)
	if err != nil || len(tails) != 8 {
		t.Fatal(err, len(tails))
	}
	for h := 0; h < 8; h++ {
		if got[h*64+62] != float32(h+1) {
			t.Fatal("callback order")
		}
	}
	tails = nil
	opt.Inverse = func(_ int, tail []float32) error {
		tails = append(tails, tail)
		if len(tails) == 8 {
			tails[0][0] = float32(math.NaN())
		}
		return nil
	}
	got, err = V41SparseAttentionSink(q, kv, nil, idx, opt)
	if got != nil || err == nil || !strings.Contains(err.Error(), "inverse rotation") || !strings.Contains(err.Error(), "h=0") {
		t.Fatal("retained tail was not revalidated", err)
	}
}

// Frozen pre-change scalar contraction: independent scheduling/error-order oracle.
func v41SparseSinkSerialOracle(q, sink []float32, idx []int32, opt V41SparseAttentionSinkOptions, rows v41SparseSinkRows) ([]float32, error) {
	negInf := float32(math.Inf(-1))
	o := make([]float32, opt.B*opt.M*opt.Heads*opt.HeadDim)
	// Each head reuses its rounded scores and then its exponential terms. The
	// inputs stay read-only until contraction ends, so recomputing either in
	// later passes only repeats work. Masked slots are never read from scratch.
	// Bound scratch by the existing index payload even for overflowed shapes.
	terms := make([]float32, min(opt.TopK, len(idx)))
	for b := 0; b < opt.B; b++ {
		for m := 0; m < opt.M; m++ {
			idxBase := (b*opt.M + m) * opt.TopK
			// validCount is the number of leading slots that count; slots at
			// index >= validCount are padding and contribute nothing.
			validCount := opt.TopK
			if opt.TopKLength != nil {
				validCount = int(opt.TopKLength[b*opt.M+m])
			}
			for h := 0; h < opt.Heads; h++ {
				qBase := ((b*opt.M+m)*opt.Heads + h) * opt.HeadDim
				// Stage 1: max over valid scaled scores, seeded with the sink.
				maxScore := negInf
				if sink != nil {
					maxScore = sink[h]
				}
				for i := 0; i < validCount; i++ {
					row := int(idx[idxBase+i])
					if row < 0 {
						continue
					}
					values := rows.row(b, row, opt.N, opt.HeadDim)
					var dot float32
					for d := 0; d < opt.HeadDim; d++ {
						dot += float32(q[qBase+d] * values[d])
					}
					dot *= opt.Softmax
					if !finite32(dot) {
						return nil, sparseSinkNonFinite(v41SinkStageScore, b, m, h, row, -1, dot)
					}
					terms[i] = dot
					if dot > maxScore {
						maxScore = dot
					}
				}
				if maxScore == negInf {
					// No valid row and no sink: the reference's finite bound
					// yields an all-zero output row.
					continue
				}
				// Stage 2: softmax denominator with the sink included.
				var sum float32
				if sink != nil {
					sum = exp32(sink[h] - maxScore)
				}
				for i := 0; i < validCount; i++ {
					row := int(idx[idxBase+i])
					if row < 0 {
						continue
					}
					// Match the rounded float32 score used by the maximum pass.
					// A finite score gap may round to -Inf; exp(-Inf)=0 is valid.
					term := exp32(terms[i] - maxScore)
					if !finite32(term) {
						return nil, sparseSinkNonFinite(v41SinkStageSoftmax, b, m, h, row, -1, term)
					}
					terms[i] = term
					sum += term
				}
				if !finite32(sum) {
					return nil, sparseSinkNonFinite(v41SinkStageSoftmax, b, m, h, -1, -1, sum)
				}
				if sum == 0 {
					continue
				}
				// Stage 3: weighted value sum, normalized by the denominator.
				oBase := qBase
				for i := 0; i < validCount; i++ {
					row := int(idx[idxBase+i])
					if row < 0 {
						continue
					}
					values := rows.row(b, row, opt.N, opt.HeadDim)
					weight := terms[i] / sum
					if !finite32(weight) {
						return nil, sparseSinkNonFinite(v41SinkStageValue, b, m, h, row, -1, weight)
					}
					for d := 0; d < opt.HeadDim; d++ {
						o[oBase+d] += float32(weight * values[d])
						if !finite32(o[oBase+d]) {
							return nil, sparseSinkNonFinite(v41SinkStageValue, b, m, h, row, d, o[oBase+d])
						}
					}
				}
			}
		}
	}
	if opt.Inverse != nil {
		for b := 0; b < opt.B; b++ {
			for m := 0; m < opt.M; m++ {
				for h := 0; h < opt.Heads; h++ {
					headBase := ((b*opt.M+m)*opt.Heads + h) * opt.HeadDim
					tail := o[headBase+opt.HeadDim-opt.RopeDim : headBase+opt.HeadDim]
					if err := opt.Inverse(opt.RopeDim, tail); err != nil {
						return nil, err
					}
				}
			}
		}
		// A callback can retain a previously supplied tail. Validate the final
		// tensor after every callback has returned, not only its latest tail.
		for i, value := range o {
			if !finite32(value) {
				head := i / opt.HeadDim
				return nil, sparseSinkNonFinite(v41SinkStageInverse, head/(opt.M*opt.Heads), head/opt.Heads%opt.M, head%opt.Heads, -1, i%opt.HeadDim, value)
			}
		}
	}
	return o, nil
}
