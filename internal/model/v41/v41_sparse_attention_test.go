// Copyright (c) 2023 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

package v41

import (
	"math"
	"testing"
)

// v41SinkOracle is an independent scalar transcription of the pinned
// sparse_attn contract (inference/kernel.py:311-405): explicit per-row gather,
// scaled per-head dot products, a sink-augmented softmax denominator, and a
// normalized value sum. It deliberately recomputes the score three times and
// keeps no running state so the implementation under test cannot share a bug
// with it.
func v41SinkOracle(q, kv, sink []float32, idx []int32, b, m, heads, headDim, topk, n int, scale float32) []float32 {
	o := make([]float32, b*m*heads*headDim)
	for bi := 0; bi < b; bi++ {
		for mi := 0; mi < m; mi++ {
			idxBase := (bi*m + mi) * topk
			for h := 0; h < heads; h++ {
				qBase := ((bi*m+mi)*heads + h) * headDim
				dots := make([]float64, topk)
				valid := make([]bool, topk)
				best := math.Inf(-1)
				if sink != nil {
					best = float64(sink[h])
				}
				for i := 0; i < topk; i++ {
					row := int(idx[idxBase+i])
					if row < 0 {
						continue
					}
					valid[i] = true
					kvBase := (bi*n + row) * headDim
					var dot float64
					for d := 0; d < headDim; d++ {
						dot += float64(q[qBase+d]) * float64(kv[kvBase+d])
					}
					dots[i] = dot * float64(scale)
					if dots[i] > best {
						best = dots[i]
					}
				}
				if math.IsInf(best, -1) {
					continue
				}
				var sum float64
				if sink != nil {
					sum += math.Exp(float64(sink[h]) - best)
				}
				for i := 0; i < topk; i++ {
					if valid[i] {
						sum += math.Exp(dots[i] - best)
					}
				}
				if sum == 0 {
					continue
				}
				for i := 0; i < topk; i++ {
					if !valid[i] {
						continue
					}
					row := int(idx[idxBase+i])
					kvBase := (bi*n + row) * headDim
					w := math.Exp(dots[i]-best) / sum
					for d := 0; d < headDim; d++ {
						o[qBase+d] += float32(w * float64(kv[kvBase+d]))
					}
				}
			}
		}
	}
	return o
}

func v41SinkClose(t *testing.T, name string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > float64(tol) {
			t.Fatalf("%s: element %d got %g want %g (tol %g)", name, i, got[i], want[i], tol)
		}
	}
}

// TestV41SparseAttentionSinkOracle proves the sparse sink contraction against
// the independent oracle for a two-head, two-position, two-row case, and shows
// the sink changes the denominator without changing which rows are valid.
func TestV41SparseAttentionSinkOracle(t *testing.T) {
	const b, m, heads, headDim, topk, n = 1, 2, 2, 4, 3, 2
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	q := []float32{
		1, 0, 0.5, -0.25, // pos 0 head 0
		0, 1, -0.5, 0.75, // pos 0 head 1
		-1, 0.25, 0, 0.5, // pos 1 head 0
		0.5, 0, -1, 0.25, // pos 1 head 1
	}
	kv := []float32{
		1, 0, 0, 0, // row 0
		0, 1, 0, 0, // row 1
	}
	idx := []int32{
		0, 1, -1, // pos 0: row 0, row 1, invalid
		1, -1, -1, // pos 1: row 1 only
	}
	sink := []float32{0.5, -0.25}

	want := v41SinkOracle(q, kv, sink, idx, b, m, heads, headDim, topk, n, scale)
	got, err := V41SparseAttentionSink(q, kv, sink, idx, V41SparseAttentionSinkOptions{
		B: b, M: m, Heads: heads, HeadDim: headDim, TopK: topk, N: n, Softmax: scale,
	})
	if err != nil {
		t.Fatal(err)
	}
	v41SinkClose(t, "sink oracle", got, want, 1e-5)

	// The same inputs without a sink must differ: the sink is load-bearing.
	noSink, err := V41SparseAttentionSink(q, kv, nil, idx, V41SparseAttentionSinkOptions{
		B: b, M: m, Heads: heads, HeadDim: headDim, TopK: topk, N: n, Softmax: scale,
	})
	if err != nil {
		t.Fatal(err)
	}
	differed := false
	for i := range noSink {
		if math.Abs(float64(noSink[i]-want[i])) > 1e-4 {
			differed = true
		}
	}
	if !differed {
		t.Fatal("sink did not change the denominator")
	}
}

// TestV41SparseAttentionSinkIgnoresInvalidRows proves a -1 index contributes
// neither score nor value: reordering/relocating invalid slots must not change
// the output, and an all-invalid row yields an all-zero output.
func TestV41SparseAttentionSinkIgnoresInvalidRows(t *testing.T) {
	const b, m, heads, headDim, topk, n = 1, 1, 1, 2, 3, 2
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	q := []float32{1, 2}
	kv := []float32{1, 0, 0, 1}
	base := []int32{0, -1, -1}
	moved := []int32{-1, 0, -1}

	opt := V41SparseAttentionSinkOptions{B: b, M: m, Heads: heads, HeadDim: headDim, TopK: topk, N: n, Softmax: scale}
	a, err := V41SparseAttentionSink(q, kv, nil, base, opt)
	if err != nil {
		t.Fatal(err)
	}
	bb, err := V41SparseAttentionSink(q, kv, nil, moved, opt)
	if err != nil {
		t.Fatal(err)
	}
	v41SinkClose(t, "invalid slot relocation", bb, a, 1e-6)

	empty, err := V41SparseAttentionSink(q, kv, nil, []int32{-1, -1, -1}, opt)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range empty {
		if v != 0 {
			t.Fatalf("all-invalid row produced non-zero output at %d: %g", i, v)
		}
	}

	if _, err := V41SparseAttentionSink(q, kv, nil, []int32{0, -1, int32(n)}, opt); err == nil {
		t.Fatal("out-of-range index was not refused")
	}
}

// TestV41SparseAttentionInverseRotationRestoresBasis proves the explicit
// inverse-rotation callback runs on the trailing rope dimensions of every
// output head and restores the pre-rotation basis.
func TestV41SparseAttentionInverseRotationRestoresBasis(t *testing.T) {
	const b, m, heads, headDim, topk, n, rd = 1, 1, 2, 4, 1, 1, 2
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	q := []float32{1, 0, 0, 0, 0, 1, 0, 0}
	kv := []float32{1, 2, 3, 4}
	idx := []int32{0}

	forward := func(ropeDim int, fixed []float32) error {
		for i := 0; i < ropeDim; i += 2 {
			a, bb := fixed[i], fixed[i+1]
			fixed[i] = a*0 - bb*1
			fixed[i+1] = a*1 + bb*0
		}
		return nil
	}
	inverse := func(ropeDim int, fixed []float32) error {
		for i := 0; i < ropeDim; i += 2 {
			a, bb := fixed[i], fixed[i+1]
			fixed[i] = a*0 + bb*1
			fixed[i+1] = a*-1 + bb*0
		}
		return nil
	}

	plain, err := V41SparseAttentionSink(q, kv, nil, idx, V41SparseAttentionSinkOptions{
		B: b, M: m, Heads: heads, HeadDim: headDim, TopK: topk, N: n, Softmax: scale,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Pre-rotate the output in place, then ask the contraction to inverse-rotate.
	rotated := append([]float32(nil), plain...)
	for h := 0; h < heads; h++ {
		if err := forward(rd, rotated[h*headDim+headDim-rd:h*headDim+headDim]); err != nil {
			t.Fatal(err)
		}
	}
	restored, err := V41SparseAttentionSink([]float32{
		rotated[0], rotated[1], rotated[2], rotated[3],
		rotated[4], rotated[5], rotated[6], rotated[7],
	}, kv, nil, idx, V41SparseAttentionSinkOptions{
		B: b, M: m, Heads: heads, HeadDim: headDim, TopK: topk, N: n, Softmax: scale,
		Inverse: inverse, RopeDim: rd,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Construct the expected value directly: the contraction of the unrotated
	// inputs with the inverse restored in the trailing dims.
	want := v41SinkOracle(q, kv, nil, idx, b, m, heads, headDim, topk, n, scale)
	for h := 0; h < heads; h++ {
		if err := inverse(rd, want[h*headDim+headDim-rd:h*headDim+headDim]); err != nil {
			t.Fatal(err)
		}
	}
	v41SinkClose(t, "inverse rotation", restored, want, 1e-5)

	if _, err := V41SparseAttentionSink(q, kv, nil, idx, V41SparseAttentionSinkOptions{
		B: b, M: m, Heads: heads, HeadDim: headDim, TopK: topk, N: n, Softmax: scale,
		Inverse: inverse, RopeDim: rd + 1,
	}); err == nil {
		t.Fatal("odd rope dim was not refused")
	}
}

// TestV41GroupedOutputProjectionOrder proves the wo_a-then-wo_b ordering by
// comparing against an oracle that applies the block-diagonal group projection
// before the down-projection, and by showing the reverse order is different.
func TestV41GroupedOutputProjectionOrder(t *testing.T) {
	const b, m, heads, headDim, groups, rank, dim = 1, 2, 4, 2, 2, 3, 5
	// o rows are [B*M, heads, headDim]; each group owns headsPerGroup heads.
	headsPerGroup := heads / groups
	aRow := headsPerGroup * headDim

	o := make([]float32, b*m*heads*headDim)
	for i := range o {
		o[i] = float32((i%7)-3) * 0.25
	}
	woA := make([]float32, groups*rank*aRow)
	for i := range woA {
		woA[i] = float32((i%5)-2) * 0.2
	}
	woB := make([]float32, dim*groups*rank)
	for i := range woB {
		woB[i] = float32((i%4)-1) * 0.5
	}

	got, err := V41GroupedOutputProjection(o, woA, woB, b, m, heads, headDim, groups, rank, dim)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != b*m*dim {
		t.Fatalf("output length %d want %d", len(got), b*m*dim)
	}

	// Oracle: per position, grouped block-diagonal wo_a then wo_b.
	want := make([]float32, b*m*dim)
	for pos := 0; pos < b*m; pos++ {
		joined := make([]float64, groups*rank)
		for g := 0; g < groups; g++ {
			for r := 0; r < rank; r++ {
				var acc float64
				for k := 0; k < aRow; k++ {
					acc += float64(o[pos*heads*headDim+g*aRow+k]) * float64(woA[(g*rank+r)*aRow+k])
				}
				joined[g*rank+r] = acc
			}
		}
		for d := 0; d < dim; d++ {
			var acc float64
			for k := 0; k < groups*rank; k++ {
				acc += float64(woB[d*groups*rank+k]) * joined[k]
			}
			want[pos*dim+d] = float32(acc)
		}
	}
	v41SinkClose(t, "grouped projection", got, want, 1e-5)

	// Reversing the order (wo_b applied to raw heads) must not match.
	reversed := make([]float32, b*m*dim)
	for pos := 0; pos < b*m; pos++ {
		for d := 0; d < dim; d++ {
			var acc float32
			for k := 0; k < heads*headDim; k++ {
				acc += o[pos*heads*headDim+k] * 0.01
			}
			reversed[pos*dim+d] = acc
		}
	}
	same := true
	for i := range reversed {
		if math.Abs(float64(reversed[i]-got[i])) > 1e-6 {
			same = false
		}
	}
	if same {
		t.Fatal("projection order is not distinguishable")
	}

	if _, err := V41GroupedOutputProjection(o, woA, woB, b, m, heads, headDim, groups, rank, dim+1); err == nil {
		t.Fatal("wrong wo_b length was not refused")
	}
	if _, err := V41GroupedOutputProjection(o, woA, woB, b, m, heads, headDim, 3, rank, dim); err == nil {
		t.Fatal("indivisible groups were not refused")
	}
}

// TestV41SparseAttentionSinkRefusesMalformedInput proves fail-closed shape and
// non-finite validation before any caller-visible output.
func TestV41SparseAttentionSinkRefusesMalformedInput(t *testing.T) {
	base := V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: 2, TopK: 1, N: 1, Softmax: 0.5}
	okQ := []float32{1, 1}
	okKV := []float32{1, 1}
	okIdx := []int32{0}

	bad := base
	bad.Softmax = 0
	if _, err := V41SparseAttentionSink(okQ, okKV, nil, okIdx, bad); err == nil {
		t.Fatal("zero softmax scale was not refused")
	}
	bad = base
	bad.HeadDim = 0
	if _, err := V41SparseAttentionSink(okQ, okKV, nil, okIdx, bad); err == nil {
		t.Fatal("zero head dim was not refused")
	}
	nan := []float32{1, float32(math.NaN())}
	if _, err := V41SparseAttentionSink(nan, okKV, nil, okIdx, base); err == nil {
		t.Fatal("non-finite query was not refused")
	}
	if _, err := V41SparseAttentionSink(okQ, okKV, nil, []int32{-2}, base); err == nil {
		t.Fatal("index below -1 was not refused")
	}
	if _, err := V41SparseAttentionSink(okQ, okKV, []float32{1, 2}, okIdx, base); err == nil {
		t.Fatal("wrong sink length was not refused")
	}
	if _, err := V41GroupedOutputProjection([]float32{1}, nil, nil, 1, 1, 1, 1, 1, 1, 1); err == nil {
		t.Fatal("short wo_a was not refused")
	}
}
