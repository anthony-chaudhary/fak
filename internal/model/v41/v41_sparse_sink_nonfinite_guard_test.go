package v41

// v41_sparse_sink_nonfinite_guard_test.go - the permanent RED->GREEN regression
// for the per-position V4.1 sparse sink half of fak#13290. The layer-36
// schedule resolves Ratio=1 onto this contraction (see the package-model
// witness TestV41Layer36RealScheduleBranch): with HeadDim=512 a finite
// per-element magnitude above ~8.15e17 overflows the 512-term score accumulate
// to +Inf, which makes exp32(Inf-Inf)=NaN poison the softmax denominator and the
// weighted value sum. Before the guard the contraction returned a NaN output
// with no error; the failure then surfaced misleadingly at the downstream
// V41GroupedOutputProjection as "non-finite value element 0".
//
// POST-FIX WITNESS: the sink refuses with ErrV41SparseSinkNonFinite naming the
// TRUE producing stage, batch, position, head and offending element.

import (
	"errors"
	"math"
	"strings"
	"testing"

	model "github.com/anthony-chaudhary/fak/internal/model"
)

// TestV41SparseSinkNonFiniteGuard is the RED->GREEN witness. At parent
// 193c17360 finite inputs produced a non-finite output with a nil error; the
// post-fix contract is a typed refusal from the contraction itself.
func TestV41SparseSinkNonFiniteGuard(t *testing.T) {
	const (
		heads   = 64
		headDim = 512
		rows    = 4
	)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	overflowMag := float32(1e20) // 1e20*1e20 = 1e40 > MaxFloat32

	q := make([]float32, heads*headDim)
	for i := range q {
		q[i] = overflowMag
	}
	flatKV := make([]float32, rows*headDim)
	for i := range flatKV {
		flatKV[i] = overflowMag
	}
	idx := make([]int32, rows+1)
	for i := 0; i < rows; i++ {
		idx[i] = int32(i)
	}
	idx[rows] = -1
	sink := make([]float32, heads)
	for h := range sink {
		sink[h] = 0.25
	}

	out, err := V41SparseAttentionSink(q, flatKV, sink, idx, V41SparseAttentionSinkOptions{
		B: 1, M: 1, Heads: heads, HeadDim: headDim, TopK: rows + 1, N: rows, Softmax: scale,
	})
	if err == nil {
		t.Fatalf("sparse sink returned %d outputs from overflowing finite inputs; want a typed refusal (RED-on-parent)", len(out))
	}
	if !errors.Is(err, ErrV41SparseSinkNonFinite) {
		t.Fatalf("sparse sink error = %v, want errors.Is(err, ErrV41SparseSinkNonFinite)", err)
	}
	msg := err.Error()
	for _, want := range []string{"sparse sink", "produced a non-finite value", "b=0", "m=0", "h=", "value="} {
		if !strings.Contains(msg, want) {
			t.Fatalf("sparse sink refusal message %q does not name %q", msg, want)
		}
	}
	t.Logf("GREEN sparse-sink refusal: %v", err)
}

// TestV41SparseSinkFiniteMatchesOracle pins that a NORMAL O(1) row is
// unaffected: it still matches an independent scalar transcription and stays
// finite, so the guard is purely additive on the finite path.
func TestV41SparseSinkFiniteMatchesOracle(t *testing.T) {
	const (
		heads   = 4
		headDim = 8
		topk    = 3
		n       = 5
	)
	scale := float32(0.125)

	q := make([]float32, heads*headDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i))) * 0.5
	}
	kv := make([]float32, n*headDim)
	for i := range kv {
		kv[i] = float32(math.Cos(float64(i))) * 0.5
	}
	idx := []int32{0, 2, -1}
	sink := []float32{0.1, -0.2, 0.3, 0.4}

	got, err := V41SparseAttentionSink(q, kv, sink, idx, V41SparseAttentionSinkOptions{
		B: 1, M: 1, Heads: heads, HeadDim: headDim, TopK: topk, N: n, Softmax: scale,
	})
	if err != nil {
		t.Fatalf("sparse sink (finite O(1)) refused: %v", err)
	}
	for h := 0; h < heads; h++ {
		var rowsSeen []int
		for _, r := range idx {
			if r >= 0 {
				rowsSeen = append(rowsSeen, int(r))
			}
		}
		maxScore := float64(sink[h])
		dots := make([]float64, len(rowsSeen))
		for i, r := range rowsSeen {
			var d float64
			for j := 0; j < headDim; j++ {
				d += float64(q[h*headDim+j]) * float64(kv[r*headDim+j])
			}
			d *= float64(scale)
			dots[i] = d
			if d > maxScore {
				maxScore = d
			}
		}
		sum := math.Exp(float64(sink[h]) - maxScore)
		for _, d := range dots {
			sum += math.Exp(d - maxScore)
		}
		// Accumulate the expected weighted value over every visible row, exactly
		// as the production weighted-value loop does.
		exp := make([]float64, headDim)
		for i, r := range rowsSeen {
			w := math.Exp(dots[i]-maxScore) / sum
			for j := 0; j < headDim; j++ {
				exp[j] += w * float64(kv[r*headDim+j])
			}
		}
		for j := 0; j < headDim; j++ {
			want := float32(exp[j])
			if d := math.Abs(float64(got[h*headDim+j] - want)); d > 1e-5 {
				t.Fatalf("h=%d d=%d: got %g want %g (|delta|=%.3e)", h, j, got[h*headDim+j], want, d)
			}
			if !model.Finite32(got[h*headDim+j]) {
				t.Fatalf("h=%d d=%d: finite inputs produced non-finite %v", h, j, got[h*headDim+j])
			}
		}
	}
}
