package compute

import (
	"math"
	"testing"
)

// Closed host oracle for microsoft/MInference PIT mixed-sparse prefill attention (ADAPT).
// [SW-VERIFIED] structure and oracle only: no hardware, no speed claim.
func pitPaddedHeadDim(d int) int {
	for _, bucket := range []int{16, 32, 64, 128} {
		if d <= bucket {
			return bucket
		}
	}
	return 128
}

func pitSparseAttention(q, k, v [][]float32, verticalCols []int, slashOffsets []int) [][]float32 {
	seqLen := len(q)
	rawDim := len(q[0])
	padDim := pitPaddedHeadDim(rawDim)
	scale := 1.0 / math.Sqrt(float64(rawDim))

	// Zero-pad Q, K, V to {16,32,64,128} as PIT does.
	pad := func(src [][]float32) [][]float64 {
		dst := make([][]float64, len(src))
		for i := range src {
			dst[i] = make([]float64, padDim)
			for j := range src[i] {
				dst[i][j] = float64(src[i][j])
			}
		}
		return dst
	}
	qp, kp, vp := pad(q), pad(k), pad(v)

	out := make([][]float32, seqLen)
	for qi := 0; qi < seqLen; qi++ {
		active := make([]bool, seqLen)
		for _, col := range verticalCols {
			if col >= 0 && col <= qi {
				active[col] = true
			}
		}
		for _, off := range slashOffsets {
			kj := qi - off
			if kj >= 0 && kj <= qi {
				active[kj] = true
			}
		}
		maxScore := math.Inf(-1)
		scores := make([]float64, seqLen)
		for kj := 0; kj <= qi; kj++ {
			if !active[kj] {
				continue
			}
			var dot float64
			for d := 0; d < padDim; d++ {
				dot += qp[qi][d] * kp[kj][d]
			}
			scores[kj] = dot * scale
			if scores[kj] > maxScore {
				maxScore = scores[kj]
			}
		}
		var sumExp float64
		for kj := 0; kj <= qi; kj++ {
			if active[kj] {
				scores[kj] = math.Exp(scores[kj] - maxScore)
				sumExp += scores[kj]
			}
		}
		out[qi] = make([]float32, rawDim)
		for d := 0; d < rawDim; d++ {
			var acc float64
			for kj := 0; kj <= qi; kj++ {
				if active[kj] {
					acc += (scores[kj] / sumExp) * vp[kj][d]
				}
			}
			out[qi][d] = float32(acc)
		}
	}
	return out
}

func TestPITSparsePrefillKernel(t *testing.T) {
	t.Parallel()
	if cuhDefine(t, "pit_sparse_prefill.cuh", "FAK_PIT_BLOCK_M") != 64 ||
		cuhDefine(t, "pit_sparse_prefill.cuh", "FAK_PIT_BLOCK_N") != 64 ||
		cuhDefine(t, "pit_sparse_prefill.cuh", "FAK_PIT_HEAD_DIM_MIN") != 16 ||
		cuhDefine(t, "pit_sparse_prefill.cuh", "FAK_PIT_HEAD_DIM_MAX") != 128 {
		t.Fatal("pit_sparse_prefill.cuh constants drifted")
	}
	if pitPaddedHeadDim(12) != 16 || pitPaddedHeadDim(48) != 64 || pitPaddedHeadDim(128) != 128 {
		t.Fatal("head_dim padding bucket mismatch")
	}
	// Non-bucket head_dim=3 padded to 16; vertical col 0 + slash offset 0 (diagonal) & 1.
	q := [][]float32{{1, 0, 1}, {0, 1, 1}, {1, 1, 0}, {1, 0, 0}}
	k := [][]float32{{1, 1, 0}, {0, 1, 1}, {1, 0, 1}, {0, 0, 1}}
	v := [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}
	got := pitSparseAttention(q, k, v, []int{0}, []int{0, 1})
	// Row 0 only attends col 0 -> output equals v[0].
	for d := range v[0] {
		if math.Abs(float64(got[0][d]-v[0][d])) > 1e-5 {
			t.Fatalf("row 0 d=%d: got %v, want %v", d, got[0][d], v[0][d])
		}
	}
	// Row 3 attends {0, 2, 3} (vertical 0 + slash 0,1) -> output must be a convex combo of v[0],v[2],v[3].
	if got[3][0] <= 1.0 || got[3][0] >= 10.0 {
		t.Fatalf("row 3 output %v not a convex combination of active rows", got[3][0])
	}
}
