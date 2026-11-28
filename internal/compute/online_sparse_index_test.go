package compute

import (
	"math"
	"slices"
	"sort"
	"testing"
)

// Closed host oracle for microsoft/MInference online sparse-index approximation (ADAPT).
// [SW-VERIFIED] structure and oracle only: last-q block vertical + slash top-k without full recompute.
func onlineCalcIndex(lastQ, k [][]float32, topV, topS int) (vCols, sOffsets []int) {
	qLen := len(lastQ)
	kLen := len(k)
	dim := len(lastQ[0])
	scale := 1.0 / math.Sqrt(float64(dim))

	vScores := make([]float64, kLen)
	sScores := make([]float64, kLen)
	// Causal last-q block: query i corresponds to global position (kLen - qLen + i).
	for i := 0; i < qLen; i++ {
		gPos := kLen - qLen + i
		for j := 0; j <= gPos; j++ {
			var dot float64
			for d := 0; d < dim; d++ {
				dot += float64(lastQ[i][d]) * float64(k[j][d])
			}
			w := math.Exp(dot * scale)
			vScores[j] += w
			sScores[gPos-j] += w // slash diagonal offset = gPos - j
		}
	}

	topKIdx := func(scores []float64, kCount int) []int {
		idx := make([]int, len(scores))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool {
			if scores[idx[a]] != scores[idx[b]] {
				return scores[idx[a]] > scores[idx[b]]
			}
			return idx[a] < idx[b]
		})
		if kCount > len(idx) {
			kCount = len(idx)
		}
		out := idx[:kCount]
		slices.Sort(out)
		return out
	}
	return topKIdx(vScores, topV), topKIdx(sScores, topS)
}

func TestOnlineSparseIndexApprox(t *testing.T) {
	t.Parallel()
	topV := cuhDefine(t, "online_sparse_index.cuh", "FAK_ONLINE_SPARSE_TOP_V")
	topS := cuhDefine(t, "online_sparse_index.cuh", "FAK_ONLINE_SPARSE_TOP_S")
	if cuhDefine(t, "online_sparse_index.cuh", "FAK_ONLINE_SPARSE_LAST_Q_ONLY") != 1 {
		t.Fatal("online_sparse_index.cuh constants drifted")
	}

	// 4 keys, last-q block of 2 queries (global pos 2, 3).
	// Key 0 has strong alignment with both queries (vertical sink); diagonal offset 0 (self) is strong.
	lastQ := [][]float32{{1, 0}, {1, 0}}
	k := [][]float32{
		{5, 0}, // col 0: attended by both queries
		{5, 0}, // col 1: attended by both queries
		{4, 0}, // col 2: attended by pos 2,3
		{1, 0}, // col 3: attended by pos 3 only
	}
	vCols, sOffsets := onlineCalcIndex(lastQ, k, topV, topS)
	if !slices.Equal(vCols, []int{0, 1}) {
		t.Fatalf("vertical top-2 = %v, want [0 1]", vCols)
	}
	if !slices.Contains(sOffsets, 1) || !slices.Contains(sOffsets, 2) {
		t.Fatalf("slash top-2 = %v, want [1 2]", sOffsets)
	}
}
