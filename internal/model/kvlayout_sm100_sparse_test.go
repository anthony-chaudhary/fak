package model

import (
	"reflect"
	"testing"
)

func TestSM100FP8SparseIndexScorerSelectorFailsClosed(t *testing.T) {
	valid := sparseIndexScorerRequest{
		SMVersion:   100,
		QueryDType:  sparseIndexDTypeFP8E4M3FN,
		KeyDType:    sparseIndexDTypeFP8E4M3FN,
		Depth:       128,
		NumHeads:    64,
		PageSize:    128,
		SparseTopK:  4,
		KeysPerPool: 1,
	}
	if got := selectSparseIndexScorer(valid); got != sparseIndexScorerSM100FP8 {
		t.Fatalf("valid SM100 FP8 request selected %v, want %v", got, sparseIndexScorerSM100FP8)
	}

	tests := []struct {
		name   string
		mutate func(*sparseIndexScorerRequest)
	}{
		{"pre-sm100", func(r *sparseIndexScorerRequest) { r.SMVersion = 90 }},
		{"future-architecture-not-explicit", func(r *sparseIndexScorerRequest) { r.SMVersion = 110 }},
		{"query-not-fp8", func(r *sparseIndexScorerRequest) { r.QueryDType = sparseIndexDTypeF32 }},
		{"key-not-fp8", func(r *sparseIndexScorerRequest) { r.KeyDType = sparseIndexDTypeF32 }},
		{"depth-not-128", func(r *sparseIndexScorerRequest) { r.Depth = 64 }},
		{"unsupported-head-count", func(r *sparseIndexScorerRequest) { r.NumHeads = 16 }},
		{"page-straddles-128-row-tile", func(r *sparseIndexScorerRequest) { r.PageSize = 192 }},
		{"sparse-width-missing", func(r *sparseIndexScorerRequest) { r.SparseTopK = 0 }},
		{"pooled-key-shape-not-explicit", func(r *sparseIndexScorerRequest) { r.KeysPerPool = 2 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := valid
			tt.mutate(&req)
			if got := selectSparseIndexScorer(req); got != sparseIndexScorerF32 {
				t.Fatalf("selected %v, want fail-closed f32 path %v", got, sparseIndexScorerF32)
			}
		})
	}

	for _, heads := range []int{4, 8, 32, 64} {
		req := valid
		req.NumHeads = heads
		if got := selectSparseIndexScorer(req); got != sparseIndexScorerSM100FP8 {
			t.Errorf("validated head count %d selected %v", heads, got)
		}
	}
	for _, pageSize := range []int{0, 128, 256} {
		req := valid
		req.PageSize = pageSize
		if got := selectSparseIndexScorer(req); got != sparseIndexScorerSM100FP8 {
			t.Errorf("supported page size %d selected %v", pageSize, got)
		}
	}
}

func TestSM100FP8SparseIndexScorerF32OracleAcrossPages(t *testing.T) {
	const (
		depth    = 128
		pageSize = 128
		nKeys    = 259
		topK     = 7
	)
	q := make([]float32, depth)
	for d := range q {
		q[d] = float32((d%11)-5) / 8
	}
	keys := make([][]float32, nKeys)
	for row := range keys {
		keys[row] = make([]float32, depth)
		for d := range keys[row] {
			keys[row][d] = float32(((row+3)*(d+5))%37-18) / 16
		}
	}
	// Put distinct maxima immediately around both page boundaries.
	for _, row := range []int{pageSize - 1, pageSize, 2*pageSize - 1, 2 * pageSize} {
		for d := range keys[row] {
			keys[row][d] = q[d] * float32(row+1)
		}
	}

	gotScores := scoreSparseIndexF32(q, keys, depth)
	wantScores, wantTop := referenceSparseIndexScoreAndTopK(q, keys, topK)
	if !reflect.DeepEqual(gotScores, wantScores) {
		t.Fatal("f32 fallback scores differ from independently implemented CPU oracle")
	}
	if gotTop := topSparseIndexF32(gotScores, topK); !reflect.DeepEqual(gotTop, wantTop) {
		t.Fatalf("f32 fallback top-k = %v, CPU oracle = %v", gotTop, wantTop)
	}
}

// referenceSparseIndexScoreAndTopK intentionally does not call production dot,
// scorer, sorting, or selection helpers: it is the independent CPU witness.
func referenceSparseIndexScoreAndTopK(q []float32, keys [][]float32, topK int) ([]float32, []int) {
	scores := make([]float32, len(keys))
	for row := range keys {
		for lane, qv := range q {
			scores[row] = scores[row] + qv*keys[row][lane]
		}
	}
	chosen := make([]int, 0, topK)
	used := make([]bool, len(scores))
	for len(chosen) < topK {
		best := -1
		for i, score := range scores {
			if used[i] || (best >= 0 && (score < scores[best] || (score == scores[best] && i > best))) {
				continue
			}
			best = i
		}
		used[best] = true
		chosen = append(chosen, best)
	}
	return scores, chosen
}
