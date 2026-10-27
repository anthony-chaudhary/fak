package model

import (
	"math"
	"slices"
	"testing"
)

// fak-test:runtime fast est=200ms lane=default
func TestV41IndexerParallelRowsPreserveSerialOrder(t *testing.T) {
	oldWorkers := NumWorkers()
	defer SetWorkers(oldWorkers)
	const heads, dim = 8, 64
	q, weights := make([]float32, heads*dim), make([]float32, heads)
	for i := range q {
		q[i] = float32(i%19-9) / 16
	}
	for i := range weights {
		weights[i] = float32(i%5-2) / 8
	}
	qBefore, weightsBefore := slices.Clone(q), slices.Clone(weights)
	for _, rows := range []int{0, 1, 7, 257} {
		keys := make([]float32, rows*dim)
		for i := range keys {
			keys[i] = float32(i%23-11) / 32
		}
		keysBefore := slices.Clone(keys)
		// Independent serial oracle: no production scoring helper or worker pool.
		want := make([]float32, rows)
		for row := range want {
			var acc float32
			for h := 0; h < heads; h++ {
				var dot float32
				for d := 0; d < dim; d++ {
					dot = float32(dot + float32(q[h*dim+d]*keys[row*dim+d]))
				}
				if dot < 0 {
					dot = 0
				}
				acc = float32(acc + float32(dot*weights[h]))
			}
			want[row] = acc
		}
		for _, workers := range []int{1, 4} {
			if err := SetWorkers(workers); err != nil {
				t.Fatal(err)
			}
			var observed []string
			got, err := v41IndexerScoreObserved(q, keys, weights, heads, dim, rows, func(arm string) {
				observed = append(observed, arm)
			})
			if err != nil || len(got) != rows {
				t.Fatalf("rows=%d workers=%d: len=%d err=%v", rows, workers, len(got), err)
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("rows=%d workers=%d score%d=%g want%g", rows, workers, i, got[i], want[i])
				}
			}
			if rows == 0 && len(observed) != 0 || rows > 0 && !slices.Equal(observed, []string{"attention-indexer-score"}) {
				t.Fatalf("rows=%d changed observation: %v", rows, observed)
			}
			if !slices.Equal(q, qBefore) || !slices.Equal(weights, weightsBefore) || !slices.Equal(keys, keysBefore) {
				t.Fatal("row partition mutated an input")
			}
		}
	}
	keys := make([]float32, 257*dim)
	keys[len(keys)-1] = float32(math.NaN())
	observed := false
	if got, err := v41IndexerScoreObserved(q, keys, weights, heads, dim, 257, func(string) { observed = true }); err == nil || got != nil || observed {
		t.Fatal("non-finite input was not refused before scoring observation")
	}
}
