package v41

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestV41IndexerScoreRejectsGeometryOverflow(t *testing.T) {
	t.Parallel()

	// Four times this dimension wraps to zero on both 32- and 64-bit int.
	// Empty input slices must not make that impossible geometry look valid.
	wide := int(^uint(0)>>1)/2 + 1
	for _, tc := range []struct {
		name                         string
		q, keys, weights             []float32
		nHeads, headDim, compressLen int
	}{
		{
			name:    "query_product_wraps_to_zero",
			weights: []float32{1, 1, 1, 1},
			nHeads:  4, headDim: wide,
		},
		{
			name: "key_product_wraps_to_zero",
			q:    []float32{1, 1, 1, 1}, weights: []float32{1},
			nHeads: 1, headDim: 4, compressLen: wide,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The old key path reaches make([]float32, wide), whose byte
			// size overflows uintptr before any allocation can take place.
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("overflowing geometry panicked instead of returning an error: %v", p)
				}
			}()
			scores, err := V41IndexerScore(tc.q, tc.keys, tc.weights, tc.nHeads, tc.headDim, tc.compressLen)
			if err == nil || scores != nil {
				t.Fatalf("overflowing geometry returned scores %v, error %v; want nil scores and an error", scores, err)
			}
		})
	}

	// The guard must preserve both an empty compression prefix and ordinary
	// finite scoring, including negative head weights after the ReLU.
	q, weights := []float32{2, -1}, []float32{-3}
	empty, err := V41IndexerScore(q, nil, weights, 1, 2, 0)
	if err != nil || len(empty) != 0 {
		t.Fatalf("empty compression = %v, error %v; want no scores and no error", empty, err)
	}
	scores, err := V41IndexerScore(q, []float32{2, 1, -1, 1}, weights, 1, 2, 2)
	if err != nil || len(scores) != 2 || scores[0] != -9 || scores[1] != 0 {
		t.Fatalf("valid scores = %v, error %v; want [-9 0] and no error", scores, err)
	}
}
