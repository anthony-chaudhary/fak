package model

import (
	"math"
	"strings"
	"testing"
)

// Overflow controls are derived from the platform int width and allocate only
// four input elements. The old query product wraps to zero and falsely accepts
// empty Q; the old key product wraps to zero and reaches a makeslice byte-size
// overflow panic. That runtime guard fires before allocation, so the negative
// control does not request a huge heap allocation on either 32- or 64-bit Go.
// fak-test:runtime fast est=1ms lane=default
func TestV41IndexerScoreGeometryOverflow(t *testing.T) {
	t.Parallel()
	maxInt := int(^uint(0) >> 1)
	wide := maxInt/2 + 1
	for _, tc := range []struct {
		name             string
		q, keys, weights []float32
		heads, dim, rows int
		message          string
	}{
		{"query wraps zero", nil, nil, []float32{1, 1, 1, 1}, 4, wide, 0, "query geometry overflows int"},
		{"keys wrap zero", []float32{1, 1, 1, 1}, nil, []float32{1}, 1, 4, wide, "key geometry overflows int"},
		{"query before shape", nil, nil, nil, maxInt, maxInt, 0, "query geometry overflows int"},
		{"keys before shape", nil, nil, nil, 1, 4, wide, "key geometry overflows int"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("geometry must return an error before allocation or indexing, panic=%v", r)
				}
			}()
			got, err := V41IndexerScore(tc.q, tc.keys, tc.weights, tc.heads, tc.dim, tc.rows)
			if got != nil || err == nil || !strings.Contains(err.Error(), tc.message) {
				t.Fatalf("score=%v err=%v, want nil and %q", got, err, tc.message)
			}
		})
	}

	// The zero-row boundary still validates finite operands and returns a fresh
	// non-nil empty logical score vector rather than requiring nonempty keys.
	got, err := V41IndexerScore([]float32{1, 2, 3, 4}, nil, []float32{1}, 1, 4, 0)
	if err != nil || got == nil || len(got) != 0 {
		t.Fatalf("valid empty rows changed: %v %v", got, err)
	}
	if got, err := V41IndexerScore([]float32{float32(math.NaN())}, nil, []float32{1}, 1, 1, 0); got != nil || err == nil {
		t.Fatal("empty rows skipped finite-input validation")
	}
	for _, geometry := range [][3]int{{0, 1, 0}, {1, 0, 0}, {-1, 1, 0}, {1, -1, 0}, {1, 1, -1}} {
		if got, err := V41IndexerScore(nil, nil, nil, geometry[0], geometry[1], geometry[2]); got != nil || err == nil {
			t.Fatalf("invalid dimensions accepted: %v", geometry)
		}
	}
}
