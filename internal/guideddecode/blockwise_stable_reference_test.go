package guideddecode

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestBlockwiseReferencePreservesDisplacedTies(t *testing.T) {
	t.Parallel()
	starts := []BlockwiseStartSlot{{Index: 0, Valid: true}}
	// Enumeration order, not the numeric end index, breaks equal-score ties.
	ends := []int{9, 3, 7}
	scores := []float32{1, 1, 2}
	ordered := []BlockwisePair{
		{Start: 0, End: 7, Score: 2},
		{Start: 0, End: 9, Score: 1},
		{Start: 0, End: 3, Score: 1},
	}
	for _, ks := range []int{2, 3} {
		want := [][]BlockwisePair{ordered[:ks]}
		got := BlockwiseTopKReference(starts, ends, scores, ks, nil)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("reference ks=%d: got %v, want %v", ks, got, want)
		}
		for _, blockSize := range []int{1, 2, 3} {
			b := NewBlockwiseTopK(1, ks)
			if !b.SetStart(0, starts[0]) {
				t.Fatal("SetStart refused the only slot")
			}
			for first := 0; first < len(ends); first += blockSize {
				last := first + blockSize
				if last > len(ends) {
					last = len(ends)
				}
				if count, ok := b.ScoreBlock(ends[first:last], scores[first:last], nil); !ok || count != last-first {
					t.Fatalf("ScoreBlock ks=%d blockSize=%d: count=%d ok=%v", ks, blockSize, count, ok)
				}
			}
			if got := b.Results(); !reflect.DeepEqual(got, want) {
				t.Fatalf("streaming ks=%d blockSize=%d: got %v, want %v", ks, blockSize, got, want)
			}
		}
	}
}
