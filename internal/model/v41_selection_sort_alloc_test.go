package model

import (
	"math"
	"slices"
	"testing"
)

// Selection should allocate its output/storage buffers, not reflection sort wrappers.
// fak-test:runtime fast est=10ms lane=default
func TestV41SelectionSortAllocations(t *testing.T) {
	logits := []float32{float32(math.Inf(1)), 5, 5, float32(math.Inf(-1)), 8, 9}
	wantRows := []int32{7, 8, 11}
	rowAllocs := testing.AllocsPerRun(100, func() {
		rows, err := V41SelectIndexRows(logits, 5, 3, 7, nil)
		if err != nil || !slices.Equal(rows, wantRows) {
			panic("row selection changed")
		}
	})
	if rowAllocs > 2 {
		t.Fatalf("row selection allocated %g objects, want at most two buffers", rowAllocs)
	}
	wantBlocks := []bool{true, true, false, false, true, true}
	blockAllocs := testing.AllocsPerRun(100, func() {
		blocks, err := V41SelectCandidateBlocks(logits, 5, 2, 2)
		if err != nil || !slices.Equal(blocks, wantBlocks) {
			panic("candidate block selection changed")
		}
	})
	if blockAllocs > 2 {
		t.Fatalf("candidate blocks allocated %g objects, want at most two buffers", blockAllocs)
	}
}
