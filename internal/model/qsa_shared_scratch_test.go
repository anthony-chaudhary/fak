package model

import (
	"errors"
	"testing"
)

// TestQSASharedScratchSizeCoversBudgetSum pins the sizing contract: the one
// hoisted buffer spans exactly maxBatchedTokens rows of
// (indexerBudget + compressRatio) int32 elements — the per-token sum of the
// two budgets — and every row is individually addressable with no slack and
// no overflow.
func TestQSASharedScratchSizeCoversBudgetSum(t *testing.T) {
	const rows, indexerBudget, compressRatio = 8, 5, 3
	s, err := NewQSASharedScratch(rows, indexerBudget, compressRatio)
	if err != nil {
		t.Fatalf("NewQSASharedScratch(%d,%d,%d) error = %v", rows, indexerBudget, compressRatio, err)
	}
	if got := s.Width(); got != indexerBudget+compressRatio {
		t.Fatalf("Width() = %d, want %d (budget sum)", got, indexerBudget+compressRatio)
	}
	if got := s.Rows(); got != rows {
		t.Fatalf("Rows() = %d, want %d", got, rows)
	}
	if got := s.Len(); got != rows*(indexerBudget+compressRatio) {
		t.Fatalf("Len() = %d, want %d", got, rows*(indexerBudget+compressRatio))
	}
	// Fill every cell through the bounded Row view and read it back.
	for i := 0; i < rows; i++ {
		row, err := s.Row(i)
		if err != nil {
			t.Fatalf("Row(%d) error = %v", i, err)
		}
		if len(row) != indexerBudget+compressRatio {
			t.Fatalf("Row(%d) len = %d, want %d", i, len(row), indexerBudget+compressRatio)
		}
		for j := range row {
			row[j] = int32(i*100 + j)
		}
	}
	for i := 0; i < rows; i++ {
		row, err := s.Row(i)
		if err != nil {
			t.Fatalf("Row(%d) reread error = %v", i, err)
		}
		for j := range row {
			if row[j] != int32(i*100+j) {
				t.Fatalf("Row(%d)[%d] = %d, want %d", i, j, row[j], i*100+j)
			}
		}
	}
}

// TestQSASharedScratchLayerViewsAliasSharedBuffer pins the core borrow:
// concurrent layer uses within one step do NOT get private copies — every QSA
// layer view aliases the single hoisted buffer, so a write through one layer's
// view is visible through every other layer's view.
func TestQSASharedScratchLayerViewsAliasSharedBuffer(t *testing.T) {
	s, err := NewQSASharedScratch(4, 2, 1)
	if err != nil {
		t.Fatalf("NewQSASharedScratch error = %v", err)
	}
	layers := []int{0, 3, 7, 11} // QSA layer indices, in trunk order
	views := make([][]int32, len(layers))
	for i, l := range layers {
		views[i] = s.LayerView(l)
		if len(views[i]) != 4*3 {
			t.Fatalf("LayerView(%d) len = %d, want %d", l, len(views[i]), 4*3)
		}
	}
	base := &views[0][0]
	for i, l := range layers {
		if &views[i][0] != base {
			t.Fatalf("LayerView(%d) does not alias the shared buffer (&view[0]=%p, want %p)", l, &views[i][0], base)
		}
	}
	// Write through one layer's view; the marker must be visible through all
	// others — aliasing, not copying.
	views[2][5] = -7
	for i, l := range layers {
		if i == 2 {
			continue
		}
		if views[i][5] != -7 {
			t.Fatalf("marker written through LayerView(7) not visible through LayerView(%d): %d", l, views[i][5])
		}
	}
	// A negative layer index is a programming error: it gets no workspace.
	if v := s.LayerView(-1); v != nil {
		t.Fatalf("LayerView(-1) = non-nil (%d elements), want nil", len(v))
	}
}

// TestQSASharedScratchDrafterReusesSharedBuffer pins the MTP-drafter half of
// the borrow: the drafter's view is the same backing array the layers use,
// not a private copy.
func TestQSASharedScratchDrafterReusesSharedBuffer(t *testing.T) {
	s, err := NewQSASharedScratch(2, 3, 2)
	if err != nil {
		t.Fatalf("NewQSASharedScratch error = %v", err)
	}
	drafter := s.DrafterView()
	layer := s.LayerView(0)
	if len(drafter) != 2*5 {
		t.Fatalf("DrafterView len = %d, want %d", len(drafter), 2*5)
	}
	if &drafter[0] != &layer[0] {
		t.Fatalf("drafter view %p does not alias the layer buffer %p", &drafter[0], &layer[0])
	}
	drafter[0] = 42
	if layer[0] != 42 {
		t.Fatalf("write through drafter view not visible through layer view: %d", layer[0])
	}
}

// TestQSASharedScratchSentinels pins the closed refusal contract: unusable
// dimensions and out-of-range rows refuse with the package sentinels, matched
// with errors.Is.
func TestQSASharedScratchSentinels(t *testing.T) {
	cases := []struct {
		name          string
		rows, idx, cr int
	}{
		{"zero rows", 0, 2, 1},
		{"negative rows", -1, 2, 1},
		{"zero indexer budget", 4, 0, 1},
		{"negative compress ratio", 4, 2, -1},
	}
	for _, tc := range cases {
		_, err := NewQSASharedScratch(tc.rows, tc.idx, tc.cr)
		if !errors.Is(err, ErrQSASharedScratchDimension) {
			t.Fatalf("%s: NewQSASharedScratch error = %v, want ErrQSASharedScratchDimension", tc.name, err)
		}
	}
	s, err := NewQSASharedScratch(2, 1, 1)
	if err != nil {
		t.Fatalf("NewQSASharedScratch error = %v", err)
	}
	for _, i := range []int{-1, 2} {
		if _, err := s.Row(i); !errors.Is(err, ErrQSASharedScratchRow) {
			t.Fatalf("Row(%d) error = %v, want ErrQSASharedScratchRow", i, err)
		}
	}
}
