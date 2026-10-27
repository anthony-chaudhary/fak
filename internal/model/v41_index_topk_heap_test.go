package model

import (
	"math"
	"slices"
	"sort"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41IndexTopKHeapPreservesSelection(t *testing.T) {
	logits := []float32{float32(math.Inf(1)), 5, 5, float32(math.Inf(-1)), 9, 9, float32(math.Copysign(0, -1)), 0, 6, 1000, float32(math.Inf(1)), float32(math.Inf(-1))}
	mask := []bool{true, false, true, true, false, true, true, true, true, true, true, false}
	for _, tc := range []struct {
		name    string
		visible int
		k       int
		mask    []bool
	}{
		{"ties", 9, 4, nil},
		{"masked", 9, 4, mask},
		{"sentinels", 9, 7, mask},
		{"invisible", 0, 4, nil},
		{"one", 9, 1, nil},
		{"zero", 9, 0, mask},
		{"all", 9, len(logits), mask},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Independent full stable sort: ascending input positions supply
			// the tie break, without the production rank comparator or heap.
			positions := make([]int, len(logits))
			masked := slices.Clone(logits)
			for i := range positions {
				positions[i] = i
				if i >= tc.visible || (tc.mask != nil && !tc.mask[i]) {
					masked[i] = float32(math.Inf(-1))
				}
			}
			sort.SliceStable(positions, func(i, j int) bool { return masked[positions[i]] > masked[positions[j]] })
			positions = positions[:tc.k]
			slices.Sort(positions)
			want := make([]int32, len(positions))
			for i, position := range positions {
				want[i] = int32(position + 17)
				if math.IsInf(float64(masked[position]), -1) {
					want[i] = -1
				}
			}
			before, maskBefore := slices.Clone(logits), slices.Clone(tc.mask)
			var observed []string
			got, err := v41SelectIndexRowsObserved(logits, tc.visible, tc.k, 17, tc.mask, func(arm string) {
				observed = append(observed, arm)
			})
			if err != nil || !slices.Equal(got, want) {
				t.Fatalf("selection = %v, %v; want %v", got, err, want)
			}
			if !slices.Equal(observed, []string{"attention-index-list"}) {
				t.Fatalf("host observations = %v", observed)
			}
			if !slices.Equal(logits, before) || !slices.Equal(tc.mask, maskBefore) {
				t.Fatal("selection mutated caller-owned scores or mask")
			}
		})
	}
	// A discarded, masked, causally invisible NaN must still reject the call
	// before emitting the host-arm event, including the zero-selection case.
	bad := slices.Clone(logits)
	bad[len(bad)-1] = float32(math.NaN())
	for _, k := range []int{0, 4} {
		observed := false
		got, err := v41SelectIndexRowsObserved(bad, 9, k, 17, mask, func(string) { observed = true })
		if got != nil || err == nil || observed {
			t.Fatalf("NaN with k=%d: rows=%v err=%v observed=%t", k, got, err, observed)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41CandidateBlocksFullCoverage(t *testing.T) {
	negInf := float32(math.Inf(-1))
	logits := []float32{negInf, negInf, 3, negInf, negInf, negInf, 10}
	for _, tc := range []struct {
		visible, blocks int
		want            []bool
		observed        bool
	}{
		{5, 4, []bool{false, false, true, true, true, true, false}, true},
		{5, 8, []bool{false, false, true, true, true, true, false}, true},
		{7, 4, []bool{false, false, true, true, false, false, true}, true},
		{0, 4, make([]bool, len(logits)), false},
		{5, 0, make([]bool, len(logits)), false},
	} {
		observed := 0
		got, err := v41SelectCandidateBlocksObserved(logits, tc.visible, tc.blocks, 2, func(arm string) {
			if arm != "attention-index-list" {
				t.Fatalf("unexpected host arm %q", arm)
			}
			observed++
		})
		if err != nil || !slices.Equal(got, tc.want) || observed > 1 || (observed == 1) != tc.observed {
			t.Fatalf("visible=%d blocks=%d: mask=%v err=%v observed=%d; want %v", tc.visible, tc.blocks, got, err, observed, tc.want)
		}
	}
	logits[len(logits)-1] = float32(math.NaN())
	observed := false
	got, err := v41SelectCandidateBlocksObserved(logits, 5, 4, 2, func(string) { observed = true })
	if got != nil || err == nil || observed {
		t.Fatalf("invisible NaN: mask=%v err=%v observed=%t", got, err, observed)
	}
}
