package main

import (
	"testing"
)

// TestMonotonicSearch verifies the monotonic boundary search (ported from
// ai-dynamo/aiperf monotonic.py, Apache-2.0) returns the correct split index
// and boundary values for all-true, all-false, single-element, and mixed
// monotonic sequences, and never revisits a probed value.
//
// The expected split indices and boundary values are hand-computed from the
// source's boundary semantics: SplitIndex is the index of the first failing
// candidate, FeasibleMax is the highest passing candidate (lo'), and
// InfeasibleMin is the lowest failing candidate (hi').
func TestMonotonicSearch(t *testing.T) {
	passAtOrBelow := func(atOrBelow int) func(int) bool {
		return func(v int) bool { return v <= atOrBelow }
	}
	passBelow := func(below int) func(int) bool {
		return func(v int) bool { return v < below }
	}

	tests := []struct {
		name          string
		candidates    []int
		feasible      func(int) bool
		wantSplit     int
		wantFeasMax   int
		wantInfeasMin int
		wantReason    MonotonicSearchReason
	}{
		{
			name:          "all-true",
			candidates:    []int{1, 2, 3, 4, 5},
			feasible:      func(int) bool { return true },
			wantSplit:     5,
			wantFeasMax:   5,
			wantInfeasMin: -1,
			wantReason:    MonotonicNoFailureInRange,
		},
		{
			name:          "all-false",
			candidates:    []int{1, 2, 3, 4, 5},
			feasible:      func(int) bool { return false },
			wantSplit:     0,
			wantFeasMax:   -1,
			wantInfeasMin: 1,
			wantReason:    MonotonicNoPassInRange,
		},
		{
			name:          "single-element-pass",
			candidates:    []int{42},
			feasible:      func(int) bool { return true },
			wantSplit:     1,
			wantFeasMax:   42,
			wantInfeasMin: -1,
			wantReason:    MonotonicNoFailureInRange,
		},
		{
			name:          "single-element-fail",
			candidates:    []int{42},
			feasible:      func(int) bool { return false },
			wantSplit:     0,
			wantFeasMax:   -1,
			wantInfeasMin: 42,
			wantReason:    MonotonicNoPassInRange,
		},
		{
			name:          "mixed-boundary-middle",
			candidates:    []int{1, 2, 3, 4, 5, 6, 7, 8},
			feasible:      passAtOrBelow(5),
			wantSplit:     5,
			wantFeasMax:   5,
			wantInfeasMin: 6,
			wantReason:    MonotonicPrecisionReached,
		},
		{
			name:          "mixed-boundary-early",
			candidates:    []int{10, 20, 30, 40, 50},
			feasible:      passBelow(40),
			wantSplit:     3,
			wantFeasMax:   30,
			wantInfeasMin: 40,
			wantReason:    MonotonicPrecisionReached,
		},
		{
			name:          "mixed-boundary-after-first",
			candidates:    []int{1, 2, 3, 4},
			feasible:      passAtOrBelow(1),
			wantSplit:     1,
			wantFeasMax:   1,
			wantInfeasMin: 2,
			wantReason:    MonotonicPrecisionReached,
		},
		{
			name:          "mixed-boundary-before-last",
			candidates:    []int{1, 2, 3, 4},
			feasible:      passBelow(4),
			wantSplit:     3,
			wantFeasMax:   3,
			wantInfeasMin: 4,
			wantReason:    MonotonicPrecisionReached,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := MonotonicSearch(tt.candidates, tt.feasible)
			if res.SplitIndex != tt.wantSplit {
				t.Errorf("SplitIndex = %d, want %d (probed=%v)", res.SplitIndex, tt.wantSplit, res.Probed)
			}
			if res.FeasibleMax != tt.wantFeasMax {
				t.Errorf("FeasibleMax = %d, want %d (probed=%v)", res.FeasibleMax, tt.wantFeasMax, res.Probed)
			}
			if res.InfeasibleMin != tt.wantInfeasMin {
				t.Errorf("InfeasibleMin = %d, want %d (probed=%v)", res.InfeasibleMin, tt.wantInfeasMin, res.Probed)
			}
			if res.Reason != tt.wantReason {
				t.Errorf("Reason = %q, want %q (probed=%v)", res.Reason, tt.wantReason, res.Probed)
			}
			// Never revisit a probed value: each value appears at most once.
			seen := map[int]bool{}
			for _, v := range res.Probed {
				if seen[v] {
					t.Errorf("revisited probed value %d (probed=%v)", v, res.Probed)
				}
				seen[v] = true
			}
			// The split index must partition the candidates exactly: every
			// candidate before it passes, every candidate at/after it fails.
			for i, v := range tt.candidates {
				wantPass := i < res.SplitIndex
				if got := tt.feasible(v); got != wantPass {
					t.Errorf("candidate[%d]=%d feasible=%v, want %v (split=%d)", i, v, got, wantPass, res.SplitIndex)
				}
			}
		})
	}
}
