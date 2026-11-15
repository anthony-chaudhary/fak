// fak-test:runtime fast est=10ms lane=default
// fak-test:justify why=contract when=changed:internal/perfledger/**
package perfledger

import (
	"math"
	"testing"
)

// TestEarlyStop verifies the multi-run early-stop decision against hand-computed
// outputs derived from the source's _track_scalar_improvement bookkeeping and
// evaluate_three_signal_convergence stop signals (ai-dynamo/aiperf@ab610add).
func TestEarlyStop(t *testing.T) {
	cases := []struct {
		name          string
		values        []float64
		lowerIsBetter bool
		patience      int
		maxRuns       int
		wantStop      bool
		wantReason    string
	}{
		{
			// 10 -> 9 improves (since=0); 9,9,9 do not (since=1,2,3).
			// patience=3 fires at since=3. Budget (maxRuns=10) not reached.
			name:          "patience_exhausted_stops",
			values:        []float64{10, 9, 9, 9, 9},
			lowerIsBetter: true,
			patience:      3,
			maxRuns:       10,
			wantStop:      true,
			wantReason:    ReasonNoImprovement,
		},
		{
			// Every run improves, so since stays 0; neither signal fires.
			name:          "still_improving_continues",
			values:        []float64{10, 9, 8, 7},
			lowerIsBetter: true,
			patience:      3,
			maxRuns:       10,
			wantStop:      false,
			wantReason:    "",
		},
		{
			// All improving (since=0) but the 5th run hits the maxRuns=5 budget.
			name:          "budget_exhausted_stops",
			values:        []float64{10, 9, 8, 7, 6},
			lowerIsBetter: true,
			patience:      10,
			maxRuns:       5,
			wantStop:      true,
			wantReason:    ReasonMaxRuns,
		},
		{
			// Higher-is-better: 100 -> 120 improves (since=0); 120,120 do not
			// (since=1,2). patience=2 fires at since=2.
			name:          "higher_is_better",
			values:        []float64{100, 120, 120, 120},
			lowerIsBetter: false,
			patience:      2,
			maxRuns:       10,
			wantStop:      true,
			wantReason:    ReasonNoImprovement,
		},
		{
			// Both signals would fire (since=2 >= patience=1 and runs=3 >=
			// maxRuns=3); the source checks the budget first, so max_runs wins.
			name:          "budget_checked_before_patience",
			values:        []float64{10, 10, 10},
			lowerIsBetter: true,
			patience:      1,
			maxRuns:       3,
			wantStop:      true,
			wantReason:    ReasonMaxRuns,
		},
		{
			// A single run establishes the best (since=0); nothing can fire yet.
			name:          "single_run_continues",
			values:        []float64{10},
			lowerIsBetter: true,
			patience:      1,
			maxRuns:       5,
			wantStop:      false,
			wantReason:    "",
		},
		{
			// patience=0 disables the patience signal even though since=3.
			name:          "zero_patience_disables_signal",
			values:        []float64{10, 10, 10, 10},
			lowerIsBetter: true,
			patience:      0,
			maxRuns:       10,
			wantStop:      false,
			wantReason:    "",
		},
		{
			// maxRuns=0 disables the budget signal even though runs=4.
			name:          "zero_maxruns_disables_signal",
			values:        []float64{10, 10, 10, 10},
			lowerIsBetter: true,
			patience:      10,
			maxRuns:       0,
			wantStop:      false,
			wantReason:    "",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotStop, gotReason := ShouldEarlyStop(tc.values, tc.lowerIsBetter, tc.patience, tc.maxRuns)
			if gotStop != tc.wantStop {
				t.Fatalf("ShouldEarlyStop(%v, lower=%v, patience=%d, maxRuns=%d) stop = %v, want %v",
					tc.values, tc.lowerIsBetter, tc.patience, tc.maxRuns, gotStop, tc.wantStop)
			}
			if gotReason != tc.wantReason {
				t.Fatalf("reason = %q, want %q", gotReason, tc.wantReason)
			}
		})
	}

	t.Run("tracker_best_and_patience", func(t *testing.T) {
		// Walk the same sequence as patience_exhausted_stops and assert the
		// best-so-far and runs-since-improvement the source would record.
		tr := NewEarlyStopTracker(true)
		wantBest := []float64{10, 9, 9, 9, 9}
		wantSince := []int{0, 0, 1, 2, 3}
		for i, v := range []float64{10, 9, 9, 9, 9} {
			tr.Observe(v)
			if got, ok := tr.Best(); !ok || got != wantBest[i] {
				t.Fatalf("after run %d: Best = (%v,%v), want (%v,true)", i+1, got, ok, wantBest[i])
			}
			if tr.SinceImprovement() != wantSince[i] {
				t.Fatalf("after run %d: SinceImprovement = %d, want %d", i+1, tr.SinceImprovement(), wantSince[i])
			}
			if tr.Runs() != i+1 {
				t.Fatalf("after run %d: Runs = %d, want %d", i+1, tr.Runs(), i+1)
			}
		}
	})

	t.Run("failed_run_is_no_improvement", func(t *testing.T) {
		// A NaN run is counted but never becomes the best, mirroring the
		// source's failed-iteration (objective=None) handling.
		tr := NewEarlyStopTracker(true)
		tr.Observe(10)         // best=10, since=0
		tr.Observe(math.NaN()) // since=1, best stays 10
		if got, ok := tr.Best(); !ok || got != 10 {
			t.Fatalf("Best after NaN = (%v,%v), want (10,true)", got, ok)
		}
		if tr.SinceImprovement() != 1 {
			t.Fatalf("SinceImprovement after NaN = %d, want 1", tr.SinceImprovement())
		}
	})
}
