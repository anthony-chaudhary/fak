// fak-test:runtime fast est=1s lane=default
package perfledger

import "testing"

func TestTerminalPhase(t *testing.T) {
	cases := []struct {
		name           string
		cfg            RollupConfig
		counts         RollupCounts
		wantPhase      TerminalPhase
		wantBudgetOver bool
		wantAborted    bool
	}{
		{
			name:      "all succeeded",
			cfg:       RollupConfig{MaxFailures: 0, Policy: FailurePolicyContinue},
			counts:    RollupCounts{Succeeded: 5},
			wantPhase: PhaseSucceeded,
		},
		{
			name:           "partial failure within maxFailures budget",
			cfg:            RollupConfig{MaxFailures: 2, Policy: FailurePolicyContinue},
			counts:         RollupCounts{Succeeded: 4, Failed: 1},
			wantPhase:      PhasePartiallyFailed,
			wantBudgetOver: false,
		},
		{
			name:           "failure exceeding maxFailures budget fails sweep",
			cfg:            RollupConfig{MaxFailures: 1, Policy: FailurePolicyContinue},
			counts:         RollupCounts{Succeeded: 3, Failed: 2},
			wantPhase:      PhaseFailed,
			wantBudgetOver: true,
		},
		{
			name:           "abort policy fails sweep even when remaining children become cancelled",
			cfg:            RollupConfig{MaxFailures: 0, Policy: FailurePolicyAbort},
			counts:         RollupCounts{Succeeded: 1, Failed: 1, Cancelled: 3},
			wantPhase:      PhaseFailed,
			wantBudgetOver: true,
			wantAborted:    true,
		},
		{
			name:           "all attempted failed yields Failed even within maxFailures",
			cfg:            RollupConfig{MaxFailures: 3, Policy: FailurePolicyContinue},
			counts:         RollupCounts{Succeeded: 0, Failed: 2},
			wantPhase:      PhaseFailed,
			wantBudgetOver: false,
		},
		{
			name:           "out-of-band cancelled children isolated from failed do not trip maxFailures=0",
			cfg:            RollupConfig{MaxFailures: 0, Policy: FailurePolicyAbort},
			counts:         RollupCounts{Succeeded: 4, Failed: 0, Cancelled: 1},
			wantPhase:      PhaseCancelled,
			wantBudgetOver: false,
			wantAborted:    false,
		},
		{
			name:      "external sweep cancellation yields Cancelled",
			cfg:       RollupConfig{MaxFailures: 2, Policy: FailurePolicyContinue, SweepCancelled: true},
			counts:    RollupCounts{Succeeded: 2, Failed: 1, Cancelled: 2},
			wantPhase: PhaseCancelled,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ResolveTerminalPhase(tc.cfg, tc.counts)
			if err != nil {
				t.Fatalf("ResolveTerminalPhase: %v", err)
			}
			if got.Phase != tc.wantPhase {
				t.Fatalf("Phase = %q, want %q (result=%+v)", got.Phase, tc.wantPhase, got)
			}
			if got.BudgetExceeded != tc.wantBudgetOver || got.Aborted != tc.wantAborted {
				t.Fatalf("BudgetExceeded/Aborted = (%v, %v), want (%v, %v)", got.BudgetExceeded, got.Aborted, tc.wantBudgetOver, tc.wantAborted)
			}
		})
	}

	t.Run("outcome slice tallies isolated buckets and rejects pending without abort/cancel", func(t *testing.T) {
		res, err := ResolveTerminalPhaseFromOutcomes(
			RollupConfig{MaxFailures: 1, Policy: FailurePolicyContinue},
			[]ChildOutcome{ChildSucceeded, ChildFailed, ChildCancelled, ChildSucceeded},
		)
		if err != nil {
			t.Fatalf("ResolveTerminalPhaseFromOutcomes: %v", err)
		}
		if res.Phase != PhasePartiallyFailed || res.Failed != 1 || res.Cancelled != 1 || res.BudgetExceeded {
			t.Fatalf("unexpected outcome rollup: %+v", res)
		}

		if _, err := ResolveTerminalPhase(RollupConfig{MaxFailures: 0}, RollupCounts{Succeeded: 1, Pending: 1}); err == nil {
			t.Fatal("expected error when non-aborted non-cancelled sweep still has pending children")
		}
		if _, err := ResolveTerminalPhase(RollupConfig{MaxFailures: -1}, RollupCounts{}); err == nil {
			t.Fatal("expected error for negative maxFailures")
		}
	})
}
