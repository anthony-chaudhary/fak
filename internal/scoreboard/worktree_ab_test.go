package scoreboard

import (
	"math"
	"strings"
	"testing"
)

func TestWorktreeABRendersBothArms(t *testing.T) {
	r := FoldWorktreeAB(
		WorktreeABArm{WaveID: "fixed-wave", Resolved: 3, DurationSeconds: 1800, PoisonIncidents: 2, PeakConcurrency: 5},
		WorktreeABArm{WaveID: "fixed-wave", Resolved: 3, DurationSeconds: 1200, PoisonIncidents: 0, PeakConcurrency: 8},
	)
	if r.Schema != WorktreeABSchema || r.Verdict != "ISOLATION_POISON_FREE" || !WorktreeABEquivalentWave(r.Baseline, r.Isolated) {
		t.Fatalf("report=%+v", r)
	}
	text := WorktreeABUpdate(r).Text()
	for _, want := range []string{"baseline: 6.00 issues/h, 2 poison, 1800.0s, peak 5", "isolated: 9.00 issues/h, 0 poison, 1200.0s, peak 8"} {
		if !strings.Contains(text, want) {
			t.Fatalf("render missing %q:\n%s", want, text)
		}
	}
}

func TestWorktreeABRequiresMeasuredPoisonFreeIsolatedArm(t *testing.T) {
	r := FoldWorktreeAB(WorktreeABArm{WaveID: "fixed-wave", Resolved: 1, DurationSeconds: 10}, WorktreeABArm{WaveID: "fixed-wave", Resolved: 1, DurationSeconds: 10, PoisonIncidents: 1})
	if r.Verdict != "NOT_PROVEN" {
		t.Fatalf("verdict=%s", r.Verdict)
	}
}

func TestWorktreeABAcceptedDeliveryAccounting(t *testing.T) {
	t.Run("outcomes_table_fixtures", func(t *testing.T) {
		tests := []struct {
			name                   string
			records                []DeliveryLifecycleRecord
			window                 float64
			wantTotal              int
			wantAccepted           int
			wantRejected           int
			wantDuplicate          int
			wantUnverified         int
			wantAcceptedPerElapsed float64
			wantVerified           bool
			wantStatus             string
		}{
			{
				name: "distinct accepted deliveries exclude duplicates, rejections, unverified",
				records: []DeliveryLifecycleRecord{
					{IssueID: 101, Outcome: OutcomeAccepted, Spend: 1.0},
					{IssueID: 102, Outcome: OutcomeAccepted, Spend: 1.5},
					{IssueID: 101, Outcome: OutcomeAccepted, Spend: 0.5},   // duplicate issue ID
					{IssueID: 103, Outcome: OutcomeDuplicate, Spend: 0.2},  // explicit duplicate
					{IssueID: 104, Outcome: OutcomeRejected, Spend: 0.8},   // rejected
					{IssueID: 105, Outcome: OutcomeUnverified, Spend: 0.4}, // unverified
				},
				window:                 3600.0,
				wantTotal:              6,
				wantAccepted:           2,
				wantRejected:           1,
				wantDuplicate:          2,
				wantUnverified:         1,
				wantAcceptedPerElapsed: 2.0,
				wantVerified:           true,
				wantStatus:             "COMPLETE",
			},
			{
				name: "all rejected and unverified yields zero accepted and incomplete status",
				records: []DeliveryLifecycleRecord{
					{IssueID: 201, Outcome: OutcomeRejected, Spend: 1.0},
					{IssueID: 202, Outcome: OutcomeUnverified, Spend: 1.0},
					{IssueID: 203, Outcome: OutcomeDuplicate, Spend: 0.5},
				},
				window:                 1800.0,
				wantTotal:              3,
				wantAccepted:           0,
				wantRejected:           1,
				wantDuplicate:          1,
				wantUnverified:         1,
				wantAcceptedPerElapsed: 0.0,
				wantVerified:           false,
				wantStatus:             "INCOMPLETE",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := AccountAcceptedDeliveries(tc.records, tc.window)
				if got.TotalDeliveries != tc.wantTotal {
					t.Errorf("TotalDeliveries = %d, want %d", got.TotalDeliveries, tc.wantTotal)
				}
				if got.AcceptedDeliveries != tc.wantAccepted {
					t.Errorf("AcceptedDeliveries = %d, want %d", got.AcceptedDeliveries, tc.wantAccepted)
				}
				if got.RejectedDeliveries != tc.wantRejected {
					t.Errorf("RejectedDeliveries = %d, want %d", got.RejectedDeliveries, tc.wantRejected)
				}
				if got.DuplicateDeliveries != tc.wantDuplicate {
					t.Errorf("DuplicateDeliveries = %d, want %d", got.DuplicateDeliveries, tc.wantDuplicate)
				}
				if got.UnverifiedDeliveries != tc.wantUnverified {
					t.Errorf("UnverifiedDeliveries = %d, want %d", got.UnverifiedDeliveries, tc.wantUnverified)
				}
				if got.TotalDeliveries != (got.AcceptedDeliveries + got.RejectedDeliveries + got.DuplicateDeliveries + got.UnverifiedDeliveries) {
					t.Errorf("conservation breach: TotalDeliveries %d != sum of parts %d",
						got.TotalDeliveries,
						got.AcceptedDeliveries+got.RejectedDeliveries+got.DuplicateDeliveries+got.UnverifiedDeliveries)
				}
				if math.Abs(got.AcceptedPerElapsedHour-tc.wantAcceptedPerElapsed) > 1e-6 {
					t.Errorf("AcceptedPerElapsedHour = %.4f, want %.4f", got.AcceptedPerElapsedHour, tc.wantAcceptedPerElapsed)
				}
				if got.Verified != tc.wantVerified {
					t.Errorf("Verified = %v, want %v", got.Verified, tc.wantVerified)
				}
				if got.Status != tc.wantStatus {
					t.Errorf("Status = %q, want %q", got.Status, tc.wantStatus)
				}
			})
		}
	})

	t.Run("incomplete_lifecycle_boundaries", func(t *testing.T) {
		validRecords := []DeliveryLifecycleRecord{
			{IssueID: 101, Outcome: OutcomeAccepted, Spend: 1.0, TotalElapsed: 60},
		}

		boundaryCases := []struct {
			name    string
			records []DeliveryLifecycleRecord
			window  float64
		}{
			{"zero window", validRecords, 0.0},
			{"negative window", validRecords, -120.0},
			{"empty records", nil, 3600.0},
			{"negative phase duration", []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, ExecutionDuration: -10, TotalElapsed: 50},
			}, 3600.0},
			{"sub-phase duration exceeds total elapsed", []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, SetupDuration: 10, ExecutionDuration: 50, TotalElapsed: 50},
			}, 3600.0},
			{"+Inf total elapsed duration", []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, TotalElapsed: math.Inf(1)},
			}, 3600.0},
			{"+Inf setup duration", []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, SetupDuration: math.Inf(1), TotalElapsed: 50},
			}, 3600.0},
			{"+Inf execution duration", []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, ExecutionDuration: math.Inf(1), TotalElapsed: 50},
			}, 3600.0},
			{"+Inf landing duration", []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, LandingDuration: math.Inf(1), TotalElapsed: 50},
			}, 3600.0},
			{"+Inf verification duration", []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, VerificationDuration: math.Inf(1), TotalElapsed: 50},
			}, 3600.0},
			{"negative spend", []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: -10.0, TotalElapsed: 50},
			}, 3600.0},
		}

		for _, bc := range boundaryCases {
			t.Run(bc.name, func(t *testing.T) {
				got := AccountAcceptedDeliveries(bc.records, bc.window)
				if got.Verified {
					t.Errorf("expected Verified=false for %s, got true", bc.name)
				}
				if got.Status != "INCOMPLETE" {
					t.Errorf("expected Status=INCOMPLETE for %s, got %q", bc.name, got.Status)
				}
				if bc.window <= 0 && got.AcceptedPerElapsedHour != 0 {
					t.Errorf("expected AcceptedPerElapsedHour=0 for non-positive window, got %f", got.AcceptedPerElapsedHour)
				}
			})
		}
	})

	t.Run("overlapping_phases_window_rate", func(t *testing.T) {
		// 3 concurrent workers in isolated worktrees:
		// Sum of execution durations: 120 + 110 + 130 = 360s
		// Sum of individual total elapsed: 160 + 160 + 160 = 480s
		// Total elapsed window for the concurrent wave: 180s (3 minutes)
		records := []DeliveryLifecycleRecord{
			{IssueID: 301, Outcome: OutcomeAccepted, SetupDuration: 15, ExecutionDuration: 120, LandingDuration: 10, VerificationDuration: 15, TotalElapsed: 160},
			{IssueID: 302, Outcome: OutcomeAccepted, SetupDuration: 20, ExecutionDuration: 110, LandingDuration: 15, VerificationDuration: 15, TotalElapsed: 160},
			{IssueID: 303, Outcome: OutcomeAccepted, SetupDuration: 10, ExecutionDuration: 130, LandingDuration: 10, VerificationDuration: 10, TotalElapsed: 160},
		}
		totalWindow := 180.0

		got := AccountAcceptedDeliveries(records, totalWindow)
		if !got.Verified || got.Status != "COMPLETE" {
			t.Fatalf("expected complete/verified, got status=%q verified=%v", got.Status, got.Verified)
		}
		if got.AcceptedDeliveries != 3 {
			t.Fatalf("expected 3 accepted deliveries, got %d", got.AcceptedDeliveries)
		}
		// Rate should be 3 accepted / 180s * 3600s/h = 60.0 accepted/h
		wantRate := 3.0 * 3600.0 / 180.0
		if math.Abs(got.AcceptedPerElapsedHour-wantRate) > 1e-6 {
			t.Fatalf("AcceptedPerElapsedHour = %.4f, want %.4f (must use window, not sum of overlapping phases)",
				got.AcceptedPerElapsedHour, wantRate)
		}
		// Confirm it does NOT equal rate derived from summing individual elapsed times (480s -> 22.5/h)
		summedElapsedRate := 3.0 * 3600.0 / 480.0
		if math.Abs(got.AcceptedPerElapsedHour-summedElapsedRate) < 1e-6 {
			t.Fatalf("AcceptedPerElapsedHour incorrectly matched summed overlapping phase rate %.4f", summedElapsedRate)
		}
	})

	t.Run("spend_accounting_and_unknown_propagation", func(t *testing.T) {
		// Case A: All spends known
		knownRecords := []DeliveryLifecycleRecord{
			{IssueID: 401, Outcome: OutcomeAccepted, Spend: 1.25, SpendUnknown: false},
			{IssueID: 402, Outcome: OutcomeAccepted, Spend: 2.75, SpendUnknown: false},
			{IssueID: 403, Outcome: OutcomeRejected, Spend: 0.50, SpendUnknown: false},
		}
		gotKnown := AccountAcceptedDeliveries(knownRecords, 3600.0)
		if gotKnown.SpendUnknown {
			t.Errorf("expected SpendUnknown=false, got true")
		}
		if math.Abs(gotKnown.Spend-4.50) > 1e-6 {
			t.Errorf("Spend = %.4f, want 4.50", gotKnown.Spend)
		}

		// Case B: At least one record has SpendUnknown=true
		unknownRecords := []DeliveryLifecycleRecord{
			{IssueID: 501, Outcome: OutcomeAccepted, Spend: 1.25, SpendUnknown: false},
			{IssueID: 502, Outcome: OutcomeAccepted, Spend: 0.00, SpendUnknown: true},
			{IssueID: 503, Outcome: OutcomeRejected, Spend: 0.75, SpendUnknown: false},
		}
		gotUnknown := AccountAcceptedDeliveries(unknownRecords, 3600.0)
		if !gotUnknown.SpendUnknown {
			t.Errorf("expected SpendUnknown=true, got false")
		}
		if math.Abs(gotUnknown.Spend-2.00) > 1e-6 {
			t.Errorf("Spend = %.4f, want 2.00 (accumulated known spends)", gotUnknown.Spend)
		}

		// Case C: Negative spend triggers SpendUnknown and Status=INCOMPLETE
		negativeSpendRecords := []DeliveryLifecycleRecord{
			{IssueID: 601, Outcome: OutcomeAccepted, Spend: 1.25, TotalElapsed: 60},
			{IssueID: 602, Outcome: OutcomeAccepted, Spend: -0.50, TotalElapsed: 60},
		}
		gotNeg := AccountAcceptedDeliveries(negativeSpendRecords, 3600.0)
		if !gotNeg.SpendUnknown {
			t.Errorf("expected SpendUnknown=true for negative spend, got false")
		}
		if gotNeg.Verified {
			t.Errorf("expected Verified=false for negative spend, got true")
		}
		if gotNeg.Status != "INCOMPLETE" {
			t.Errorf("expected Status=INCOMPLETE for negative spend, got %q", gotNeg.Status)
		}
		if math.Abs(gotNeg.Spend-1.25) > 1e-6 {
			t.Errorf("Spend = %.4f, want 1.25 (negative spend must not be accumulated)", gotNeg.Spend)
		}

		// Case D: Non-finite spend (NaN / +Inf) triggers SpendUnknown and Status=INCOMPLETE
		nanSpendRecords := []DeliveryLifecycleRecord{
			{IssueID: 701, Outcome: OutcomeAccepted, Spend: 1.50, TotalElapsed: 60},
			{IssueID: 702, Outcome: OutcomeAccepted, Spend: math.NaN(), TotalElapsed: 60},
		}
		gotNaN := AccountAcceptedDeliveries(nanSpendRecords, 3600.0)
		if !gotNaN.SpendUnknown {
			t.Errorf("expected SpendUnknown=true for NaN spend, got false")
		}
		if gotNaN.Verified {
			t.Errorf("expected Verified=false for NaN spend, got true")
		}
		if gotNaN.Status != "INCOMPLETE" {
			t.Errorf("expected Status=INCOMPLETE for NaN spend, got %q", gotNaN.Status)
		}
		if math.Abs(gotNaN.Spend-1.50) > 1e-6 {
			t.Errorf("Spend = %.4f, want 1.50 (NaN spend must not be accumulated)", gotNaN.Spend)
		}

		infSpendRecords := []DeliveryLifecycleRecord{
			{IssueID: 801, Outcome: OutcomeAccepted, Spend: 2.00, TotalElapsed: 60},
			{IssueID: 802, Outcome: OutcomeAccepted, Spend: math.Inf(1), TotalElapsed: 60},
		}
		gotInf := AccountAcceptedDeliveries(infSpendRecords, 3600.0)
		if !gotInf.SpendUnknown {
			t.Errorf("expected SpendUnknown=true for +Inf spend, got false")
		}
		if gotInf.Verified {
			t.Errorf("expected Verified=false for +Inf spend, got true")
		}
		if gotInf.Status != "INCOMPLETE" {
			t.Errorf("expected Status=INCOMPLETE for +Inf spend, got %q", gotInf.Status)
		}
		if math.Abs(gotInf.Spend-2.00) > 1e-6 {
			t.Errorf("Spend = %.4f, want 2.00 (+Inf spend must not be accumulated)", gotInf.Spend)
		}
	})
}

func TestWorktreeABFoldAcceptedDeliveryAccounting(t *testing.T) {
	t.Run("computes_and_deduplicates_both_arms_complete", func(t *testing.T) {
		trunk := WorktreeABArm{
			WaveID:          "wave-test",
			DurationSeconds: 3600.0,
			PoisonIncidents: 1,
			PeakConcurrency: 4,
			DeliveryRecords: []AcceptedDeliveryRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: 1.0, TotalElapsed: 60},
				{IssueID: 102, Outcome: OutcomeAccepted, Spend: 1.5, TotalElapsed: 60},
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: 0.5, TotalElapsed: 60},   // duplicate IssueID
				{IssueID: 103, Outcome: OutcomeDuplicate, Spend: 0.2, TotalElapsed: 60},  // explicit duplicate
				{IssueID: 104, Outcome: OutcomeRejected, Spend: 0.8, TotalElapsed: 60},   // rejected
				{IssueID: 105, Outcome: OutcomeUnverified, Spend: 0.4, TotalElapsed: 60}, // unverified
			},
		}

		worktree := WorktreeABArm{
			WaveID:          "wave-test",
			DurationSeconds: 1800.0,
			PoisonIncidents: 0,
			PeakConcurrency: 8,
			DeliveryRecords: []AcceptedDeliveryRecord{
				{IssueID: 201, Outcome: OutcomeAccepted, Spend: 1.0, TotalElapsed: 40},
				{IssueID: 202, Outcome: OutcomeAccepted, Spend: 1.0, TotalElapsed: 40},
				{IssueID: 203, Outcome: OutcomeAccepted, Spend: 1.0, TotalElapsed: 40},
				{IssueID: 201, Outcome: OutcomeAccepted, Spend: 0.5, TotalElapsed: 40},   // duplicate IssueID
				{IssueID: 204, Outcome: OutcomeUnverified, Spend: 0.5, TotalElapsed: 40}, // unverified
			},
		}

		r := FoldWorktreeAB(trunk, worktree)

		// Trunk accounting assertions
		if r.Baseline.Accounting.TotalDeliveries != 6 {
			t.Errorf("Baseline.Accounting.TotalDeliveries = %d, want 6", r.Baseline.Accounting.TotalDeliveries)
		}
		if r.Baseline.Accounting.AcceptedDeliveries != 2 {
			t.Errorf("Baseline.Accounting.AcceptedDeliveries = %d, want 2 (deduplicated)", r.Baseline.Accounting.AcceptedDeliveries)
		}
		if r.Baseline.Accounting.DuplicateDeliveries != 2 {
			t.Errorf("Baseline.Accounting.DuplicateDeliveries = %d, want 2", r.Baseline.Accounting.DuplicateDeliveries)
		}
		if r.Baseline.Accounting.RejectedDeliveries != 1 {
			t.Errorf("Baseline.Accounting.RejectedDeliveries = %d, want 1", r.Baseline.Accounting.RejectedDeliveries)
		}
		if r.Baseline.Accounting.UnverifiedDeliveries != 1 {
			t.Errorf("Baseline.Accounting.UnverifiedDeliveries = %d, want 1", r.Baseline.Accounting.UnverifiedDeliveries)
		}
		if r.Baseline.Accounting.Status != "COMPLETE" {
			t.Errorf("Baseline.Accounting.Status = %q, want COMPLETE", r.Baseline.Accounting.Status)
		}
		if !r.Baseline.Accounting.Verified {
			t.Errorf("Baseline.Accounting.Verified = false, want true")
		}
		if r.TrunkAccounting != r.Baseline.Accounting {
			t.Errorf("TrunkAccounting mismatch: %+v != %+v", r.TrunkAccounting, r.Baseline.Accounting)
		}

		// Worktree accounting assertions
		if r.Isolated.Accounting.TotalDeliveries != 5 {
			t.Errorf("Isolated.Accounting.TotalDeliveries = %d, want 5", r.Isolated.Accounting.TotalDeliveries)
		}
		if r.Isolated.Accounting.AcceptedDeliveries != 3 {
			t.Errorf("Isolated.Accounting.AcceptedDeliveries = %d, want 3 (deduplicated)", r.Isolated.Accounting.AcceptedDeliveries)
		}
		if r.Isolated.Accounting.DuplicateDeliveries != 1 {
			t.Errorf("Isolated.Accounting.DuplicateDeliveries = %d, want 1", r.Isolated.Accounting.DuplicateDeliveries)
		}
		if r.Isolated.Accounting.UnverifiedDeliveries != 1 {
			t.Errorf("Isolated.Accounting.UnverifiedDeliveries = %d, want 1", r.Isolated.Accounting.UnverifiedDeliveries)
		}
		if r.Isolated.Accounting.Status != "COMPLETE" {
			t.Errorf("Isolated.Accounting.Status = %q, want COMPLETE", r.Isolated.Accounting.Status)
		}
		if !r.Isolated.Accounting.Verified {
			t.Errorf("Isolated.Accounting.Verified = false, want true")
		}
		if r.WorktreeAccounting != r.Isolated.Accounting {
			t.Errorf("WorktreeAccounting mismatch: %+v != %+v", r.WorktreeAccounting, r.Isolated.Accounting)
		}

		// Formatted text reflections
		text := WorktreeABUpdate(r).Text()
		for _, want := range []string{"2 accepted, COMPLETE", "3 accepted, COMPLETE"} {
			if !strings.Contains(text, want) {
				t.Fatalf("render missing %q:\n%s", want, text)
			}
		}

		// Verify CompareWorktreeAB returns identical comparison
		comp, err := CompareWorktreeAB(trunk, worktree)
		if err != nil {
			t.Fatalf("CompareWorktreeAB error: %v", err)
		}
		if comp.TrunkAccounting != r.TrunkAccounting || comp.WorktreeAccounting != r.WorktreeAccounting {
			t.Fatalf("CompareWorktreeAB mismatch: %+v vs %+v", comp, r)
		}
	})

	t.Run("incomplete_status_reflection", func(t *testing.T) {
		trunk := WorktreeABArm{
			DurationSeconds: 1800.0,
			DeliveryRecords: []AcceptedDeliveryRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, TotalElapsed: 60},
			},
		}

		worktree := WorktreeABArm{
			DurationSeconds: 1800.0,
			DeliveryRecords: []AcceptedDeliveryRecord{
				{IssueID: 301, Outcome: OutcomeRejected, TotalElapsed: 60},
				{IssueID: 302, Outcome: OutcomeUnverified, TotalElapsed: 60},
			},
		}

		r := FoldWorktreeAB(trunk, worktree)
		if r.Isolated.Accounting.AcceptedDeliveries != 0 {
			t.Errorf("Isolated.Accounting.AcceptedDeliveries = %d, want 0", r.Isolated.Accounting.AcceptedDeliveries)
		}
		if r.Isolated.Accounting.Status != "INCOMPLETE" {
			t.Errorf("Isolated.Accounting.Status = %q, want INCOMPLETE", r.Isolated.Accounting.Status)
		}
		if r.Isolated.Accounting.Verified {
			t.Errorf("Isolated.Accounting.Verified = true, want false")
		}
		if r.WorktreeAccounting.Status != "INCOMPLETE" {
			t.Errorf("WorktreeAccounting.Status = %q, want INCOMPLETE", r.WorktreeAccounting.Status)
		}

		text := WorktreeABUpdate(r).Text()
		if !strings.Contains(text, "0 accepted, INCOMPLETE") {
			t.Fatalf("render missing incomplete status:\n%s", text)
		}
	})

	t.Run("preserves_pre_populated_accounting", func(t *testing.T) {
		custom := AcceptedDeliveryAccounting{
			Status:             "PRE_POPULATED",
			AcceptedDeliveries: 42,
			Verified:           true,
		}
		trunk := WorktreeABArm{
			DurationSeconds: 3600.0,
			DeliveryRecords: []AcceptedDeliveryRecord{
				{IssueID: 101, Outcome: OutcomeAccepted},
			},
			Accounting: custom,
		}
		worktree := WorktreeABArm{
			DurationSeconds: 3600.0,
		}

		r := FoldWorktreeAB(trunk, worktree)
		if r.Baseline.Accounting.Status != "PRE_POPULATED" || r.Baseline.Accounting.AcceptedDeliveries != 42 {
			t.Fatalf("expected pre-populated accounting to be preserved, got %+v", r.Baseline.Accounting)
		}
	})

	t.Run("lifecycle_records_alias", func(t *testing.T) {
		trunk := WorktreeABArm{
			WaveID:          "wave-lifecycle",
			DurationSeconds: 1800.0,
			LifecycleRecords: []AcceptedDeliveryRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, TotalElapsed: 60},
			},
		}
		worktree := WorktreeABArm{
			WaveID:          "wave-lifecycle",
			DurationSeconds: 900.0,
			LifecycleRecords: []AcceptedDeliveryRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, TotalElapsed: 60},
			},
		}
		r := FoldWorktreeAB(trunk, worktree)
		if r.Baseline.Accounting.AcceptedDeliveries != 1 || r.Isolated.Accounting.AcceptedDeliveries != 1 {
			t.Fatalf("expected 1 accepted delivery in both arms, got baseline=%d isolated=%d",
				r.Baseline.Accounting.AcceptedDeliveries, r.Isolated.Accounting.AcceptedDeliveries)
		}
		if !WorktreeABEquivalentWave(r.Baseline, r.Isolated) {
			t.Fatal("expected arms to be equivalent wave")
		}
	})
}

func TestWorktreeABRequiresComparableMeasuredArms(t *testing.T) {
	validBaseline := WorktreeABArm{
		WaveID:          "fixed-wave",
		Resolved:        4,
		DurationSeconds: 1200.0,
		PoisonIncidents: 2,
		PeakConcurrency: 4,
		HostID:          "node-1",
	}
	validIsolated := WorktreeABArm{
		WaveID:          "fixed-wave",
		Resolved:        4,
		DurationSeconds: 800.0,
		PoisonIncidents: 0,
		PeakConcurrency: 8,
		HostID:          "node-1",
	}

	tests := []struct {
		name           string
		mutateBaseline func(a *WorktreeABArm)
		mutateIsolated func(a *WorktreeABArm)
		wantVerdict    string
		wantEquivalent bool
	}{
		{
			name:           "valid comparable measured arms emit positive verdict",
			wantVerdict:    "ISOLATION_POISON_FREE",
			wantEquivalent: true,
		},
		{
			name: "missing wave ID on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.WaveID = ""
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "missing wave ID on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.WaveID = ""
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "both wave IDs empty",
			mutateBaseline: func(a *WorktreeABArm) {
				a.WaveID = ""
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.WaveID = ""
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "whitespace wave ID on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.WaveID = "   "
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "mismatched wave ID",
			mutateIsolated: func(a *WorktreeABArm) {
				a.WaveID = "different-wave"
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "zero completed work on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.Resolved = 0
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "zero completed work on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.Resolved = 0
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "zero completed work on both",
			mutateBaseline: func(a *WorktreeABArm) {
				a.Resolved = 0
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.Resolved = 0
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "negative completed work on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.Resolved = -2
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "mismatched completed work between arms",
			mutateIsolated: func(a *WorktreeABArm) {
				a.Resolved = 2 // baseline has 4
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "zero duration on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DurationSeconds = 0
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "zero duration on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.DurationSeconds = 0
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "negative duration on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DurationSeconds = -100.0
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "negative duration on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.DurationSeconds = -100.0
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "NaN duration on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DurationSeconds = math.NaN()
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "NaN duration on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.DurationSeconds = math.NaN()
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "infinite duration on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DurationSeconds = math.Inf(1)
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "infinite duration on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.DurationSeconds = math.Inf(1)
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "mismatched host IDs",
			mutateIsolated: func(a *WorktreeABArm) {
				a.HostID = "node-2" // baseline has node-1
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "compatible empty host IDs on both",
			mutateBaseline: func(a *WorktreeABArm) {
				a.HostID = ""
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.HostID = ""
			},
			wantVerdict:    "ISOLATION_POISON_FREE",
			wantEquivalent: true,
		},
		{
			name: "compatible one empty host ID",
			mutateBaseline: func(a *WorktreeABArm) {
				a.HostID = ""
			},
			wantVerdict:    "ISOLATION_POISON_FREE",
			wantEquivalent: true,
		},
		{
			name: "compatible case-insensitive host IDs",
			mutateBaseline: func(a *WorktreeABArm) {
				a.HostID = "NODE-1"
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.HostID = "node-1"
			},
			wantVerdict:    "ISOLATION_POISON_FREE",
			wantEquivalent: true,
		},
		{
			name: "poison incidents on isolated arm",
			mutateIsolated: func(a *WorktreeABArm) {
				a.PoisonIncidents = 1
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: true, // arms are equivalent wave, but isolated is poisoned so verdict is NOT_PROVEN
		},
		{
			name: "negative poison incidents on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.PoisonIncidents = -1
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "negative poison incidents on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.PoisonIncidents = -1
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "incomplete delivery accounting on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 1, Outcome: OutcomeRejected, TotalElapsed: 60},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "incomplete delivery accounting on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 1, Outcome: OutcomeRejected, TotalElapsed: 60},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "complete delivery accounting on both matching",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 101, Outcome: OutcomeAccepted, TotalElapsed: 60},
					{IssueID: 102, Outcome: OutcomeAccepted, TotalElapsed: 60},
				}
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 201, Outcome: OutcomeAccepted, TotalElapsed: 60},
					{IssueID: 202, Outcome: OutcomeAccepted, TotalElapsed: 60},
				}
			},
			wantVerdict:    "ISOLATION_POISON_FREE",
			wantEquivalent: true,
		},
		{
			name: "whitespace wave ID on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.WaveID = "   "
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "matching wave IDs with whitespace padding",
			mutateBaseline: func(a *WorktreeABArm) {
				a.WaveID = "  fixed-wave "
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.WaveID = "fixed-wave  "
			},
			wantVerdict:    "ISOLATION_POISON_FREE",
			wantEquivalent: true,
		},
		{
			name: "negative completed work on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.Resolved = -4
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "negative infinite duration on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DurationSeconds = math.Inf(-1)
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "negative infinite duration on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.DurationSeconds = math.Inf(-1)
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "compatible one empty host ID on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.HostID = ""
			},
			wantVerdict:    "ISOLATION_POISON_FREE",
			wantEquivalent: true,
		},
		{
			name: "compatible host IDs with whitespace padding",
			mutateBaseline: func(a *WorktreeABArm) {
				a.HostID = "  NODE-1 "
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.HostID = "node-1  "
			},
			wantVerdict:    "ISOLATION_POISON_FREE",
			wantEquivalent: true,
		},
		{
			name: "negative peak concurrency on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.PeakConcurrency = -1
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "negative peak concurrency on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.PeakConcurrency = -1
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "zero peak concurrency is valid non-negative",
			mutateBaseline: func(a *WorktreeABArm) {
				a.PeakConcurrency = 0
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.PeakConcurrency = 0
			},
			wantVerdict:    "ISOLATION_POISON_FREE",
			wantEquivalent: true,
		},
		{
			name: "delivery records with negative phase duration on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 101, Outcome: OutcomeAccepted, SetupDuration: -10, TotalElapsed: 60},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "delivery records with negative phase duration on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 101, Outcome: OutcomeAccepted, ExecutionDuration: -10, TotalElapsed: 60},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "delivery records with NaN phase duration on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 101, Outcome: OutcomeAccepted, TotalElapsed: math.NaN()},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "delivery records with NaN phase duration on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 101, Outcome: OutcomeAccepted, TotalElapsed: math.NaN()},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "complete delivery accounting on baseline but mismatched accepted count with isolated",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 101, Outcome: OutcomeAccepted, TotalElapsed: 60},
					{IssueID: 102, Outcome: OutcomeAccepted, TotalElapsed: 60},
				}
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 201, Outcome: OutcomeAccepted, TotalElapsed: 60},
					{IssueID: 202, Outcome: OutcomeAccepted, TotalElapsed: 60},
					{IssueID: 203, Outcome: OutcomeAccepted, TotalElapsed: 60},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "lifecycle records alias complete and matching",
			mutateBaseline: func(a *WorktreeABArm) {
				a.LifecycleRecords = []AcceptedDeliveryRecord{
					{IssueID: 101, Outcome: OutcomeAccepted, TotalElapsed: 60},
					{IssueID: 102, Outcome: OutcomeAccepted, TotalElapsed: 60},
				}
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.LifecycleRecords = []AcceptedDeliveryRecord{
					{IssueID: 201, Outcome: OutcomeAccepted, TotalElapsed: 60},
					{IssueID: 202, Outcome: OutcomeAccepted, TotalElapsed: 60},
				}
			},
			wantVerdict:    "ISOLATION_POISON_FREE",
			wantEquivalent: true,
		},
		{
			name: "lifecycle records incomplete on isolated",
			mutateIsolated: func(a *WorktreeABArm) {
				a.LifecycleRecords = []AcceptedDeliveryRecord{
					{IssueID: 201, Outcome: OutcomeRejected, TotalElapsed: 60},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "poison incidents on baseline arm permitted for isolated poison-free verdict",
			mutateBaseline: func(a *WorktreeABArm) {
				a.PoisonIncidents = 5
			},
			wantVerdict:    "ISOLATION_POISON_FREE",
			wantEquivalent: true,
		},
		{
			name: "both wave IDs whitespace",
			mutateBaseline: func(a *WorktreeABArm) {
				a.WaveID = "   "
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.WaveID = "   "
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "delivery accounting on baseline but missing on isolated",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 101, Outcome: OutcomeAccepted, TotalElapsed: 60},
					{IssueID: 102, Outcome: OutcomeAccepted, TotalElapsed: 60},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "delivery accounting on isolated but missing on baseline",
			mutateIsolated: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 201, Outcome: OutcomeAccepted, TotalElapsed: 60},
					{IssueID: 202, Outcome: OutcomeAccepted, TotalElapsed: 60},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "infinite phase duration in baseline delivery records",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 101, Outcome: OutcomeAccepted, ExecutionDuration: math.Inf(1), TotalElapsed: 60},
				}
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 201, Outcome: OutcomeAccepted, TotalElapsed: 60},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "infinite phase duration in isolated delivery records",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 101, Outcome: OutcomeAccepted, TotalElapsed: 60},
				}
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 201, Outcome: OutcomeAccepted, SetupDuration: math.Inf(1), TotalElapsed: 60},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "negative spend in delivery records on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 101, Outcome: OutcomeAccepted, Spend: -2.5, TotalElapsed: 60},
				}
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 201, Outcome: OutcomeAccepted, TotalElapsed: 60},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "negative spend in delivery records on isolated",
			mutateBaseline: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 101, Outcome: OutcomeAccepted, TotalElapsed: 60},
				}
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.DeliveryRecords = []AcceptedDeliveryRecord{
					{IssueID: 201, Outcome: OutcomeAccepted, Spend: -2.5, TotalElapsed: 60},
				}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "pre-populated accounting with NaN elapsed seconds on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.Accounting = AcceptedDeliveryAccounting{Status: "COMPLETE", Verified: true, AcceptedDeliveries: 4, TotalElapsedSeconds: math.NaN()}
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.Accounting = AcceptedDeliveryAccounting{Status: "COMPLETE", Verified: true, AcceptedDeliveries: 4, TotalElapsedSeconds: 800.0}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "pre-populated accounting with infinite elapsed seconds on isolated",
			mutateBaseline: func(a *WorktreeABArm) {
				a.Accounting = AcceptedDeliveryAccounting{Status: "COMPLETE", Verified: true, AcceptedDeliveries: 4, TotalElapsedSeconds: 1200.0}
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.Accounting = AcceptedDeliveryAccounting{Status: "COMPLETE", Verified: true, AcceptedDeliveries: 4, TotalElapsedSeconds: math.Inf(1)}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "pre-populated accounting with unverified status on baseline",
			mutateBaseline: func(a *WorktreeABArm) {
				a.Accounting = AcceptedDeliveryAccounting{Status: "COMPLETE", Verified: false, AcceptedDeliveries: 4, TotalElapsedSeconds: 1200.0}
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.Accounting = AcceptedDeliveryAccounting{Status: "COMPLETE", Verified: true, AcceptedDeliveries: 4, TotalElapsedSeconds: 800.0}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "pre-populated accounting with mismatched status",
			mutateBaseline: func(a *WorktreeABArm) {
				a.Accounting = AcceptedDeliveryAccounting{Status: "COMPLETE", Verified: true, AcceptedDeliveries: 4, TotalElapsedSeconds: 1200.0}
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.Accounting = AcceptedDeliveryAccounting{Status: "CUSTOM", Verified: true, AcceptedDeliveries: 4, TotalElapsedSeconds: 800.0}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "pre-populated accounting with negative accepted deliveries on isolated",
			mutateBaseline: func(a *WorktreeABArm) {
				a.Accounting = AcceptedDeliveryAccounting{Status: "COMPLETE", Verified: true, AcceptedDeliveries: 4, TotalElapsedSeconds: 1200.0}
			},
			mutateIsolated: func(a *WorktreeABArm) {
				a.Accounting = AcceptedDeliveryAccounting{Status: "COMPLETE", Verified: true, AcceptedDeliveries: -1, TotalElapsedSeconds: 800.0}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "pre-populated accounting on baseline but missing on isolated",
			mutateBaseline: func(a *WorktreeABArm) {
				a.Accounting = AcceptedDeliveryAccounting{Status: "COMPLETE", Verified: true, AcceptedDeliveries: 4, TotalElapsedSeconds: 1200.0}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
		{
			name: "pre-populated accounting on isolated but missing on baseline",
			mutateIsolated: func(a *WorktreeABArm) {
				a.Accounting = AcceptedDeliveryAccounting{Status: "COMPLETE", Verified: true, AcceptedDeliveries: 4, TotalElapsedSeconds: 800.0}
			},
			wantVerdict:    "NOT_PROVEN",
			wantEquivalent: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bArm := validBaseline
			iArm := validIsolated
			if tc.mutateBaseline != nil {
				tc.mutateBaseline(&bArm)
			}
			if tc.mutateIsolated != nil {
				tc.mutateIsolated(&iArm)
			}

			gotReport := FoldWorktreeAB(bArm, iArm)
			if gotReport.Verdict != tc.wantVerdict {
				t.Errorf("FoldWorktreeAB().Verdict = %q, want %q", gotReport.Verdict, tc.wantVerdict)
			}

			gotEq := WorktreeABEquivalentWave(bArm, iArm)
			if gotEq != tc.wantEquivalent {
				t.Errorf("WorktreeABEquivalentWave() = %v, want %v", gotEq, tc.wantEquivalent)
			}

			gotComp := WorktreeABComparableArms(bArm, iArm)
			if gotComp != tc.wantEquivalent {
				t.Errorf("WorktreeABComparableArms() = %v, want %v", gotComp, tc.wantEquivalent)
			}
		})
	}
}

func TestWorktreeABIssuesPerHourFailsClosed(t *testing.T) {
	cases := []struct {
		name string
		arm  WorktreeABArm
		want float64
	}{
		{
			name: "zero duration",
			arm:  WorktreeABArm{Resolved: 4, DurationSeconds: 0},
			want: 0,
		},
		{
			name: "negative duration",
			arm:  WorktreeABArm{Resolved: 4, DurationSeconds: -100},
			want: 0,
		},
		{
			name: "NaN duration",
			arm:  WorktreeABArm{Resolved: 4, DurationSeconds: math.NaN()},
			want: 0,
		},
		{
			name: "infinite duration",
			arm:  WorktreeABArm{Resolved: 4, DurationSeconds: math.Inf(1)},
			want: 0,
		},
		{
			name: "zero resolved",
			arm:  WorktreeABArm{Resolved: 0, DurationSeconds: 100},
			want: 0,
		},
		{
			name: "negative resolved",
			arm:  WorktreeABArm{Resolved: -2, DurationSeconds: 100},
			want: 0,
		},
		{
			name: "valid calculation",
			arm:  WorktreeABArm{Resolved: 2, DurationSeconds: 3600},
			want: 2.0,
		},
		{
			name: "accounting status with non-finite rate",
			arm: WorktreeABArm{
				DurationSeconds: 3600,
				Accounting: AcceptedDeliveryAccounting{
					Status:                 "COMPLETE",
					AcceptedPerElapsedHour: math.NaN(),
				},
			},
			want: 0,
		},
		{
			name: "accounting status with negative rate",
			arm: WorktreeABArm{
				DurationSeconds: 3600,
				Accounting: AcceptedDeliveryAccounting{
					Status:                 "COMPLETE",
					AcceptedPerElapsedHour: -5.0,
				},
			},
			want: 0,
		},
		{
			name: "accounting status with valid rate",
			arm: WorktreeABArm{
				DurationSeconds: 3600,
				Accounting: AcceptedDeliveryAccounting{
					Status:                 "COMPLETE",
					Verified:               true,
					AcceptedPerElapsedHour: 10.0,
				},
			},
			want: 10.0,
		},
		{
			name: "accounting status incomplete with positive rate suppresses throughput",
			arm: WorktreeABArm{
				DurationSeconds: 3600,
				Accounting: AcceptedDeliveryAccounting{
					Status:                 "INCOMPLETE",
					Verified:               false,
					AcceptedDeliveries:     2,
					AcceptedPerElapsedHour: 10.0,
				},
			},
			want: 0,
		},
		{
			name: "accounting status unverified suppresses throughput",
			arm: WorktreeABArm{
				DurationSeconds: 3600,
				Accounting: AcceptedDeliveryAccounting{
					Status:                 "COMPLETE",
					Verified:               false,
					AcceptedDeliveries:     2,
					AcceptedPerElapsedHour: 10.0,
				},
			},
			want: 0,
		},
		{
			name: "accounting status VERIFIED with valid rate",
			arm: WorktreeABArm{
				DurationSeconds: 3600,
				Accounting: AcceptedDeliveryAccounting{
					Status:                 "VERIFIED",
					AcceptedPerElapsedHour: 10.0,
				},
			},
			want: 10.0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.arm.IssuesPerHour()
			if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-tc.want) > 1e-6 {
				t.Fatalf("IssuesPerHour() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestWorktreeABIncompleteAccountingSuppressesThroughput(t *testing.T) {
	arm := WorktreeABArm{
		Name:            "isolated",
		DurationSeconds: 3600.0,
		Accounting: AcceptedDeliveryAccounting{
			Status:                 "INCOMPLETE",
			Verified:               false,
			AcceptedDeliveries:     3,
			AcceptedPerElapsedHour: 3.0,
		},
		PoisonIncidents: 0,
		PeakConcurrency: 4,
	}

	if got := arm.IssuesPerHour(); got != 0.0 {
		t.Fatalf("arm.IssuesPerHour() = %v, want 0.0", got)
	}

	rep := WorktreeABReport{
		Baseline: WorktreeABArm{
			Name:            "baseline",
			DurationSeconds: 3600.0,
			Accounting: AcceptedDeliveryAccounting{
				Status:                 "COMPLETE",
				Verified:               true,
				AcceptedDeliveries:     3,
				AcceptedPerElapsedHour: 3.0,
			},
			PoisonIncidents: 1,
			PeakConcurrency: 2,
		},
		Isolated: arm,
		Verdict:  "NOT_PROVEN",
	}

	update := WorktreeABUpdate(rep)
	text := update.Text()
	if !strings.Contains(text, "0.00 issues/h") {
		t.Fatalf("WorktreeABUpdate output missing '0.00 issues/h':\n%s", text)
	}
	wantLine := "isolated: 0.00 issues/h (3 accepted, INCOMPLETE), 0 poison, 3600.0s, peak 4"
	if !strings.Contains(text, wantLine) {
		t.Fatalf("WorktreeABUpdate output missing %q:\n%s", wantLine, text)
	}
}

func TestWorktreeABIncompleteBoundarySuppressesIssuesPerHour(t *testing.T) {
	// Delivery with negative execution duration causes incomplete boundary
	records := []DeliveryLifecycleRecord{
		{IssueID: 101, Outcome: OutcomeAccepted, ExecutionDuration: -10, TotalElapsed: 60},
	}
	acc := AccountAcceptedDeliveries(records, 3600.0)
	if acc.Status != "INCOMPLETE" || acc.Verified {
		t.Fatalf("expected incomplete unverified accounting, got status=%q verified=%v", acc.Status, acc.Verified)
	}
	if acc.AcceptedPerElapsedHour <= 0 {
		t.Fatalf("expected positive raw accepted per elapsed hour, got %v", acc.AcceptedPerElapsedHour)
	}

	arm := WorktreeABArm{
		Name:            "isolated",
		DurationSeconds: 3600.0,
		Accounting:      acc,
	}
	if got := arm.IssuesPerHour(); got != 0.0 {
		t.Fatalf("expected IssuesPerHour() = 0 for incomplete boundary, got %v", got)
	}

	rep := FoldWorktreeAB(
		WorktreeABArm{Name: "baseline", DurationSeconds: 3600.0, WaveID: "w1", Resolved: 1},
		arm,
	)
	text := WorktreeABUpdate(rep).Text()
	if !strings.Contains(text, "0.00 issues/h") {
		t.Fatalf("WorktreeABUpdate text missing '0.00 issues/h':\n%s", text)
	}
}

func TestWorktreeABSubPhaseDurationConsistency(t *testing.T) {
	tests := []struct {
		name         string
		records      []DeliveryLifecycleRecord
		window       float64
		wantVerified bool
		wantStatus   string
	}{
		{
			name: "sub-phases exceed total elapsed rejected as incomplete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:              101,
					Outcome:              OutcomeAccepted,
					SetupDuration:        15.0,
					ExecutionDuration:    80.0,
					LandingDuration:      10.0,
					VerificationDuration: 15.0,
					TotalElapsed:         100.0, // 15 + 80 + 10 + 15 = 120 > 100
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "single execution phase exceeds total elapsed rejected as incomplete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:           102,
					Outcome:           OutcomeAccepted,
					ExecutionDuration: 120.0,
					TotalElapsed:      100.0, // 120 > 100
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "setup phase alone exceeds total elapsed rejected as incomplete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:       103,
					Outcome:       OutcomeAccepted,
					SetupDuration: 105.0,
					TotalElapsed:  100.0, // 105 > 100
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "landing phase alone exceeds total elapsed rejected as incomplete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:         104,
					Outcome:         OutcomeAccepted,
					LandingDuration: 105.0,
					TotalElapsed:    100.0, // 105 > 100
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "verification phase alone exceeds total elapsed rejected as incomplete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:              105,
					Outcome:              OutcomeAccepted,
					VerificationDuration: 105.0,
					TotalElapsed:         100.0, // 105 > 100
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "sub-phases exactly equal to total elapsed accepted as complete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:              106,
					Outcome:              OutcomeAccepted,
					SetupDuration:        10.0,
					ExecutionDuration:    70.0,
					LandingDuration:      10.0,
					VerificationDuration: 10.0,
					TotalElapsed:         100.0, // 10 + 70 + 10 + 10 = 100 == 100
				},
			},
			window:       3600.0,
			wantVerified: true,
			wantStatus:   "COMPLETE",
		},
		{
			name: "sub-phases strictly less than total elapsed accepted as complete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:              107,
					Outcome:              OutcomeAccepted,
					SetupDuration:        10.0,
					ExecutionDuration:    50.0,
					LandingDuration:      10.0,
					VerificationDuration: 10.0,
					TotalElapsed:         100.0, // 10 + 50 + 10 + 10 = 80 < 100
				},
			},
			window:       3600.0,
			wantVerified: true,
			wantStatus:   "COMPLETE",
		},
		{
			name: "multiple records with one exceeding rejected as incomplete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:           201,
					Outcome:           OutcomeAccepted,
					ExecutionDuration: 50.0,
					TotalElapsed:      60.0,
				},
				{
					IssueID:           202,
					Outcome:           OutcomeAccepted,
					SetupDuration:     20.0,
					ExecutionDuration: 60.0,
					TotalElapsed:      60.0, // 20 + 60 = 80 > 60
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := AccountAcceptedDeliveries(tc.records, tc.window)
			if got.Verified != tc.wantVerified {
				t.Errorf("Verified = %v, want %v", got.Verified, tc.wantVerified)
			}
			if got.Status != tc.wantStatus {
				t.Errorf("Status = %q, want %q", got.Status, tc.wantStatus)
			}
		})
	}
}

func TestWorktreeABRejectsInfiniteDurationsAndNegativeSpend(t *testing.T) {
	tests := []struct {
		name             string
		records          []DeliveryLifecycleRecord
		window           float64
		wantStatus       string
		wantVerified     bool
		wantSpendUnknown bool
	}{
		{
			name: "positive infinity total elapsed duration",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: 1.0, TotalElapsed: math.Inf(1)},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: false,
		},
		{
			name: "positive infinity setup duration",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: 1.0, SetupDuration: math.Inf(1), TotalElapsed: 60.0},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: false,
		},
		{
			name: "positive infinity execution duration",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: 1.0, ExecutionDuration: math.Inf(1), TotalElapsed: 60.0},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: false,
		},
		{
			name: "positive infinity landing duration",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: 1.0, LandingDuration: math.Inf(1), TotalElapsed: 60.0},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: false,
		},
		{
			name: "positive infinity verification duration",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: 1.0, VerificationDuration: math.Inf(1), TotalElapsed: 60.0},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: false,
		},
		{
			name: "negative infinity total elapsed duration",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: 1.0, TotalElapsed: math.Inf(-1)},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: false,
		},
		{
			name: "negative infinity setup duration",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: 1.0, SetupDuration: math.Inf(-1), TotalElapsed: 60.0},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: false,
		},
		{
			name: "negative infinity execution duration",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: 1.0, ExecutionDuration: math.Inf(-1), TotalElapsed: 60.0},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: false,
		},
		{
			name: "negative infinity landing duration",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: 1.0, LandingDuration: math.Inf(-1), TotalElapsed: 60.0},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: false,
		},
		{
			name: "negative infinity verification duration",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: 1.0, VerificationDuration: math.Inf(-1), TotalElapsed: 60.0},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: false,
		},
		{
			name: "negative spend marks incomplete and spend unknown",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: -2.5, TotalElapsed: 60.0},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: true,
		},
		{
			name: "NaN spend marks incomplete and spend unknown",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: math.NaN(), TotalElapsed: 60.0},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: true,
		},
		{
			name: "positive infinity spend marks incomplete and spend unknown",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: math.Inf(1), TotalElapsed: 60.0},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: true,
		},
		{
			name: "negative infinity spend marks incomplete and spend unknown",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: math.Inf(-1), TotalElapsed: 60.0},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: true,
		},
		{
			name: "positive infinity duration and negative spend combined",
			records: []DeliveryLifecycleRecord{
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: -1.0, ExecutionDuration: math.Inf(1), TotalElapsed: 60.0},
			},
			window:           3600.0,
			wantStatus:       "INCOMPLETE",
			wantVerified:     false,
			wantSpendUnknown: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := AccountAcceptedDeliveries(tc.records, tc.window)
			if got.Status != tc.wantStatus {
				t.Errorf("Status = %q, want %q", got.Status, tc.wantStatus)
			}
			if got.Verified != tc.wantVerified {
				t.Errorf("Verified = %v, want %v", got.Verified, tc.wantVerified)
			}
			if got.SpendUnknown != tc.wantSpendUnknown {
				t.Errorf("SpendUnknown = %v, want %v", got.SpendUnknown, tc.wantSpendUnknown)
			}
			if tc.wantSpendUnknown && got.Spend < 0 {
				t.Errorf("accumulated spend is negative: %f", got.Spend)
			}
		})
	}
}

// TestAccountAcceptedDeliveries_SubPhaseDurationConsistency tests issue #11939:
// audit(scoreboard): validate sub-phase duration consistency against total elapsed time in delivery lifecycle records.
// Demonstrates that lifecycle records with sub-phase durations exceeding total elapsed window
// (and edge cases like exact equality vs excess) are rejected as incomplete boundaries.
func TestAccountAcceptedDeliveries_SubPhaseDurationConsistency(t *testing.T) {
	tests := []struct {
		name         string
		records      []DeliveryLifecycleRecord
		window       float64
		wantVerified bool
		wantStatus   string
	}{
		{
			name: "#11939 sub-phase durations exceeding total elapsed rejected as incomplete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:              11939,
					Outcome:              OutcomeAccepted,
					SetupDuration:        20.0,
					ExecutionDuration:    80.0,
					LandingDuration:      15.0,
					VerificationDuration: 10.0,
					TotalElapsed:         100.0, // 20 + 80 + 15 + 10 = 125 > 100
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "#11939 minute excess over total elapsed rejected as incomplete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:              11939,
					Outcome:              OutcomeAccepted,
					SetupDuration:        10.0,
					ExecutionDuration:    70.0,
					LandingDuration:      10.0,
					VerificationDuration: 10.001,
					TotalElapsed:         100.0, // sum = 100.001 > 100.0
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "#11939 floating point rounding sum within tolerance accepted as complete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:              11939,
					Outcome:              OutcomeAccepted,
					SetupDuration:        0.1,
					ExecutionDuration:    0.2,
					LandingDuration:      0.3,
					VerificationDuration: 0.4,
					TotalElapsed:         1.0, // 0.1+0.2+0.3+0.4 = 1.0000000000000002 in IEEE 754
				},
			},
			window:       3600.0,
			wantVerified: true,
			wantStatus:   "COMPLETE",
		},
		{
			name: "#11939 sub-phase excess within allowable tolerance accepted as complete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:              11939,
					Outcome:              OutcomeAccepted,
					SetupDuration:        10.0,
					ExecutionDuration:    70.0,
					LandingDuration:      10.0,
					VerificationDuration: 10.0000005, // excess 5e-7 <= 1e-6 tolerance
					TotalElapsed:         100.0,
				},
			},
			window:       3600.0,
			wantVerified: true,
			wantStatus:   "COMPLETE",
		},
		{
			name: "#11939 sub-phase excess exceeding allowable tolerance rejected as incomplete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:              11939,
					Outcome:              OutcomeAccepted,
					SetupDuration:        10.0,
					ExecutionDuration:    70.0,
					LandingDuration:      10.0,
					VerificationDuration: 10.000002, // excess 2e-6 > 1e-6 tolerance
					TotalElapsed:         100.0,
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "#11939 exact equality of sub-phases to total elapsed accepted as complete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:              11939,
					Outcome:              OutcomeAccepted,
					SetupDuration:        10.0,
					ExecutionDuration:    70.0,
					LandingDuration:      10.0,
					VerificationDuration: 10.0,
					TotalElapsed:         100.0, // 10 + 70 + 10 + 10 = 100.0 == 100.0
				},
			},
			window:       3600.0,
			wantVerified: true,
			wantStatus:   "COMPLETE",
		},
		{
			name: "#11939 sub-phases strictly less than total elapsed accepted as complete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:              11939,
					Outcome:              OutcomeAccepted,
					SetupDuration:        5.0,
					ExecutionDuration:    50.0,
					LandingDuration:      5.0,
					VerificationDuration: 5.0,
					TotalElapsed:         100.0, // sum = 65.0 < 100.0
				},
			},
			window:       3600.0,
			wantVerified: true,
			wantStatus:   "COMPLETE",
		},
		{
			name: "#11939 setup duration alone exceeds total elapsed",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:       11939,
					Outcome:       OutcomeAccepted,
					SetupDuration: 101.0,
					TotalElapsed:  100.0,
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "#11939 execution duration alone exceeds total elapsed",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:           11939,
					Outcome:           OutcomeAccepted,
					ExecutionDuration: 101.0,
					TotalElapsed:      100.0,
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "#11939 landing duration alone exceeds total elapsed",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:         11939,
					Outcome:         OutcomeAccepted,
					LandingDuration: 101.0,
					TotalElapsed:    100.0,
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "#11939 verification duration alone exceeds total elapsed",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:              11939,
					Outcome:              OutcomeAccepted,
					VerificationDuration: 101.0,
					TotalElapsed:         100.0,
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "#11939 zero total elapsed with positive sub-phase duration rejected as incomplete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:           11939,
					Outcome:           OutcomeAccepted,
					ExecutionDuration: 1.0,
					TotalElapsed:      0.0,
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "#11939 zero total elapsed with zero sub-phases accepted as complete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:      11939,
					Outcome:      OutcomeAccepted,
					TotalElapsed: 0.0,
				},
			},
			window:       3600.0,
			wantVerified: true,
			wantStatus:   "COMPLETE",
		},
		{
			name: "#11939 multi-record batch with one exceeding rejected as incomplete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:           11939,
					Outcome:           OutcomeAccepted,
					ExecutionDuration: 40.0,
					TotalElapsed:      60.0,
				},
				{
					IssueID:              11940,
					Outcome:              OutcomeAccepted,
					SetupDuration:        20.0,
					ExecutionDuration:    50.0,
					LandingDuration:      10.0,
					VerificationDuration: 10.0,
					TotalElapsed:         60.0, // 20 + 50 + 10 + 10 = 90 > 60
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
		{
			name: "#11939 non-accepted outcome with sub-phase excess marks boundary incomplete",
			records: []DeliveryLifecycleRecord{
				{
					IssueID:           11939,
					Outcome:           OutcomeAccepted,
					ExecutionDuration: 50.0,
					TotalElapsed:      100.0,
				},
				{
					IssueID:           11941,
					Outcome:           OutcomeRejected,
					ExecutionDuration: 150.0,
					TotalElapsed:      100.0, // 150 > 100
				},
			},
			window:       3600.0,
			wantVerified: false,
			wantStatus:   "INCOMPLETE",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := AccountAcceptedDeliveries(tc.records, tc.window)
			if got.Verified != tc.wantVerified {
				t.Errorf("Verified = %v, want %v", got.Verified, tc.wantVerified)
			}
			if got.Status != tc.wantStatus {
				t.Errorf("Status = %q, want %q", got.Status, tc.wantStatus)
			}
			if !tc.wantVerified {
				arm := WorktreeABArm{
					Name:            "isolated",
					Worktree:        true,
					DurationSeconds: tc.window,
					Accounting:      got,
				}
				if iph := arm.IssuesPerHour(); iph != 0 {
					t.Errorf("IssuesPerHour = %f, want 0.0 on incomplete boundary", iph)
				}
			}
		})
	}
}
