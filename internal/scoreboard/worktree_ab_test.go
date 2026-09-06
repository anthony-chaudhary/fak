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
				{IssueID: 101, Outcome: OutcomeAccepted, Spend: 0.5, TotalElapsed: 60},  // duplicate IssueID
				{IssueID: 103, Outcome: OutcomeDuplicate, Spend: 0.2, TotalElapsed: 60}, // explicit duplicate
				{IssueID: 104, Outcome: OutcomeRejected, Spend: 0.8, TotalElapsed: 60},  // rejected
				{IssueID: 105, Outcome: OutcomeUnverified, Spend: 0.4, TotalElapsed: 60},// unverified
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
				{IssueID: 201, Outcome: OutcomeAccepted, Spend: 0.5, TotalElapsed: 40},  // duplicate IssueID
				{IssueID: 204, Outcome: OutcomeUnverified, Spend: 0.5, TotalElapsed: 40},// unverified
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
}
