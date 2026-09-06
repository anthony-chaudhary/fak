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
