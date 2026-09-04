package dispatchconservation

import (
	"fmt"
	"testing"
	"time"
)

func BenchmarkDispatchConservation(b *testing.B) {
	now := time.Date(2026, 8, 7, 12, 0, 0, 0, time.UTC)
	nowISO := now.Format(time.RFC3339)

	reasons := []string{
		ReasonProviderQuotaWall,
		ReasonWallClockExhausted,
		ReasonCleanExitNoCommit,
		ReasonDiedBeforeEpilogue,
		"self_modify",
		"policy_block",
		"restart_exhausted",
		"unknown",
	}

	units := make([]Unit, 0, 120)
	for i := 0; i < 120; i++ {
		issue := (i % 40) + 1
		var outcome, reason string
		switch i % 6 {
		case 0:
			outcome = OutcomeLive
		case 1:
			outcome = OutcomeWitnessed
		case 2:
			outcome = OutcomeUnwitnessed
		case 3:
			outcome = OutcomeNoCommit
			reason = reasons[i%len(reasons)]
		case 4:
			outcome = OutcomeSpawnFailed
		case 5:
			outcome = OutcomeLeaked
		}

		stamp := now.Add(-time.Duration(i) * 3 * time.Minute)
		units = append(units, Unit{
			Log:        fmt.Sprintf("worker-dispatch-issue-%d-20260807-120000.log", issue),
			Kind:       "resolve",
			Issue:      issue,
			Lane:       "testlane",
			Backend:    "claude",
			SpawnedUTC: stamp.Format(time.RFC3339),
			Outcome:    outcome,
			Reason:     reason,
			PID:        1000 + i,
			stamp:      stamp,
		})
	}

	openCount := 10
	baselineCount := 15
	closes := Closes{
		ClosedInWindow: 5,
		OpenNow:        &openCount,
		BaselineOpen:   &baselineCount,
	}
	holds := Holds{
		Rows:           5,
		DistinctIssues: 2,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r := FoldConservation(units, closes, holds, 6.0, nowISO)
		if r.Units.ResolveTotal != 120 {
			b.Fatalf("unexpected total units: got %d, want 120", r.Units.ResolveTotal)
		}
	}
}

func TestDispatchConservationBenchmarkSanity(t *testing.T) {
	now := time.Date(2026, 8, 7, 12, 0, 0, 0, time.UTC)
	units := []Unit{
		{Kind: "resolve", Outcome: OutcomeWitnessed, Issue: 1},
		{Kind: "resolve", Outcome: OutcomeLeaked, Issue: 2},
	}
	closes := Closes{ClosedInWindow: 1}
	holds := Holds{Rows: 0, DistinctIssues: 0}

	report := FoldConservation(units, closes, holds, 1.0, now.Format(time.RFC3339))
	if report.Units.ResolveTotal != 2 {
		t.Fatalf("expected 2 units, got %d", report.Units.ResolveTotal)
	}
	if report.Units.ShippedWitnessed != 1 || report.Units.LeakedUnswept != 1 {
		t.Fatalf("unexpected breakdown: %+v", report.Units)
	}
}
