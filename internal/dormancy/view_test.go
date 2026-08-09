package dormancy

import (
	"strings"
	"testing"
	"time"
)

func TestFoldDistinguishesPlannedSleepFromHungLoop(t *testing.T) {
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	ledger := `{"loop_id":"planned","last_active_at":"2026-08-08T10:00:00Z","expected_interval_seconds":300,"sleep_until":"2026-08-08T13:00:00Z","restore_count":2}
{"loop_id":"hung","last_active_at":"2026-08-08T11:00:00Z","expected_interval_seconds":300,"restore_count":1}`
	records, err := ReadLedger(strings.NewReader(ledger))
	if err != nil {
		t.Fatal(err)
	}
	got := Fold(records, now)
	if got.Dormant != 1 || got.Stuck != 1 {
		t.Fatalf("dormant=%d stuck=%d", got.Dormant, got.Stuck)
	}
	if got.Loops[0].LoopID != "hung" || got.Loops[0].Status != "stuck" {
		t.Fatalf("hung=%+v", got.Loops[0])
	}
	if got.Loops[1].LoopID != "planned" || got.Loops[1].Status != "intentionally_dormant" {
		t.Fatalf("planned=%+v", got.Loops[1])
	}
	if got.OldestLoopID != "planned" {
		t.Fatalf("oldest=%q", got.OldestLoopID)
	}
	metrics := got.Prometheus()
	for _, want := range []string{"fak_dormancy_gap_seconds", "status=\"stuck\"", "status=\"intentionally_dormant\"", "fak_dormancy_restores_total"} {
		if !strings.Contains(metrics, want) {
			t.Errorf("metrics missing %q", want)
		}
	}
}

func TestReadLedgerRejectsMissingClock(t *testing.T) {
	_, err := ReadLedger(strings.NewReader(`{"loop_id":"x"}`))
	if err == nil {
		t.Fatal("expected error")
	}
}
