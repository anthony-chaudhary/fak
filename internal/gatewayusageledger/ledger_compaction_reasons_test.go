package gatewayusageledger

import (
	"path/filepath"
	"testing"
	"time"
)

// TestCompactionBailReasonsRoundTrip witnesses the guard-usage-plane diagnostic axis:
// a "guard" exit row carrying the per-reason compaction bail breakdown (the closed
// agent.CompactReason* vocabulary) and the anchor-starved subset survives Append and
// reads back intact. Before these fields existed the WHY behind a zero fak shed slice
// (burst_unprofitable vs under_budget vs anchor-starved, #1407/#1408) lived only on
// the in-process /metrics and the console exit summary — gone the moment the guard
// process exited, which is exactly the population the managed-cache proving ground
// needs to read.
func TestCompactionBailReasonsRoundTrip(t *testing.T) {
	ledgerPath := filepath.Join(t.TempDir(), "gateway-usage.jsonl")

	row := NewRow("exit", "guard", "claude", "", 90*time.Second, Counters{
		CompactionBailed: 14,
		CompactionBailReasons: map[string]uint64{
			"under_budget":       11,
			"burst_unprofitable": 2,
			"too_few_msgs":       1,
		},
		CompactionAnchorStarved: 3,
	}, time.Unix(3000, 0))
	if err := Append(ledgerPath, row); err != nil {
		t.Fatalf("Append: %v", err)
	}

	rows := ReadLedgerFile(ledgerPath)
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	got := rows[0].Counters
	if got.CompactionBailed != 14 {
		t.Fatalf("CompactionBailed = %d, want 14", got.CompactionBailed)
	}
	if got.CompactionBailReasons["burst_unprofitable"] != 2 ||
		got.CompactionBailReasons["under_budget"] != 11 ||
		got.CompactionBailReasons["too_few_msgs"] != 1 {
		t.Fatalf("CompactionBailReasons did not round-trip: %+v", got.CompactionBailReasons)
	}
	if got.CompactionAnchorStarved != 3 {
		t.Fatalf("CompactionAnchorStarved = %d, want 3", got.CompactionAnchorStarved)
	}
	if rows[0].SessionType != "guard" {
		t.Fatalf("SessionType = %q, want guard", rows[0].SessionType)
	}
}
