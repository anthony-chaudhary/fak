package ops

import (
	"path/filepath"
	"testing"
	"time"
)

func TestSchemaAndLedgerRoundTrip(t *testing.T) {
	tmp := t.TempDir()
	logPath := filepath.Join(tmp, "ops-events.jsonl")

	led, err := OpenLedger(logPath)
	if err != nil {
		t.Fatalf("OpenLedger: %v", err)
	}

	ev1 := Event{
		ActionType:     ActionStorageReclaim,
		Details:        "Tier 0 clean-bins pruned 3 files",
		BytesReclaimed: 1048576,
		DurationMS:     12,
	}
	if err := led.Record(ev1); err != nil {
		t.Fatalf("Record ev1: %v", err)
	}

	ev2 := Event{
		ActionType:   ActionProcessReap,
		Details:      "Reaped orphan helper",
		PIDsAffected: []int{1234, 5678},
		DurationMS:   5,
	}
	if err := led.Record(ev2); err != nil {
		t.Fatalf("Record ev2: %v", err)
	}

	events, err := led.QueryEvents(1 * time.Hour)
	if err != nil {
		t.Fatalf("QueryEvents: %v", err)
	}
	if len(events) != 2 {
		t.Fatalf("expected 2 events, got %d", len(events))
	}

	if events[0].Schema != EventSchemaV1 {
		t.Errorf("expected schema %s, got %s", EventSchemaV1, events[0].Schema)
	}
	if events[0].BytesReclaimed != 1048576 {
		t.Errorf("expected 1048576 bytes reclaimed, got %d", events[0].BytesReclaimed)
	}
	if len(events[1].PIDsAffected) != 2 || events[1].PIDsAffected[0] != 1234 {
		t.Errorf("unexpected PIDs affected: %v", events[1].PIDsAffected)
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.WarningFreeBytes != 4*1024*1024*1024 {
		t.Errorf("unexpected warning bytes: %d", cfg.WarningFreeBytes)
	}
	if cfg.RefuseFreeBytes != 2*1024*1024*1024 {
		t.Errorf("unexpected refuse bytes: %d", cfg.RefuseFreeBytes)
	}
	if !cfg.OrphanReapEnabled {
		t.Errorf("expected orphan reap enabled by default")
	}
}
