package main

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

// TestRecordFakDevUsageJoinsSharedJournal pins that a fak-dev invocation writes
// one row to the SAME usagelog journal cmd/fak writes, carrying the fak-dev verb,
// so the fak_cli_* Grafana families count developer tooling alongside runtime
// verbs. The strix known-hosts broker child is excluded as an internal transport.
func TestRecordFakDevUsageJoinsSharedJournal(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "usage.jsonl")
	t.Setenv("FAK_USAGE_LOG_PATH", logPath)
	t.Setenv("FAK_USAGE_LOG", "")

	recordFakDevUsage([]string{"index", "verbs", "intent"}, 0)

	rows, err := usagelog.ReadRows(logPath)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
	if rows[0].Verb != "index" || rows[0].ExitCode != 0 {
		t.Fatalf("row = %+v, want verb=index exit 0", rows[0])
	}
	if rows[0].Argc != 2 {
		t.Fatalf("argc = %d, want 2", rows[0].Argc)
	}
}

// TestRecordFakDevUsageSkipsBrokerChild pins the exclusion of the internal
// known-hosts broker transport, which is not an operator-visible verb.
func TestRecordFakDevUsageSkipsBrokerChild(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "usage.jsonl")
	t.Setenv("FAK_USAGE_LOG_PATH", logPath)
	t.Setenv("FAK_USAGE_LOG", "")

	recordFakDevUsage([]string{amdgpu.StrixKnownHostsOperand, "a", "b"}, 0)

	rows, err := usagelog.ReadRows(logPath)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	if len(rows) != 0 {
		t.Fatalf("broker child must not be recorded, got %+v", rows)
	}
}
