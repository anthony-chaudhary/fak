package modelperfobs

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestQwenSwapUsageWeeklyFoldSeparatesCommittedRefusedAndErrors(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "qwen-swap.jsonl")
	rows := []QwenSwapUsageRow{
		newQwenSwapUsageRow(time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC), QwenSwapDirectionOut, QwenSwapOutcomeSuccess, QwenSwapResultCommitted, 158488532),
		newQwenSwapUsageRow(time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC), QwenSwapDirectionIn, QwenSwapOutcomeSuccess, QwenSwapResultRefused, 158488532),
		newQwenSwapUsageRow(time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC), QwenSwapDirectionIn, QwenSwapOutcomeError, QwenSwapResultRefused, 32),
	}
	for _, row := range rows {
		if err := AppendQwenSwapUsage(ledger, row); err != nil {
			t.Fatal(err)
		}
	}

	got, err := FoldQwenSwapUsage(ledger)
	if err != nil {
		t.Fatal(err)
	}
	want := []QwenSwapWeeklyUsage{
		{WeekStart: "2026-08-17", Invocations: 1, Bytes: 158488532, SwapOut: 1, Succeeded: 1},
		{WeekStart: "2026-08-24", Invocations: 2, Bytes: 158488564, RestoreIn: 2, Refused: 2, Errors: 1},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("weekly fold = %+v, want %+v", got, want)
	}
}

func TestQwenSwapUsageIsOptInAndRejectsSuccessForRefusal(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	row := newQwenSwapUsageRow(time.Now(), QwenSwapDirectionOut, QwenSwapOutcomeSuccess, QwenSwapResultCommitted, 64)
	if err := AppendQwenSwapUsage("", row); err != nil {
		t.Fatalf("disabled ledger: %v", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("disabled ledger wrote %d entries", len(entries))
	}

	bad := row
	bad.Outcome = QwenSwapOutcomeError
	bad.Result = QwenSwapResultCommitted
	if err := AppendQwenSwapUsage(filepath.Join(dir, "bad.jsonl"), bad); err == nil {
		t.Fatal("error outcome claimed committed success")
	}
	if _, err := os.Stat(filepath.Join(dir, "bad.jsonl")); !os.IsNotExist(err) {
		t.Fatalf("invalid row created a ledger: %v", err)
	}
}

func TestQwenSwapUsageRowHasClosedPrivacySafeShape(t *testing.T) {
	row := newQwenSwapUsageRow(time.Unix(1, 0), QwenSwapDirectionOut, QwenSwapOutcomeSuccess, QwenSwapResultCommitted, 64)
	raw, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]any
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	want := []string{"schema", "observed_at", "version", "outcome", "bytes", "direction", "result"}
	if len(fields) != len(want) {
		t.Fatalf("usage row fields = %v, want only %v", fields, want)
	}
	for _, key := range want {
		if _, ok := fields[key]; !ok {
			t.Fatalf("usage row missing %q: %v", key, fields)
		}
	}
}

func newQwenSwapUsageRow(at time.Time, direction, outcome, result string, bytes int64) QwenSwapUsageRow {
	return QwenSwapUsageRow{
		Schema: QwenSwapUsageSchema, ObservedAt: at.UTC(), Version: QwenSwapCodecVersion,
		Direction: direction, Outcome: outcome, Result: result, Bytes: bytes,
	}
}
