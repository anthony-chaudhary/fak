package perfrsiscore

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAppendUsageWritesOneSanitizedRowPerReceipt(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	privatePath := "/private/fleet/host-alpha/evidence.json"
	receipt := LoopTurnReceipt{
		Schema:                LoopTurnSchema,
		Status:                LoopTurnScored,
		Reason:                "SCORE_COMPLETE",
		Input:                 privatePath,
		Snapshot:              "run-42",
		UnavailableDiagnostic: "failed on host-alpha",
		InvocationOutcomes:    OutcomeCounts{Success: 1},
	}
	at := time.Date(2026, 8, 30, 12, 0, 0, 0, time.FixedZone("test", -7*60*60))
	if err := AppendUsage(path, at, receipt); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if lines := strings.Count(string(b), "\n"); lines != 1 {
		t.Fatalf("ledger rows=%d, want 1: %q", lines, b)
	}
	for _, secret := range []string{privatePath, "host-alpha", "unavailable_diagnostic", `"input"`} {
		if strings.Contains(string(b), secret) {
			t.Fatalf("usage row leaked %q: %s", secret, b)
		}
	}
	fold, err := FoldUsage(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := FormatUsageFold(fold); got != "2026-W35 invocations=1 scored=1 unavailable=0 success=1 refusal=0 error=0" {
		t.Fatalf("fold=%q", got)
	}
}

func TestFoldUsageCountsInvocationsByISOWeek(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	rows := []struct {
		at      time.Time
		receipt LoopTurnReceipt
	}{
		{time.Date(2026, 12, 31, 12, 0, 0, 0, time.UTC), LoopTurnReceipt{Status: LoopTurnScored, Reason: "SCORE_COMPLETE", InvocationOutcomes: OutcomeCounts{Success: 1}}},
		{time.Date(2027, 1, 1, 12, 0, 0, 0, time.UTC), LoopTurnReceipt{Status: LoopTurnUnavailable, Reason: "SCORE_INPUT_UNAVAILABLE", InvocationOutcomes: OutcomeCounts{Refusal: 1}}},
		{time.Date(2027, 1, 4, 12, 0, 0, 0, time.UTC), LoopTurnReceipt{Status: LoopTurnUnavailable, Reason: "SCORE_INPUT_UNAVAILABLE", InvocationOutcomes: OutcomeCounts{Error: 1}}},
	}
	for _, row := range rows {
		if err := AppendUsage(path, row.at, row.receipt); err != nil {
			t.Fatal(err)
		}
	}
	fold, err := FoldUsage(path)
	if err != nil {
		t.Fatal(err)
	}
	got := FormatUsageFold(fold)
	want := strings.Join([]string{
		"2026-W53 invocations=2 scored=1 unavailable=1 success=1 refusal=1 error=0",
		"2027-W01 invocations=1 scored=0 unavailable=1 success=0 refusal=0 error=1",
	}, "\n")
	if got != want {
		t.Fatalf("fold:\n%s\nwant:\n%s", got, want)
	}
}

func TestFoldUsageSkipsMalformedTailWithoutInventingCounts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	if err := AppendUsage(path, time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC), LoopTurnReceipt{
		Status: LoopTurnScored, Reason: "SCORE_COMPLETE", InvocationOutcomes: OutcomeCounts{Success: 1},
	}); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(`{"schema":"fak-performance-rsi-usage/1"`); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	fold, err := FoldUsage(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := FormatUsageFold(fold); got != "2026-W35 invocations=1 scored=1 unavailable=0 success=1 refusal=0 error=0" {
		t.Fatalf("fold=%q", got)
	}
}
