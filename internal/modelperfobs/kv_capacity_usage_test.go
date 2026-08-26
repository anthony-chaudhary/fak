package modelperfobs

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestKVCapacityUsageLedgerAndWeeklyFold(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "kv-capacity-direct.json"))
	if err != nil {
		t.Fatal(err)
	}
	sample, err := DecodeKVMetricSample(data, KVDialectDirect)
	if err != nil {
		t.Fatal(err)
	}
	ledger := filepath.Join(t.TempDir(), "usage.jsonl")

	for _, at := range []time.Time{
		time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC),
		time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC),
		time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC),
	} {
		if _, err := NormalizeKVCapacity(ledger, at, sample, nil); err != nil {
			t.Fatal(err)
		}
	}

	raw, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	lines := bytesLines(raw)
	if len(lines) != 3 {
		t.Fatalf("ledger rows = %d, want 3", len(lines))
	}
	for i, line := range lines {
		var row KVCapacityUsageRow
		if err := json.Unmarshal(line, &row); err != nil {
			t.Fatalf("row %d: %v", i+1, err)
		}
		if row.Schema != KVCapacityUsageSchema || row.Dialect != KVDialectDirect || row.Outcome != "valid" {
			t.Fatalf("row %d = %+v", i+1, row)
		}
	}

	fold, err := FoldKVCapacityUsage(ledger)
	if err != nil {
		t.Fatal(err)
	}
	want := []KVCapacityWeeklyCount{
		{WeekStart: "2026-08-17", Invocations: 1, Valid: 1},
		{WeekStart: "2026-08-24", Invocations: 2, Valid: 2},
	}
	if len(fold) != len(want) {
		t.Fatalf("fold = %+v", fold)
	}
	for i := range want {
		if fold[i] != want[i] {
			t.Fatalf("fold[%d] = %+v, want %+v", i, fold[i], want[i])
		}
	}
}

func TestKVCapacityRefusesToClaimSuccessWithoutLedger(t *testing.T) {
	_, err := NormalizeKVCapacity("", time.Now(), KVMetricSample{}, nil)
	if err == nil {
		t.Fatal("expected missing ledger error")
	}
}

func bytesLines(data []byte) [][]byte {
	var lines [][]byte
	start := 0
	for i, b := range data {
		if b == '\n' {
			if i > start {
				lines = append(lines, data[start:i])
			}
			start = i + 1
		}
	}
	return lines
}

func TestKVCapacityUsageDogfoodFold(t *testing.T) {
	fold, err := FoldKVCapacityUsage(filepath.Join("testdata", "kv-capacity-usage.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	want := []KVCapacityWeeklyCount{
		{WeekStart: "2026-08-17", Invocations: 1, Valid: 1},
		{WeekStart: "2026-08-24", Invocations: 2, Valid: 1, Invalid: 1},
	}
	if len(fold) != len(want) {
		t.Fatalf("fold = %+v", fold)
	}
	for i := range want {
		if fold[i] != want[i] {
			t.Fatalf("fold[%d] = %+v, want %+v", i, fold[i], want[i])
		}
	}
}
