package ultracodebench

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAppendAccountingUsageWritesDurablePublicSafeJSONL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	at := time.Date(2026, 8, 26, 12, 30, 0, 0, time.FixedZone("private-zone", -7*60*60))
	if _, err := AppendAccountingUsage(path, at, knownAccounting(1, 1, 0, 0, 2, .01, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")); err != nil {
		t.Fatal(err)
	}
	if _, err := AppendAccountingUsage(path, at.Add(time.Hour), missingAccountingReceipt()); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("rows=%d want 2: %s", len(lines), data)
	}
	for _, forbidden := range []string{"usage.jsonl", "private-zone", "hostname", "provider"} {
		if strings.Contains(string(data), forbidden) {
			t.Fatalf("ledger leaked %q: %s", forbidden, data)
		}
	}
	var first AccountingUsageRow
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatal(err)
	}
	if first.Schema != AccountingUsageSchema || first.Invocations != 1 || first.Outcomes.Success != 1 {
		t.Fatalf("unexpected first row: %+v", first)
	}
	if !strings.HasSuffix(lines[0], `"error":0}}`) {
		t.Fatalf("row is not one complete JSON object: %s", lines[0])
	}
}

func TestFoldAccountingUsageByISOWeekAndOutcome(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	available := knownAccounting(1, 1, 0, 0, 2, .01, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	partial := available
	partial.InputTokens.Availability = AccountingPartial
	partial.InputTokens.Coverage = .5
	partial.InputTokens.Reason = "provider sample covered half the invocation"
	for _, tc := range []struct {
		at      time.Time
		receipt AccountingReceipt
	}{
		{time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC), available},
		{time.Date(2026, 1, 2, 0, 0, 0, 0, time.UTC), missingAccountingReceipt()},
		{time.Date(2026, 1, 5, 0, 0, 0, 0, time.UTC), partial},
	} {
		if _, err := AppendAccountingUsage(path, tc.at, tc.receipt); err != nil {
			t.Fatal(err)
		}
	}
	folds, err := FoldAccountingUsage(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(folds) != 2 {
		t.Fatalf("folds=%+v", folds)
	}
	if got := folds[0]; got.Week != "2026-W01" || got.Invocations != 2 || got.Outcomes.Success != 1 || got.Outcomes.Refusal != 1 || got.Outcomes.Error != 0 {
		t.Fatalf("week one=%+v", got)
	}
	if got := folds[1]; got.Week != "2026-W02" || got.Invocations != 1 || got.Outcomes.Success != 0 || got.Outcomes.Refusal != 0 || got.Outcomes.Error != 1 {
		t.Fatalf("week two=%+v", got)
	}
}
