package archreport

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestUsageLedgerAppendsPrivacySafeRowsAndFoldsWeeks(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	rows := []Usage{
		{At: "2026-08-03T10:00:00Z", Mode: "full", Format: "text", Outcome: "ok", Diagnostics: 1},
		{At: "2026-08-04T10:00:00Z", Mode: "scoped", Format: "json", Outcome: "error"},
		{At: "2026-08-10T10:00:00Z", Mode: "scoped", Format: "json", Outcome: "ok", Violations: 2},
	}
	for _, row := range rows {
		if err := AppendUsage(path, row); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"workspace", "hostname", "username", "leaf_name", `C:\\`, "/home/"} {
		if strings.Contains(string(raw), forbidden) {
			t.Fatalf("ledger leaks %q: %s", forbidden, raw)
		}
	}
	for i, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatalf("line %d: %v", i+1, err)
		}
		if row["schema"] != UsageSchema {
			t.Fatalf("line %d schema=%v", i+1, row["schema"])
		}
	}
	weeks, err := FoldUsage(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(weeks) != 2 {
		t.Fatalf("weeks=%+v", weeks)
	}
	if got := weeks[0]; got.Week != "2026-W32" || got.Invocations != 2 || got.Full != 1 || got.Scoped != 1 || got.Text != 1 || got.JSON != 1 || got.OK != 1 || got.Error != 1 {
		t.Fatalf("week 32=%+v", got)
	}
	if got := weeks[1]; got.Week != "2026-W33" || got.Invocations != 1 || got.Scoped != 1 || got.JSON != 1 || got.OK != 1 {
		t.Fatalf("week 33=%+v", got)
	}
}

func TestUsagePathOverrideAndDisable(t *testing.T) {
	want := filepath.Join(t.TempDir(), "custom.jsonl")
	t.Setenv("FAK_ARCHITECTURE_USAGE_FILE", want)
	got, err := UsagePath()
	if err != nil || got != want {
		t.Fatalf("path=%q err=%v", got, err)
	}
	t.Setenv("FAK_ARCHITECTURE_USAGE_FILE", "off")
	got, err = UsagePath()
	if err != nil || got != "" {
		t.Fatalf("disabled path=%q err=%v", got, err)
	}
}

func TestAppendUsageRejectsInvalidRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	for _, row := range []Usage{
		{At: "not-time", Mode: "full", Format: "text", Outcome: "ok"},
		{At: "2026-08-03T10:00:00Z", Mode: "leaf-name", Format: "text", Outcome: "ok"},
	} {
		if err := AppendUsage(path, row); err == nil {
			t.Fatalf("accepted invalid row %+v", row)
		}
	}
}
