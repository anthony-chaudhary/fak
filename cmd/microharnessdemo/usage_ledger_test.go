package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestUsageLedgerAppendsAndFoldsWeekly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	rows := []usageRow{
		{At: time.Date(2026, 8, 17, 1, 0, 0, 0, time.UTC), Mode: "fixture", Outcome: "completed", Completions: 6},
		{At: time.Date(2026, 8, 22, 1, 0, 0, 0, time.UTC), Mode: "live", Outcome: "completed", PromptTokens: 66, CompletionTokens: 18},
		{At: time.Date(2026, 8, 24, 1, 0, 0, 0, time.UTC), Mode: "live", Outcome: "failed"},
	}
	for _, row := range rows {
		if err := appendUsage(path, row); err != nil {
			t.Fatal(err)
		}
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(splitNonemptyLines(string(body))); got != 3 {
		t.Fatalf("rows = %d", got)
	}
	weeks, err := foldWeeklyUsage(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(weeks) != 2 || weeks[0].Invocations != 2 || weeks[0].Completed != 2 || weeks[0].Live != 1 || weeks[0].Fixture != 1 {
		t.Fatalf("weeks = %#v", weeks)
	}
	if weeks[0].PromptTokens != 66 || weeks[0].CompletionTokens != 18 || weeks[1].Failed != 1 {
		t.Fatalf("weeks = %#v", weeks)
	}
}

func TestUsageLedgerRejectsMalformedRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	if err := os.WriteFile(path, []byte("not-json\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := foldWeeklyUsage(path); err == nil {
		t.Fatal("malformed ledger accepted")
	}
}

func splitNonemptyLines(s string) []string {
	var out []string
	start := 0
	for i, r := range s {
		if r == '\n' {
			if i > start {
				out = append(out, s[start:i])
			}
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}
