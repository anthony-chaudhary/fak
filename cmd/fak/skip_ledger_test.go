package main

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestRunDispatchSkipLedger_RendersSelectedAndSkippedRows(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/candidates.json"
	fixture := `[
		{"id":"1","key":"a","updated_unix":900},
		{"id":"2","key":"a","updated_unix":800},
		{"id":"3","live":true,"updated_unix":700}
	]`
	if err := os.WriteFile(path, []byte(fixture), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := runDispatchSkipLedger(&stdout, &stderr, []string{"--in", path, "--workspace", dir, "--now", "1000"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "1 selected") {
		t.Fatalf("want 1 selected candidate, got %q", out)
	}
	if !strings.Contains(out, "2 skipped") {
		t.Fatalf("want 2 skipped candidates, got %q", out)
	}
	if !strings.Contains(out, "superseded_by_fresher") || !strings.Contains(out, "worker_live") {
		t.Fatalf("want both skip reasons rendered, got %q", out)
	}

	ledger, err := os.ReadFile(dir + "/.dispatch-runs/skip-ledger.jsonl")
	if err != nil {
		t.Fatalf("read persisted ledger: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(ledger)), "\n")
	if len(lines) != 3 {
		t.Fatalf("want 3 persisted rows, got %d: %s", len(lines), ledger)
	}
}

func TestRunDispatchSkipLedger_JSONOutput(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/candidates.json"
	if err := os.WriteFile(path, []byte(`[{"id":"5","updated_unix":100}]`), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := runDispatchSkipLedger(&stdout, &stderr, []string{"--in", path, "--workspace", dir, "--now", "1000", "--json"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"selected_count": 1`) {
		t.Fatalf("want selected_count=1 in the json report, got %s", stdout.String())
	}
}

func TestRunDispatchSkipLedger_AppendsAcrossMultipleTicks(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/candidates.json"
	if err := os.WriteFile(path, []byte(`[{"id":"1","updated_unix":100}]`), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	for i := 0; i < 2; i++ {
		if code := runDispatchSkipLedger(&stdout, &stderr, []string{"--in", path, "--workspace", dir, "--now", "1000"}); code != 0 {
			t.Fatalf("tick %d: want exit 0, got %d (stderr=%s)", i, code, stderr.String())
		}
	}
	ledger, err := os.ReadFile(dir + "/.dispatch-runs/skip-ledger.jsonl")
	if err != nil {
		t.Fatalf("read persisted ledger: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(ledger)), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 persisted rows across 2 ticks (append, not overwrite), got %d: %s", len(lines), ledger)
	}
}
