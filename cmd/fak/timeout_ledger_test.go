package main

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestRunDispatchTimeoutLedger_RendersMultiplePhases(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/attempts.json"
	fixture := `[
		{"id":"101","started":false},
		{"id":"102","started":true,"last_stage":"edit"},
		{"id":"103","started":true,"last_stage":"push","failure_class":"push_rejected"}
	]`
	if err := os.WriteFile(path, []byte(fixture), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := runDispatchTimeoutLedger(&stdout, &stderr, []string{"--in", path, "--workspace", dir, "--now", "1000"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	out := stdout.String()
	// The witness: at least two distinct timeout phases must render.
	if !strings.Contains(out, "unknown") || !strings.Contains(out, "during_edit") || !strings.Contains(out, "during_push") {
		t.Fatalf("want unknown, during_edit, and during_push phases rendered, got %q", out)
	}
	if !strings.Contains(out, "push_rejected") {
		t.Fatalf("want failure_class rendered, got %q", out)
	}

	ledger, err := os.ReadFile(dir + "/.dispatch-runs/timeout-ledger.jsonl")
	if err != nil {
		t.Fatalf("read persisted ledger: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(ledger)), "\n")
	if len(lines) != 3 {
		t.Fatalf("want 3 persisted rows, got %d: %s", len(lines), ledger)
	}
}

func TestRunDispatchTimeoutLedger_JSONOutput(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/attempts.json"
	if err := os.WriteFile(path, []byte(`[{"id":"5","started":true,"last_stage":"commit"}]`), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := runDispatchTimeoutLedger(&stdout, &stderr, []string{"--in", path, "--workspace", dir, "--now", "1000", "--json"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"during_commit": 1`) {
		t.Fatalf("want during_commit=1 in the json report, got %s", stdout.String())
	}
}

func TestRunDispatchTimeoutLedger_AppendsAcrossMultipleTicks(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/attempts.json"
	if err := os.WriteFile(path, []byte(`[{"id":"1","started":true,"last_stage":"test"}]`), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	for i := 0; i < 2; i++ {
		if code := runDispatchTimeoutLedger(&stdout, &stderr, []string{"--in", path, "--workspace", dir, "--now", "1000"}); code != 0 {
			t.Fatalf("tick %d: want exit 0, got %d (stderr=%s)", i, code, stderr.String())
		}
	}
	ledger, err := os.ReadFile(dir + "/.dispatch-runs/timeout-ledger.jsonl")
	if err != nil {
		t.Fatalf("read persisted ledger: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(ledger)), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 persisted rows across 2 ticks (append, not overwrite), got %d: %s", len(lines), ledger)
	}
}

func TestRunDispatchTimeoutLedger_NoObservedStageDefaultsToUnknownOrBeforeStartup(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/attempts.json"
	// Neither started nor any stage marker observed -- must classify, never crash.
	if err := os.WriteFile(path, []byte(`[{"id":"9"}]`), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := runDispatchTimeoutLedger(&stdout, &stderr, []string{"--in", path, "--workspace", dir, "--now", "1000", "--json"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"unknown": 1`) {
		t.Fatalf("want unknown=1 for an attempt with no observed evidence, got %s", stdout.String())
	}
}
