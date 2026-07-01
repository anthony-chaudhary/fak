package main

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestRunDispatchAttemptBudget_MovesRepeatedAttemptsToHeld(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/issues.json"
	fixture := `[
		{"issue_id":"1","budget":3,"attempts":[{"failure_class":"test_failure","at_unix":100}]},
		{"issue_id":"2","budget":3,"attempts":[
			{"failure_class":"test_failure","at_unix":100},
			{"failure_class":"test_failure","at_unix":200},
			{"failure_class":"timeout","at_unix":300}
		]}
	]`
	if err := os.WriteFile(path, []byte(fixture), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := runDispatchAttemptBudget(&stdout, &stderr, []string{"--in", path})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "1 dispatchable") || !strings.Contains(out, "1 held") {
		t.Fatalf("want 1 dispatchable, 1 held, got %q", out)
	}
	if !strings.Contains(out, "timeout") {
		t.Fatalf("want the held issue's last failure class rendered, got %q", out)
	}
}

func TestRunDispatchAttemptBudget_DefaultBudgetFlagAppliesWhenUnset(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/issues.json"
	fixture := `[{"issue_id":"9","attempts":[
		{"failure_class":"a","at_unix":1},
		{"failure_class":"b","at_unix":2}
	]}]`
	if err := os.WriteFile(path, []byte(fixture), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := runDispatchAttemptBudget(&stdout, &stderr, []string{"--in", path, "--budget", "2"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "1 held") {
		t.Fatalf("want the --budget default to hold issue 9, got %q", stdout.String())
	}
}

func TestRunDispatchAttemptBudget_JSONOutput(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/issues.json"
	if err := os.WriteFile(path, []byte(`[{"issue_id":"1","budget":5,"attempts":[]}]`), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := runDispatchAttemptBudget(&stdout, &stderr, []string{"--in", path, "--json"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"dispatchable_count": 1`) {
		t.Fatalf("want dispatchable_count=1 in the json report, got %s", stdout.String())
	}
}
