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

// TestRunDispatchAttemptBudget_ReportShowsDistinctBackoffWindowsByClass is the
// CLI-level half of the #1778 witness: the rendered candidate report -- not
// just the internal/attemptbudget package -- must show different backoff
// windows for different failure classes, since the issue's Done condition is
// "the policy is documented AND reflected in the candidate report."
func TestRunDispatchAttemptBudget_ReportShowsDistinctBackoffWindowsByClass(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/issues.json"
	fixture := `[
		{"issue_id":"auth-1","budget":10,"attempts":[{"failure_class":"auth_error","at_unix":1000}]},
		{"issue_id":"merge-1","budget":10,"attempts":[{"failure_class":"merge_conflict","at_unix":1000}]},
		{"issue_id":"test-1","budget":10,"attempts":[{"failure_class":"test_failure","at_unix":1000}]},
		{"issue_id":"scope-1","budget":10,"attempts":[{"failure_class":"ambiguous_scope","at_unix":1000}]}
	]`
	if err := os.WriteFile(path, []byte(fixture), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := runDispatchAttemptBudget(&stdout, &stderr, []string{"--in", path, "--now", "1000"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	out := stdout.String()

	// Each failure class's window (4h, 30m, 10m, 2h respectively) must appear,
	// distinctly, in the rendered report.
	wantWindows := []string{"4h0m0s", "30m0s", "10m0s", "2h0m0s"}
	for _, w := range wantWindows {
		if !strings.Contains(out, w) {
			t.Fatalf("want backoff window %q rendered in the report, got:\n%s", w, out)
		}
	}
	for _, class := range []string{"auth", "merge", "test", "ambiguous_scope"} {
		if !strings.Contains(out, class) {
			t.Fatalf("want backoff_class %q rendered in the report, got:\n%s", class, out)
		}
	}
	// All four issues are still under budget (10) and their windows have not
	// elapsed as of --now=1000 (== at_unix), so all four are cooling down, not
	// dispatchable or held.
	if !strings.Contains(out, "0 dispatchable") || !strings.Contains(out, "4 cooling down") || !strings.Contains(out, "0 held") {
		t.Fatalf("want 0 dispatchable, 4 cooling down, 0 held, got %q", out)
	}
}

// TestRunDispatchAttemptBudget_CooldownElapsesBackToDispatchable proves the
// CLI's --now flag drives the same class-specific cooldown math as the pure
// package: once enough wall-clock time has passed, a cooling-down issue
// becomes dispatchable again without crossing the attempt budget.
func TestRunDispatchAttemptBudget_CooldownElapsesBackToDispatchable(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/issues.json"
	// test_failure's default window is 10 minutes (600s).
	fixture := `[{"issue_id":"1","budget":10,"attempts":[{"failure_class":"test_failure","at_unix":1000}]}]`
	if err := os.WriteFile(path, []byte(fixture), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runDispatchAttemptBudget(&stdout, &stderr, []string{"--in", path, "--now", "1500"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "1 cooling down") {
		t.Fatalf("before the 600s window elapses: want cooling down, got %q", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = runDispatchAttemptBudget(&stdout, &stderr, []string{"--in", path, "--now", "1600"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "1 dispatchable") {
		t.Fatalf("after the 600s window elapses: want dispatchable, got %q", stdout.String())
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
