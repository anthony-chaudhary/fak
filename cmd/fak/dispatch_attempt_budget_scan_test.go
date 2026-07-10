package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestAttemptExhaustedIssues pins the poison-issue cap: an open issue with at
// least `budget` recorded worker attempts is held out of the wave, .out/.err
// splits of one attempt count once, and a non-positive budget disables the cap.
func TestAttemptExhaustedIssues(t *testing.T) {
	runs := t.TempDir()
	write := func(name string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(runs, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// issue 100: 3 distinct attempts (the poison case at budget 3).
	write("resolve-100-20260101-000001.log")
	write("resolve-100-20260101-000002.log")
	write("resolve-100-20260101-000003.log")
	// issue 200: a single attempt — under budget.
	write("resolve-200-20260101-000001.log")
	// issue 300: 2 distinct attempts, one split into .out.log/.err.log (counts once).
	write("resolve-300-fleet-20260101-000001.out.log")
	write("resolve-300-fleet-20260101-000001.err.log")
	write("resolve-300-fleet-20260101-000002.log")
	// noise that must not be counted as an attempt.
	write("dispatch-progress.jsonl")

	got := attemptExhaustedIssues(runs, 3)
	if !got[100] {
		t.Fatalf("issue 100 has 3 attempts >= budget 3, want held; got %v", got)
	}
	if got[200] || got[300] {
		t.Fatalf("issues 200(1) and 300(2 distinct) are under budget 3, want not held; got %v", got)
	}
	if len(got) != 1 {
		t.Fatalf("want only issue 100 held at budget 3, got %v", got)
	}

	// A lower budget catches issue 300's two collapsed attempts too.
	if g2 := attemptExhaustedIssues(runs, 2); !g2[100] || !g2[300] || g2[200] {
		t.Fatalf("budget 2 should hold 100 and 300 but not 200: %v", g2)
	}

	// A non-positive budget disables the cap entirely (legacy behavior).
	if g0 := attemptExhaustedIssues(runs, 0); len(g0) != 0 {
		t.Fatalf("budget 0 must disable the cap, got %v", g0)
	}
}

// TestDispatchAttemptBudget covers the default and the FAK_DISPATCH_ATTEMPT_BUDGET
// override, including the 0-disables and garbage-falls-back-to-default paths.
func TestDispatchAttemptBudget(t *testing.T) {
	t.Setenv("FAK_DISPATCH_ATTEMPT_BUDGET", "")
	if got := dispatchAttemptBudget(); got != dispatchAttemptBudgetDefault {
		t.Fatalf("unset env: got %d want default %d", got, dispatchAttemptBudgetDefault)
	}
	t.Setenv("FAK_DISPATCH_ATTEMPT_BUDGET", "3")
	if got := dispatchAttemptBudget(); got != 3 {
		t.Fatalf("env=3: got %d want 3", got)
	}
	t.Setenv("FAK_DISPATCH_ATTEMPT_BUDGET", "0")
	if got := dispatchAttemptBudget(); got != 0 {
		t.Fatalf("env=0 (disable): got %d want 0", got)
	}
	t.Setenv("FAK_DISPATCH_ATTEMPT_BUDGET", "garbage")
	if got := dispatchAttemptBudget(); got != dispatchAttemptBudgetDefault {
		t.Fatalf("env=garbage: got %d want default %d", got, dispatchAttemptBudgetDefault)
	}
}
