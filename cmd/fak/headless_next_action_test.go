package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestRunHeadlessLintNextAction exercises the run-level stop-shape fold behind
// `fak headless-lint --next-action`: a final summary that stops in terminal-block
// grammar with no next action exits 1 (refused); the same obstacle once a next
// action is named exits 0 (clean). It pins the doctrine that there is no blocked
// state — every obstacle resolves to a disposition.
func TestRunHeadlessLintNextAction(t *testing.T) {
	run := func(argv ...string) (int, string) {
		var out, errb bytes.Buffer
		code := runHeadlessLint(&out, &errb, strings.NewReader(""), argv)
		return code, out.String() + errb.String()
	}

	// Arm 1 — terminal-block grammar, no next action -> refused (exit 1).
	if code, s := run("--next-action", "The build is blocked and we cannot proceed."); code != 1 {
		t.Fatalf("arm1: want exit 1 (refused), got %d\noutput: %s", code, s)
	}

	// Arm 2 — same obstacle with the next action named -> clean (exit 0).
	if code, s := run("--next-action", "Blocked by the lease; next action: route around to item 4."); code != 0 {
		t.Fatalf("arm2: want exit 0 (clean) with a named next action, got %d\noutput: %s", code, s)
	}

	// Arm 3 — a cited ticket is a reachable next action -> clean.
	if code, s := run("--next-action", "Cannot proceed; tracked in #1234."); code != 0 {
		t.Fatalf("arm3: want exit 0 (clean) with a cited ticket, got %d\noutput: %s", code, s)
	}

	// Arm 4 — no block grammar at all -> clean.
	if code, _ := run("--next-action", "Implemented the parser, tests pass, pushed."); code != 0 {
		t.Fatalf("arm4: want exit 0, got %d", code)
	}

	// JSON mode still refuses arm 1 and emits the schema tag + verdict.
	code, s := run("--next-action", "--json", "Stuck on the migration.")
	if code != 1 {
		t.Fatalf("json arm1: want exit 1, got %d", code)
	}
	if !strings.Contains(s, "fak-next-action-fold/1") || !strings.Contains(s, "terminal_block") {
		t.Errorf("json arm1: expected schema + verdict in output, got: %s", s)
	}
}
