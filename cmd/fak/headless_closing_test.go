package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestRunHeadlessLintClosing exercises the run-level closing-shape fold behind
// `fak headless-lint --closing` end to end: a final summary whose last block is a
// trailing prose wall (over the 40-word threshold, no leading bullet) exits 1
// (CLOSING_PROSE_WALL); the operator escape (--override), a bulleted final block, and a
// short single-line closer each exit 0 (clean). It is the CLI sibling of
// TestRunHeadlessLintLeftovers.
func TestRunHeadlessLintClosing(t *testing.T) {
	// A trailing paragraph well over closingProseWords (40) with no bullets — the pathology.
	const proseWall = "I looked into the failing test and it turns out the root cause was a " +
		"race condition in the scheduler where two goroutines could both observe the queue as " +
		"empty and then both attempt to advance the cursor which corrupts the offset and produces " +
		"the flaky behavior we saw so I changed the locking to hold the mutex across the whole " +
		"compare-and-advance and reran it fifty times and it passed every time now."

	// The scannable shape: the LAST block leads with a bullet, verdict first, next step last.
	const scannable = "Fixed the flaky scheduler test.\n\n" +
		"- Root cause: race on the queue cursor (two goroutines advanced it).\n" +
		"- Fix: hold the mutex across compare-and-advance.\n" +
		"- Next: re-run the full suite before shipping."

	run := func(argv ...string) (int, string) {
		var out, errb bytes.Buffer
		code := runHeadlessLint(&out, &errb, strings.NewReader(""), argv)
		return code, out.String() + errb.String()
	}

	// Arm 1 — trailing prose wall -> refused (exit 1).
	if code, s := run("--closing", proseWall); code != 1 {
		t.Fatalf("arm1: want exit 1 (prose wall refused), got %d\noutput: %s", code, s)
	}

	// Operator escape — intentional prose closer -> clean (exit 0).
	if code, s := run("--closing", "--override", proseWall); code != 0 {
		t.Fatalf("override: want exit 0 (clean), got %d\noutput: %s", code, s)
	}

	// A bulleted final block is the scannable shape -> clean (exit 0).
	if code, s := run("--closing", scannable); code != 0 {
		t.Fatalf("scannable: want exit 0 (clean), got %d\noutput: %s", code, s)
	}

	// A short single-line closer (under the 40-word threshold) -> clean.
	if code, s := run("--closing", "Done — fix shipped, tests pass, committed abc1234."); code != 0 {
		t.Fatalf("short closer: want exit 0 (clean), got %d\noutput: %s", code, s)
	}

	// JSON mode still refuses arm 1 and emits the versioned schema tag + verdict.
	code, s := run("--closing", "--json", proseWall)
	if code != 1 {
		t.Fatalf("json arm1: want exit 1, got %d", code)
	}
	if !strings.Contains(s, "fak-closing-fold/1") || !strings.Contains(s, "closing_prose_wall") {
		t.Errorf("json arm1: expected schema + verdict in output, got: %s", s)
	}
}
