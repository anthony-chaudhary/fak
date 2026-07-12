package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestRunHeadlessLintLeftovers exercises the run-level end-of-run fold behind
// `fak headless-lint --leftovers` end to end (#3670): a final summary that narrates
// deferred work with zero issues filed exits 1 (refused); the same summary once the
// follow-ups are filed (--issues-filed) or an operator escape (--override) is given
// exits 0 (clean).
func TestRunHeadlessLintLeftovers(t *testing.T) {
	const summary = "Shipped the fix, tests pass, committed abc1234.\n" +
		"There are two more things worth doing: backoff and a docs pass."

	run := func(argv ...string) (int, string) {
		var out, errb bytes.Buffer
		code := runHeadlessLint(&out, &errb, strings.NewReader(""), argv)
		return code, out.String() + errb.String()
	}

	// Arm 1 — narrated leftovers, zero issues filed -> refused (exit 1).
	if code, s := run("--leftovers", summary); code != 1 {
		t.Fatalf("arm1: want exit 1 (refused), got %d\noutput: %s", code, s)
	}

	// Arm 2 — same summary, follow-ups filed as gh issues -> clean (exit 0).
	if code, s := run("--leftovers", "--issues-filed", "2", summary); code != 0 {
		t.Fatalf("arm2: want exit 0 (clean) with 2 issues filed, got %d\noutput: %s", code, s)
	}

	// Operator escape — "genuinely nothing left" -> clean (exit 0).
	if code, s := run("--leftovers", "--override", summary); code != 0 {
		t.Fatalf("override: want exit 0 (clean), got %d\noutput: %s", code, s)
	}

	// A completed-work summary carries no leftover narration -> clean.
	if code, _ := run("--leftovers", "Implemented the parser, tests pass, pushed."); code != 0 {
		t.Fatalf("clean summary: want exit 0, got %d", code)
	}

	// JSON mode still refuses arm 1 and emits the schema tag.
	code, s := run("--leftovers", "--json", summary)
	if code != 1 {
		t.Fatalf("json arm1: want exit 1, got %d", code)
	}
	if !strings.Contains(s, "fak-leftovers-fold/1") || !strings.Contains(s, "leftovers_unfiled") {
		t.Errorf("json arm1: expected schema + verdict in output, got: %s", s)
	}
}
