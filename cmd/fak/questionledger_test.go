package main

import (
	"os"
	"strings"
	"testing"
)

// TestQuestionLedgerGHSeamIsDeadlineBound guards the #3509 regression. The
// `ensure-label --apply` seam shells out to `gh label create`, a GitHub network
// round-trip; built from a bare exec.Command it inherits no deadline, so a
// wedged gh hangs the invocation (and any caller scripting it) forever. The
// bound has to live at this call site, which is what ghexec.CommandTimeout
// gives it — a source guard, because the behavioural proof that the deadline
// reaches the process belongs to internal/ghexec's own tests.
func TestQuestionLedgerGHSeamIsDeadlineBound(t *testing.T) {
	src, err := os.ReadFile("questionledger.go")
	if err != nil {
		t.Fatalf("read questionledger.go: %v", err)
	}
	text := string(src)
	if strings.Contains(text, `exec.Command("gh"`) {
		t.Error(`questionledger.go builds a bare exec.Command("gh", ...): the ensure-label round-trip is unbounded again (#3509) — build it with ghexec.CommandTimeout`)
	}
	if !strings.Contains(text, "ghexec.CommandTimeout(") {
		t.Error("questionledger.go no longer routes its gh call through ghexec.CommandTimeout: the ensure-label network call must carry a deadline (#3509)")
	}
}
