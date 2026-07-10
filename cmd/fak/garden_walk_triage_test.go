package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestRunGardenWalkSelfcheck proves the CLI wiring: `fak garden walk selfcheck`
// dispatches the no-I/O decenter proof (gardenbundle.TriageSelfcheck) before any
// gh/config load or the FAK_GARDEN off-gate, exits 0, and prints the OK banner.
func TestRunGardenWalkSelfcheck(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runGardenWalk(&stdout, &stderr, []string{"selfcheck"})
	if rc != 0 {
		t.Fatalf("selfcheck rc=%d stderr=%q", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "triage selfcheck OK") {
		t.Fatalf("stdout=%q, want triage selfcheck OK banner", stdout.String())
	}
}

// TestRunGardenWalkSelfcheckRejectsArg keeps the subcommand strict about extra
// args, matching the other report selfchecks (blockers|cadence|fleetpane).
func TestRunGardenWalkSelfcheckRejectsArg(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if rc := runGardenWalk(&stdout, &stderr, []string{"selfcheck", "extra"}); rc != 2 {
		t.Fatalf("selfcheck with extra arg rc=%d, want 2 (stderr=%q)", rc, stderr.String())
	}
}
