package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestRunFleetPaneSelfcheck proves the CLI wiring: `fleetpane selfcheck` dispatches
// the no-I/O worker-health decenter proof before any config/runner load, exits 0,
// and prints the OK banner.
func TestRunFleetPaneSelfcheck(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runFleetPane(&stdout, &stderr, []string{"selfcheck"})
	if rc != 0 {
		t.Fatalf("selfcheck rc=%d stderr=%q", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "SELFCHECK OK") {
		t.Fatalf("stdout=%q, want SELFCHECK OK banner", stdout.String())
	}
}

// TestRunFleetPaneSelfcheckRejectsArg keeps the subcommand strict about extra args,
// matching the other report selfchecks.
func TestRunFleetPaneSelfcheckRejectsArg(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if rc := runFleetPane(&stdout, &stderr, []string{"selfcheck", "extra"}); rc != 2 {
		t.Fatalf("selfcheck with extra arg rc=%d, want 2 (stderr=%q)", rc, stderr.String())
	}
}
