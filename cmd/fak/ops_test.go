package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestOpsCLIStatusAndSweep(t *testing.T) {
	var out, errb bytes.Buffer

	// status
	rc := runOps(&out, &errb, []string{"status", "--json"})
	if rc != 0 {
		t.Fatalf("runOps status failed with %d: %s", rc, errb.String())
	}
	if !strings.Contains(out.String(), `"status":"healthy"`) {
		t.Errorf("expected healthy status in JSON: %s", out.String())
	}

	out.Reset()
	errb.Reset()

	// sweep dry run
	rc = runOps(&out, &errb, []string{"sweep", "--dry-run", "--json"})
	if rc != 0 {
		t.Fatalf("runOps sweep failed with %d: %s", rc, errb.String())
	}
	if !strings.Contains(out.String(), `"dry_run":true`) {
		t.Errorf("expected dry_run:true in JSON: %s", out.String())
	}
}
