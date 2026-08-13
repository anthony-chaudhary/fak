package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestContinuitySelfcheckCapturedJourney(t *testing.T) {
	var out, errout bytes.Buffer
	if code := runContinuitySelfcheck(&out, &errout, false); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	for _, want := range []string{"PASS personal continuity", "3 real objects", "2 isolated homes", "no service", "behavior skill=review-concisely workflow=triage-before-fix policy=deny-destructive", "receipts export=", "rollback restored prior inactive context"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q in:\n%s", want, out.String())
		}
	}
}
func TestContinuityHelpIsTaskOrientedAndDryRunExplicit(t *testing.T) {
	var out bytes.Buffer
	continuityHelp(&out)
	for _, want := range []string{"move a safe managed context between homes", "mutations preview unless --commit", "preview", "export", "apply", "switch", "status", "rollback", "--json", "--select"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q", want)
		}
	}
}
