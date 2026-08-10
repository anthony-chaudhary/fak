package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestRunIssueUsesFakDevRoute(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"issue", "help"}); code != 0 {
		t.Fatalf("run(issue help) = %d, stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "fak-dev issue") {
		t.Fatalf("issue help does not advertise fak-dev route:\n%s", out.String())
	}
}

func TestRunProjectCompletionRejectsMissingInput(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"project", "completion"}); code != 2 {
		t.Fatalf("run(project completion) = %d, want 2; stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "fak-dev project completion") {
		t.Fatalf("project completion error does not advertise fak-dev route:\n%s", errOut.String())
	}
}
