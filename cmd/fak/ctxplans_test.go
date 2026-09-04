package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplans"
)

func TestRunCtxPlansHelp(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCtxPlans(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("runCtxPlans --help returned %d, want 0", code)
	}
	if !strings.Contains(stderr.String(), "usage: fak ctxplans") {
		t.Fatalf("runCtxPlans --help missing usage in stderr:\n%s", stderr.String())
	}
}

func TestRunCtxPlansUnexpectedArg(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCtxPlans(&stdout, &stderr, []string{"unexpected"})
	if code != 2 {
		t.Fatalf("runCtxPlans with unexpected arg returned %d, want 2", code)
	}
}

func TestRunCtxPlansFixture(t *testing.T) {
	var stdout, stderr bytes.Buffer
	fixture := filepath.Join(repoRoot(), "internal", "ctxplans", "testdata", "repo")
	code := runCtxPlans(&stdout, &stderr, []string{"--json", "--root", fixture})
	if code != 0 {
		t.Fatalf("runCtxPlans --json returned %d, stderr:\n%s", code, stderr.String())
	}
	var rep ctxplans.Report
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v\noutput:\n%s", err, stdout.String())
	}
	if rep.DeclaredVerbs != 3 {
		t.Errorf("rep.DeclaredVerbs = %d, want 3", rep.DeclaredVerbs)
	}
	if rep.Debt != 3 {
		t.Errorf("rep.Debt = %d, want 3", rep.Debt)
	}
}
