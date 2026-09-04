package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
)

func TestScoreboardDebtPageCLI(t *testing.T) {
	var stdout, stderr bytes.Buffer
	root := repoRoot()

	// Test --check-doc on repository root
	code := runScoreboardDebtPage(&stdout, &stderr, []string{"--workspace", root, "--check-doc"})
	if code != 0 {
		t.Fatalf("expected --check-doc exit code 0, got %d; stdout=%s, stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "OK") {
		t.Fatalf("expected OK output, got %s", stdout.String())
	}

	// Test --block / --markdown emits markdown
	stdout.Reset()
	stderr.Reset()
	code = runScoreboardDebtPage(&stdout, &stderr, []string{"--workspace", root, "--markdown"})
	if code != 0 {
		t.Fatalf("expected --markdown exit code 0, got %d; stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "# Scoreboard debt — unbounded portfolio summary") {
		t.Fatalf("expected H1 title in markdown output, got %s", out)
	}
	if !strings.Contains(out, "code-slop") || !strings.Contains(out, "slop_debt") {
		t.Fatalf("expected code-slop category in markdown output, got %s", out)
	}

	// Test category count exceeds 20
	if len(scorecardpane.DebtCategories) < 20 {
		t.Fatalf("expected at least 20 debt categories, got %d", len(scorecardpane.DebtCategories))
	}
}
