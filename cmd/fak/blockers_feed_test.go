package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunBlockersFeedUnknownFailsClosed(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBlockersFeed(&stdout, &stderr, []string{
		"--source-status", "unknown",
		"--source-error", "configured label does not exist",
	})
	if code != 1 {
		t.Fatalf("unknown source exit = %d, want 1; stderr:\n%s", code, stderr.String())
	}
	got := stdout.String()
	for _, want := range []string{":warning:", "UNKNOWN", "No all-clear was evaluated"} {
		if !strings.Contains(got, want) {
			t.Fatalf("unknown source output missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, ":white_check_mark:") || strings.Contains(got, "all clear") {
		t.Fatalf("unknown source rendered a false all-clear:\n%s", got)
	}
	if !strings.Contains(stderr.String(), "refusing to post or report all-clear") {
		t.Fatalf("unknown source stderr lacks refusal:\n%s", stderr.String())
	}
}

func TestRunBlockersFeedSuccessfulEmptyQueryIsClear(t *testing.T) {
	dir := t.TempDir()
	issues := filepath.Join(dir, "issues.json")
	if err := os.WriteFile(issues, []byte("[]\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runBlockersFeed(&stdout, &stderr, []string{
		"--source-status", "ok",
		"--issues", issues,
		"--label", "blocked",
		"--dry-run",
	})
	if code != 0 {
		t.Fatalf("successful empty source exit = %d, want 0; stderr:\n%s", code, stderr.String())
	}
	got := stdout.String()
	if !strings.Contains(got, ":white_check_mark:") || !strings.Contains(got, "no standing blockers") {
		t.Fatalf("successful empty source did not render clear:\n%s", got)
	}
	if strings.Contains(got, "UNKNOWN") {
		t.Fatalf("successful empty source rendered unknown:\n%s", got)
	}
}

func TestRunBlockersFeedRejectsUnknownSourceStatus(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBlockersFeed(&stdout, &stderr, []string{"--source-status", "maybe"})
	if code != 2 {
		t.Fatalf("invalid source status exit = %d, want 2", code)
	}
	if stdout.Len() != 0 || !strings.Contains(stderr.String(), "want: ok | unknown") {
		t.Fatalf("invalid source status output = stdout %q stderr %q", stdout.String(), stderr.String())
	}
}
