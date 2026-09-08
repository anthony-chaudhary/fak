package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuesolved"
)

func TestIssuesSolved_Help(t *testing.T) {
	for _, flag := range []string{"--help", "-h"} {
		var stdout, stderr bytes.Buffer
		rc := runIssuesSolved(&stdout, &stderr, []string{flag})
		if rc != 0 {
			t.Errorf("expected exit code 0 for %s, got %d (stderr: %s)", flag, rc, stderr.String())
		}
	}
}

func TestIssuesSolved_InvalidFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runIssuesSolved(&stdout, &stderr, []string{"--not-a-valid-flag"})
	if rc != 2 {
		t.Errorf("expected exit code 2 for invalid flag, got %d", rc)
	}
}

func TestIssuesSolved_UnexpectedArg(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runIssuesSolved(&stdout, &stderr, []string{"extra-arg"})
	if rc != 2 {
		t.Errorf("expected exit code 2 for unexpected arg, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "unexpected argument") {
		t.Errorf("expected stderr to mention unexpected argument, got: %s", stderr.String())
	}
}

func TestIssuesSolved_InvalidHours(t *testing.T) {
	for _, val := range []string{"-5", "0"} {
		var stdout, stderr bytes.Buffer
		rc := runIssuesSolved(&stdout, &stderr, []string{"--hours", val})
		if rc != 2 {
			t.Errorf("expected exit code 2 for --hours %s, got %d", val, rc)
		}
	}
}

func TestIssuesSolved_ConflictingHoursAndSince(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runIssuesSolved(&stdout, &stderr, []string{"--hours", "5", "--since", "2h"})
	if rc != 2 {
		t.Errorf("expected exit code 2 for conflicting --hours and --since, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "cannot specify both") {
		t.Errorf("expected stderr to mention cannot specify both, got: %s", stderr.String())
	}
}

func TestIssuesSolved_FutureSince(t *testing.T) {
	future := time.Now().Add(24 * time.Hour).UTC().Format(time.RFC3339)
	var stdout, stderr bytes.Buffer
	rc := runIssuesSolved(&stdout, &stderr, []string{"--since", future})
	if rc != 2 {
		t.Errorf("expected exit code 2 for future --since, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "future") {
		t.Errorf("expected stderr to mention future, got: %s", stderr.String())
	}
}

func TestIssuesSolved_InvalidSource(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runIssuesSolved(&stdout, &stderr, []string{"--source", "unsupported"})
	if rc != 2 {
		t.Errorf("expected exit code 2 for invalid source, got %d", rc)
	}
}

func TestIssuesSolved_InvalidSince(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runIssuesSolved(&stdout, &stderr, []string{"--since", "not-a-valid-date-or-duration"})
	if rc != 2 {
		t.Errorf("expected exit code 2 for invalid since, got %d", rc)
	}
}

func TestIssuesSolved_Hours1(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runIssuesSolved(&stdout, &stderr, []string{"--no-private", "--source", "git", "--hours", "1"})
	if rc != 0 {
		t.Fatalf("expected exit code 0, got %d (stderr: %s)", rc, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Issues solved in last 1.0h") {
		t.Errorf("expected output to contain 'Issues solved in last 1.0h', got: %s", out)
	}
}

func TestIssuesSolved_JSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runIssuesSolved(&stdout, &stderr, []string{"--no-private", "--source", "git", "--json", "--hours", "1"})
	if rc != 0 {
		t.Fatalf("expected exit code 0, got %d (stderr: %s)", rc, stderr.String())
	}

	var rep issuesolved.Report
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to parse JSON report: %v\nOutput: %s", err, stdout.String())
	}
	if rep.Schema != issuesolved.Schema {
		t.Errorf("expected schema %q, got %q", issuesolved.Schema, rep.Schema)
	}
	if len(rep.Repos) != 1 {
		t.Errorf("expected 1 repo with --no-private, got %d", len(rep.Repos))
	}
}

func TestIssuesSolved_DetailedAndList(t *testing.T) {
	for _, flag := range []string{"--detailed", "--list"} {
		var stdout, stderr bytes.Buffer
		rc := runIssuesSolved(&stdout, &stderr, []string{"--no-private", "--source", "git", flag, "--hours", "1"})
		if rc != 0 {
			t.Fatalf("expected exit code 0 for %s, got %d (stderr: %s)", flag, rc, stderr.String())
		}
		if !strings.Contains(stdout.String(), "Issues solved in last 1.0h") {
			t.Errorf("expected summary in detailed output for %s", flag)
		}
	}
}

func TestIssuesSolved_Since(t *testing.T) {
	// Duration format
	var stdout1, stderr1 bytes.Buffer
	rc1 := runIssuesSolved(&stdout1, &stderr1, []string{"--no-private", "--source", "git", "--since", "2h"})
	if rc1 != 0 {
		t.Fatalf("expected exit code 0 for --since 2h, got %d (stderr: %s)", rc1, stderr1.String())
	}

	// RFC3339 format
	sinceRFC := time.Now().Add(-2 * time.Hour).UTC().Format(time.RFC3339)
	var stdout2, stderr2 bytes.Buffer
	rc2 := runIssuesSolved(&stdout2, &stderr2, []string{"--no-private", "--source", "git", "--since", sinceRFC})
	if rc2 != 0 {
		t.Fatalf("expected exit code 0 for RFC3339 since, got %d (stderr: %s)", rc2, stderr2.String())
	}
}

func TestIssuesSolved_RuntimeError(t *testing.T) {
	orig := collectIssuesSolvedFunc
	defer func() { collectIssuesSolvedFunc = orig }()

	collectIssuesSolvedFunc = func(ctx context.Context, opts issuesolved.Options) (*issuesolved.Report, error) {
		return nil, errors.New("simulated collect failure")
	}

	var stdout, stderr bytes.Buffer
	rc := runIssuesSolved(&stdout, &stderr, []string{"--no-private"})
	if rc != 1 {
		t.Errorf("expected exit code 1 on runtime error, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "simulated collect failure") {
		t.Errorf("expected stderr to contain error message, got: %s", stderr.String())
	}
}

func TestIssuesSolved_ParseSinceHelper(t *testing.T) {
	now := time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)

	// Empty string
	tEmpty, err := parseSince("", now)
	if err != nil || !tEmpty.IsZero() {
		t.Errorf("expected zero time for empty since, got %v (err: %v)", tEmpty, err)
	}

	// Duration 3h
	tDur, err := parseSince("3h", now)
	if err != nil {
		t.Fatalf("parseSince(3h) failed: %v", err)
	}
	if want := now.Add(-3 * time.Hour); !tDur.Equal(want) {
		t.Errorf("parseSince(3h) = %v, want %v", tDur, want)
	}

	// RFC3339
	rfc := "2026-09-08T09:00:00Z"
	tRFC, err := parseSince(rfc, now)
	if err != nil {
		t.Fatalf("parseSince(%s) failed: %v", rfc, err)
	}
	if want := time.Date(2026, 9, 8, 9, 0, 0, 0, time.UTC); !tRFC.Equal(want) {
		t.Errorf("parseSince(%s) = %v, want %v", rfc, tRFC, want)
	}

	// RFC3339Nano
	rfcNano := "2026-09-08T09:00:00.123456789Z"
	tRFCNano, err := parseSince(rfcNano, now)
	if err != nil {
		t.Fatalf("parseSince(%s) failed: %v", rfcNano, err)
	}
	if want := time.Date(2026, 9, 8, 9, 0, 0, 123456789, time.UTC); !tRFCNano.Equal(want) {
		t.Errorf("parseSince(%s) = %v, want %v", rfcNano, tRFCNano, want)
	}

	// Invalid string
	_, err = parseSince("not-a-date", now)
	if err == nil {
		t.Error("expected error for invalid since, got nil")
	}
}
