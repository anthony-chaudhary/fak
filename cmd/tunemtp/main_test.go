package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/mtptune"
)

func TestRunDefault(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"-k-max", "2", "-p-step", "0.5"}
	code := run(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d. stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "PARETO OPTIMAL FRONTIER") {
		t.Fatalf("expected stdout to contain table header, got:\n%s", out)
	}
	if !strings.Contains(out, "OPTIMAL PROFILE OVERALL") {
		t.Fatalf("expected stdout to contain optimal profile summary, got:\n%s", out)
	}
}

func TestRunJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"-json", "-k-max", "2", "-p-step", "0.5"}
	code := run(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d. stderr: %s", code, stderr.String())
	}

	var report mtptune.SweepReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to parse JSON output: %v\nOutput: %s", err, stdout.String())
	}
	if report.OptimalProfile.K < 1 {
		t.Fatalf("invalid report parsed, K = %d", report.OptimalProfile.K)
	}
}

func TestRunInvalidArgs(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"-k-min", "10", "-k-max", "2"}
	code := run(&stdout, &stderr, args)
	if code != 1 {
		t.Fatalf("expected exit code 1 for invalid config, got %d", code)
	}
	if !strings.Contains(stderr.String(), "Error running MTP tuning sweep") {
		t.Fatalf("expected error in stderr, got: %s", stderr.String())
	}
}

func TestRunBadFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"-invalid-unknown-flag"}
	code := run(&stdout, &stderr, args)
	if code != 2 {
		t.Fatalf("expected exit code 2 for flag parse error, got %d", code)
	}
}
