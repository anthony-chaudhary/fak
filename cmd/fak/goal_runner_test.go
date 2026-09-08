package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGoalLaunchSubcommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runGoalLaunchSubcommand(&stdout, &stderr, []string{"--help"})
	if code != 2 {
		t.Fatalf("expected code 2 for --help, got %d", code)
	}
	if !strings.Contains(stderr.String(), "-pointer") {
		t.Fatalf("expected help text to contain -pointer, got:\n%s", stderr.String())
	}
}

func TestGoalFleetSubcommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runGoalFleetSubcommand(&stdout, &stderr, []string{"--help"})
	if code != 2 {
		t.Fatalf("expected code 2 for --help, got %d", code)
	}
	if !strings.Contains(stderr.String(), "-contracts-dir") {
		t.Fatalf("expected help text to contain -contracts-dir, got:\n%s", stderr.String())
	}
}

func TestGoalDispatchLaunchAndFleet(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runGoal(&stdout, &stderr, []string{"launch", "--plan-only", "--workspace", "../.."})
	if code != 0 {
		t.Fatalf("runGoal launch --plan-only failed (code %d):\nstderr: %s\nstdout: %s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), "PLAN_ONLY") {
		t.Fatalf("expected PLAN_ONLY in output, got: %s", stdout.String())
	}

	// Test fleet dry-run
	tmpDir := t.TempDir()
	contractsDir := filepath.Join(tmpDir, "contracts")
	if err := os.MkdirAll(contractsDir, 0755); err != nil {
		t.Fatal(err)
	}
	c1 := filepath.Join(contractsDir, "issue-1.md")
	if err := os.WriteFile(c1, []byte("/goal solve issue 1"), 0644); err != nil {
		t.Fatal(err)
	}

	stdout.Reset()
	stderr.Reset()
	code = runGoal(&stdout, &stderr, []string{"fleet", "--dry-run", "--workspace", tmpDir, "--contracts-dir", contractsDir})
	if code != 0 {
		t.Fatalf("runGoal fleet --dry-run failed (code %d):\nstderr: %s\nstdout: %s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), "completed 1 contracts") {
		t.Fatalf("expected 'completed 1 contracts' in output, got: %s", stdout.String())
	}
}
