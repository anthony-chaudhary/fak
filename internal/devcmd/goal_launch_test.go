package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/goalrunner"
)

func TestGoalLaunchHelpAndInvalid(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// 1. --help should return exit code 0
	code := RunGoalLaunch(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("expected 0 for --help, got %d", code)
	}
	if !strings.Contains(stderr.String(), "usage: fak-dev goal-launch") {
		t.Fatalf("expected usage message, got %q", stderr.String())
	}

	// 2. Empty args should return exit code 2
	stdout.Reset()
	stderr.Reset()
	code = RunGoalLaunch(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("expected 2 for empty args, got %d", code)
	}
}

func TestGoalLaunchPlanOnly(t *testing.T) {
	tmpDir := t.TempDir()
	ptrFile := filepath.Join(tmpDir, "issue-42.md")
	if err := os.WriteFile(ptrFile, []byte("/goal solve issue 42"), 0644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := RunGoalLaunch(&stdout, &stderr, []string{
		"--workspace", tmpDir,
		"--pointer", ptrFile,
		"--plan-only",
		"--tag", "test-worker",
		"--json",
	})

	if code != 0 {
		t.Fatalf("expected exit code 0 for plan-only, got %d; stderr=%s", code, stderr.String())
	}

	var res goalrunner.LaunchResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("failed to decode JSON output: %v\nstdout=%s", err, stdout.String())
	}

	if !res.PlanOnly {
		t.Errorf("expected PlanOnly to be true")
	}
	if res.Tag != "test-worker" {
		t.Errorf("expected tag test-worker, got %s", res.Tag)
	}
	if !strings.Contains(res.LaunchWitness, "PLAN_ONLY") {
		t.Errorf("expected launch witness to mention PLAN_ONLY, got %s", res.LaunchWitness)
	}
}

func TestGoalLaunchFleetHelpAndInvalid(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// 1. --help should return exit code 0
	code := RunGoalFleet(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("expected 0 for --help, got %d", code)
	}
	if !strings.Contains(stderr.String(), "usage: fak-dev goal-fleet") {
		t.Fatalf("expected usage message, got %q", stderr.String())
	}

	// 2. Empty args should return exit code 2
	stdout.Reset()
	stderr.Reset()
	code = RunGoalFleet(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("expected 2 for empty args, got %d", code)
	}
}

func TestGoalLaunchFleetDryRun(t *testing.T) {
	tmpDir := t.TempDir()
	contractsDir := filepath.Join(tmpDir, "contracts")
	if err := os.MkdirAll(contractsDir, 0755); err != nil {
		t.Fatal(err)
	}

	contractsJSON := filepath.Join(contractsDir, "contracts.json")
	contractsData := `[
		{"n": 10, "pointer": "issue-10.md", "host": "full"},
		{"n": 20, "pointer": "issue-20.md", "host": "blocked"}
	]`
	if err := os.WriteFile(contractsJSON, []byte(contractsData), 0644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := RunGoalFleet(&stdout, &stderr, []string{
		"--workspace", tmpDir,
		"--contracts-dir", contractsDir,
		"--dry-run",
		"--json",
	})

	if code != 0 {
		t.Fatalf("expected exit code 0 for dry-run fleet, got %d; stderr=%s", code, stderr.String())
	}

	var results []*goalrunner.WitnessResult
	if err := json.Unmarshal(stdout.Bytes(), &results); err != nil {
		t.Fatalf("failed to decode JSON output: %v\nstdout=%s", err, stdout.String())
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	for _, r := range results {
		if r.Outcome != "dry-run (skipped launch)" {
			t.Errorf("expected dry-run outcome, got %s", r.Outcome)
		}
	}
}
