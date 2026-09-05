package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestRedteamCLI(t *testing.T) {
	// 1. Dry run
	var stdout, stderr bytes.Buffer
	code := runRedTeam([]string{"--dry-run"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("runRedTeam --dry-run failed with exit %d: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "FAK ADVERSARIAL RED-TEAM ARENA (DRY RUN)") {
		t.Errorf("expected dry run header, got: %s", stdout.String())
	}

	// 2. Dry run with JSON
	stdout.Reset()
	stderr.Reset()
	code = runRedTeam([]string{"--dry-run", "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("runRedTeam --dry-run --json failed with exit %d: %s", code, stderr.String())
	}
	var dryRunData map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &dryRunData); err != nil {
		t.Fatalf("failed parsing dry-run JSON: %v", err)
	}
	if dryRunData["dry_run"] != true {
		t.Errorf("expected dry_run=true, got: %v", dryRunData["dry_run"])
	}

	// 3. Full battery execution with JSON
	stdout.Reset()
	stderr.Reset()
	code = runRedTeam([]string{"--suite", "battery", "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("runRedTeam --suite battery --json failed with exit %d: %s", code, stderr.String())
	}
	var summary redTeamBatterySummary
	if err := json.Unmarshal(stdout.Bytes(), &summary); err != nil {
		t.Fatalf("failed parsing battery summary JSON: %v\nOutput: %s", err, stdout.String())
	}
	if !summary.Passed {
		t.Errorf("expected summary.Passed to be true, got false")
	}
	if summary.TotalAttacks != 3 {
		t.Errorf("expected 3 total attacks, got %d", summary.TotalAttacks)
	}
	if summary.ContainedAttacks != 3 {
		t.Errorf("expected 3 contained attacks, got %d", summary.ContainedAttacks)
	}
	if summary.CanariesTripped < 1 {
		t.Errorf("expected at least 1 canary tripped, got %d", summary.CanariesTripped)
	}
	if summary.EgressBlocked < 1 {
		t.Errorf("expected at least 1 egress blocked, got %d", summary.EgressBlocked)
	}
	if summary.ResidualFilesOnHost != 0 {
		t.Errorf("expected 0 residual files on host, got %d", summary.ResidualFilesOnHost)
	}

	// 4. Unknown suite error handling
	stdout.Reset()
	stderr.Reset()
	code = runRedTeam([]string{"--suite", "unknown-suite"}, &stdout, &stderr)
	if code != 2 {
		t.Errorf("expected exit code 2 for unknown suite, got %d", code)
	}
}
