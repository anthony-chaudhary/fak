package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/factorymigrate"
)

func TestRunFactoryMigrate_Help(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunFactoryMigrate(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("expected code 0 for --help, got %d", code)
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev factory-migrate") {
		t.Errorf("expected usage help message, got: %s", stdout.String())
	}
}

func TestRunFactoryMigrate_Status(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunFactoryMigrate(&stdout, &stderr, []string{"status"})
	if code != 0 {
		t.Fatalf("expected code 0 for status, got %d. stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Autonomous Dev Factory Migration Status") {
		t.Errorf("expected status header, got: %s", out)
	}
	if !strings.Contains(out, "Cohort Breakdown:") {
		t.Errorf("expected cohort breakdown section, got: %s", out)
	}
	if !strings.Contains(out, "Total: 100") {
		t.Errorf("expected Total: 100, got: %s", out)
	}

	// Test default when no arguments given
	stdout.Reset()
	stderr.Reset()
	code = RunFactoryMigrate(&stdout, &stderr, nil)
	if code != 0 {
		t.Fatalf("expected code 0 for default status, got %d", code)
	}
	if !strings.Contains(stdout.String(), "Autonomous Dev Factory Migration Status") {
		t.Errorf("expected status header on default, got: %s", stdout.String())
	}
}

func TestRunFactoryMigrate_Status_JSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunFactoryMigrate(&stdout, &stderr, []string{"status", "--json"})
	if code != 0 {
		t.Fatalf("expected code 0 for status --json, got %d. stderr: %s", code, stderr.String())
	}

	var report factorymigrate.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to parse JSON status: %v", err)
	}
	if report.Total != 100 {
		t.Errorf("report.Total = %d, want 100", report.Total)
	}
	if len(report.Cohorts) == 0 {
		t.Errorf("expected cohorts in report JSON, got 0")
	}
}

func TestRunFactoryMigrate_List(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunFactoryMigrate(&stdout, &stderr, []string{"list"})
	if code != 0 {
		t.Fatalf("expected code 0 for list, got %d. stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "NUM") || !strings.Contains(out, "TARGET PATH") {
		t.Errorf("expected list header, got: %s", out)
	}

	// Test filtered list by cohort
	stdout.Reset()
	stderr.Reset()
	code = RunFactoryMigrate(&stdout, &stderr, []string{"list", "--cohort", "watchdogs"})
	if code != 0 {
		t.Fatalf("expected code 0 for list --cohort watchdogs, got %d", code)
	}
	lines := strings.Split(strings.TrimSpace(stdout.String()), "\n")
	// 1 header line + 20 watchdogs items = 21 lines
	if len(lines) != 21 {
		t.Errorf("expected 21 lines (header + 20 items), got %d lines", len(lines))
	}

	// Test list --json
	stdout.Reset()
	stderr.Reset()
	code = RunFactoryMigrate(&stdout, &stderr, []string{"list", "--cohort", "memsync", "--json"})
	if code != 0 {
		t.Fatalf("expected code 0 for list --json, got %d", code)
	}
	var items []factorymigrate.Item
	if err := json.Unmarshal(stdout.Bytes(), &items); err != nil {
		t.Fatalf("failed to unmarshal list JSON: %v", err)
	}
	if len(items) != 2 {
		t.Errorf("expected 2 memsync items, got %d", len(items))
	}
}

func TestRunFactoryMigrate_Next(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunFactoryMigrate(&stdout, &stderr, []string{"next", "--count", "5"})
	if code != 0 {
		t.Fatalf("expected code 0 for next, got %d. stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Next 5 Migration Candidates") {
		t.Errorf("expected 'Next 5 Migration Candidates', got: %s", out)
	}

	// Test next with JSON
	stdout.Reset()
	stderr.Reset()
	code = RunFactoryMigrate(&stdout, &stderr, []string{"next", "--count", "3", "--json"})
	if code != 0 {
		t.Fatalf("expected code 0 for next --json, got %d", code)
	}
	var candidates []factorymigrate.Item
	if err := json.Unmarshal(stdout.Bytes(), &candidates); err != nil {
		t.Fatalf("failed to unmarshal next candidates: %v", err)
	}
	if len(candidates) != 3 {
		t.Errorf("expected 3 candidates, got %d", len(candidates))
	}
}

func TestRunFactoryMigrate_AuditBoundary(t *testing.T) {
	tmpDir := t.TempDir()
	privRoot := filepath.Join(tmpDir, "fak-private")
	platformDir := filepath.Join(privRoot, "platform", "worker")
	if err := os.MkdirAll(platformDir, 0755); err != nil {
		t.Fatal(err)
	}

	// Clean tree
	cleanCode := `package worker

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/pkg/abi"
)

var _ = fmt.Println
var _ = abi.Version
`
	if err := os.WriteFile(filepath.Join(platformDir, "worker.go"), []byte(cleanCode), 0644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := RunFactoryMigrate(&stdout, &stderr, []string{"audit-boundary", "--private-root", privRoot})
	if code != 0 {
		t.Fatalf("expected code 0 on clean tree, got %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Boundary audit passed") {
		t.Errorf("expected 'Boundary audit passed', got: %s", stdout.String())
	}

	// Dirty tree: introduce internal import
	badCode := `package worker

import (
	"github.com/anthony-chaudhary/fak/internal/engine"
)

var _ = engine.Run
`
	if err := os.WriteFile(filepath.Join(platformDir, "bad.go"), []byte(badCode), 0644); err != nil {
		t.Fatal(err)
	}

	stdout.Reset()
	stderr.Reset()
	code = RunFactoryMigrate(&stdout, &stderr, []string{"audit-boundary", "--private-root", privRoot})
	if code != 1 {
		t.Fatalf("expected code 1 on boundary violation, got %d", code)
	}
	if !strings.Contains(stdout.String(), "Found 1 boundary violation(s)") {
		t.Errorf("expected violation count, got: %s", stdout.String())
	}

	// JSON format on violation
	stdout.Reset()
	stderr.Reset()
	code = RunFactoryMigrate(&stdout, &stderr, []string{"audit-boundary", "--private-root", privRoot, "--json"})
	if code != 1 {
		t.Fatalf("expected code 1 on violation with --json, got %d", code)
	}
	var violations []factorymigrate.BoundaryViolation
	if err := json.Unmarshal(stdout.Bytes(), &violations); err != nil {
		t.Fatalf("failed to unmarshal violations JSON: %v", err)
	}
	if len(violations) != 1 {
		t.Errorf("expected 1 violation, got %d", len(violations))
	}
	if violations[0].Rule != "NO_INTERNAL_IMPORT" {
		t.Errorf("violation rule = %q, want NO_INTERNAL_IMPORT", violations[0].Rule)
	}
}

func TestRunFactoryMigrate_Scaffold(t *testing.T) {
	tmpDir := t.TempDir()
	privRoot := filepath.Join(tmpDir, "fak-private")

	// Missing item number
	var stdout, stderr bytes.Buffer
	code := RunFactoryMigrate(&stdout, &stderr, []string{"scaffold", "--private-root", privRoot})
	if code != 2 {
		t.Fatalf("expected code 2 for missing scaffold arg, got %d", code)
	}

	// Invalid item number
	stdout.Reset()
	stderr.Reset()
	code = RunFactoryMigrate(&stdout, &stderr, []string{"scaffold", "999", "--private-root", privRoot})
	if code != 2 {
		t.Fatalf("expected code 2 for out of range item number, got %d", code)
	}

	// Dry run scaffold item 1
	stdout.Reset()
	stderr.Reset()
	code = RunFactoryMigrate(&stdout, &stderr, []string{"scaffold", "1", "--private-root", privRoot, "--dry-run"})
	if code != 0 {
		t.Fatalf("expected code 0 for dry-run scaffold, got %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "[dry-run] Would scaffold item 1") {
		t.Errorf("expected dry run message, got: %s", stdout.String())
	}
	targetFile := filepath.Join(privRoot, "platform", "watchdogs", "crash_audit.go")
	if _, err := os.Stat(targetFile); !os.IsNotExist(err) {
		t.Errorf("target file should not exist after dry run")
	}

	// Real scaffold item 1
	stdout.Reset()
	stderr.Reset()
	code = RunFactoryMigrate(&stdout, &stderr, []string{"scaffold", "1", "--private-root", privRoot})
	if code != 0 {
		t.Fatalf("expected code 0 for real scaffold, got %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Scaffolded item 1") {
		t.Errorf("expected scaffold success message, got: %s", stdout.String())
	}
	if _, err := os.Stat(targetFile); err != nil {
		t.Errorf("target file was not created on disk: %v", err)
	}

	// Duplicate scaffold returns error
	stdout.Reset()
	stderr.Reset()
	code = RunFactoryMigrate(&stdout, &stderr, []string{"scaffold", "1", "--private-root", privRoot})
	if code != 1 {
		t.Errorf("expected code 1 when scaffolding over existing file, got %d", code)
	}
}

func TestRunFactoryMigrate_UnknownSubcommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunFactoryMigrate(&stdout, &stderr, []string{"foobar_unknown"})
	if code != 2 {
		t.Fatalf("expected code 2 for unknown subcommand, got %d", code)
	}
	if !strings.Contains(stderr.String(), "unknown subcommand") {
		t.Errorf("expected unknown subcommand error, got: %s", stderr.String())
	}
}
