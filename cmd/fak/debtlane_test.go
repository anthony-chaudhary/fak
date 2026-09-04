package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/debtlane"
)

func TestDebtLanesCLIJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{"--workspace", repoRoot(), "--json", "--top", "5"})
	// By default, report command must exit 0 so defaults "just work" in scripts/CLI
	if code != 0 {
		t.Fatalf("runDebtLanes default failed with exit code %d; stderr: %s", code, stderr.String())
	}

	var report debtlane.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to parse JSON stdout: %v; raw: %s", err, stdout.String())
	}
	if report.Schema != debtlane.Schema {
		t.Errorf("expected schema %q, got %q", debtlane.Schema, report.Schema)
	}
	if report.ProductionGrade.DenominatorPoints <= 0 {
		t.Errorf("expected positive production grade denominator points")
	}
	if report.ProductionGrade.TotalUnits == 0 {
		t.Errorf("expected total units > 0")
	}

	// Verify --check exits 1 when active debt exists
	var checkStdout, checkStderr bytes.Buffer
	checkCode := runDebtLanes(&checkStdout, &checkStderr, []string{"--workspace", repoRoot(), "--check"})
	if checkCode != 1 {
		t.Fatalf("runDebtLanes with --check should exit 1 on active debt, got %d", checkCode)
	}
}

func TestDebtLanesCLIFilterLane(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{"--workspace", repoRoot(), "--lane", "gateway", "--json"})
	if code == 2 {
		t.Fatalf("runDebtLanes failed with exit code 2; stderr: %s", stderr.String())
	}

	var report debtlane.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to parse JSON stdout: %v", err)
	}
	if len(report.Lanes) != 1 {
		t.Fatalf("expected 1 filtered lane, got %d", len(report.Lanes))
	}
	if report.Lanes[0].Lane != "gateway" {
		t.Errorf("expected lane 'gateway', got %q", report.Lanes[0].Lane)
	}
}

func TestMaturitySubcommandDebtLanes(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runMaturity(&stdout, &stderr, []string{"debt-lanes", "--workspace", repoRoot(), "--json"})
	if code == 2 {
		t.Fatalf("runMaturity debt-lanes failed with exit code 2; stderr: %s", stderr.String())
	}

	var report debtlane.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to parse JSON stdout: %v", err)
	}
	if report.Schema != debtlane.Schema {
		t.Errorf("expected schema %q, got %q", debtlane.Schema, report.Schema)
	}
}

func TestDebtLanesCLIMarkdown(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{"--workspace", repoRoot(), "--markdown", "--top", "3"})
	if code == 2 {
		t.Fatalf("runDebtLanes markdown failed with exit code 2; stderr: %s", stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "Maturity Debt Lanes Scorecard") {
		t.Errorf("expected title in markdown output: %s", out)
	}
	if !strings.Contains(out, "Production Grade & WIP Dilution") {
		t.Errorf("expected section in markdown output: %s", out)
	}
}
