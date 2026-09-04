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

func TestDebtLanesCLIPlanWavesJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", repoRoot(),
		"--plan-waves",
		"--wave-size", "3",
		"--max-waves", "2",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runDebtLanes --plan-waves failed with exit code %d; stderr: %s", code, stderr.String())
	}

	var plan debtlane.WavePlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("failed to parse JSON wave plan: %v; raw: %s", err, stdout.String())
	}
	if plan.Schema != debtlane.WavePlanSchema {
		t.Errorf("expected schema %q, got %q", debtlane.WavePlanSchema, plan.Schema)
	}
	if plan.WaveSizeCap != 3 {
		t.Errorf("expected wave size cap 3, got %d", plan.WaveSizeCap)
	}
	if plan.TotalWaves > 2 {
		t.Errorf("expected at most 2 waves, got %d", plan.TotalWaves)
	}
	if len(plan.Waves) == 0 {
		t.Errorf("expected at least 1 planned wave, got 0")
	}
	for _, w := range plan.Waves {
		if w.WaveSize > 3 {
			t.Errorf("wave size %d exceeds cap 3", w.WaveSize)
		}
		if len(w.Lanes) != w.WaveSize {
			t.Errorf("mismatch between lane count %d and wave size %d", len(w.Lanes), w.WaveSize)
		}
	}
}

func TestDebtLanesCLIPlanWavesText(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", repoRoot(),
		"--plan-waves",
		"--wave-size", "4",
		"--max-waves", "2",
	})
	if code != 0 {
		t.Fatalf("runDebtLanes text wave plan failed with exit code %d; stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "CONCURRENT SAFE WAVE PLAN") {
		t.Errorf("expected wave plan header in text output: %s", out)
	}
	if !strings.Contains(out, "WAVE-1") {
		t.Errorf("expected WAVE-1 in text output: %s", out)
	}
}

func TestDebtLanesCLIPlanWavesMarkdown(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", repoRoot(),
		"--plan-waves",
		"--wave-size", "4",
		"--max-waves", "2",
		"--markdown",
	})
	if code != 0 {
		t.Fatalf("runDebtLanes markdown wave plan failed with exit code %d; stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Concurrent Safe Debt Retirement Wave Plan") {
		t.Errorf("expected markdown header: %s", out)
	}
	if !strings.Contains(out, "Wave-1") {
		t.Errorf("expected Wave-1 in markdown output: %s", out)
	}
}

func TestDebtLanesCLIPlanWavesTargetGrade(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", repoRoot(),
		"--plan-waves",
		"--target-grade", "80%",
		"--wave-size", "4",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runDebtLanes --target-grade failed with exit code %d; stderr: %s", code, stderr.String())
	}

	var plan debtlane.WavePlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("failed to decode json wave plan: %v", err)
	}
	if plan.TargetGrade != "80%" {
		t.Errorf("expected target grade '80%%', got %q", plan.TargetGrade)
	}
	if plan.ProjectedPercent < 80.0 {
		t.Errorf("expected projected percent >= 80.0, got %.1f", plan.ProjectedPercent)
	}
	if plan.ProjectedGrade != "B" && plan.ProjectedGrade != "A" {
		t.Errorf("expected projected grade B or A, got %s", plan.ProjectedGrade)
	}
	// Total waves should be bounded, not the entire backlog.
	if plan.TotalWaves <= 0 || plan.TotalWaves > 50 {
		t.Errorf("expected 1-50 planned waves for target grade 80%%, got %d", plan.TotalWaves)
	}
	for _, w := range plan.Waves {
		if w.WaveSize > 4 {
			t.Errorf("wave size %d exceeds wave size cap 4", w.WaveSize)
		}
	}
}

func TestDebtLanesCLIPlanWavesTargetPointsAlias(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", repoRoot(),
		"--plan-waves",
		"--points", "50",
		"--wave-size", "3",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runDebtLanes --points failed with exit code %d; stderr: %s", code, stderr.String())
	}

	var plan debtlane.WavePlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("failed to decode json wave plan: %v", err)
	}
	if plan.TotalDebtInPlan < 50.0 && plan.PotentialPoints < 50.0 {
		t.Errorf("expected total debt in plan or potential points >= 50.0, got debt=%.1f pot=%.1f", plan.TotalDebtInPlan, plan.PotentialPoints)
	}
	for _, w := range plan.Waves {
		if w.WaveSize > 3 {
			t.Errorf("wave size %d exceeds wave size cap 3", w.WaveSize)
		}
	}
}
