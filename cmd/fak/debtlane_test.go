package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
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

func setupDebtLaneWaveWorkspace(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()

	lanes := []string{"mocka", "mockb", "mockc", "mockd", "mocke", "mockf", "mockg", "mockh"}
	var dosContent strings.Builder
	dosContent.WriteString("workspace = \".\"\n\n[lanes]\nconcurrent = [\n")
	for _, l := range lanes {
		dosContent.WriteString("  \"" + l + "\",\n")
	}
	dosContent.WriteString("]\n\n[lanes.trees]\n")
	for _, l := range lanes {
		dosContent.WriteString("\"" + l + "\" = [\"internal/" + l + "/**\"]\n")
	}

	if err := os.WriteFile(filepath.Join(tmp, "dos.toml"), []byte(dosContent.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, l := range lanes {
		laneDir := filepath.Join(tmp, "internal", l)
		if err := os.MkdirAll(laneDir, 0o755); err != nil {
			t.Fatal(err)
		}
		code := "package " + l + "\n\nfunc Work() string {\n\treturn \"wip\"\n}\n"
		if err := os.WriteFile(filepath.Join(laneDir, l+".go"), []byte(code), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	return tmp
}

func TestDebtLanesCLIPlanWavesJSON(t *testing.T) {
	tmp := setupDebtLaneWaveWorkspace(t)
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", tmp,
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
	tmp := setupDebtLaneWaveWorkspace(t)
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", tmp,
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
	tmp := setupDebtLaneWaveWorkspace(t)
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", tmp,
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
	tmp := setupDebtLaneWaveWorkspace(t)
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", tmp,
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
	// If starting grade is already >= 80%, 0 waves are needed; otherwise waves are planned.
	if plan.StartingPercent < 80.0 {
		if plan.TotalWaves <= 0 || plan.TotalWaves > 50 {
			t.Errorf("expected 1-50 planned waves for target grade 80%%, got %d", plan.TotalWaves)
		}
	} else {
		if plan.TotalWaves != 0 {
			t.Errorf("expected 0 planned waves when already >= 80%%, got %d", plan.TotalWaves)
		}
	}
	for _, w := range plan.Waves {
		if w.WaveSize > 4 {
			t.Errorf("wave size %d exceeds wave size cap 4", w.WaveSize)
		}
	}
}

func TestDebtLanesCLIPlanWavesTargetPointsAlias(t *testing.T) {
	tmp := setupDebtLaneWaveWorkspace(t)
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", tmp,
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

func TestDebtOrchestratorCLI(t *testing.T) {
	tmp := setupDebtLaneWaveWorkspace(t)
	var stdout, stderr bytes.Buffer
	code := runDebtOrchestrator(&stdout, &stderr, []string{
		"--workspace", tmp,
		"--wave-size", "5",
		"--max-waves", "2",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runDebtOrchestrator failed with exit code %d; stderr: %s", code, stderr.String())
	}

	var plan debtlane.WavePlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("failed to decode json wave plan: %v", err)
	}
	if plan.Schema != debtlane.WavePlanSchema {
		t.Errorf("expected schema %q, got %q", debtlane.WavePlanSchema, plan.Schema)
	}
	if plan.TotalWaves != 2 {
		t.Errorf("expected 2 waves with --max-waves 2, got %d", plan.TotalWaves)
	}
	if plan.WaveSizeCap != 5 {
		t.Errorf("expected wave size cap 5, got %d", plan.WaveSizeCap)
	}
}

func TestDebtOrchestratorCLIDualRepo(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDebtOrchestrator(&stdout, &stderr, []string{
		"--workspace", repoRoot(),
		"--target-repo", "both",
		"--wave-size", "6",
		"--max-waves", "3",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runDebtOrchestrator --target-repo both failed: code %d, stderr: %s", code, stderr.String())
	}

	var plan debtlane.WavePlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("failed to decode json wave plan: %v", err)
	}
	if plan.TargetRepo != "both" {
		t.Errorf("expected TargetRepo 'both', got %q", plan.TargetRepo)
	}
	if len(plan.Waves) == 0 {
		t.Fatalf("expected waves planned in dual-repo mode")
	}

	// Verify repo tags exist on lanes in the plan
	foundFak := false
	for _, w := range plan.Waves {
		for _, l := range w.Lanes {
			if l.Repo == "fak" {
				foundFak = true
			}
		}
	}
	if !foundFak {
		t.Errorf("expected to find at least one lane with repo 'fak'")
	}
}

func TestDebtLanesCLITargetRepoPrivate(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", repoRoot(),
		"--target-repo", "fak-private",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runDebtLanes --target-repo fak-private failed: code %d, stderr: %s", code, stderr.String())
	}

	var report debtlane.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}
	if report.TargetRepo != "fak-private" {
		t.Errorf("expected TargetRepo 'fak-private', got %q", report.TargetRepo)
	}
	if report.ProductionGrade.TotalUnits == 0 {
		t.Errorf("expected total units > 0 in fak-private scan")
	}
	for _, l := range report.Lanes {
		if l.Repo != "fak-private" {
			t.Errorf("expected lane %s repo to be 'fak-private', got %q", l.Lane, l.Repo)
			break
		}
	}
}

func TestDebtLanesCLIQuery(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", repoRoot(),
		"--query", "gateway",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runDebtLanes --query gateway failed: code %d, stderr: %s", code, stderr.String())
	}

	var report debtlane.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}
	if len(report.Lanes) == 0 {
		t.Fatalf("expected at least 1 lane matching query 'gateway', got 0")
	}
	for _, l := range report.Lanes {
		if !strings.Contains(strings.ToLower(l.Lane), "gateway") &&
			!strings.Contains(strings.ToLower(l.UnitOfWork), "gateway") &&
			!strings.Contains(strings.ToLower(l.Related.CompanionLane), "gateway") {
			t.Errorf("lane %q does not match query 'gateway'", l.Lane)
		}
	}
}

func TestDebtLanesCLIHealth(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", repoRoot(),
		"--health", "healthy,degraded",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runDebtLanes --health failed: code %d, stderr: %s", code, stderr.String())
	}

	var report debtlane.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}
	for _, l := range report.Lanes {
		if l.Health.Status != debtlane.HealthHealthy && l.Health.Status != debtlane.HealthDegraded {
			t.Errorf("lane %s health status %q not in [healthy, degraded]", l.Lane, l.Health.Status)
		}
	}
}

func TestDebtLanesCLICrossIndex(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", repoRoot(),
		"--cross-index",
		"--top", "5",
	})
	if code != 0 {
		t.Fatalf("runDebtLanes --cross-index failed: code %d, stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "FAK DEBT CROSS-INDEX & COMPANIONS") {
		t.Errorf("expected cross-index header, got:\n%s", out)
	}
	if !strings.Contains(out, "COMPANION (REPO:UNIT)") {
		t.Errorf("expected COMPANION column header, got:\n%s", out)
	}
}

func TestDebtOrchestratorCLIOpencodeCommands(t *testing.T) {
	tmp := setupDebtLaneWaveWorkspace(t)
	var stdout, stderr bytes.Buffer
	code := runDebtOrchestrator(&stdout, &stderr, []string{
		"--workspace", tmp,
		"--wave-size", "2",
		"--max-waves", "1",
		"--opencode-commands",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runDebtOrchestrator --opencode-commands failed: code %d, stderr: %s", code, stderr.String())
	}

	var plan debtlane.WavePlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("failed to parse JSON wave plan: %v; raw: %s", err, stdout.String())
	}
	if len(plan.Waves) == 0 {
		t.Fatalf("expected planned waves, got 0")
	}
	if len(plan.Waves[0].OpencodeChats) == 0 {
		t.Errorf("expected OpencodeChats to be populated on Wave 0")
	}
	if len(plan.OpencodeCommands) == 0 {
		t.Errorf("expected OpencodeCommands summary on plan")
	}

	// Test text rendering includes OpenCode commands
	var textOut, textErr bytes.Buffer
	textCode := runDebtOrchestrator(&textOut, &textErr, []string{
		"--workspace", tmp,
		"--wave-size", "2",
		"--max-waves", "1",
		"--opencode-commands",
	})
	if textCode != 0 {
		t.Fatalf("runDebtOrchestrator text failed: code %d, stderr: %s", textCode, textErr.String())
	}
	if !strings.Contains(textOut.String(), "OpenCode Chat Commands:") {
		t.Errorf("expected text output to include OpenCode Chat Commands:\n%s", textOut.String())
	}
}

func TestDebtOrchestratorCLIPerfFocus(t *testing.T) {
	tmp := setupDebtLaneWaveWorkspace(t)
	var stdout, stderr bytes.Buffer
	code := runDebtOrchestrator(&stdout, &stderr, []string{
		"--workspace", tmp,
		"--wave-size", "2",
		"--max-waves", "1",
		"--perf-focus",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runDebtOrchestrator --perf-focus failed: code %d, stderr: %s", code, stderr.String())
	}

	var plan debtlane.WavePlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}
	if plan.Schema != debtlane.WavePlanSchema {
		t.Errorf("expected schema %s, got %s", debtlane.WavePlanSchema, plan.Schema)
	}
}

func TestDebtLanesCLICoverageReceipt(t *testing.T) {
	tmp := setupDebtLaneWaveWorkspace(t)
	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", tmp,
		"--coverage",
	})
	if code != 0 {
		t.Fatalf("runDebtLanes --coverage failed: code %d, stderr: %s", code, stderr.String())
	}

	var receipt debtlane.CoverageReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to parse JSON coverage receipt: %v; raw: %s", err, stdout.String())
	}
	if receipt.Schema != debtlane.CoverageReceiptSchema {
		t.Errorf("expected schema %s, got %s", debtlane.CoverageReceiptSchema, receipt.Schema)
	}
	if receipt.TargetDepth != 15 {
		t.Errorf("expected target depth 15, got %d", receipt.TargetDepth)
	}
	if len(receipt.DeclaredDimensions) != 15 {
		t.Errorf("expected 15 declared dimensions in receipt, got %d", len(receipt.DeclaredDimensions))
	}
}

func TestDebtLanesCLIExpandedSurfaces(t *testing.T) {
	tmp := setupDebtLaneWaveWorkspace(t)
	// Add a cmd surface
	cmdDir := filepath.Join(tmp, "cmd", "fakecmd")
	if err := os.MkdirAll(cmdDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cmdDir, "main.go"), []byte("package main\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", tmp,
		"--json",
	})
	if code != 0 {
		t.Fatalf("runDebtLanes default failed: code %d, stderr: %s", code, stderr.String())
	}

	var report debtlane.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}
	foundCmd := false
	for _, l := range report.Lanes {
		if l.Lane == "cmd_fakecmd" {
			foundCmd = true
			break
		}
	}
	if !foundCmd {
		t.Errorf("expected cmd_fakecmd to be discovered by default under expanded 9-surface scanning")
	}

	// Test opt-out with --no-expanded-surfaces
	stdout.Reset()
	stderr.Reset()
	codeOptOut := runDebtLanes(&stdout, &stderr, []string{
		"--workspace", tmp,
		"--no-expanded-surfaces",
		"--json",
	})
	if codeOptOut != 0 {
		t.Fatalf("runDebtLanes --no-expanded-surfaces failed: code %d, stderr: %s", codeOptOut, stderr.String())
	}
	var reportOptOut debtlane.Report
	if err := json.Unmarshal(stdout.Bytes(), &reportOptOut); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}
	foundCmdOptOut := false
	for _, l := range reportOptOut.Lanes {
		if l.Lane == "cmd_fakecmd" {
			foundCmdOptOut = true
			break
		}
	}
	if foundCmdOptOut {
		t.Errorf("expected cmd_fakecmd to be excluded when --no-expanded-surfaces is passed")
	}
}
