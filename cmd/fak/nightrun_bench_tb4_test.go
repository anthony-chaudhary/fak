package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/benchcatalog"
	"github.com/anthony-chaudhary/fak/internal/nightrun"
	"github.com/anthony-chaudhary/fak/internal/tb4bench"
)

func init() {
	// Ensure TB4 in-process runner is wired for nightrun regression helper
	nightrun.RegisterTB4Runner(func(ctx context.Context, datasetPath string, outDir string) (*nightrun.TB4RegressionResult, error) {
		suite, err := tb4bench.LoadManifestFile(datasetPath)
		if err != nil {
			return nil, err
		}

		var taskIDs []string
		for _, t := range suite.Tasks {
			taskIDs = append(taskIDs, t.TaskID)
		}
		contract := tb4bench.DefaultRunContract("qwen3.8-coder-7b-q4_k_m.gguf", "sha256:pinned", "Q4_K_M", taskIDs)

		runCfg := tb4bench.RunCampaignConfig{
			Tasks:       suite.Tasks,
			Arm:         "both",
			OutDir:      outDir,
			MockMode:    true,
			Determinism: contract.Determinism,
			Contract:    contract,
		}

		if _, err := tb4bench.RunCampaign(ctx, runCfg); err != nil {
			return nil, err
		}

		evalCfg := tb4bench.EvaluateCampaignConfig{
			RunDir:  outDir,
			Dataset: datasetPath,
			Tasks:   suite.Tasks,
		}
		if _, err := tb4bench.EvaluateCampaign(ctx, evalCfg); err != nil {
			return nil, err
		}

		compCfg := tb4bench.CompareCampaignConfig{
			FakDir:      filepath.Join(outDir, "fak"),
			OpenCodeDir: filepath.Join(outDir, "opencode"),
			Dataset:     datasetPath,
			Tasks:       suite.Tasks,
			OutJSON:     filepath.Join(outDir, "compare.json"),
			OutMD:       filepath.Join(outDir, "compare.md"),
		}
		report, err := tb4bench.CompareCampaign(ctx, compCfg)
		if err != nil {
			return nil, err
		}

		return &nightrun.TB4RegressionResult{
			ArmASolveRate:    report.ArmAMetrics.Official.SolveRate,
			ArmBSolveRate:    report.ArmBMetrics.Official.SolveRate,
			SolveRateDelta:   report.SolveRateDelta,
			ArmAPromptTokens: report.ArmAMetrics.Telemetry.TotalPromptTokens,
			ArmBPromptTokens: report.ArmBMetrics.Telemetry.TotalPromptTokens,
			VDSOHits:         report.ArmAMetrics.Telemetry.VDSOHits,
			MarkdownReport:   report.GenerateMarkdown(),
		}, nil
	})
}

// TestNightrunBenchTB4CatalogRegistration verifies that tb4bench is registered in benchcatalog
// and discoverable through the fak benchmarks CLI.
func TestNightrunBenchTB4CatalogRegistration(t *testing.T) {
	b, ok := benchcatalog.Get("tb4bench")
	if !ok {
		t.Fatal("tb4bench benchmark missing from benchcatalog registry")
	}

	if b.Name != "tb4bench" {
		t.Errorf("Name = %q, want 'tb4bench'", b.Name)
	}
	if b.Kind != benchcatalog.KindVerb {
		t.Errorf("Kind = %s, want %s (KindVerb)", b.Kind, benchcatalog.KindVerb)
	}
	if b.Need != benchcatalog.NeedNone {
		t.Errorf("Need = %s, want %s (NeedNone)", b.Need, benchcatalog.NeedNone)
	}
	if b.Level != benchcatalog.LevelE2E {
		t.Errorf("Level = %s, want %s (LevelE2E)", b.Level, benchcatalog.LevelE2E)
	}
	if !strings.Contains(b.Summary, "Terminal-Bench 4") {
		t.Errorf("Summary = %q, want mention of 'Terminal-Bench 4'", b.Summary)
	}
	if !strings.Contains(b.Run, "fak bench tb4 run") {
		t.Errorf("Run = %q, want 'fak bench tb4 run' command", b.Run)
	}
	if b.Doc != "BENCHMARK-AUTHORITY.md" {
		t.Errorf("Doc = %q, want 'BENCHMARK-AUTHORITY.md'", b.Doc)
	}
	for _, requiredFlag := range []string{"--arm", "--mock", "--dataset"} {
		found := false
		for _, f := range b.Flags {
			if strings.Contains(f, requiredFlag) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Flags = %v, missing flag %s", b.Flags, requiredFlag)
		}
	}
	if !b.Offline() {
		t.Error("tb4bench mock mode must be classified as offline (NeedNone)")
	}

	// Verify alias lookup
	if bAlias, ok := benchcatalog.Get("bench-tb4"); !ok || bAlias.Name != "tb4bench" {
		t.Errorf("benchcatalog.Get('bench-tb4') alias failed, got ok=%t, name=%s", ok, bAlias.Name)
	}

	// Verify CLI listing
	var stdout, stderr bytes.Buffer
	rc := runBenchmarks(&stdout, &stderr, []string{"list", "--offline"})
	if rc != 0 {
		t.Fatalf("fak benchmarks list failed with code %d: %s", rc, stderr.String())
	}
	listOut := stdout.String()
	if !strings.Contains(listOut, "tb4bench") {
		t.Errorf("fak benchmarks list output missing 'tb4bench':\n%s", listOut)
	}

	// Verify CLI describe
	stdout.Reset()
	stderr.Reset()
	rc = runBenchmarks(&stdout, &stderr, []string{"describe", "tb4bench"})
	if rc != 0 {
		t.Fatalf("fak benchmarks describe tb4bench failed with code %d: %s", rc, stderr.String())
	}
	descOut := stdout.String()
	if !strings.Contains(descOut, "Terminal-Bench 4") {
		t.Errorf("fak benchmarks describe output missing 'Terminal-Bench 4':\n%s", descOut)
	}
	if !strings.Contains(descOut, "BENCHMARK-AUTHORITY.md") {
		t.Errorf("fak benchmarks describe output missing doc reference:\n%s", descOut)
	}
}

// TestNightrunBenchTB4TaskResolution verifies that the nightly collector backlog
// resolves the TB4 benchmark into a feasible collection task.
func TestNightrunBenchTB4TaskResolution(t *testing.T) {
	tasks, err := nightrun.Backlog("")
	if err != nil {
		t.Fatalf("nightrun.Backlog() failed: %v", err)
	}

	task, ok := nightrun.FindTB4Task(tasks)
	if !ok {
		t.Fatal("nightrun.Backlog() missing TB4 task")
	}

	if task.ID != nightrun.TB4TaskID {
		t.Errorf("task ID = %q, want %q", task.ID, nightrun.TB4TaskID)
	}
	if task.Source != nightrun.SourceBenchmark {
		t.Errorf("task Source = %q, want %q", task.Source, nightrun.SourceBenchmark)
	}
	if task.Value != nightrun.ValueCoverage {
		t.Errorf("task Value = %q, want %q (LevelE2E maps to ValueCoverage)", task.Value, nightrun.ValueCoverage)
	}
	if !task.AutoRunnable() {
		t.Error("TB4 task must be AutoRunnable()")
	}
	if !nightrun.IsTB4Task(task) {
		t.Error("nightrun.IsTB4Task must return true for resolved task")
	}
	if len(task.Requires) > 0 {
		t.Errorf("task.Requires = %v, want empty (offline)", task.Requires)
	}

	// Verify nightrun plan CLI surfaces TB4
	var stdout, stderr bytes.Buffer
	rc := runNightrun(&stdout, &stderr, []string{"plan", "--json"})
	if rc != 0 {
		t.Fatalf("fak nightrun plan failed with code %d: %s", rc, stderr.String())
	}

	var plan nightrun.PlanReport
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("failed to decode nightrun plan JSON: %v", err)
	}

	foundInPlan := false
	for _, s := range plan.Ranked {
		if s.Task.ID == nightrun.TB4TaskID {
			foundInPlan = true
			if !s.Feasible {
				t.Errorf("TB4 task should be feasible on offline box, got blocked reason: %v", s.Reason)
			}
			break
		}
	}
	if !foundInPlan {
		t.Errorf("nightrun plan missing task %s", nightrun.TB4TaskID)
	}
}

// TestNightrunBenchTB4RegressionSmoke verifies that RunTB4NightrunRegression executes
// and verifies the synthetic regression suite.
func TestNightrunBenchTB4RegressionSmoke(t *testing.T) {
	tempDir := t.TempDir()
	ctx := context.Background()

	res, err := nightrun.RunTB4NightrunRegression(ctx, "", tempDir)
	if err != nil {
		t.Fatalf("RunTB4NightrunRegression failed: %v", err)
	}
	if res == nil {
		t.Fatal("expected non-nil TB4RegressionResult")
	}

	if res.ArmASolveRate < 1.0 {
		t.Errorf("arm A solve rate = %.2f, want >= 1.0", res.ArmASolveRate)
	}
	if res.TokenReduction < 0.80 {
		t.Errorf("token reduction = %.2f, want >= 0.80", res.TokenReduction)
	}
	if !res.ThresholdsMet {
		t.Errorf("expected ThresholdsMet = true, error = %s", res.Error)
	}
	if res.Outcome != "collected" {
		t.Errorf("expected outcome = 'collected', got %q", res.Outcome)
	}
	if res.LogPath == "" {
		t.Fatal("expected non-empty LogPath")
	}

	t.Cleanup(func() {
		_ = os.Remove(res.LogPath)
	})

	logData, err := os.ReadFile(res.LogPath)
	if err != nil {
		t.Fatalf("failed to read emitted nightrun log %s: %v", res.LogPath, err)
	}
	logContent := string(logData)
	if !strings.Contains(logContent, `outcome="collected"`) {
		t.Errorf("log missing outcome=\"collected\":\n%s", logContent)
	}
	if !strings.Contains(logContent, "task_id: "+nightrun.TB4TaskID) {
		t.Errorf("log missing task_id:\n%s", logContent)
	}
}
