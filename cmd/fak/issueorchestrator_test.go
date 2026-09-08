package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/debtlane"
	"github.com/anthony-chaudhary/fak/internal/issueorchestrator"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func writeTestIssuesFile(t *testing.T, issues []issueorchestrator.Issue) string {
	t.Helper()
	b, err := json.Marshal(issues)
	if err != nil {
		t.Fatalf("marshal issues: %v", err)
	}
	path := filepath.Join(t.TempDir(), "issues.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write issues: %v", err)
	}
	return path
}

func TestIssueOrchestratorCLIPlanWavesJSON(t *testing.T) {
	issues := []issueorchestrator.Issue{
		{
			Number:          101,
			Key:             "issue-101",
			Title:           "Add gateway route cache",
			Lane:            "gateway",
			Paths:           []string{"internal/gateway/route.go"},
			ExpectedSteps:   3,
			Dispatchability: "dispatchable",
		},
		{
			Number:          102,
			Key:             "issue-102",
			Title:           "Optimize model attention kernel",
			Lane:            "model",
			Paths:           []string{"internal/model/attn.go"},
			ExpectedSteps:   4,
			Dispatchability: "dispatchable",
		},
		{
			Number:          103,
			Key:             "issue-103",
			Title:           "Fix compute buffer leak",
			Lane:            "compute",
			Paths:           []string{"internal/compute/buffer.go"},
			ExpectedSteps:   2,
			Dispatchability: "dispatchable",
		},
	}

	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--wave-size", "2",
		"--adaptive-concurrency=false",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runIssueOrchestrator failed with exit code %d; stderr: %s", code, stderr.String())
	}

	var plan issueorchestrator.Plan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("failed to decode JSON plan: %v; raw: %s", err, stdout.String())
	}

	if plan.Schema != issueorchestrator.WavePlanSchema {
		t.Errorf("expected schema %q, got %q", issueorchestrator.WavePlanSchema, plan.Schema)
	}
	if plan.TotalIssues != 3 {
		t.Errorf("expected 3 total issues, got %d", plan.TotalIssues)
	}
	if plan.WaveSizeCap != 2 {
		t.Errorf("expected wave size cap 2, got %d", plan.WaveSizeCap)
	}
	if len(plan.Waves) == 0 {
		t.Fatalf("expected at least 1 planned wave, got 0")
	}
}

func TestIssueOrchestratorCLIMarkdown(t *testing.T) {
	issues := []issueorchestrator.Issue{
		{
			Number:          201,
			Key:             "issue-201",
			Title:           "Add token optimizer",
			Lane:            "token",
			Paths:           []string{"internal/token/opt.go"},
			ExpectedSteps:   3,
			Dispatchability: "dispatchable",
		},
	}

	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--markdown",
	})
	if code != 0 {
		t.Fatalf("runIssueOrchestrator failed with exit code %d; stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "Issue Orchestrator: Concurrent Safe Wave Plan") {
		t.Errorf("markdown output missing title: %s", out)
	}
	if !strings.Contains(out, "#201") {
		t.Errorf("markdown output missing issue #201: %s", out)
	}
}

func TestIssueOrchestratorCLICompare(t *testing.T) {
	baselinePlan := issueorchestrator.Plan{
		Schema:        issueorchestrator.WavePlanSchema,
		TotalIssues:   2,
		PlannedIssues: 2,
		PlannedSteps:  6,
		TotalWaves:    1,
		Waves: []issueorchestrator.Wave{
			{
				ID:       "wave-1",
				WaveSize: 2,
				Issues: []issueorchestrator.Issue{
					{Number: 301, Title: "Issue 301"},
					{Number: 302, Title: "Issue 302"},
				},
			},
		},
	}
	baselineBytes, err := json.Marshal(baselinePlan)
	if err != nil {
		t.Fatal(err)
	}
	baselinePath := filepath.Join(t.TempDir(), "baseline.json")
	if err := os.WriteFile(baselinePath, baselineBytes, 0o644); err != nil {
		t.Fatal(err)
	}

	// Current issues: only 302 remains (301 closed)
	currentIssues := []issueorchestrator.Issue{
		{
			Number:          302,
			Key:             "issue-302",
			Title:           "Issue 302",
			Lane:            "model",
			Paths:           []string{"internal/model/m.go"},
			ExpectedSteps:   3,
			Dispatchability: "dispatchable",
		},
	}
	currentPath := writeTestIssuesFile(t, currentIssues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", currentPath,
		"--compare", baselinePath,
	})
	if code != 0 {
		t.Fatalf("runIssueOrchestrator --compare failed: code=%d, stderr=%s", code, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "CAMPAIGN BURNDOWN COMPARISON") {
		t.Errorf("missing comparison header in output: %s", out)
	}
	if !strings.Contains(out, "1/2 issue(s) closed (50.0%)") {
		t.Errorf("missing burndown stats in output: %s", out)
	}
	if !strings.Contains(out, "#301") {
		t.Errorf("missing closed issue #301 in output: %s", out)
	}
}

func TestIssueOrchestratorCLISubdivideAndTriageFilters(t *testing.T) {
	issues := []issueorchestrator.Issue{
		{
			Number:          401,
			Key:             "epic-401",
			Title:           "Huge multi-subsystem epic",
			ExpectedSteps:   25,
			Dispatchability: "dispatchable",
		},
		{
			Number:          402,
			Key:             "triage-402",
			Title:           "",
			Dispatchability: "triage_only",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var subOut, subErr bytes.Buffer
	subCode := runIssueOrchestrator(&subOut, &subErr, []string{
		"--from-issues", issuesPath,
		"--subdivide",
	})
	if subCode != 0 {
		t.Fatalf("--subdivide failed: %s", subErr.String())
	}
	if !strings.Contains(subOut.String(), "#401") {
		t.Errorf("--subdivide output missing #401: %s", subOut.String())
	}

	var triOut, triErr bytes.Buffer
	triCode := runIssueOrchestrator(&triOut, &triErr, []string{
		"--from-issues", issuesPath,
		"--triage",
	})
	if triCode != 0 {
		t.Fatalf("--triage failed: %s", triErr.String())
	}
	if !strings.Contains(triOut.String(), "#402") {
		t.Errorf("--triage output missing #402: %s", triOut.String())
	}
}

func TestIssueOrchestratorSpawnOpencodeDryRun(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	defer func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
	}()
	controllerStampFunc = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: "532688a0a04ba669a20d2c7f353150d044ae8be8", HasVCS: true}
	}
	controllerHeadRevFunc = func(string) string {
		return "532688a0a04ba669a20d2c7f353150d044ae8be8"
	}

	issues := []issueorchestrator.Issue{
		{
			Number:          501,
			Key:             "issue-501",
			Title:           "Implement OpenCode worker spawner",
			Lane:            "issueorchestrator",
			Paths:           []string{"internal/issueorchestrator/opencode.go"},
			ExpectedSteps:   3,
			Dispatchability: "dispatchable",
		},
		{
			Number:          502,
			Key:             "issue-502",
			Title:           "Add chat options tests",
			Lane:            "testing",
			Paths:           []string{"cmd/fak/issueorchestrator_test.go"},
			ExpectedSteps:   2,
			Dispatchability: "dispatchable",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--spawn-opencode",
		"--dry-run",
		"--json",
		"--model", "gpt-4o",
		"--agent", "worker",
	})
	if code != 0 {
		t.Fatalf("runIssueOrchestrator failed with exit code %d; stderr: %s", code, stderr.String())
	}

	var receipt OpencodeSpawnReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to decode JSON receipt: %v; raw: %s", err, stdout.String())
	}

	if receipt.Schema != "fak.issue-orchestrator-opencode-spawn.v1" {
		t.Errorf("expected schema %q, got %q", "fak.issue-orchestrator-opencode-spawn.v1", receipt.Schema)
	}
	if !receipt.DryRun {
		t.Errorf("expected DryRun to be true, got false")
	}
	if receipt.TotalSpawned != 2 {
		t.Errorf("expected 2 spawned chats, got %d", receipt.TotalSpawned)
	}
	if len(receipt.Chats) != 2 {
		t.Fatalf("expected 2 chats in slice, got %d", len(receipt.Chats))
	}
	for _, c := range receipt.Chats {
		if c.Status != "dry_run" {
			t.Errorf("expected chat status dry_run, got %q", c.Status)
		}
		if len(c.Command) == 0 {
			t.Errorf("expected non-empty command for chat")
		}
	}
}

func TestIssueOrchestratorDefaultWorktreePrepareFailureIsTerminal(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	origPrepare := prepareManagedWorkerWorktreeFunc
	origStart := startDispatchWorkerFunc
	origAcquire := acquireTreeLeaseFunc
	t.Cleanup(func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
		prepareManagedWorkerWorktreeFunc = origPrepare
		startDispatchWorkerFunc = origStart
		acquireTreeLeaseFunc = origAcquire
	})

	const testRev = "532688a0a04ba669a20d2c7f353150d044ae8be8"
	controllerStampFunc = func() binstamp.Stamp { return binstamp.Stamp{Revision: testRev, HasVCS: true} }
	controllerHeadRevFunc = func(string) string { return testRev }
	var gotRoot string
	prepareManagedWorkerWorktreeFunc = func(root, lane, key, baseSHA, wtRoot string, git workerworktree.GitRunner) workerworktree.Result {
		gotRoot = root
		return workerworktree.Result{OK: false, Code: "PREPARE_REFUSED", Reason: "synthetic refusal"}
	}
	started := false
	startDispatchWorkerFunc = func(*exec.Cmd) error { started = true; return nil }
	acquired := false
	acquireTreeLeaseFunc = func(string, string, []string, int) error { acquired = true; return nil }

	workspace := t.TempDir()
	issuesPath := writeTestIssuesFile(t, []issueorchestrator.Issue{{
		Number: 12379, Key: "portable-default", Title: "Portable default", Lane: "issueorchestrator",
		Paths: []string{"cmd/fak/issueorchestrator.go"}, ExpectedSteps: 1, Dispatchability: "dispatchable",
	}})
	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath, "--spawn-opencode", "--workspace", workspace,
		"--no-detect-held", "--supervise=false", "--json",
	})
	if code == 0 {
		t.Fatalf("prepare refusal must return non-zero; stdout=%s", stdout.String())
	}
	if started || acquired {
		t.Fatalf("prepare refusal must precede lease and process start: acquired=%v started=%v", acquired, started)
	}
	if gotRoot != workspace {
		t.Fatalf("prepare root = %q, want selected workspace %q", gotRoot, workspace)
	}
	var receipt OpencodeSpawnReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode receipt: %v; raw=%s stderr=%s", err, stdout.String(), stderr.String())
	}
	if receipt.WorktreeMode != worktreeModeManagedDefault || !receipt.AutoLand {
		t.Fatalf("default lifecycle receipt = %+v", receipt)
	}
	if len(receipt.Chats) != 1 || receipt.Chats[0].Status != "error" || !strings.Contains(receipt.Chats[0].Error, "WORKTREE_PREPARE_FAILED") {
		t.Fatalf("prepare refusal chat = %+v", receipt.Chats)
	}
}

func TestIssueOrchestratorExplicitSharedOptOutIsVisible(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	origPrepare := prepareManagedWorkerWorktreeFunc
	t.Cleanup(func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
		prepareManagedWorkerWorktreeFunc = origPrepare
	})
	const testRev = "532688a0a04ba669a20d2c7f353150d044ae8be8"
	controllerStampFunc = func() binstamp.Stamp { return binstamp.Stamp{Revision: testRev, HasVCS: true} }
	controllerHeadRevFunc = func(string) string { return testRev }
	prepareManagedWorkerWorktreeFunc = func(string, string, string, string, string, workerworktree.GitRunner) workerworktree.Result {
		t.Fatal("explicit shared opt-out must not prepare a worktree")
		return workerworktree.Result{}
	}
	issuesPath := writeTestIssuesFile(t, []issueorchestrator.Issue{{
		Number: 12380, Key: "portable-optout", Title: "Portable opt-out", Lane: "issueorchestrator",
		Paths: []string{"cmd/fak/issueorchestrator.go"}, ExpectedSteps: 1, Dispatchability: "dispatchable",
	}})
	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath, "--spawn-opencode", "--dry-run", "--worktree=false", "--auto-land=false",
		"--no-detect-held", "--json",
	})
	if code != 0 {
		t.Fatalf("explicit opt-out failed: code=%d stderr=%s", code, stderr.String())
	}
	var receipt OpencodeSpawnReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode receipt: %v", err)
	}
	if receipt.WorktreeMode != worktreeModeSharedExplicitOptOut || receipt.AutoLand {
		t.Fatalf("explicit opt-out receipt = %+v", receipt)
	}
	if len(receipt.Chats) != 1 || !receipt.Chats[0].UnsafeSharedWorkspace {
		t.Fatalf("shared mode must be explicitly marked unsafe: %+v", receipt.Chats)
	}
}

func TestIssueOrchestratorOpencodeCommandsFlag(t *testing.T) {
	issues := []issueorchestrator.Issue{
		{
			Number:          601,
			Key:             "issue-601",
			Title:           "Build OpenCode commands in plan",
			Lane:            "testing-opencode-lane",
			Paths:           []string{"internal/testing-opencode-lane/gw.go"},
			ExpectedSteps:   3,
			Dispatchability: "dispatchable",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--opencode-commands",
		"--json",
		"--no-detect-held",
	})
	if code != 0 {
		t.Fatalf("runIssueOrchestrator failed with exit code %d; stderr: %s", code, stderr.String())
	}

	var plan issueorchestrator.Plan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("failed to decode JSON plan: %v; raw: %s", err, stdout.String())
	}

	if len(plan.Waves) == 0 {
		t.Fatalf("expected waves in plan, got 0")
	}
	wave := plan.Waves[0]
	if len(wave.OpencodeChats) == 0 {
		t.Errorf("expected opencode_chats to be populated on wave")
	}
	if len(wave.Issues) == 0 || len(wave.Issues[0].OpencodeCommand) == 0 {
		t.Errorf("expected opencode_command to be populated on issue")
	}
	command := wave.OpencodeChats[0].Command
	if got := issueOrchestratorArgCount(command, "--variant"); got != 0 {
		t.Fatalf("default generated command contains %d --variant overrides, want 0: %v", got, command)
	}
	if !issueOrchestratorContainsArgPair(command, "--agent", "worker") {
		t.Fatalf("default generated command missing --agent worker: %v", command)
	}
}

func TestIssueOrchestratorSpawnOpencodeTextRender(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	defer func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
	}()
	controllerStampFunc = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: "532688a0a04ba669a20d2c7f353150d044ae8be8", HasVCS: true}
	}
	controllerHeadRevFunc = func(string) string {
		return "532688a0a04ba669a20d2c7f353150d044ae8be8"
	}

	issues := []issueorchestrator.Issue{
		{
			Number:          701,
			Key:             "issue-701",
			Title:           "Test text rendering banner",
			Lane:            "cli",
			Paths:           []string{"cmd/fak/main.go"},
			ExpectedSteps:   1,
			Dispatchability: "dispatchable",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--spawn-opencode",
		"--dry-run",
	})
	if code != 0 {
		t.Fatalf("runIssueOrchestrator failed with exit code %d; stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "OpenCode Chat Spawner") {
		t.Errorf("expected OpenCode Chat Spawner banner in output: %s", out)
	}
	if !strings.Contains(out, "#701") {
		t.Errorf("expected issue #701 in output: %s", out)
	}
	if !strings.Contains(out, "DRY RUN") {
		t.Errorf("expected DRY RUN indicator in output: %s", out)
	}
}

func TestIssueOrchestratorCLIDynamicSlidingWindow(t *testing.T) {
	var issues []issueorchestrator.Issue
	for i := 1; i <= 25; i++ {
		if i == 1 || i == 5 || i == 9 || i == 13 || i == 17 || i == 21 || i == 23 || i == 25 {
			issues = append(issues, issueorchestrator.Issue{
				Number:          i,
				Key:             fmt.Sprintf("leaf-%d", i),
				Title:           fmt.Sprintf("Leaf %d", i),
				Lane:            fmt.Sprintf("lane%d", i),
				Paths:           []string{fmt.Sprintf("internal/pkg%d/file.go", i)},
				ExpectedSteps:   2,
				Dispatchability: "dispatchable",
			})
		} else {
			issues = append(issues, issueorchestrator.Issue{
				Number:          i,
				Key:             fmt.Sprintf("epic-%d", i),
				Title:           fmt.Sprintf("Epic %d", i),
				Lane:            "epiclane",
				Paths:           []string{"internal/a/a.go", "internal/b/b.go", "internal/c/c.go"},
				ExpectedSteps:   25,
				Dispatchability: "needs_scope",
			})
		}
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--target-issues", "6",
		"--min-window", "10",
		"--max-window", "30",
		"--auto-expand",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runIssueOrchestrator failed with exit code %d; stderr: %s", code, stderr.String())
	}

	var plan issueorchestrator.Plan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("failed to decode JSON plan: %v; raw: %s", err, stdout.String())
	}

	if plan.Diagnostics == nil {
		t.Fatalf("expected plan.Diagnostics to be populated")
	}
	if plan.Diagnostics.DiscoveryWindowSize <= 10 {
		t.Errorf("expected DiscoveryWindowSize > 10, got %d", plan.Diagnostics.DiscoveryWindowSize)
	}
	if plan.PlannedIssues < 6 {
		t.Errorf("expected at least 6 planned issues, got %d", plan.PlannedIssues)
	}
}

func TestIssueOrchestratorCLILiveViewValidation(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--live",
		"--view", "nonexistent-test-view-slug",
		"--json",
	})
	if code != 2 {
		t.Fatalf("expected exit code 2 on invalid view slug, got %d; stdout: %s; stderr: %s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "view slug \"nonexistent-test-view-slug\" not found") {
		t.Errorf("expected view error in stderr: %s", stderr.String())
	}
}

func TestIssueOrchestratorAdjustStepsSubcommandWithPlan(t *testing.T) {
	plan := issueorchestrator.Plan{
		Schema:        issueorchestrator.WavePlanSchema,
		TotalIssues:   2,
		PlannedIssues: 2,
		PlannedSteps:  7,
		TotalWaves:    1,
		Waves: []issueorchestrator.Wave{
			{
				ID:         "wave-1",
				WaveSize:   2,
				StepBudget: 7,
				Issues: []issueorchestrator.Issue{
					{Number: 101, Title: "Issue 101", ExpectedSteps: 3, Dispatchability: "dispatchable"},
					{Number: 102, Title: "Issue 102", ExpectedSteps: 4, Dispatchability: "dispatchable"},
				},
			},
		},
	}
	b, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	planPath := filepath.Join(t.TempDir(), "plan.json")
	if err := os.WriteFile(planPath, b, 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"adjust-steps",
		"--issue", "101",
		"--steps", "9",
		"--reason", "AST expansion",
		"--plan", planPath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("adjust-steps failed: code=%d, stderr=%s", code, stderr.String())
	}

	// 1. Verify stdout JSON matches updated plan
	var updatedStdoutPlan issueorchestrator.Plan
	if err := json.Unmarshal(stdout.Bytes(), &updatedStdoutPlan); err != nil {
		t.Fatalf("failed to decode stdout plan JSON: %v; raw: %s", err, stdout.String())
	}
	if updatedStdoutPlan.PlannedSteps != 13 {
		t.Errorf("expected PlannedSteps 13, got %d", updatedStdoutPlan.PlannedSteps)
	}
	if updatedStdoutPlan.Waves[0].Issues[0].ExpectedSteps != 9 {
		t.Errorf("expected issue 101 steps 9, got %d", updatedStdoutPlan.Waves[0].Issues[0].ExpectedSteps)
	}
	if updatedStdoutPlan.Waves[0].StepBudget != 13 {
		t.Errorf("expected wave 1 StepBudget 13, got %d", updatedStdoutPlan.Waves[0].StepBudget)
	}

	// 2. Verify file was modified in-place on disk
	fileBytes, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("failed to read plan file: %v", err)
	}
	var filePlan issueorchestrator.Plan
	if err := json.Unmarshal(fileBytes, &filePlan); err != nil {
		t.Fatalf("failed to decode disk plan JSON: %v", err)
	}
	if filePlan.PlannedSteps != 13 {
		t.Errorf("expected disk PlannedSteps 13, got %d", filePlan.PlannedSteps)
	}
	if filePlan.Waves[0].Issues[0].ExpectedSteps != 9 {
		t.Errorf("expected disk issue 101 steps 9, got %d", filePlan.Waves[0].Issues[0].ExpectedSteps)
	}
	if filePlan.Diagnostics == nil || len(filePlan.Diagnostics.AdvisoryWarnings) == 0 {
		t.Fatalf("expected diagnostics advisory warnings on disk plan")
	}
	if !strings.Contains(filePlan.Diagnostics.AdvisoryWarnings[0], "AST expansion") {
		t.Errorf("expected advisory warning to contain AST expansion: %v", filePlan.Diagnostics.AdvisoryWarnings)
	}
}

func TestIssueOrchestratorAdjustStepsSubcommandWithoutPlan(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"adjust-steps",
		"--issue", "205",
		"--steps", "5",
		"--reason", "minimal plan test",
	})
	if code != 0 {
		t.Fatalf("adjust-steps failed: code=%d, stderr=%s", code, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "Adjusted issue #205 steps to 5") {
		t.Errorf("stdout missing adjustment summary: %s", out)
	}
	if !strings.Contains(out, "Reason: minimal plan test") {
		t.Errorf("stdout missing reason: %s", out)
	}
}

func TestIssueOrchestratorAdjustStepsSubcommandErrors(t *testing.T) {
	// Missing --issue
	var out1, err1 bytes.Buffer
	code1 := runIssueOrchestrator(&out1, &err1, []string{
		"adjust-steps",
		"--steps", "5",
	})
	if code1 != 2 {
		t.Errorf("expected exit code 2 for missing --issue, got %d", code1)
	}

	// Missing --steps
	var out2, err2 bytes.Buffer
	code2 := runIssueOrchestrator(&out2, &err2, []string{
		"adjust-steps",
		"--issue", "101",
	})
	if code2 != 2 {
		t.Errorf("expected exit code 2 for missing --steps, got %d", code2)
	}

	// Issue not found in plan
	plan := issueorchestrator.Plan{
		Schema:      issueorchestrator.WavePlanSchema,
		TotalIssues: 1,
		TotalWaves:  1,
		Waves: []issueorchestrator.Wave{
			{
				ID: "wave-1",
				Issues: []issueorchestrator.Issue{
					{Number: 101, ExpectedSteps: 3},
				},
			},
		},
	}
	b, _ := json.Marshal(plan)
	planPath := filepath.Join(t.TempDir(), "plan.json")
	_ = os.WriteFile(planPath, b, 0o644)

	var out3, err3 bytes.Buffer
	code3 := runIssueOrchestrator(&out3, &err3, []string{
		"adjust-steps",
		"--issue", "999",
		"--steps", "5",
		"--plan", planPath,
	})
	if code3 != 1 {
		t.Errorf("expected exit code 1 for issue not found, got %d", code3)
	}
}

func TestIssueOrchestratorHarvestFlagExplicitReceipt(t *testing.T) {
	receipt := issueorchestrator.HarvestReceipt{
		Schema: issueorchestrator.HarvestReceiptSchema,
		WaveID: "wave-test-harvest-explicit",
		Leaves: []issueorchestrator.LeafReceipt{
			{IssueNumber: 301, Title: "Leaf 301", Lane: "gateway", State: issueorchestrator.StateVerifiedCleared, CommitSHA: "sha301"},
			{IssueNumber: 302, Title: "Leaf 302", Lane: "model", State: issueorchestrator.StateResidualReview, CommitSHA: "sha302"},
			{IssueNumber: 303, Title: "Leaf 303", Lane: "compute", State: issueorchestrator.StateQuietIncomplete},
			{IssueNumber: 304, Title: "Leaf 304", Lane: "token", State: issueorchestrator.StateSpinningStalled},
		},
	}
	b, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	receiptPath := filepath.Join(t.TempDir(), "wave_receipt.json")
	if err := os.WriteFile(receiptPath, b, 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--harvest",
		"--receipt", receiptPath,
		"--auto-land",
		"--min-clear-rate", "0.20",
		"--json",
	})
	if code != 0 {
		t.Fatalf("--harvest failed: code=%d, stderr=%s", code, stderr.String())
	}

	var res issueorchestrator.HarvestReceipt
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("failed to decode harvest receipt JSON: %v; raw: %s", err, stdout.String())
	}

	if res.WaveID != "wave-test-harvest-explicit" {
		t.Errorf("expected wave_id %q, got %q", "wave-test-harvest-explicit", res.WaveID)
	}
	if res.TotalLeaves != 4 {
		t.Errorf("expected 4 total leaves, got %d", res.TotalLeaves)
	}
	if res.ClearedCount != 1 {
		t.Errorf("expected 1 cleared, got %d", res.ClearedCount)
	}
	if res.ResidualCount != 1 {
		t.Errorf("expected 1 residual, got %d", res.ResidualCount)
	}
	if res.QuietCount != 1 {
		t.Errorf("expected 1 quiet, got %d", res.QuietCount)
	}
	if res.StalledCount != 1 {
		t.Errorf("expected 1 stalled, got %d", res.StalledCount)
	}
	if res.ClearRate != 0.25 {
		t.Errorf("expected clear rate 0.25, got %f", res.ClearRate)
	}
	if len(res.LandedSHAs) != 1 || res.LandedSHAs[0] != "sha301" {
		t.Errorf("expected LandedSHAs [sha301], got %v", res.LandedSHAs)
	}
	if len(res.ReviewQueue) != 1 || res.ReviewQueue[0].IssueNumber != 302 {
		t.Errorf("expected ReviewQueue with issue #302, got %v", res.ReviewQueue)
	}
}

func TestIssueOrchestratorHarvestFlagTextOutput(t *testing.T) {
	receipt := issueorchestrator.HarvestReceipt{
		Schema: issueorchestrator.HarvestReceiptSchema,
		WaveID: "wave-test-text",
		Leaves: []issueorchestrator.LeafReceipt{
			{IssueNumber: 401, Title: "Leaf 401", Lane: "gateway", State: issueorchestrator.StateVerifiedCleared, CommitSHA: "sha401"},
		},
	}
	b, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	receiptPath := filepath.Join(t.TempDir(), "wave_receipt.json")
	if err := os.WriteFile(receiptPath, b, 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--harvest",
		"--receipt", receiptPath,
	})
	if code != 0 {
		t.Fatalf("--harvest text failed: code=%d, stderr=%s", code, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "Wave Harvest Reconciliation: wave-test-text") {
		t.Errorf("stdout missing header: %s", out)
	}
	if !strings.Contains(out, "Cleared:        1 (100.0%)") {
		t.Errorf("stdout missing cleared count: %s", out)
	}
}

func TestIssueOrchestratorHarvestFlagDefaultReceiptDiscovery(t *testing.T) {
	wsDir := t.TempDir()
	dispatchRunsDir := filepath.Join(wsDir, ".dispatch-runs")
	if err := os.MkdirAll(dispatchRunsDir, 0o755); err != nil {
		t.Fatal(err)
	}

	receipt := issueorchestrator.HarvestReceipt{
		Schema: issueorchestrator.HarvestReceiptSchema,
		WaveID: "wave-auto-discovered",
		Leaves: []issueorchestrator.LeafReceipt{
			{IssueNumber: 501, Title: "Leaf 501", Lane: "gateway", State: issueorchestrator.StateVerifiedCleared},
		},
	}
	b, _ := json.Marshal(receipt)
	receiptPath := filepath.Join(dispatchRunsDir, "wave_receipt_auto.json")
	if err := os.WriteFile(receiptPath, b, 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--workspace", wsDir,
		"--harvest",
		"--json",
	})
	if code != 0 {
		t.Fatalf("harvest auto-discovery failed: code=%d, stderr=%s", code, stderr.String())
	}

	var res issueorchestrator.HarvestReceipt
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("failed to decode json: %v", err)
	}
	if res.WaveID != "wave-auto-discovered" {
		t.Errorf("expected discovered wave_id %q, got %q", "wave-auto-discovered", res.WaveID)
	}
}

func TestIssueOrchestratorHarvestFlagMissingReceiptError(t *testing.T) {
	emptyWs := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--workspace", emptyWs,
		"--harvest",
	})
	if code != 2 {
		t.Errorf("expected exit code 2 when no receipt found, got %d", code)
	}
	if !strings.Contains(stderr.String(), "no wave receipt found") {
		t.Errorf("expected error message in stderr: %s", stderr.String())
	}
}

func TestIssueOrchestratorCLIAutoExpandDisabled(t *testing.T) {
	var issues []issueorchestrator.Issue
	for i := 1; i <= 25; i++ {
		if i == 1 || i == 5 || i == 9 || i == 13 || i == 17 || i == 21 || i == 23 || i == 25 {
			issues = append(issues, issueorchestrator.Issue{
				Number:          i,
				Key:             fmt.Sprintf("leaf-%d", i),
				Title:           fmt.Sprintf("Leaf %d", i),
				Lane:            fmt.Sprintf("lane%d", i),
				Paths:           []string{fmt.Sprintf("internal/pkg%d/file.go", i)},
				ExpectedSteps:   2,
				Dispatchability: "dispatchable",
			})
		} else {
			issues = append(issues, issueorchestrator.Issue{
				Number:          i,
				Key:             fmt.Sprintf("epic-%d", i),
				Title:           fmt.Sprintf("Epic %d", i),
				Lane:            "epiclane",
				Paths:           []string{"internal/a/a.go", "internal/b/b.go", "internal/c/c.go"},
				ExpectedSteps:   25,
				Dispatchability: "needs_scope",
			})
		}
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--target-issues", "6",
		"--top", "10",
		"--auto-expand=false",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runIssueOrchestrator failed with exit code %d; stderr: %s", code, stderr.String())
	}

	var plan issueorchestrator.Plan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("failed to decode JSON plan: %v; raw: %s", err, stdout.String())
	}

	// Without auto-expand, only issues 1..10 were evaluated (issues 1, 5, 9 are the only leaves in 1..10)
	if plan.PlannedIssues > 3 {
		t.Errorf("expected at most 3 planned issues without auto-expand in top 10, got %d", plan.PlannedIssues)
	}
}

func TestIssueOrchestratorSuperviseFlagCLI(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	defer func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
	}()
	controllerStampFunc = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: "532688a0a04ba669a20d2c7f353150d044ae8be8", HasVCS: true}
	}
	controllerHeadRevFunc = func(string) string {
		return "532688a0a04ba669a20d2c7f353150d044ae8be8"
	}

	issues := []issueorchestrator.Issue{
		{
			Number:          901,
			Key:             "issue-901",
			Title:           "Test supervise flag",
			Lane:            "cli",
			Paths:           []string{"cmd/fak/main.go"},
			ExpectedSteps:   1,
			Dispatchability: "dispatchable",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--spawn-opencode",
		"--dry-run",
		"--supervise=true",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runIssueOrchestrator with --supervise failed: code=%d, stderr=%s", code, stderr.String())
	}

	tmpDir := t.TempDir()
	logFile := filepath.Join(tmpDir, "worker.log")
	_ = os.WriteFile(logFile, []byte("init log"), 0o644)
	cfg := issueorchestrator.WorkerSupervisorConfig{
		PID:         os.Getpid(),
		WorktreeDir: tmpDir,
		LogFile:     logFile,
	}
	sup := newWorkerSupervisorFunc(cfg)
	if sup == nil {
		t.Fatalf("expected initialized supervisor, got nil")
	}
}

func TestIssueOrchestratorControllerStaleRefusal(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	defer func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
	}()
	controllerStampFunc = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: "stale-revision-1111111", HasVCS: true}
	}
	controllerHeadRevFunc = func(string) string {
		return "head-revision-2222222"
	}

	issues := []issueorchestrator.Issue{
		{
			Number:          1001,
			Key:             "issue-1001",
			Title:           "Stale controller test",
			Lane:            "issueorchestrator",
			Paths:           []string{"internal/issueorchestrator/opencode.go"},
			ExpectedSteps:   1,
			Dispatchability: "dispatchable",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--spawn-opencode",
		"--dry-run",
	})
	if code == 0 {
		t.Fatalf("expected non-zero exit code for stale controller, got 0")
	}

	errStr := stderr.String()
	if !strings.Contains(errStr, "CONTROLLER_STALE") {
		t.Errorf("stderr missing 'CONTROLLER_STALE': %s", errStr)
	}
	if !strings.Contains(errStr, "go build ./cmd/fak") {
		t.Errorf("stderr missing rebuild instructions 'go build ./cmd/fak': %s", errStr)
	}
	if !strings.Contains(errStr, "stale-revision-1111111") {
		t.Errorf("stderr missing running revision: %s", errStr)
	}
	if !strings.Contains(errStr, "head-revision-2222222") {
		t.Errorf("stderr missing head revision: %s", errStr)
	}
}

func TestIssueOrchestratorControllerAmbiguousRefusal(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	defer func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
	}()

	issues := []issueorchestrator.Issue{
		{
			Number:          1002,
			Key:             "issue-1002",
			Title:           "Ambiguous controller test",
			Lane:            "issueorchestrator",
			Paths:           []string{"internal/issueorchestrator/opencode.go"},
			ExpectedSteps:   1,
			Dispatchability: "dispatchable",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	t.Run("unstamped running binary", func(t *testing.T) {
		controllerStampFunc = func() binstamp.Stamp {
			return binstamp.Stamp{Revision: "", HasVCS: false}
		}
		controllerHeadRevFunc = func(string) string {
			return "head-revision-2222222"
		}

		var stdout, stderr bytes.Buffer
		code := runIssueOrchestrator(&stdout, &stderr, []string{
			"--from-issues", issuesPath,
			"--spawn-opencode",
			"--dry-run",
		})
		if code == 0 {
			t.Fatalf("expected non-zero exit code for unstamped controller, got 0")
		}

		errStr := stderr.String()
		if !strings.Contains(errStr, "CONTROLLER_AMBIGUOUS") {
			t.Errorf("stderr missing 'CONTROLLER_AMBIGUOUS': %s", errStr)
		}
		if !strings.Contains(errStr, "go build ./cmd/fak") {
			t.Errorf("stderr missing rebuild instructions 'go build ./cmd/fak': %s", errStr)
		}
	})

	t.Run("unresolvable checkout HEAD", func(t *testing.T) {
		controllerStampFunc = func() binstamp.Stamp {
			return binstamp.Stamp{Revision: "running-rev-3333333", HasVCS: true}
		}
		controllerHeadRevFunc = func(string) string {
			return ""
		}

		var stdout, stderr bytes.Buffer
		code := runIssueOrchestrator(&stdout, &stderr, []string{
			"--from-issues", issuesPath,
			"--spawn-opencode",
			"--dry-run",
		})
		if code == 0 {
			t.Fatalf("expected non-zero exit code for unresolvable HEAD, got 0")
		}

		errStr := stderr.String()
		if !strings.Contains(errStr, "CONTROLLER_AMBIGUOUS") {
			t.Errorf("stderr missing 'CONTROLLER_AMBIGUOUS': %s", errStr)
		}
		if !strings.Contains(errStr, "go build ./cmd/fak") {
			t.Errorf("stderr missing rebuild instructions 'go build ./cmd/fak': %s", errStr)
		}
	})
}

func TestIssueOrchestratorSourceMatchedSpawn(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	origSHA := controllerBinarySHAFunc
	defer func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
		controllerBinarySHAFunc = origSHA
	}()

	const testRev = "feedbeef12345678abcdef0123456789abcdef01"
	const testSHA = "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"

	controllerStampFunc = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: testRev, HasVCS: true}
	}
	controllerHeadRevFunc = func(string) string {
		return testRev
	}
	controllerBinarySHAFunc = func() (string, error) {
		return testSHA, nil
	}

	issues := []issueorchestrator.Issue{
		{
			Number:          1003,
			Key:             "issue-1003",
			Title:           "Source matched spawn test",
			Lane:            "issueorchestrator",
			Paths:           []string{"internal/issueorchestrator/opencode.go"},
			ExpectedSteps:   1,
			Dispatchability: "dispatchable",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--spawn-opencode",
		"--dry-run",
		"--json",
	})
	if code != 0 {
		t.Fatalf("expected exit code 0 for matched controller, got %d; stderr: %s", code, stderr.String())
	}

	var receipt OpencodeSpawnReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to decode JSON receipt: %v; raw: %s", err, stdout.String())
	}

	if receipt.ControllerRevision != testRev {
		t.Errorf("receipt ControllerRevision = %q, want %q", receipt.ControllerRevision, testRev)
	}
	if receipt.ControllerBinarySHA != testSHA {
		t.Errorf("receipt ControllerBinarySHA = %q, want %q", receipt.ControllerBinarySHA, testSHA)
	}
	if receipt.ControllerFreshness != "fresh" {
		t.Errorf("receipt ControllerFreshness = %q, want 'fresh'", receipt.ControllerFreshness)
	}
	if receipt.AgentProfile != "worker" {
		t.Errorf("receipt AgentProfile = %q, want 'worker'", receipt.AgentProfile)
	}

	if len(receipt.Chats) != 1 {
		t.Fatalf("expected 1 chat in receipt, got %d", len(receipt.Chats))
	}
	chat := receipt.Chats[0]
	if chat.ControllerRevision != testRev {
		t.Errorf("chat ControllerRevision = %q, want %q", chat.ControllerRevision, testRev)
	}
	if chat.ControllerBinarySHA != testSHA {
		t.Errorf("chat ControllerBinarySHA = %q, want %q", chat.ControllerBinarySHA, testSHA)
	}
	if chat.ControllerFreshness != "fresh" {
		t.Errorf("chat ControllerFreshness = %q, want 'fresh'", chat.ControllerFreshness)
	}
	if chat.AgentProfile != "worker" {
		t.Errorf("chat AgentProfile = %q, want 'worker'", chat.AgentProfile)
	}
}

func TestIssueOrchestratorDefaultAgent(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	defer func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
	}()

	const testRev = "532688a0a04ba669a20d2c7f353150d044ae8be8"
	controllerStampFunc = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: testRev, HasVCS: true}
	}
	controllerHeadRevFunc = func(string) string {
		return testRev
	}

	issues := []issueorchestrator.Issue{
		{
			Number:          1004,
			Key:             "issue-1004",
			Title:           "Default agent test",
			Lane:            "issueorchestrator",
			Paths:           []string{"internal/issueorchestrator/opencode.go"},
			ExpectedSteps:   1,
			Dispatchability: "dispatchable",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--spawn-opencode",
		"--dry-run",
		"--json",
	})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr: %s", code, stderr.String())
	}

	var receipt OpencodeSpawnReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to decode JSON receipt: %v; raw: %s", err, stdout.String())
	}

	if receipt.AgentProfile != "worker" {
		t.Errorf("expected default AgentProfile 'worker', got %q", receipt.AgentProfile)
	}
	if len(receipt.Chats) != 1 {
		t.Fatalf("expected 1 chat, got %d", len(receipt.Chats))
	}
	chat := receipt.Chats[0]
	if chat.AgentProfile != "worker" {
		t.Errorf("expected chat AgentProfile 'worker', got %q", chat.AgentProfile)
	}

	foundAgentFlag := false
	for i, arg := range chat.Command {
		if arg == "--agent" {
			foundAgentFlag = true
			if i+1 >= len(chat.Command) || chat.Command[i+1] != "worker" {
				t.Fatalf("expected '--agent worker', got command: %v", chat.Command)
			}
			break
		}
	}
	if !foundAgentFlag {
		t.Fatalf("expected command to contain '--agent worker', got: %v", chat.Command)
	}
	if got := issueOrchestratorArgCount(chat.Command, "--variant"); got != 0 {
		t.Fatalf("default spawn command contains %d --variant overrides, want 0: %v", got, chat.Command)
	}
}

func TestIssueOrchestratorExplicitVariantOverride(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	defer func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
	}()

	const testRev = "532688a0a04ba669a20d2c7f353150d044ae8be8"
	controllerStampFunc = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: testRev, HasVCS: true}
	}
	controllerHeadRevFunc = func(string) string { return testRev }

	issuesPath := writeTestIssuesFile(t, []issueorchestrator.Issue{{
		Number:          1005,
		Key:             "issue-1005",
		Title:           "Explicit variant override test",
		Lane:            "issueorchestrator",
		Paths:           []string{"internal/issueorchestrator/opencode.go"},
		ExpectedSteps:   1,
		Dispatchability: "dispatchable",
	}})

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--spawn-opencode",
		"--dry-run",
		"--json",
		"--variant", "high",
	})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr: %s", code, stderr.String())
	}

	var receipt OpencodeSpawnReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to decode JSON receipt: %v; raw: %s", err, stdout.String())
	}
	if len(receipt.Chats) != 1 {
		t.Fatalf("expected 1 chat, got %d", len(receipt.Chats))
	}
	command := receipt.Chats[0].Command
	if got := issueOrchestratorArgCount(command, "--variant"); got != 1 {
		t.Fatalf("explicit command contains %d --variant overrides, want 1: %v", got, command)
	}
	if !issueOrchestratorContainsArgPair(command, "--variant", "high") {
		t.Fatalf("explicit command missing --variant high: %v", command)
	}
	if !issueOrchestratorContainsArgPair(command, "--agent", "worker") {
		t.Fatalf("explicit command missing --agent worker: %v", command)
	}
}

func issueOrchestratorArgCount(args []string, target string) int {
	count := 0
	for _, arg := range args {
		if arg == target {
			count++
		}
	}
	return count
}

func issueOrchestratorContainsArgPair(args []string, key, value string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == key && args[i+1] == value {
			return true
		}
	}
	return false
}

func TestIssueOrchestratorExplicitAgent(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	defer func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
	}()

	const testRev = "532688a0a04ba669a20d2c7f353150d044ae8be8"
	controllerStampFunc = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: testRev, HasVCS: true}
	}
	controllerHeadRevFunc = func(string) string {
		return testRev
	}

	issues := []issueorchestrator.Issue{
		{
			Number:          1005,
			Key:             "issue-1005",
			Title:           "Explicit agent test",
			Lane:            "issueorchestrator",
			Paths:           []string{"internal/issueorchestrator/opencode.go"},
			ExpectedSteps:   1,
			Dispatchability: "dispatchable",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--spawn-opencode",
		"--dry-run",
		"--json",
		"--agent", "custom-profile",
	})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr: %s", code, stderr.String())
	}

	var receipt OpencodeSpawnReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to decode JSON receipt: %v; raw: %s", err, stdout.String())
	}

	if receipt.AgentProfile != "custom-profile" {
		t.Errorf("expected explicit AgentProfile 'custom-profile', got %q", receipt.AgentProfile)
	}
	if len(receipt.Chats) != 1 {
		t.Fatalf("expected 1 chat, got %d", len(receipt.Chats))
	}
	chat := receipt.Chats[0]
	if chat.AgentProfile != "custom-profile" {
		t.Errorf("expected chat AgentProfile 'custom-profile', got %q", chat.AgentProfile)
	}

	foundAgentFlag := false
	for i, arg := range chat.Command {
		if arg == "--agent" {
			foundAgentFlag = true
			if i+1 >= len(chat.Command) || chat.Command[i+1] != "custom-profile" {
				t.Fatalf("expected '--agent custom-profile', got command: %v", chat.Command)
			}
			break
		}
	}
	if !foundAgentFlag {
		t.Fatalf("expected command to contain '--agent custom-profile', got: %v", chat.Command)
	}
}

func TestIssueOrchestrator_ExactTreeLeaseAcquisitionBeforeSpawn(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	origAcquire := acquireTreeLeaseFunc
	origRelease := releaseTreeLeaseFunc
	origStart := startDispatchWorkerFunc
	defer func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
		acquireTreeLeaseFunc = origAcquire
		releaseTreeLeaseFunc = origRelease
		startDispatchWorkerFunc = origStart
	}()

	const testRev = "532688a0a04ba669a20d2c7f353150d044ae8be8"
	controllerStampFunc = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: testRev, HasVCS: true}
	}
	controllerHeadRevFunc = func(string) string {
		return testRev
	}

	tempDir := t.TempDir()
	logDir := filepath.Join(tempDir, "logs")

	var acquiredLane string
	var acquiredPaths []string
	acquireTreeLeaseFunc = func(workspace, lane string, paths []string, pid int) error {
		acquiredLane = lane
		acquiredPaths = append([]string(nil), paths...)
		return nil
	}

	startDispatchWorkerFunc = func(cmd *exec.Cmd) error {
		dummy := exec.Command("cmd.exe", "/c", "exit 0")
		if err := dummy.Start(); err != nil {
			return err
		}
		cmd.Process = dummy.Process
		return nil
	}

	issues := []issueorchestrator.Issue{
		{
			Number:          1201,
			Key:             "issue-1201",
			Title:           "Exact tree lease acquisition test",
			Lane:            "compute",
			Paths:           []string{"internal/compute/leaf.go"},
			ExpectedSteps:   1,
			Dispatchability: "dispatchable",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--spawn-opencode",
		"--worktree=false",
		"--workspace", tempDir,
		"--log-dir", logDir,
		"--supervise=false",
		"--json",
	})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr: %s", code, stderr.String())
	}

	if acquiredLane != "compute" {
		t.Errorf("expected acquired lane 'compute', got %q", acquiredLane)
	}
	if !reflect.DeepEqual(acquiredPaths, []string{"internal/compute/leaf.go"}) {
		t.Errorf("expected acquired paths [internal/compute/leaf.go], got %v", acquiredPaths)
	}

	var receipt OpencodeSpawnReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode receipt: %v; raw: %s", err, stdout.String())
	}
	if !receipt.TreeLeaseAcquired {
		t.Errorf("expected receipt TreeLeaseAcquired to be true")
	}
	if !reflect.DeepEqual(receipt.ExactPaths, []string{"internal/compute/leaf.go"}) {
		t.Errorf("expected receipt ExactPaths [internal/compute/leaf.go], got %v", receipt.ExactPaths)
	}
	if len(receipt.Chats) != 1 {
		t.Fatalf("expected 1 chat, got %d", len(receipt.Chats))
	}
	if !receipt.Chats[0].TreeLeaseAcquired {
		t.Errorf("expected chat TreeLeaseAcquired to be true")
	}
	if !reflect.DeepEqual(receipt.Chats[0].ExactPaths, []string{"internal/compute/leaf.go"}) {
		t.Errorf("expected chat ExactPaths [internal/compute/leaf.go], got %v", receipt.Chats[0].ExactPaths)
	}
}

func TestIssueOrchestrator_OverlapRefusal(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	origDiscover := discoverHeldLeasesFunc
	defer func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
		discoverHeldLeasesFunc = origDiscover
	}()

	const testRev = "532688a0a04ba669a20d2c7f353150d044ae8be8"
	controllerStampFunc = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: testRev, HasVCS: true}
	}
	controllerHeadRevFunc = func(string) string {
		return testRev
	}

	tempDir := t.TempDir()

	// Mock held lease on internal/compute/a.go
	discoverHeldLeasesFunc = func(workspace string) ([]debtlane.HeldLease, error) {
		return []debtlane.HeldLease{
			{Lane: "compute", Tree: []string{"internal/compute/a.go"}, PID: 1234},
		}, nil
	}

	issues := []issueorchestrator.Issue{
		{
			Number:          1202,
			Key:             "issue-1202",
			Title:           "Overlap refusal test",
			Lane:            "compute",
			Paths:           []string{"internal/compute/a.go"},
			ExpectedSteps:   1,
			Dispatchability: "dispatchable",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--spawn-opencode",
		"--worktree=false",
		"--workspace", tempDir,
		"--dry-run",
		"--json",
	})
	if code == 0 {
		t.Fatalf("expected non-zero exit code due to overlap refusal, got 0")
	}

	var receipt OpencodeSpawnReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode receipt: %v; raw: %s", err, stdout.String())
	}
	if len(receipt.Chats) != 1 {
		t.Fatalf("expected 1 chat in receipt, got %d", len(receipt.Chats))
	}
	chat := receipt.Chats[0]
	if chat.Status != "error" {
		t.Errorf("expected status 'error', got %q", chat.Status)
	}
	if !strings.Contains(chat.Error, "EXACT_TREE_OVERLAP_REFUSAL") {
		t.Errorf("expected error to contain 'EXACT_TREE_OVERLAP_REFUSAL', got %q", chat.Error)
	}
}

func TestIssueOrchestrator_CleanupOnSpawnFailure(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	origAcquire := acquireTreeLeaseFunc
	origRelease := releaseTreeLeaseFunc
	origStart := startDispatchWorkerFunc
	defer func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
		acquireTreeLeaseFunc = origAcquire
		releaseTreeLeaseFunc = origRelease
		startDispatchWorkerFunc = origStart
	}()

	const testRev = "532688a0a04ba669a20d2c7f353150d044ae8be8"
	controllerStampFunc = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: testRev, HasVCS: true}
	}
	controllerHeadRevFunc = func(string) string {
		return testRev
	}

	tempDir := t.TempDir()
	logDir := filepath.Join(tempDir, "logs")

	leaseAcquired := false
	acquireTreeLeaseFunc = func(workspace, lane string, paths []string, pid int) error {
		leaseAcquired = true
		return nil
	}

	leaseReleased := false
	var releasedPaths []string
	releaseTreeLeaseFunc = func(workspace, lane string, paths []string, pid int) error {
		leaseReleased = true
		releasedPaths = append([]string(nil), paths...)
		return nil
	}

	// Mock start worker to fail
	startDispatchWorkerFunc = func(cmd *exec.Cmd) error {
		return fmt.Errorf("simulated process start failure")
	}

	issues := []issueorchestrator.Issue{
		{
			Number:          1203,
			Key:             "issue-1203",
			Title:           "Spawn failure cleanup test",
			Lane:            "compute",
			Paths:           []string{"internal/compute/fail.go"},
			ExpectedSteps:   1,
			Dispatchability: "dispatchable",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--spawn-opencode",
		"--worktree=false",
		"--workspace", tempDir,
		"--log-dir", logDir,
		"--supervise=false",
		"--json",
	})
	if code == 0 {
		t.Fatalf("expected non-zero exit code on spawn failure, got 0")
	}

	if !leaseAcquired {
		t.Errorf("expected lease to be acquired before spawn")
	}
	if !leaseReleased {
		t.Errorf("expected lease to be released after spawn failure")
	}
	if !reflect.DeepEqual(releasedPaths, []string{"internal/compute/fail.go"}) {
		t.Errorf("expected released paths [internal/compute/fail.go], got %v", releasedPaths)
	}

	var receipt OpencodeSpawnReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode receipt: %v; raw: %s", err, stdout.String())
	}
	if len(receipt.Chats) != 1 {
		t.Fatalf("expected 1 chat in receipt, got %d", len(receipt.Chats))
	}
	if receipt.Chats[0].TreeLeaseAcquired {
		t.Errorf("expected TreeLeaseAcquired to be false after cleanup")
	}
	if receipt.Chats[0].Status != "error" {
		t.Errorf("expected chat status 'error', got %q", receipt.Chats[0].Status)
	}
	if !strings.Contains(receipt.Chats[0].Error, "simulated process start failure") {
		t.Errorf("expected error to contain simulated failure, got %q", receipt.Chats[0].Error)
	}
}

func TestIssueOrchestrator_NarrowedPathRecording(t *testing.T) {
	origStamp := controllerStampFunc
	origHead := controllerHeadRevFunc
	origDiscover := discoverHeldLeasesFunc
	origAcquire := acquireTreeLeaseFunc
	origRelease := releaseTreeLeaseFunc
	origStart := startDispatchWorkerFunc
	defer func() {
		controllerStampFunc = origStamp
		controllerHeadRevFunc = origHead
		discoverHeldLeasesFunc = origDiscover
		acquireTreeLeaseFunc = origAcquire
		releaseTreeLeaseFunc = origRelease
		startDispatchWorkerFunc = origStart
	}()

	const testRev = "532688a0a04ba669a20d2c7f353150d044ae8be8"
	controllerStampFunc = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: testRev, HasVCS: true}
	}
	controllerHeadRevFunc = func(string) string {
		return testRev
	}

	tempDir := t.TempDir()
	logDir := filepath.Join(tempDir, "logs")

	// Held lease specifically on internal/compute/a.go
	discoverHeldLeasesFunc = func(workspace string) ([]debtlane.HeldLease, error) {
		return []debtlane.HeldLease{
			{Lane: "compute", Tree: []string{"internal/compute/a.go"}, PID: 999},
		}, nil
	}

	var acquiredPaths []string
	acquireTreeLeaseFunc = func(workspace, lane string, paths []string, pid int) error {
		acquiredPaths = append([]string(nil), paths...)
		return nil
	}

	startDispatchWorkerFunc = func(cmd *exec.Cmd) error {
		dummy := exec.Command("cmd.exe", "/c", "exit 0")
		if err := dummy.Start(); err != nil {
			return err
		}
		cmd.Process = dummy.Process
		return nil
	}

	// Issue declared paths: both a.go (collides) and b.go (disjoint)
	issues := []issueorchestrator.Issue{
		{
			Number:          1204,
			Key:             "issue-1204",
			Title:           "Narrowed path recording test",
			Lane:            "compute",
			Paths:           []string{"internal/compute/a.go", "internal/compute/b.go"},
			ExpectedSteps:   1,
			Dispatchability: "dispatchable",
		},
	}
	issuesPath := writeTestIssuesFile(t, issues)

	var stdout, stderr bytes.Buffer
	code := runIssueOrchestrator(&stdout, &stderr, []string{
		"--from-issues", issuesPath,
		"--spawn-opencode",
		"--worktree=false",
		"--workspace", tempDir,
		"--log-dir", logDir,
		"--supervise=false",
		"--json",
	})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr: %s", code, stderr.String())
	}

	// Tree lease acquired should be ONLY the narrowed disjoint subset: ["internal/compute/b.go"]
	if !reflect.DeepEqual(acquiredPaths, []string{"internal/compute/b.go"}) {
		t.Errorf("expected acquiredPaths to be narrowed to [internal/compute/b.go], got %v", acquiredPaths)
	}

	var receipt OpencodeSpawnReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode receipt: %v; raw: %s", err, stdout.String())
	}
	if !reflect.DeepEqual(receipt.ExactPaths, []string{"internal/compute/a.go", "internal/compute/b.go"}) {
		t.Errorf("expected receipt ExactPaths [a.go, b.go], got %v", receipt.ExactPaths)
	}
	if !reflect.DeepEqual(receipt.NarrowedPaths, []string{"internal/compute/b.go"}) {
		t.Errorf("expected receipt NarrowedPaths [b.go], got %v", receipt.NarrowedPaths)
	}
	if len(receipt.Chats) != 1 {
		t.Fatalf("expected 1 chat in receipt, got %d", len(receipt.Chats))
	}
	chat := receipt.Chats[0]
	if !reflect.DeepEqual(chat.ExactPaths, []string{"internal/compute/a.go", "internal/compute/b.go"}) {
		t.Errorf("expected chat ExactPaths [a.go, b.go], got %v", chat.ExactPaths)
	}
	if !reflect.DeepEqual(chat.NarrowedPaths, []string{"internal/compute/b.go"}) {
		t.Errorf("expected chat NarrowedPaths [b.go], got %v", chat.NarrowedPaths)
	}
	if chat.Status != "spawned" {
		t.Errorf("expected chat status 'spawned', got %q", chat.Status)
	}
}
