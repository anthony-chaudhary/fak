package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issueorchestrator"
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
