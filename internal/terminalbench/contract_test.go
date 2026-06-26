package terminalbench

import (
	"strings"
	"testing"
)

func TestBuildOfficialRunContractKeepsResultGated(t *testing.T) {
	c := BuildOfficialRunContract(OfficialRunContractInput{
		GeneratedAt:          "2026-06-26T00:00:00Z",
		Suite:                sampleContractSuite(),
		SuitePath:            "testdata/terminalbench/command_boundary_smoke.json",
		LocalFixtureArtifact: "experiments/agent-live/terminalbench-command-boundary-smoke-20260625.json",
		DatasetName:          "terminal-bench-core",
		DatasetVersion:       "0.1.1",
		Model:                "gpt-4.1",
		Agent:                "terminus",
		FakAgent:             "terminus-through-fak",
		NConcurrent:          1,
		RawCommand:           "tb run raw",
		FakCommand:           "tb run fak",
		RawOutputDir:         "experiments/raw",
		FakOutputDir:         "experiments/fak",
		FakGateway:           "http://localhost:8080/v1",
	})
	if c.Schema != OfficialRunContractSchema {
		t.Fatalf("schema = %q", c.Schema)
	}
	if c.Status != "READY_FOR_EXTERNAL_HARNESS" {
		t.Fatalf("status = %q", c.Status)
	}
	if c.EvidenceClass != "EXTERNAL_RUN_CONTRACT" {
		t.Fatalf("evidence class = %q", c.EvidenceClass)
	}
	if c.ResultClaimAllowed {
		t.Fatal("official-run contract must not allow a result claim")
	}
	if len(c.TaskSelection.CandidateTaskIDs) != 2 || !c.TaskSelection.OfficialTaskIDsRequired || !c.TaskSelection.SameImageRequired {
		t.Fatalf("task selection = %+v", c.TaskSelection)
	}
	if len(c.RequiredBeforeClaim) == 0 || !strings.Contains(strings.Join(c.RequiredBeforeClaim, " "), "tb run") {
		t.Fatalf("requirements do not name Terminal-Bench evidence: %+v", c.RequiredBeforeClaim)
	}
}

func TestBuildOfficialRunContractIncompleteWithoutCommands(t *testing.T) {
	c := BuildOfficialRunContract(OfficialRunContractInput{Suite: sampleContractSuite()})
	if c.Status != "INCOMPLETE_CONTRACT" {
		t.Fatalf("status = %q", c.Status)
	}
	var missingRaw bool
	for _, gate := range c.Gates {
		if gate.Name == "raw_arm_command" && !gate.OK {
			missingRaw = true
		}
	}
	if !missingRaw {
		t.Fatalf("missing raw command gate not recorded: %+v", c.Gates)
	}
}

func TestBuildOfficialRunContractSortsCandidates(t *testing.T) {
	c := BuildOfficialRunContract(OfficialRunContractInput{
		Suite:      sampleContractSuite(),
		RawCommand: "raw",
		FakCommand: "fak",
	})
	got := c.TaskSelection.CandidateTaskIDs
	if len(got) != 2 || got[0] != "go-cli-help-benign" || got[1] != "python-config-fix-danger-after-tests" {
		t.Fatalf("candidate ids = %+v", got)
	}
}

func TestRenderOfficialRunContractMarkdown(t *testing.T) {
	c := BuildOfficialRunContract(OfficialRunContractInput{
		Suite:      sampleContractSuite(),
		Model:      "gpt-4.1",
		Agent:      "terminus",
		RawCommand: "raw",
		FakCommand: "fak",
	})
	md := RenderOfficialRunContractMarkdown(c)
	for _, want := range []string{"Terminal-Bench Official-Run Contract", "Required Before Any Result Claim", "raw-terminalbench"} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}

func sampleContractSuite() Suite {
	return Suite{
		Benchmark: "terminal-bench-command-smoke",
		Tasks: []Task{
			{ID: "python-config-fix-danger-after-tests", Image: "python:3.12-slim", TestOracle: "pytest", BudgetTurns: 4},
			{ID: "go-cli-help-benign", Image: "golang:1.26", TestOracle: "go-test", BudgetTurns: 2},
		},
	}
}
