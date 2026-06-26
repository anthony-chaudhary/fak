package browseraction

import (
	"strings"
	"testing"
)

func TestBuildOfficialRunContractKeepsResultGated(t *testing.T) {
	c := BuildOfficialRunContract(OfficialRunContractInput{
		GeneratedAt:          "2026-06-26T00:00:00Z",
		Suite:                sampleContractSuite(),
		SuitePath:            "testdata/webbench/action_mediation_smoke.json",
		LocalFixtureArtifact: "experiments/agent-live/browser-action-mediation-smoke-20260625.json",
		Harness:              "BrowserGym/AgentLab",
		Benchmark:            "WebArena",
		Model:                "gpt-4.1",
		Agent:                "browsergym-agent",
		FakAgent:             "browsergym-agent-through-fak",
		MaxSteps:             30,
		RawCommand:           "agentlab raw",
		FakCommand:           "agentlab fak",
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
	if len(c.TaskSelection.CandidateTaskIDs) != 2 || !c.TaskSelection.OfficialTaskIDsRequired || !c.TaskSelection.SameBrowserStateRequired {
		t.Fatalf("task selection = %+v", c.TaskSelection)
	}
	joined := strings.Join(c.RequiredBeforeClaim, " ")
	if !strings.Contains(joined, "browser state") || !strings.Contains(joined, "model perception") {
		t.Fatalf("requirements do not name browser evidence and failure split: %+v", c.RequiredBeforeClaim)
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
	if len(got) != 2 || got[0] != "knowledgebase-search-benign" || got[1] != "shopping-address-delete-minefield" {
		t.Fatalf("candidate ids = %+v", got)
	}
}

func TestRenderOfficialRunContractMarkdown(t *testing.T) {
	c := BuildOfficialRunContract(OfficialRunContractInput{
		Suite:      sampleContractSuite(),
		Model:      "gpt-4.1",
		Agent:      "browsergym-agent",
		RawCommand: "raw",
		FakCommand: "fak",
	})
	md := RenderOfficialRunContractMarkdown(c)
	for _, want := range []string{"Browser Action Official-Run Contract", "Required Before Any Result Claim", "raw-browseraction"} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}

func sampleContractSuite() ActionMediationSuite {
	return ActionMediationSuite{
		Benchmark: "browser-action-mediation-smoke",
		Tasks: []ActionMediationTask{
			{ID: "shopping-address-delete-minefield", Benchmark: "browser-agent", Domain: "shop.example", SourceURL: "https://shop.example/account", BudgetTurns: 4},
			{ID: "knowledgebase-search-benign", Benchmark: "browser-agent", Domain: "help.example", SourceURL: "https://help.example", BudgetTurns: 3},
		},
	}
}
