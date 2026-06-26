package toolsandbox

import (
	"strings"
	"testing"
)

func TestBuildOfficialRunContractKeepsResultGated(t *testing.T) {
	c := BuildOfficialRunContract(OfficialRunContractInput{
		GeneratedAt:          "2026-06-26T00:00:00Z",
		Suite:                sampleContractSuite(),
		SuitePath:            "testdata/toolsandbox/policy_state_smoke.json",
		LocalFixtureArtifact: "experiments/agent-live/toolsandbox-policy-state-smoke-20260625.json",
		OfficialHarness:      "tau3",
		Domain:               "retail",
		Model:                "gpt-4.1",
		UserModel:            "gpt-4.1",
		Trials:               1,
		RawCommand:           "tau2 run raw",
		FakCommand:           "tau2 run fak",
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
	if len(c.TaskSelection.CandidateTaskIDs) != 2 || !c.TaskSelection.OfficialTaskIDsRequired {
		t.Fatalf("task selection = %+v", c.TaskSelection)
	}
	if !c.ScoreEvidenceLink.Required || len(c.ScoreEvidenceLink.JoinKeys) == 0 {
		t.Fatalf("score evidence link = %+v", c.ScoreEvidenceLink)
	}
	if len(c.RequiredBeforeClaim) == 0 || !strings.Contains(strings.Join(c.RequiredBeforeClaim, " "), "benchmark-native") {
		t.Fatalf("requirements do not name benchmark-native evidence: %+v", c.RequiredBeforeClaim)
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

func TestBuildOfficialRunContractFiltersCandidateDomain(t *testing.T) {
	c := BuildOfficialRunContract(OfficialRunContractInput{
		Suite:      sampleDomainSuite(),
		Domain:     "retail",
		RawCommand: "raw",
		FakCommand: "fak",
	})
	if got := c.TaskSelection.CandidateTaskIDs; len(got) != 1 || got[0] != "retail-refund-policy-minefield" {
		t.Fatalf("candidate ids = %+v", got)
	}
}

func TestRenderOfficialRunContractMarkdown(t *testing.T) {
	c := BuildOfficialRunContract(OfficialRunContractInput{
		Suite:      sampleContractSuite(),
		Model:      "gpt-4.1",
		UserModel:  "gpt-4.1",
		RawCommand: "raw",
		FakCommand: "fak",
	})
	md := RenderOfficialRunContractMarkdown(c)
	for _, want := range []string{"ToolSandbox/tau3 Official-Run Contract", "Score Evidence Link", "Required Before Any Result Claim", "raw-toolsandbox"} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}

func sampleContractSuite() Suite {
	return Suite{
		Benchmark: "toolsandbox-smoke",
		Tasks: []Task{
			{ID: "retail-refund-policy-minefield"},
			{ID: "banking-address-update-benign"},
		},
	}
}

func sampleDomainSuite() Suite {
	return Suite{
		Benchmark: "toolsandbox-smoke",
		Tasks: []Task{
			{ID: "retail-refund-policy-minefield", Domain: "retail"},
			{ID: "banking-address-update-benign", Domain: "banking"},
		},
	}
}
