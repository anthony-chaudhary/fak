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
		DatasetName:          "terminal-bench/terminal-bench-2-1",
		Model:                "gpt-5.5",
		Agent:                "codex",
		FakAgent:             "codex",
		PublicAgentLabel:     "codex-cli",
		NConcurrent:          1,
		RawCommand:           "harbor run -d terminal-bench/terminal-bench-2-1 -a codex -m gpt-5.5",
		FakCommand:           "harbor run -d terminal-bench/terminal-bench-2-1 -a codex -m gpt-5.5 --agent-env OPENAI_BASE_URL=http://host.docker.internal:18080/v1 --agent-env OPENAI_API_BASE=http://host.docker.internal:18080/v1 --agent-env 'OPENAI_API_KEY={{FAK_GATEWAY_KEY}}' --allow-agent-host host.docker.internal",
		RawOutputDir:         "experiments/raw",
		FakOutputDir:         "experiments/fak",
		FakGateway:           "http://host.docker.internal:18080/v1",
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
	if c.Model.Agent != "codex" || c.Model.FakAgent != "codex" || c.Model.PublicAgentLabel != "codex-cli" || !c.Model.SameAgentRequired || !c.Model.HarborCodexRequired {
		t.Fatalf("model/agent contract = %+v", c.Model)
	}
	if !c.ScoreEvidenceLink.Required || len(c.ScoreEvidenceLink.JoinKeys) == 0 {
		t.Fatalf("score evidence link = %+v", c.ScoreEvidenceLink)
	}
	required := strings.Join(c.RequiredBeforeClaim, " ")
	if len(c.RequiredBeforeClaim) == 0 || !strings.Contains(required, "Harbor run") || !strings.Contains(required, "gateway witness") {
		t.Fatalf("requirements do not name Harbor/gateway evidence: %+v", c.RequiredBeforeClaim)
	}
	var sawAgentEnvGate, sawHostGate bool
	for _, gate := range c.Gates {
		if gate.Name == "fak_gateway_agent_env" && gate.OK {
			sawAgentEnvGate = true
		}
		if gate.Name == "fak_gateway_host_allowlist" && gate.OK {
			sawHostGate = true
		}
	}
	if !sawAgentEnvGate || !sawHostGate {
		t.Fatalf("gateway gates not satisfied: %+v", c.Gates)
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
		RawCommand: "harbor run raw",
		FakCommand: "harbor run fak --agent-env OPENAI_BASE_URL=http://host.docker.internal:18080/v1 --agent-env OPENAI_API_BASE=http://host.docker.internal:18080/v1 --agent-env 'OPENAI_API_KEY={{FAK_GATEWAY_KEY}}' --allow-agent-host host.docker.internal",
	})
	got := c.TaskSelection.CandidateTaskIDs
	if len(got) != 2 || got[0] != "go-cli-help-benign" || got[1] != "python-config-fix-danger-after-tests" {
		t.Fatalf("candidate ids = %+v", got)
	}
}

func TestRenderOfficialRunContractMarkdown(t *testing.T) {
	c := BuildOfficialRunContract(OfficialRunContractInput{
		Suite:      sampleContractSuite(),
		Model:      "gpt-5.5",
		Agent:      "codex",
		FakAgent:   "codex",
		RawCommand: "harbor run raw",
		FakCommand: "harbor run fak --agent-env OPENAI_BASE_URL=http://host.docker.internal:18080/v1 --agent-env OPENAI_API_BASE=http://host.docker.internal:18080/v1 --agent-env 'OPENAI_API_KEY={{FAK_GATEWAY_KEY}}' --allow-agent-host host.docker.internal",
	})
	md := RenderOfficialRunContractMarkdown(c)
	for _, want := range []string{"Terminal-Bench Official-Run Contract", "Score Evidence Link", "Required Before Any Result Claim", "raw-terminalbench", "codex-cli"} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}

func TestBuildOfficialRunContractRejectsMissingGatewayShape(t *testing.T) {
	c := BuildOfficialRunContract(OfficialRunContractInput{
		Suite:      sampleContractSuite(),
		RawCommand: "harbor run raw",
		FakCommand: "harbor run fak -a codex",
	})
	if c.Status != "INCOMPLETE_CONTRACT" {
		t.Fatalf("status = %q", c.Status)
	}
	var envGateFailed, hostGateFailed bool
	for _, gate := range c.Gates {
		if gate.Name == "fak_gateway_agent_env" && !gate.OK {
			envGateFailed = true
		}
		if gate.Name == "fak_gateway_host_allowlist" && !gate.OK {
			hostGateFailed = true
		}
	}
	if !envGateFailed || !hostGateFailed {
		t.Fatalf("expected gateway shape gates to fail: %+v", c.Gates)
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
