package terminalbench

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildOfficialRunContractKeepsResultGated(t *testing.T) {
	c := BuildOfficialRunContract(OfficialRunContractInput{
		GeneratedAt:          "2026-06-26T00:00:00Z",
		Suite:                sampleContractSuite(),
		SuitePath:            "testdata/terminalbench/command_boundary_smoke.json",
		LocalFixtureArtifact: "experiments/agent-live/terminalbench-command-boundary-smoke-20260625.json",
		DatasetName:          OfficialTerminalBench21Dataset,
		Model:                OfficialTerminalBench21TopAgentModel,
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
	if c.Model.Agent != "codex" || c.Model.FakAgent != "codex" || c.Model.PublicAgentLabel != "codex-cli" || c.Model.Model != OfficialTerminalBench21TopAgentModel || c.Model.TopAgentModel != OfficialTerminalBench21TopAgentModel || !c.Model.SameAgentRequired || !c.Model.HarborCodexRequired {
		t.Fatalf("model/agent contract = %+v", c.Model)
	}
	if !c.ScoreEvidenceLink.Required || len(c.ScoreEvidenceLink.JoinKeys) == 0 {
		t.Fatalf("score evidence link = %+v", c.ScoreEvidenceLink)
	}
	required := strings.Join(c.RequiredBeforeClaim, " ")
	if len(c.RequiredBeforeClaim) == 0 || !strings.Contains(required, "Harbor run") || !strings.Contains(required, "gateway witness") {
		t.Fatalf("requirements do not name Harbor/gateway evidence: %+v", c.RequiredBeforeClaim)
	}
	var sawAgentEnvGate, sawHostGate, sawDatasetGate, sawTopModelGate bool
	for _, gate := range c.Gates {
		if gate.Name == "fak_gateway_agent_env" && gate.OK {
			sawAgentEnvGate = true
		}
		if gate.Name == "fak_gateway_host_allowlist" && gate.OK {
			sawHostGate = true
		}
		if gate.Name == "official_dataset_pin" && gate.OK && gate.Detail == OfficialTerminalBench21Dataset {
			sawDatasetGate = true
		}
		if gate.Name == "top_agent_model_current" && gate.OK && gate.Detail == OfficialTerminalBench21TopAgentModel {
			sawTopModelGate = true
		}
	}
	if !sawAgentEnvGate || !sawHostGate || !sawDatasetGate || !sawTopModelGate {
		t.Fatalf("submission gates not satisfied: %+v", c.Gates)
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

func TestBuildOfficialRunContractRejectsUnpinnedDataset(t *testing.T) {
	tests := []struct {
		name           string
		datasetName    string
		datasetVersion string
	}{
		{
			name:        "wrong dataset",
			datasetName: "terminal-bench/terminal-bench-2-0",
		},
		{
			name:           "version suffix",
			datasetName:    OfficialTerminalBench21Dataset,
			datasetVersion: "latest",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := BuildOfficialRunContract(OfficialRunContractInput{
				Suite:          sampleContractSuite(),
				DatasetName:    tt.datasetName,
				DatasetVersion: tt.datasetVersion,
				Model:          OfficialTerminalBench21TopAgentModel,
				Agent:          "codex",
				FakAgent:       "codex",
				RawCommand:     "harbor run raw",
				FakCommand:     "harbor run fak --agent-env OPENAI_BASE_URL=http://host.docker.internal:18080/v1 --agent-env OPENAI_API_BASE=http://host.docker.internal:18080/v1 --agent-env 'OPENAI_API_KEY={{FAK_GATEWAY_KEY}}' --allow-agent-host host.docker.internal",
			})
			if c.Status != "INCOMPLETE_CONTRACT" {
				t.Fatalf("status = %q", c.Status)
			}
			var sawFailedDatasetGate bool
			for _, gate := range c.Gates {
				if gate.Name == "official_dataset_pin" && !gate.OK {
					sawFailedDatasetGate = true
				}
			}
			if !sawFailedDatasetGate {
				t.Fatalf("expected official dataset pin gate to fail: %+v", c.Gates)
			}
		})
	}
}

func TestOfficialRunContractArtifactPinsSubmissionGates(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "experiments", "agent-live", "terminalbench-official-run-contract-20260626.json"))
	if err != nil {
		t.Fatal(err)
	}
	var c OfficialRunContract
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	if c.Status != "READY_FOR_EXTERNAL_HARNESS" {
		t.Fatalf("artifact status = %q", c.Status)
	}
	if c.ResultClaimAllowed {
		t.Fatal("artifact must not allow a result claim")
	}
	if c.TaskSelection.OfficialDataset != OfficialTerminalBench21Dataset || c.TaskSelection.OfficialDatasetVersion != "" {
		t.Fatalf("artifact dataset pin = %q version %q", c.TaskSelection.OfficialDataset, c.TaskSelection.OfficialDatasetVersion)
	}
	if c.Model.Model != OfficialTerminalBench21TopAgentModel || c.Model.TopAgentModel != OfficialTerminalBench21TopAgentModel {
		t.Fatalf("artifact model pin = %+v", c.Model)
	}
	for _, arm := range c.Arms {
		if !strings.Contains(arm.Command, "-d "+OfficialTerminalBench21Dataset) {
			t.Fatalf("artifact arm %q command does not target %s:\n%s", arm.Name, OfficialTerminalBench21Dataset, arm.Command)
		}
	}
	requiredGates := map[string]bool{
		"official_dataset_pin":      false,
		"official_harness_required": false,
		"top_agent_model_current":   false,
	}
	for _, gate := range c.Gates {
		if _, ok := requiredGates[gate.Name]; ok && gate.OK {
			requiredGates[gate.Name] = true
		}
	}
	for name, ok := range requiredGates {
		if !ok {
			t.Fatalf("artifact missing ready gate %q in %+v", name, c.Gates)
		}
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
	for _, want := range []string{"Terminal-Bench Official-Run Contract", "Score Evidence Link", "Required Before Any Result Claim", "raw-terminalbench", "codex-cli", "Terminal-Bench 2.1 top-agent model"} {
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
