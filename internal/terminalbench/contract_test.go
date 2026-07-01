package terminalbench

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const officialRawHarborCommand = "harbor run -d " + OfficialTerminalBench21Dataset + " -a codex -m " + OfficialTerminalBench21TopAgentModel

const officialFakHarborCommand = "harbor run -d " + OfficialTerminalBench21Dataset + " -a codex -m " + OfficialTerminalBench21TopAgentModel + " --agent-env OPENAI_BASE_URL=http://host.docker.internal:18080/v1 --agent-env OPENAI_API_BASE=http://host.docker.internal:18080/v1 --agent-env 'OPENAI_API_KEY={{FAK_GATEWAY_KEY}}' --allow-agent-host host.docker.internal"

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
		RawCommand:           officialRawHarborCommand,
		FakCommand:           officialFakHarborCommand,
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
	if c.TaskSelection.OfficialDataset != OfficialTerminalBench21Dataset || c.TaskSelection.OfficialDatasetVersion != "" {
		t.Fatalf("dataset pin = %q version %q", c.TaskSelection.OfficialDataset, c.TaskSelection.OfficialDatasetVersion)
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
	if !hasString(c.ScoreEvidenceLink.FakCommandEvidenceFiles, "experiments/fak/fak-gateway-witness.json") {
		t.Fatalf("gateway witness artifact missing from score evidence link: %+v", c.ScoreEvidenceLink.FakCommandEvidenceFiles)
	}
	if !armRequiredArtifactContains(c.Arms, "fak-terminalbench", "fak gateway log witness") {
		t.Fatalf("fak arm missing required gateway log witness: %+v", c.Arms)
	}
	required := strings.Join(c.RequiredBeforeClaim, " ")
	if len(c.RequiredBeforeClaim) == 0 || !strings.Contains(required, "Harbor run") || !strings.Contains(required, "gateway witness") {
		t.Fatalf("requirements do not name Harbor/gateway evidence: %+v", c.RequiredBeforeClaim)
	}
	var sawAgentEnvGate, sawHostGate, sawDatasetGate, sawRawDatasetCommandGate, sawFakDatasetCommandGate, sawTopModelGate bool
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
		if gate.Name == "raw_arm_official_dataset" && gate.OK {
			sawRawDatasetCommandGate = true
		}
		if gate.Name == "fak_arm_official_dataset" && gate.OK {
			sawFakDatasetCommandGate = true
		}
		if gate.Name == "top_agent_model_current" && gate.OK && gate.Detail == OfficialTerminalBench21TopAgentModel {
			sawTopModelGate = true
		}
	}
	if !sawAgentEnvGate || !sawHostGate || !sawDatasetGate || !sawRawDatasetCommandGate || !sawFakDatasetCommandGate || !sawTopModelGate {
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
				RawCommand:     officialRawHarborCommand,
				FakCommand:     officialFakHarborCommand,
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

func TestBuildOfficialRunContractAcceptsVersionlessHarborDatasetID(t *testing.T) {
	c := BuildOfficialRunContract(OfficialRunContractInput{
		Suite:          sampleContractSuite(),
		DatasetName:    OfficialTerminalBench21Dataset,
		DatasetVersion: "",
		Model:          OfficialTerminalBench21TopAgentModel,
		Agent:          "codex",
		FakAgent:       "codex",
		RawCommand:     "harbor run --dataset=" + OfficialTerminalBench21Dataset + " -a codex -m " + OfficialTerminalBench21TopAgentModel,
		FakCommand:     "harbor run --dataset " + OfficialTerminalBench21Dataset + " -a codex -m " + OfficialTerminalBench21TopAgentModel + " --agent-env OPENAI_BASE_URL=http://host.docker.internal:18080/v1 --agent-env OPENAI_API_BASE=http://host.docker.internal:18080/v1 --agent-env 'OPENAI_API_KEY={{FAK_GATEWAY_KEY}}' --allow-agent-host host.docker.internal",
	})
	if c.Status != "READY_FOR_EXTERNAL_HARNESS" {
		t.Fatalf("status = %q gates=%+v", c.Status, c.Gates)
	}
	if !gateOK(c.Gates, "official_dataset_pin") || !gateOK(c.Gates, "raw_arm_official_dataset") || !gateOK(c.Gates, "fak_arm_official_dataset") {
		t.Fatalf("versionless dataset gates not satisfied: %+v", c.Gates)
	}
}

func TestBuildOfficialRunContractRejectsCommandsWithoutOfficialDataset(t *testing.T) {
	tests := []struct {
		name       string
		rawCommand string
		fakCommand string
		wantGate   string
	}{
		{
			name:       "raw command missing dataset",
			rawCommand: "harbor run -a codex -m " + OfficialTerminalBench21TopAgentModel,
			fakCommand: officialFakHarborCommand,
			wantGate:   "raw_arm_official_dataset",
		},
		{
			name:       "raw command wrong dataset",
			rawCommand: "harbor run -d terminal-bench/terminal-bench-2-0 -a codex -m " + OfficialTerminalBench21TopAgentModel,
			fakCommand: officialFakHarborCommand,
			wantGate:   "raw_arm_official_dataset",
		},
		{
			name:       "fak command missing dataset",
			rawCommand: officialRawHarborCommand,
			fakCommand: "harbor run -a codex -m " + OfficialTerminalBench21TopAgentModel + " --agent-env OPENAI_BASE_URL=http://host.docker.internal:18080/v1 --agent-env OPENAI_API_BASE=http://host.docker.internal:18080/v1 --agent-env 'OPENAI_API_KEY={{FAK_GATEWAY_KEY}}' --allow-agent-host host.docker.internal",
			wantGate:   "fak_arm_official_dataset",
		},
		{
			name:       "fak command versioned dataset",
			rawCommand: officialRawHarborCommand,
			fakCommand: "harbor run --dataset=" + OfficialTerminalBench21Dataset + ":latest -a codex -m " + OfficialTerminalBench21TopAgentModel + " --agent-env OPENAI_BASE_URL=http://host.docker.internal:18080/v1 --agent-env OPENAI_API_BASE=http://host.docker.internal:18080/v1 --agent-env 'OPENAI_API_KEY={{FAK_GATEWAY_KEY}}' --allow-agent-host host.docker.internal",
			wantGate:   "fak_arm_official_dataset",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := BuildOfficialRunContract(OfficialRunContractInput{
				Suite:       sampleContractSuite(),
				DatasetName: OfficialTerminalBench21Dataset,
				Model:       OfficialTerminalBench21TopAgentModel,
				Agent:       "codex",
				FakAgent:    "codex",
				RawCommand:  tt.rawCommand,
				FakCommand:  tt.fakCommand,
			})
			if c.Status != "INCOMPLETE_CONTRACT" {
				t.Fatalf("status = %q", c.Status)
			}
			if !gateFailed(c.Gates, tt.wantGate) {
				t.Fatalf("expected %s to fail: %+v", tt.wantGate, c.Gates)
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
		RawCommand: officialRawHarborCommand,
		FakCommand: officialFakHarborCommand,
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
		RawCommand: officialRawHarborCommand,
		FakCommand: officialFakHarborCommand,
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
		RawCommand: officialRawHarborCommand,
		FakCommand: "harbor run -d " + OfficialTerminalBench21Dataset + " -a codex",
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

func gateOK(gates []ContractGate, name string) bool {
	for _, gate := range gates {
		if gate.Name == name {
			return gate.OK
		}
	}
	return false
}

func gateFailed(gates []ContractGate, name string) bool {
	for _, gate := range gates {
		if gate.Name == name {
			return !gate.OK
		}
	}
	return false
}

func hasString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func armRequiredArtifactContains(arms []ContractArm, armName, want string) bool {
	for _, arm := range arms {
		if arm.Name != armName {
			continue
		}
		for _, artifact := range arm.RequiredArtifacts {
			if strings.Contains(artifact, want) {
				return true
			}
		}
	}
	return false
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
