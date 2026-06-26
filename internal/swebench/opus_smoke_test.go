package swebench

import (
	"strings"
	"testing"
)

func TestBuildOpusSmokeContractGatesRemoteGrading(t *testing.T) {
	c := BuildOpusSmokeContract(OpusSmokeContractInput{
		GeneratedAt:  "2026-06-26T00:00:00Z",
		Dataset:      sampleDataset(),
		Source:       "testdata/swebench_smoke.json",
		Filter:       "smoke",
		Model:        "claude-opus-4-8",
		RawCommand:   "raw-opus-runner --tasks task_ids.txt",
		FakCommand:   "go run ./cmd/fak swebench run --agent fleet",
		RawOutputDir: "experiments/raw",
		FakOutputDir: "experiments/fak",
		EvalCapability: EvalCapability{
			DockerPresent:  false,
			HarnessPresent: false,
			Runnable:       false,
			Reason:         "no Docker",
		},
	})
	if c.Schema != OpusSmokeContractSchema {
		t.Fatalf("schema = %q", c.Schema)
	}
	if c.Status != "READY_FOR_REMOTE_GRADING" {
		t.Fatalf("status = %q", c.Status)
	}
	if c.EvidenceClass != "EXTERNAL_RUN_CONTRACT" {
		t.Fatalf("evidence class = %q", c.EvidenceClass)
	}
	if len(c.TaskSelection.TaskIDs) != 2 || !c.TaskSelection.SameTaskIDs {
		t.Fatalf("task selection = %+v", c.TaskSelection)
	}
	if c.ResultClaimAllowed {
		t.Fatal("pre-run contract must not allow a result claim")
	}
	if c.Arms[0].EvalCommand == "" || c.Arms[1].EvalCommand == "" {
		t.Fatalf("missing eval commands: %+v", c.Arms)
	}
	if !c.CompareEvidenceLink.Required ||
		len(c.CompareEvidenceLink.Predictions) != 2 ||
		len(c.CompareEvidenceLink.OfficialEval) != 2 ||
		len(c.CompareEvidenceLink.FakEvidence) == 0 ||
		len(c.CompareEvidenceLink.JoinKeys) == 0 {
		t.Fatalf("compare evidence link = %+v", c.CompareEvidenceLink)
	}
}

func TestBuildOpusSmokeContractIncomplete(t *testing.T) {
	c := BuildOpusSmokeContract(OpusSmokeContractInput{Dataset: sampleDataset()})
	if c.Status != "INCOMPLETE_CONTRACT" {
		t.Fatalf("status = %q", c.Status)
	}
	var sawMissingModel bool
	for _, gate := range c.Gates {
		if gate.Name == "same_model" && !gate.OK {
			sawMissingModel = true
		}
	}
	if !sawMissingModel {
		t.Fatalf("missing model gate not recorded: %+v", c.Gates)
	}
}

func TestRenderOpusSmokeContractMarkdown(t *testing.T) {
	c := BuildOpusSmokeContract(OpusSmokeContractInput{
		Dataset:        sampleDataset(),
		Model:          "claude-opus-4-8",
		RawCommand:     "raw",
		FakCommand:     "fak",
		EvalCapability: EvalCapability{Runnable: true},
	})
	md := RenderOpusSmokeContractMarkdown(c)
	for _, want := range []string{"Opus SWE-bench Smoke Contract", "Compare Evidence Link", "Required Before Any Result Claim", "same_model"} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}
