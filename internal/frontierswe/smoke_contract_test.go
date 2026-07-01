package frontierswe

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildRawFakContractGatesResultClaims(t *testing.T) {
	task, err := LoadTask(filepath.Join(fixtureDir, "git-to-zig"))
	if err != nil {
		t.Fatalf("LoadTask: %v", err)
	}
	c := BuildRawFakContract(RawFakContractInput{
		GeneratedAt:  "2026-07-01T00:00:00Z",
		Task:         task,
		Source:       fixtureDir,
		Model:        "claude-opus-4-6",
		Agent:        "claude-code",
		RawBaseURL:   "$env:RAW_FRONTIERSWE_BASE_URL",
		FakBaseURL:   "http://127.0.0.1:8080/v1",
		RawOutputDir: "experiments/frontierswe/raw",
		FakOutputDir: "experiments/frontierswe/fak",
		EvalCapability: FrontierEvalCapability{
			DockerPresent: false,
			ModalPresent:  false,
			Runnable:      false,
			Reason:        "no Docker and no Modal CLI",
		},
	})

	if c.Schema != RawFakContractSchema {
		t.Fatalf("schema = %q", c.Schema)
	}
	if c.Status != "READY_FOR_EXTERNAL_RUN" {
		t.Fatalf("status = %q", c.Status)
	}
	if c.EvidenceClass != "EXTERNAL_RUN_CONTRACT" {
		t.Fatalf("evidence class = %q", c.EvidenceClass)
	}
	if c.ResultClaimAllowed {
		t.Fatal("pre-run contract must not allow a FrontierSWE score or TTS claim")
	}
	if c.TaskSelection.Task != "git-to-zig" || !c.TaskSelection.SameTask {
		t.Fatalf("task selection = %+v", c.TaskSelection)
	}
	if !c.Model.SameModel || !c.Model.SameAgent || !c.Budget.SameBudget {
		t.Fatalf("same model/agent/budget flags not set: model=%+v budget=%+v", c.Model, c.Budget)
	}
	if len(c.Arms) != 2 || c.Arms[0].Reward == "" || c.Arms[1].TTSTrace == "" {
		t.Fatalf("arms missing reward/TTS artifacts: %+v", c.Arms)
	}
	if !c.CompareEvidenceLink.Required || len(c.CompareEvidenceLink.JoinKeys) == 0 {
		t.Fatalf("compare evidence link = %+v", c.CompareEvidenceLink)
	}
	assertGate(t, c.Gates, "score_parity_gate_declared", true)
	assertGate(t, c.Gates, "tts_metric_declared", true)
	assertGate(t, c.Gates, "official_grader_local", false)
	for _, want := range []string{"ScoreParity(", "TTS metric", "official FrontierSWE scorer"} {
		if !containsContractRequirement(c.RequiredBeforeClaim, want) {
			t.Fatalf("RequiredBeforeClaim missing %q: %#v", want, c.RequiredBeforeClaim)
		}
	}
}

func TestBuildRawFakContractIncomplete(t *testing.T) {
	task, err := LoadTask(filepath.Join(fixtureDir, "git-to-zig"))
	if err != nil {
		t.Fatalf("LoadTask: %v", err)
	}
	c := BuildRawFakContract(RawFakContractInput{Task: task})
	if c.Status != "INCOMPLETE_CONTRACT" {
		t.Fatalf("status = %q", c.Status)
	}
	assertGate(t, c.Gates, "same_model", false)
	assertGate(t, c.Gates, "raw_model_endpoint", false)
	assertGate(t, c.Gates, "fak_model_endpoint", false)
}

func TestRenderRawFakContractMarkdown(t *testing.T) {
	task, err := LoadTask(filepath.Join(fixtureDir, "git-to-zig"))
	if err != nil {
		t.Fatalf("LoadTask: %v", err)
	}
	c := BuildRawFakContract(RawFakContractInput{
		Task:       task,
		Model:      "claude-opus-4-6",
		RawBaseURL: "http://raw.example/v1",
		FakBaseURL: "http://127.0.0.1:8080/v1",
	})
	md := RenderRawFakContractMarkdown(c)
	for _, want := range []string{
		"FrontierSWE Raw-vs-fak Smoke Contract",
		"Compare Evidence Link",
		"Required Before Any Result Claim",
		"score_parity_gate_declared",
		"tts_metric_declared",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}

func assertGate(t *testing.T, gates []ContractGate, name string, ok bool) {
	t.Helper()
	for _, gate := range gates {
		if gate.Name == name {
			if gate.OK != ok {
				t.Fatalf("gate %s OK=%t, want %t (detail=%s)", name, gate.OK, ok, gate.Detail)
			}
			return
		}
	}
	t.Fatalf("gate %s not found in %+v", name, gates)
}

func containsContractRequirement(reqs []string, needle string) bool {
	for _, req := range reqs {
		if strings.Contains(req, needle) {
			return true
		}
	}
	return false
}
