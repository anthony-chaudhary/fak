package swebench

import (
	"strings"
	"testing"
)

func TestBuildDeepSWERawFakContractGatesResultClaim(t *testing.T) {
	c := BuildDeepSWERawFakContract(DeepSWERawFakContractInput{
		GeneratedAt:  "2026-06-26T00:00:00Z",
		Dataset:      sampleDataset(),
		Source:       "testdata/swebench_smoke.json",
		Filter:       "full",
		Limit:        2,
		Model:        "DeepSWE-Preview",
		RawBaseURL:   "$env:RAW_DEEPSWE_BASE_URL",
		FakBaseURL:   "http://localhost:8080/v1",
		Adapter:      "deepswe-r2e-runner",
		RawCommand:   "raw",
		FakCommand:   "fak",
		RawOutputDir: "experiments/raw",
		FakOutputDir: "experiments/fak",
		MaxSteps:     50,
		Timeout:      "30m",
		EvalCapability: EvalCapability{
			DockerPresent:  false,
			HarnessPresent: false,
			Runnable:       false,
			Reason:         "no Docker",
		},
	})
	if c.Schema != DeepSWERawFakContractSchema {
		t.Fatalf("schema = %q", c.Schema)
	}
	if c.Status != "READY_FOR_EXTERNAL_RUN" {
		t.Fatalf("status = %q", c.Status)
	}
	if c.ResultClaimAllowed {
		t.Fatal("pre-run contract must not allow a result claim")
	}
	if len(c.TaskSelection.TaskIDs) != 2 || !c.TaskSelection.SameTaskIDs {
		t.Fatalf("task selection = %+v", c.TaskSelection)
	}
	if !c.Adapter.SameAdapter || !c.Model.SameModel || !c.Budget.SameBudget {
		t.Fatalf("shared adapter/model/budget flags not set: adapter=%+v model=%+v budget=%+v", c.Adapter, c.Model, c.Budget)
	}
	if len(c.Arms) != 2 || c.Arms[0].EvalCommand == "" || c.Arms[1].EvalCommand == "" {
		t.Fatalf("arms missing eval commands: %+v", c.Arms)
	}
	if len(c.RequiredBeforeClaim) == 0 {
		t.Fatal("contract must list requirements before any result claim")
	}
}

func TestBuildDeepSWERawFakContractIncomplete(t *testing.T) {
	c := BuildDeepSWERawFakContract(DeepSWERawFakContractInput{Dataset: sampleDataset()})
	if c.Status != "INCOMPLETE_CONTRACT" {
		t.Fatalf("status = %q", c.Status)
	}
	var sawMissingAdapter bool
	for _, gate := range c.Gates {
		if gate.Name == "same_adapter" && !gate.OK {
			sawMissingAdapter = true
		}
	}
	if !sawMissingAdapter {
		t.Fatalf("missing adapter gate not recorded: %+v", c.Gates)
	}
}

func TestRenderDeepSWERawFakContractMarkdown(t *testing.T) {
	c := BuildDeepSWERawFakContract(DeepSWERawFakContractInput{
		Dataset:        sampleDataset(),
		Model:          "DeepSWE-Preview",
		RawBaseURL:     "$env:RAW_DEEPSWE_BASE_URL",
		FakBaseURL:     "http://localhost:8080/v1",
		Adapter:        "deepswe-r2e-runner",
		RawCommand:     "raw",
		FakCommand:     "fak",
		MaxSteps:       50,
		Timeout:        "30m",
		EvalCapability: EvalCapability{Runnable: true},
	})
	md := RenderDeepSWERawFakContractMarkdown(c)
	for _, want := range []string{"DeepSWE Raw-vs-fak SWE-bench Contract", "Required Before Any Result Claim", "same_adapter"} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}
