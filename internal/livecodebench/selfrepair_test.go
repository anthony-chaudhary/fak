package livecodebench

import (
	"math"
	"strings"
	"testing"
)

func TestBuildSelfRepairDeltaRefusesWithoutCodegenSource(t *testing.T) {
	if _, err := BuildSelfRepairDelta("m", 5, nil); err == nil {
		t.Fatal("expected a refusal without a codegen source run, got nil error")
	}
	// A problem carrying no source generations is also a refusal: there is
	// nothing to repair, so the codegen run must be produced first.
	_, err := BuildSelfRepairDelta("m", 1, []RepairSource{{QuestionID: "q1", SourceSamples: 0}})
	if err == nil {
		t.Fatal("expected a refusal for a problem with no source generations, got nil error")
	}
	if !strings.Contains(err.Error(), "codegen") {
		t.Fatalf("refusal should name the missing codegen source, got %q", err.Error())
	}
}

func TestBuildSelfRepairDeltaEnforcesRepairN1AndScoresDelta(t *testing.T) {
	// Two problems, codegen at n=2. q1: 0/2 correct, repair fixes both failing.
	// q2: 1/2 correct, repair fixes the 1 failing. Repaired correct = 2/2 each,
	// so repaired pass@1 = 1.0.
	d, err := BuildSelfRepairDelta("model-x", 2, []RepairSource{
		{QuestionID: "q1", SourceSamples: 2, SourceCorrect: 0, Fixed: 2},
		{QuestionID: "q2", SourceSamples: 2, SourceCorrect: 1, Fixed: 1},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.RepairN != SelfRepairRepairN {
		t.Fatalf("repair pass must be recorded at n=%d, got repair_n=%d", SelfRepairRepairN, d.RepairN)
	}
	if SelfRepairRepairN != 1 {
		t.Fatalf("self-repair is defined at repair n=1, constant is %d", SelfRepairRepairN)
	}
	if d.CodegenN != 2 {
		t.Fatalf("codegen_n = %d, want 2", d.CodegenN)
	}
	if d.Scenario != ScenarioSelfRepair {
		t.Fatalf("scenario = %q, want %q", d.Scenario, ScenarioSelfRepair)
	}
	if d.ResultClaimAllowed {
		t.Fatal("result_claim_allowed must stay false for a locally-ungraded self-repair run")
	}
	if d.EvidenceClass != EvidenceLocalUngraded {
		t.Fatalf("evidence_class = %q, want %q", d.EvidenceClass, EvidenceLocalUngraded)
	}
	// source pass@1: q1 = pass@1(2,0)=0, q2 = pass@1(2,1)=0.5 -> mean 0.25.
	if math.Abs(d.SourcePassAt1-0.25) > 1e-9 {
		t.Fatalf("source_pass_at_1 = %v, want 0.25", d.SourcePassAt1)
	}
	if math.Abs(d.RepairedPassAt1-1.0) > 1e-9 {
		t.Fatalf("repaired_pass_at_1 = %v, want 1.0", d.RepairedPassAt1)
	}
	if math.Abs(d.Delta-0.75) > 1e-9 {
		t.Fatalf("delta = %v, want 0.75 (repaired - source)", d.Delta)
	}
}

func TestBuildSelfRepairDeltaRejectsFixingMoreThanFailing(t *testing.T) {
	// Only 1 of 2 samples failed, so at most 1 can be fixed.
	_, err := BuildSelfRepairDelta("m", 2, []RepairSource{
		{QuestionID: "q1", SourceSamples: 2, SourceCorrect: 1, Fixed: 2},
	})
	if err == nil {
		t.Fatal("expected an error when fixed exceeds the failing count")
	}
	// codegen_n must match the per-problem source sample count.
	_, err = BuildSelfRepairDelta("m", 3, []RepairSource{
		{QuestionID: "q1", SourceSamples: 2, SourceCorrect: 1, Fixed: 0},
	})
	if err == nil {
		t.Fatal("expected an error when source_samples does not match codegen_n")
	}
}

func TestBuildRepairPromptFeedsWrongGenerationAndFeedback(t *testing.T) {
	p := Problem{QuestionID: "q1", Scenario: ScenarioSelfRepair, Prompt: "Add two numbers."}
	prompt, err := BuildRepairPrompt(p, "def add(a,b): return a-b", "expected 5 got -1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, want := range []string{"Add two numbers.", "def add(a,b): return a-b", "expected 5 got -1", "Corrected solution"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("repair prompt is missing %q\n---\n%s", want, prompt)
		}
	}
}

func TestBuildRepairPromptRefusesEmptyInputs(t *testing.T) {
	p := Problem{QuestionID: "q1", Prompt: "Add two numbers."}
	if _, err := BuildRepairPrompt(p, "   ", "feedback"); err == nil {
		t.Fatal("expected a refusal for an empty prior generation")
	}
	if _, err := BuildRepairPrompt(p, "code", ""); err == nil {
		t.Fatal("expected a refusal for empty failing-test feedback")
	}
}
