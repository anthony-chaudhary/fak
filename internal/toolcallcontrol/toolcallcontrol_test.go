package toolcallcontrol

import (
	"strings"
	"testing"
)

func TestEvaluateReusesExactFreshResult(t *testing.T) {
	p := Proposal{ID: "repeat", Tool: "read", Args: "{\"path\":\"a\"}", StateEpoch: "git:abc", PromptTokens: 128000, ReadOnly: true}
	got := Evaluate(Config{}, p, []Observation{{Tool: p.Tool, Args: p.Args, StateEpoch: p.StateEpoch, ResultRef: "turn-2"}}, nil)
	if got.Action != Reuse || got.ReuseRef != "turn-2" || got.ReplayUnitsSaved != 128000 {
		t.Fatalf("unexpected decision: %+v", got)
	}
	if got.ReplaySquaredSaved != "16384000000" {
		t.Fatalf("proxy=%s", got.ReplaySquaredSaved)
	}
}

func TestStateChangeMakesRepeatNovel(t *testing.T) {
	p := Proposal{ID: "repeat", Tool: "read", Args: "a", StateEpoch: "git:new", ReadOnly: true, EvidenceGap: "new file contents", EffectIfNew: "change patch"}
	got := Evaluate(Config{}, p, []Observation{{Tool: p.Tool, Args: p.Args, StateEpoch: "git:old"}}, nil)
	if got.Action != Allow {
		t.Fatalf("action=%s reason=%s", got.Action, got.Reason)
	}
}

func TestWeakSpeculativeReadDeferredButMutationAllowed(t *testing.T) {
	weak := Proposal{ID: "browse", Tool: "search", ReadOnly: true, ExpectedInfoGainBP: 100, PromptTokens: 64000}
	if got := Evaluate(Config{}, weak, nil, nil); got.Action != Defer {
		t.Fatalf("weak read: %+v", got)
	}
	mutation := weak
	mutation.ID = "write"
	mutation.ReadOnly = false
	if got := Evaluate(Config{}, mutation, nil, nil); got.Action != Allow {
		t.Fatalf("mutation: %+v", got)
	}
}

func TestIndependentReadsBatch(t *testing.T) {
	a := Proposal{ID: "a", Tool: "read", ReadOnly: true, BatchKey: "inspect"}
	b := Proposal{ID: "b", Tool: "read", ReadOnly: true, BatchKey: "inspect"}
	if got := Evaluate(Config{}, a, nil, []Proposal{a, b}); got.Action != Allow || got.Reason != "batch_leader" {
		t.Fatalf("got %+v", got)
	}
	if got := Evaluate(Config{}, b, nil, []Proposal{a, b}); got.Action != Batch || got.Reason != "merged_into_batch_leader" {
		t.Fatalf("got %+v", got)
	}
}

func TestInstructionEscalatesLongContext(t *testing.T) {
	if !strings.Contains(Instruction(128000), "long-context") {
		t.Fatal("missing long-context instruction")
	}
	if strings.Contains(Instruction(1000), "long-context") {
		t.Fatal("short context escalated")
	}
}

func TestAblationReportsFalseSuppressionSeparately(t *testing.T) {
	trace := []LabeledProposal{
		{Proposal: Proposal{ID: "needed", PromptTokens: 100}, Needed: true},
		{Proposal: Proposal{ID: "waste", PromptTokens: 200}, Needed: false},
	}
	arms := []Arm{{Name: "gate", Decisions: map[string]Verdict{
		"needed": {Action: Batch}, "waste": {Action: Reuse},
	}}}
	got := Ablate(trace, arms)[0]
	if got.NeededCallsSuppressed != 0 || got.UnneededCallsAvoided != 1 || got.ReplayUnitsSaved != 200 || got.ReplaySquaredSaved != "40000" {
		t.Fatalf("metrics=%+v", got)
	}
}
