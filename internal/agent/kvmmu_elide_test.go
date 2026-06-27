package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// kvmmu_elide_test.go witnesses the plan SegElisionPlan builds for the #579 planned-elision
// residency bridge: every elided span must carry a page-back-in handle (a content-address Digest)
// and the Selected/Elided sets must partition the transcript, so ctxplan.Audit certifies the plan
// FAITHFUL (a recoverable view, not lossy compaction). This is the demand-fault-intact half of the
// issue's acceptance criteria, checked at the plan layer where the handle lives.

func TestSegElisionPlanCarriesPageBackInHandle(t *testing.T) {
	messages := []Message{
		{Role: RoleSystem, Content: "you are a helper"},
		{Role: RoleUser, Content: "first resident question"},
		{Role: RoleAssistant, Content: "first cold candidate"},
		{Role: RoleUser, Content: "second cold candidate"},
	}
	// Provable direction: keep the early spans resident, elide the LATER ones (over-budget tail).
	elided := []bool{false, false, true, true}
	plan := SegElisionPlan(messages, elided)

	if len(plan.Elided) != 2 {
		t.Fatalf("plan elided %d spans, want 2 (the over-budget tail)", len(plan.Elided))
	}
	if len(plan.Selected) != 2 {
		t.Fatalf("plan selected %d spans, want 2 (the resident prefix)", len(plan.Selected))
	}
	// Every elided span must carry a non-empty page-back-in handle (the demand-fault path).
	for _, e := range plan.Elided {
		if e.Digest == "" {
			t.Errorf("elided span %q has no Digest page-back-in handle (demand-fault path would be broken)", e.ID)
		}
	}
	// The plan must partition the transcript AND keep every elided span recoverable — exactly
	// what distinguishes a faithful planned VIEW from lossy compaction.
	if w := ctxplan.Audit(plan); !w.Faithful {
		t.Fatalf("SegElisionPlan is not Faithful: %+v (partition=%v unrecoverable=%v)", w, w.Partition, w.Unrecoverable)
	}
}

// TestSegElisionPlanIdCorrespondence pins the kvmmu.ApplyPlan adapter contract: the plan's span
// ids are exactly the segment ids lowerSegments/segIDFor mint, so ApplyPlan evicts a segment iff
// its id is elided. A drift here would silently evict nothing (or the wrong span).
func TestSegElisionPlanIdCorrespondence(t *testing.T) {
	messages := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "resident"},
		{Role: RoleTool, ToolCallID: "call_7", Name: "fetch", Content: "cold result"},
	}
	elided := []bool{false, false, true}
	plan := SegElisionPlan(messages, elided)
	if len(plan.Elided) != 1 {
		t.Fatalf("elided %d, want 1", len(plan.Elided))
	}
	// The tool result's id must be the tool-call-tied form so it is addressable by admission identity.
	if got := plan.Elided[0].ID; got != "m2:call_7" {
		t.Errorf("tool-result segment id = %q, want m2:call_7 (tool-call-id-tied)", got)
	}
	// Selected ids are the plain message-index form for non-tool messages.
	if got := plan.Selected[0].ID; got != "m0" {
		t.Errorf("system segment id = %q, want m0", got)
	}
}
