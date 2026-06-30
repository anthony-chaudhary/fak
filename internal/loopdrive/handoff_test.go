package loopdrive

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// witnessedDoneHandoff is a fak.task-handoff.v1 record for a verified-done task,
// with next-step state left to the caller so each gate case can vary just that.
func witnessedDoneHandoff() taskmgr.Handoff {
	return taskmgr.Handoff{
		Schema: taskmgr.SchemaHandoff,
		Task: taskmgr.HandoffTask{
			TaskID:  "task-handoff-next-step-push",
			Title:   "Make agentic tasks push their next step",
			State:   taskmgr.StateDone,
			Witness: &taskmgr.WitnessRecord{VerifiedState: taskmgr.VerifiedDone, Source: "test.ps1"},
		},
		CurrentState: "typed gate and CLI exist; loop wiring landing",
	}
}

// TestHandoffGateGatesWitnessedDoneWithNoFollowUp proves case (a): a
// witnessed-done task with neither next_steps nor a no_next_step_reason is
// blocked at the loop-drive completion boundary.
func TestHandoffGateGatesWitnessedDoneWithNoFollowUp(t *testing.T) {
	res := HandoffGate(true, witnessedDoneHandoff())
	if res.Outcome != HandoffGated || res.OK() {
		t.Fatalf("outcome = %q ok=%v, want gated", res.Outcome, res.OK())
	}
	if !containsReason(res.Reasons, "MISSING_NEXT_STEP_OR_NOT_APPLICABLE_REASON") {
		t.Fatalf("reasons = %v, want MISSING_NEXT_STEP_OR_NOT_APPLICABLE_REASON", res.Reasons)
	}
}

// TestHandoffGatePassesWithNextSteps proves case (b): one concrete next step
// satisfies the gate.
func TestHandoffGatePassesWithNextSteps(t *testing.T) {
	h := witnessedDoneHandoff()
	h.NextSteps = []taskmgr.HandoffNextStep{{
		Key:    "wire-loop-drive-handoff",
		Title:  "Wire the handoff gate at loop-drive completion",
		Body:   "Apply taskmgr.ReviewHandoff at the witnessed-done boundary.",
		Reason: "the typed gate exists but no loop calls it yet",
	}}
	res := HandoffGate(true, h)
	if res.Outcome != HandoffPass || !res.OK() {
		t.Fatalf("outcome = %q ok=%v reasons=%v, want pass", res.Outcome, res.OK(), res.Reasons)
	}
	if res.Review.IssueCount != 1 {
		t.Fatalf("issue count = %d, want 1", res.Review.IssueCount)
	}
}

// TestHandoffGatePassesWithNoNextStepReason proves case (c): an explicit
// no_next_step_reason satisfies the gate without any next steps.
func TestHandoffGatePassesWithNoNextStepReason(t *testing.T) {
	h := witnessedDoneHandoff()
	h.NoNextStepReason = "feature fully shipped; no reasonable follow-up remains"
	res := HandoffGate(true, h)
	if res.Outcome != HandoffPass || !res.OK() {
		t.Fatalf("outcome = %q ok=%v reasons=%v, want pass", res.Outcome, res.OK(), res.Reasons)
	}
	if res.Review.Verdict != "not_applicable" {
		t.Fatalf("verdict = %q, want not_applicable", res.Review.Verdict)
	}
}

// TestHandoffGateFailsOpenForNonAgentStop proves case (d): an ordinary stop
// with no agent task handoff is never blocked.
func TestHandoffGateFailsOpenForNonAgentStop(t *testing.T) {
	res := HandoffGate(false, taskmgr.Handoff{})
	if res.Outcome != HandoffFailOpen || !res.OK() {
		t.Fatalf("outcome = %q ok=%v, want fail-open", res.Outcome, res.OK())
	}
	if res.Reason != ReasonNoAgentTask {
		t.Fatalf("reason = %q, want %q", res.Reason, ReasonNoAgentTask)
	}
}

func containsReason(reasons []string, want string) bool {
	for _, r := range reasons {
		if r == want {
			return true
		}
	}
	return false
}
