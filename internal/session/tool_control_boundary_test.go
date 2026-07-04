package session

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestSessionStopReasonIsNeverAToolReason is the #2630 anti-collapse witness:
// the declared session-stop reason must not be any per-tool refusal token. If a
// refactor makes a tool refusal (MALFORMED, MISROUTE, ...) leak into the
// session-stop reason slot, this fails.
func TestSessionStopReasonIsNeverAToolReason(t *testing.T) {
	if !sessionStopReasonIsNotAToolReason(ReasonRepeatedToolRejection) {
		t.Fatalf("declared session-stop reason %q collides with a per-tool refusal token", ReasonRepeatedToolRejection)
	}
	// Every closed per-tool reason name must be REJECTED as a session-stop reason,
	// so the boundary holds across the whole vocabulary, not just one sample.
	for _, name := range abi.ReasonNames() {
		if sessionStopReasonIsNotAToolReason(name) {
			t.Fatalf("per-tool refusal token %q was accepted as a session-stop reason; the control planes collapsed", name)
		}
	}
	// The empty reason is not a stop reason and must not be mistaken for one.
	if sessionStopReasonIsNotAToolReason("") {
		t.Fatal("empty token treated as a valid distinct session-stop reason")
	}
}

// TestStoppedControlNeverBorrowsToolReason proves the runtime seam matching the
// compile-time guard: a policy-driven session stop names the control-plane
// reason and the per-tool reason stays in its own field, never copied into the
// stop reason. This is the four-malformed-calls case escalated to a real stop.
func TestStoppedControlNeverBorrowsToolReason(t *testing.T) {
	tracker := RepeatedBadToolCallTracker{Policy: RepeatedBadToolCallPolicy{StopAfter: 2}}
	tracker.Observe(malformedRetryableOutcome("toolu-json"))
	ctrl := tracker.Observe(malformedRetryableOutcome("toolu-json"))

	if !ctrl.StopsSession() {
		t.Fatalf("policy StopAfter=2 did not stop on the 2nd bad call: %+v", ctrl)
	}
	reason, stopped := ctrl.SessionStopReason()
	if !stopped || reason != ReasonRepeatedToolRejection {
		t.Fatalf("stop reason = %q/%v, want %q/true", reason, stopped, ReasonRepeatedToolRejection)
	}
	if !sessionStopReasonIsNotAToolReason(reason) {
		t.Fatalf("session stop reason %q is a per-tool refusal token — planes collapsed", reason)
	}
	// The per-tool reason survives in its own slot, uncopied into the stop reason.
	if ctrl.ToolReasonToken() != abi.ReasonName(abi.ReasonMalformed) {
		t.Fatalf("per-tool reason lost: %+v", ctrl)
	}
}
