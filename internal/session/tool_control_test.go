package session

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func malformedRetryableOutcome(id string) ToolCallOutcome {
	return ToolCallOutcome{
		Tool:        "Write",
		ToolCallID:  id,
		Kind:        ToolCallOutcomeRejected,
		Reason:      abi.ReasonMalformed,
		Disposition: ToolDispositionRetryable,
	}
}

func TestMalformedToolOutcomesContinueWithoutDeclaredStopPolicy(t *testing.T) {
	var tracker RepeatedBadToolCallTracker
	for i := 1; i <= 4; i++ {
		ctrl := tracker.Observe(malformedRetryableOutcome("toolu-json"))
		if ctrl.Decision != SessionControlContinue {
			t.Fatalf("malformed outcome %d decision = %s, want CONTINUE without a declared stop policy", i, ctrl.Decision)
		}
		if ctrl.StopsSession() {
			t.Fatalf("malformed outcome %d stopped the session without a declared stop policy: %+v", i, ctrl)
		}
		if ctrl.Consecutive != i {
			t.Fatalf("malformed outcome %d consecutive = %d, want %d", i, ctrl.Consecutive, i)
		}
		if got := ctrl.ToolReasonToken(); got != "MALFORMED" {
			t.Fatalf("malformed outcome %d tool reason = %q, want MALFORMED", i, got)
		}
		if reason, stopped := ctrl.SessionStopReason(); stopped || reason != "" {
			t.Fatalf("malformed outcome %d reported session stop reason %q/%v without policy", i, reason, stopped)
		}
	}
}

func TestRepeatedBadToolPolicyEscalatesOnlyThroughDeclaredControl(t *testing.T) {
	tracker := RepeatedBadToolCallTracker{Policy: RepeatedBadToolCallPolicy{
		Name:         "declared-json-retry-policy",
		EndTurnAfter: 3,
		StopAfter:    6,
		StopReason:   ReasonRepeatedToolRejection,
	}}

	want := []SessionControlDecision{
		SessionControlContinue,
		SessionControlContinue,
		SessionControlEndTurn,
		SessionControlEndTurn,
		SessionControlEndTurn,
		SessionControlStop,
	}
	for i, decision := range want {
		ctrl := tracker.Observe(malformedRetryableOutcome("toolu-json"))
		if ctrl.Decision != decision {
			t.Fatalf("turn %d decision = %s, want %s", i+1, ctrl.Decision, decision)
		}
		if ctrl.Policy != "declared-json-retry-policy" {
			t.Fatalf("turn %d policy = %q, want declared-json-retry-policy", i+1, ctrl.Policy)
		}
		if ctrl.ToolReasonToken() != "MALFORMED" {
			t.Fatalf("turn %d lost per-tool reason: %+v", i+1, ctrl)
		}
		if ctrl.Decision.EndsTurn() && ctrl.Reason != ReasonRepeatedToolRejection {
			t.Fatalf("turn %d control reason = %q, want %q", i+1, ctrl.Reason, ReasonRepeatedToolRejection)
		}
		if reason, stopped := ctrl.SessionStopReason(); ctrl.Decision == SessionControlStop {
			if !stopped || reason != ReasonRepeatedToolRejection {
				t.Fatalf("turn %d stop reason = %q/%v, want %q/true", i+1, reason, stopped, ReasonRepeatedToolRejection)
			}
			if reason == "MALFORMED" {
				t.Fatal("session stop reason collapsed to the per-tool MALFORMED reason")
			}
			if ctrl.Threshold != 6 || ctrl.Consecutive != 6 {
				t.Fatalf("stop evidence = consecutive %d threshold %d, want 6/6", ctrl.Consecutive, ctrl.Threshold)
			}
		} else if stopped || reason != "" {
			t.Fatalf("turn %d non-stop reported stop reason %q/%v", i+1, reason, stopped)
		}
	}
}

func TestRepeatedBadToolPolicyResetsOnProgressOrDifferentReason(t *testing.T) {
	tracker := RepeatedBadToolCallTracker{Policy: RepeatedBadToolCallPolicy{EndTurnAfter: 2, StopAfter: 4}}
	if ctrl := tracker.Observe(malformedRetryableOutcome("toolu-json")); ctrl.Consecutive != 1 {
		t.Fatalf("seed consecutive = %d, want 1", ctrl.Consecutive)
	}
	progress := malformedRetryableOutcome("toolu-json")
	progress.Progress = true
	if ctrl := tracker.Observe(progress); ctrl.Decision != SessionControlContinue || ctrl.Consecutive != 0 {
		t.Fatalf("progress should reset: %+v", ctrl)
	}
	if ctrl := tracker.Observe(malformedRetryableOutcome("toolu-json")); ctrl.Consecutive != 1 {
		t.Fatalf("post-progress consecutive = %d, want fresh 1", ctrl.Consecutive)
	}
	different := malformedRetryableOutcome("toolu-json")
	different.Reason = abi.ReasonMisroute
	if ctrl := tracker.Observe(different); ctrl.Consecutive != 1 || ctrl.ToolReasonToken() != "MISROUTE" {
		t.Fatalf("different reason should reseed at 1 with MISROUTE: %+v", ctrl)
	}
}

func TestToolOutcomeDefaultControlIsNotSessionControl(t *testing.T) {
	outcome := malformedRetryableOutcome("toolu-json")
	ctrl := outcome.DefaultControl()
	if !ctrl.Continue() || ctrl.Decision != SessionControlContinue {
		t.Fatalf("default control for rejected tool call = %+v, want CONTINUE", ctrl)
	}
	if outcome.ReasonToken() != "MALFORMED" {
		t.Fatalf("tool outcome reason = %q, want MALFORMED", outcome.ReasonToken())
	}
	if reason, stopped := ctrl.SessionStopReason(); stopped || reason != "" {
		t.Fatalf("default control leaked a stop reason from a tool outcome: %q/%v", reason, stopped)
	}
}
