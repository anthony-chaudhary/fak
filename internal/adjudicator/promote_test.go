package adjudicator

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// decideEvent builds an EvDecide event carrying a tool + verdict, the shape the kernel
// emits on the adjudication stream.
func decideEvent(kind abi.EventKind, tool string, v abi.Verdict) abi.Event {
	return abi.Event{Kind: kind, Call: &abi.ToolCall{Tool: tool}, Verdict: &v}
}

// admitAndLogVerdict is the complain-mode admit the ledger counts.
func admitAndLogVerdict() abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "monitor",
		Meta: map[string]string{"posture": "admit_and_log", "would_deny": "DEFAULT_DENY"}}
}

// TestLedgerCountsOnlyComplainAdmitsOnEvDecide pins the Emit filter: a clean event is
// folded ONLY from a complain-mode admit (an Allow carrying posture=admit_and_log) on
// the verdict-resolved event. A plain Allow, a Deny, and a non-EvDecide event for the
// same admit must NOT count.
func TestLedgerCountsOnlyComplainAdmitsOnEvDecide(t *testing.T) {
	l := NewLedger()

	// 1. complain-mode admit on EvDecide -> counted.
	l.Emit(decideEvent(abi.EvDecide, "provision_widget", admitAndLogVerdict()))
	// 2. a plain Allow (no admit_and_log posture) -> not a complain admit.
	l.Emit(decideEvent(abi.EvDecide, "provision_widget", abi.Verdict{Kind: abi.VerdictAllow, By: "monitor"}))
	// 3. a Deny (soft default-deny) -> not an admit.
	l.Emit(decideEvent(abi.EvDecide, "provision_widget", abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny}))
	// 4. the SAME complain admit on a NON-decide event -> must not double-count.
	l.Emit(decideEvent(abi.EvDispatch, "provision_widget", admitAndLogVerdict()))

	if got := l.Clean("provision_widget"); got != 1 {
		t.Fatalf("clean count = %d, want exactly 1 (only the admitted EvDecide)", got)
	}
}

// TestLedgerAccumulatesPerTool proves the counter is per-tool and accumulates across
// repeated complain admits.
func TestLedgerAccumulatesPerTool(t *testing.T) {
	l := NewLedger()
	for i := 0; i < 5; i++ {
		l.Emit(decideEvent(abi.EvDecide, "tool_a", admitAndLogVerdict()))
	}
	for i := 0; i < 2; i++ {
		l.Emit(decideEvent(abi.EvDecide, "tool_b", admitAndLogVerdict()))
	}
	if l.Clean("tool_a") != 5 || l.Clean("tool_b") != 2 {
		t.Fatalf("per-tool counts a=%d b=%d, want 5/2", l.Clean("tool_a"), l.Clean("tool_b"))
	}
	if l.Clean("tool_unseen") != 0 {
		t.Fatalf("unseen tool should be 0, got %d", l.Clean("tool_unseen"))
	}
}

// TestLedgerHardRefusalResetsCleanRun is the disqualification rule: a hard-refusal Deny
// for a tool resets its clean run to zero and records a hard event. A soft DEFAULT_DENY
// does NOT reset (it is the admittable reason).
func TestLedgerHardRefusalResetsCleanRun(t *testing.T) {
	l := NewLedger()

	// Build a clean run of 3.
	for i := 0; i < 3; i++ {
		l.Emit(decideEvent(abi.EvDecide, "provision_widget", admitAndLogVerdict()))
	}
	if l.Clean("provision_widget") != 3 {
		t.Fatalf("pre-reset clean = %d, want 3", l.Clean("provision_widget"))
	}

	// A soft DEFAULT_DENY does NOT reset.
	l.Emit(decideEvent(abi.EvDecide, "provision_widget", abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny}))
	if l.Clean("provision_widget") != 3 {
		t.Fatalf("soft default-deny must not reset, clean = %d, want 3", l.Clean("provision_widget"))
	}

	// A HARD refusal (self-modify) resets to zero and records a hard event.
	l.Emit(decideEvent(abi.EvDecide, "provision_widget", abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonSelfModify}))
	if l.Clean("provision_widget") != 0 {
		t.Fatalf("hard refusal must reset clean run, clean = %d, want 0", l.Clean("provision_widget"))
	}
	if l.HardEvents("provision_widget") != 1 {
		t.Fatalf("hard refusal must be recorded, hard = %d, want 1", l.HardEvents("provision_widget"))
	}
}

// TestHardRefusalClassification pins which reasons disqualify a tool: the provable
// policy/security refusals, but NOT the soft DEFAULT_DENY / model-fixable MISROUTE /
// transient RATE_LIMITED.
func TestHardRefusalClassification(t *testing.T) {
	hard := []abi.ReasonCode{abi.ReasonPolicyBlock, abi.ReasonSelfModify, abi.ReasonSecretExfil,
		abi.ReasonTrustViolation, abi.ReasonMalformed, abi.ReasonUnwitnessed}
	soft := []abi.ReasonCode{abi.ReasonNone, abi.ReasonDefaultDeny, abi.ReasonMisroute,
		abi.ReasonRateLimited, abi.ReasonOversize, abi.ReasonLeaseHeld, abi.ReasonUnknownTool}
	for _, r := range hard {
		if !hardRefusal(r) {
			t.Errorf("%s should be a hard refusal", abi.ReasonName(r))
		}
	}
	for _, r := range soft {
		if hardRefusal(r) {
			t.Errorf("%s should NOT be a hard refusal", abi.ReasonName(r))
		}
	}
}
