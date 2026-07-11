package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

// TestSessionScopedRuleExpires is the #2406 acceptance witness: an allow rule added
// at session scope is live mid-session and provably gone after the session ends
// (`fak policy --dump` — Dump() here — shows no residue). A session-scoped widening
// is admitted WITHOUT a witness precisely because its lifetime is bounded.
func TestSessionScopedRuleExpires(t *testing.T) {
	pr := NewPolicyRegime("default", nil)
	const trace = "trace-abc"
	const tool = "shell.exec"

	v, err := pr.Apply(trace, PolicyOp{
		Kind:  PolicyAddRules,
		Scope: ScopeSession,
		Rules: []PolicyRule{{Tool: tool, Allow: true}},
	})
	if err != nil {
		t.Fatalf("session-scoped add_rules refused: %v", err)
	}
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("session-scoped add_rules verdict = %v, want Allow", v.Kind)
	}

	// Live mid-session.
	if !pr.IsAllowed(trace, tool) {
		t.Fatalf("session-scoped rule not live mid-session")
	}
	if dump := pr.Dump(); !strings.Contains(dump, tool) {
		t.Fatalf("session rule missing from dump mid-session:\n%s", dump)
	}

	// End the session: the rule expires automatically.
	pr.EndSession(trace)
	if pr.IsAllowed(trace, tool) {
		t.Fatalf("session-scoped rule still live after session end")
	}
	if dump := pr.Dump(); strings.Contains(dump, tool) {
		t.Fatalf("policy --dump shows session-rule residue after session end:\n%s", dump)
	}
}

// TestWidenRequiresWitness is the #2406 acceptance witness: a DURABLE allow (a
// widening) without a witness is refused with the closed UNWITNESSED reason, routed
// through abi.VerdictRequireWitness; the same op WITH a witness is admitted durably
// and survives the session.
func TestWidenRequiresWitness(t *testing.T) {
	pr := NewPolicyRegime("default", nil)
	const trace = "trace-xyz"
	const tool = "net.fetch"

	v, err := pr.Apply(trace, PolicyOp{
		Kind:  PolicyAddRules,
		Scope: ScopeDurable,
		Rules: []PolicyRule{{Tool: tool, Allow: true}},
	})
	if err == nil {
		t.Fatalf("durable widen without a witness was admitted (want a refusal)")
	}
	if v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("verdict = %v, want RequireWitness", v.Kind)
	}
	if got := abi.ReasonName(v.Reason); got != "UNWITNESSED" {
		t.Fatalf("closed reason = %q, want UNWITNESSED", got)
	}
	if pr.IsAllowed(trace, tool) {
		t.Fatalf("durable floor widened despite the refusal")
	}

	// With a witness it is admitted durably.
	v2, err := pr.Apply(trace, PolicyOp{
		Kind:    PolicyAddRules,
		Scope:   ScopeDurable,
		Rules:   []PolicyRule{{Tool: tool, Allow: true}},
		Witness: "operator-token-42",
	})
	if err != nil {
		t.Fatalf("witnessed durable widen refused: %v", err)
	}
	if v2.Kind != abi.VerdictAllow {
		t.Fatalf("witnessed verdict = %v, want Allow", v2.Kind)
	}
	if !pr.IsAllowed(trace, tool) {
		t.Fatalf("witnessed durable widen did not apply")
	}
	// Durable ⇒ survives the session end (unlike a session-scoped grant).
	pr.EndSession(trace)
	if !pr.IsAllowed(trace, tool) {
		t.Fatalf("durable witnessed grant expired with the session")
	}
}

// TestPolicyTightenAppliesWithoutWitness proves the tighten/widen split is read
// structurally: a durable EXPLICIT DENY narrows the floor immediately, no witness.
func TestPolicyTightenAppliesWithoutWitness(t *testing.T) {
	pr := NewPolicyRegime("default", nil)
	if _, err := pr.Apply("t", PolicyOp{
		Kind:  PolicyAddRules,
		Scope: ScopeDurable,
		Rules: []PolicyRule{{Tool: "danger.rm", Allow: false}},
	}); err != nil {
		t.Fatalf("durable tighten-only op refused: %v", err)
	}
	if pr.IsAllowed("t", "danger.rm") {
		t.Fatalf("explicit durable deny did not tighten the floor")
	}
}

// TestRevokeEvictsForwardFromEpoch proves fak_revoke's causal eviction walks FORWARD
// from a refuted epoch: an admission cited under a later epoch is evicted while an
// earlier one is untouched.
func TestRevokeEvictsForwardFromEpoch(t *testing.T) {
	pr := NewPolicyRegime("default", nil)

	mustApply(t, pr, "t", PolicyOp{Kind: PolicyAddRules, Rules: []PolicyRule{{Tool: "A", Allow: true}}})
	e1 := pr.Epoch()
	pr.RecordAdmission("t", "call-under-A")

	mustApply(t, pr, "t", PolicyOp{Kind: PolicyAddRules, Rules: []PolicyRule{{Tool: "B", Allow: true}}})
	e2 := pr.Epoch()
	pr.RecordAdmission("t", "call-under-B")

	if e2 <= e1 {
		t.Fatalf("epoch did not advance across ops: e1=%d e2=%d", e1, e2)
	}
	if n := pr.Revoke(e2); n != 1 {
		t.Fatalf("Revoke(e2) evicted %d admissions, want 1", n)
	}
	live := pr.LiveAdmissions()
	if len(live) != 1 || live[0] != "call-under-A" {
		t.Fatalf("live admissions = %v, want [call-under-A]", live)
	}
}

// TestRegimePivotAuditVerifies is the #2406 done-condition witness: `fak audit
// verify` (journal.VerifyRows here) passes across a regime pivot — every applied
// policy op is a chained POLICY_OP row and the chain verifies end to end.
func TestRegimePivotAuditVerifies(t *testing.T) {
	jnl := journal.OpenMemory()
	pr := NewPolicyRegime("default", jnl)

	mustApply(t, pr, "t", PolicyOp{Kind: PolicyAddRules, Scope: ScopeDurable, Rules: []PolicyRule{{Tool: "x", Allow: false}}})
	mustApply(t, pr, "t", PolicyOp{Kind: PolicyAddRules, Scope: ScopeSession, Rules: []PolicyRule{{Tool: "y", Allow: true}}})
	mustApply(t, pr, "t", PolicyOp{Kind: PolicySetRegime, Scope: ScopeDurable, Regime: "hardened", Witness: "operator-token"})

	rows := jnl.Recent(0)
	if len(rows) != 3 {
		t.Fatalf("journaled %d policy ops, want 3", len(rows))
	}
	if _, err := journal.VerifyRows(rows); err != nil {
		t.Fatalf("audit verify failed across regime pivot: %v", err)
	}
	for i, r := range rows {
		if r.Kind != journal.KindPolicyOp {
			t.Fatalf("row %d kind = %q, want %q", i, r.Kind, journal.KindPolicyOp)
		}
	}
	if dump := pr.Dump(); !strings.Contains(dump, "regime: hardened") {
		t.Fatalf("regime pivot not reflected in dump:\n%s", dump)
	}
}

func mustApply(t *testing.T, pr *PolicyRegime, trace string, op PolicyOp) {
	t.Helper()
	if _, err := pr.Apply(trace, op); err != nil {
		t.Fatalf("Apply(%s) refused: %v", op.Kind, err)
	}
}
