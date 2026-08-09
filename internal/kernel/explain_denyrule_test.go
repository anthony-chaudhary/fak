package kernel

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// denyWithMeta is a rung that refuses with a caller-supplied Meta map — the seam
// a real rung uses to stamp its deny_rule and its own sanctioned alternative.
type denyWithMeta struct {
	reason abi.ReasonCode
	meta   map[string]string
}

func (d denyWithMeta) Adjudicate(context.Context, *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: d.reason, By: "monitor", Meta: d.meta}
}
func (denyWithMeta) Caps() []abi.Capability { return nil }

// TestDecisionCarriesMatchedRule is the #5213 provenance floor: a refusal must
// name WHICH rule matched, not just the reason CLASS. POLICY_BLOCK alone covers
// the recursive-delete rung, the out-of-tree-write rung and seven gitgate laws,
// so an operator handed only the class cannot audit the denial.
func TestDecisionCarriesMatchedRule(t *testing.T) {
	ctx := context.Background()
	chain := []abi.Adjudicator{denyWithMeta{abi.ReasonPolicyBlock, map[string]string{
		abi.MetaDenyRule: "rm_rf recursive/forced delete",
	}}}
	_, d := FoldExplain(ctx, chain, callInline("shell", `{"cmd":"rm -rf /tmp/x"}`))

	if d.DenyRule != "rm_rf" {
		t.Fatalf("DenyRule = %q, want %q (canonicalized from the rung's authored label)", d.DenyRule, "rm_rf")
	}
	if !strings.Contains(d.Explanation, "matched rule: rm_rf") {
		t.Errorf("explanation omits the matched rule: %q", d.Explanation)
	}
	if !strings.Contains(d.Text(), "rule: rm_rf") {
		t.Errorf("Text() omits the matched rule:\n%s", d.Text())
	}
	// The args digest is the call-identity half of the provenance record, and it
	// must never be the raw args.
	if d.ArgsDigest == "" || strings.Contains(d.Text(), "rm -rf /tmp/x") {
		t.Errorf("want a digest and no raw args in the trace:\n%s", d.Text())
	}
}

// TestDenyRuleRejectsNonMemberWhole pins the closed-vocabulary contract at THIS
// consumer: a value outside abi's set is dropped whole, never trimmed or
// truncated into the field. Without this, a rung (or a model-controlled Meta map)
// could smuggle arbitrary text into a field documented as safe to log.
func TestDenyRuleRejectsNonMemberWhole(t *testing.T) {
	ctx := context.Background()
	for _, raw := range []string{
		"AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI",
		"rm_rf_but_not_really",
		"../../etc/passwd",
		"",
	} {
		chain := []abi.Adjudicator{denyWithMeta{abi.ReasonPolicyBlock, map[string]string{abi.MetaDenyRule: raw}}}
		_, d := FoldExplain(ctx, chain, callInline("shell", "{}"))
		if d.DenyRule != "" {
			t.Errorf("raw %q leaked DenyRule = %q, want dropped whole", raw, d.DenyRule)
		}
		if raw != "" && strings.Contains(d.Text(), raw) {
			t.Errorf("raw %q survived into the trace:\n%s", raw, d.Text())
		}
	}
}

// TestRemedyComesFromTheMatchedRung is the other half of #5213: the alternative
// an operator is handed must come from the rung that ACTUALLY refused.
func TestRemedyComesFromTheMatchedRung(t *testing.T) {
	ctx := context.Background()
	chain := []abi.Adjudicator{denyWithMeta{abi.ReasonPolicyBlock, map[string]string{
		abi.MetaDenyRule: "out_of_tree_write",
		"fix":            "write inside the workspace",
	}}}
	_, d := FoldExplain(ctx, chain, callInline("write", "{}"))

	if d.Remedy != "write inside the workspace" {
		t.Fatalf("Remedy = %q, want the refusing rung's own fix", d.Remedy)
	}
	if !strings.Contains(d.Explanation, "Sanctioned alternative: write inside the workspace.") {
		t.Errorf("explanation omits the rung's remedy: %q", d.Explanation)
	}
}

// TestNoRemedyIsStatedNotInvented is the regression that binds the issue's actual
// complaint. The refused call below is an ordinary SELF_MODIFY on a guarded path
// whose rung declares NO alternative. The old behavior back-filled advice from a
// reason-CLASS table, which is how a refused file edit was answered with "fak
// slack send" and "fak commit --core-lock-maintenance-witness" — advice that had
// nothing to do with the call. Absence must be reported as absence.
func TestNoRemedyIsStatedNotInvented(t *testing.T) {
	ctx := context.Background()
	chain := []abi.Adjudicator{denyWithMeta{abi.ReasonSelfModify, map[string]string{
		abi.MetaDenyRule: "self_modify_path",
	}}}
	_, d := FoldExplain(ctx, chain, callInline("edit", `{"path":"install-openssh-linux.sh"}`))

	if d.Remedy != "" {
		t.Fatalf("Remedy = %q, want empty when the matched rule declares none", d.Remedy)
	}
	if !strings.Contains(d.Explanation, "No sanctioned alternative is known for this call.") {
		t.Errorf("absence not stated out loud: %q", d.Explanation)
	}
	// The specific unrelated advice from the rollout in #5213 must not appear.
	for _, unrelated := range []string{"slack", "core-lock-maintenance-witness", "-r/-f", "moving a path aside"} {
		if strings.Contains(strings.ToLower(d.Text()), strings.ToLower(unrelated)) {
			t.Errorf("unrelated advice %q emitted for a call it does not address:\n%s", unrelated, d.Text())
		}
	}
	// A refusal always prints a remedy line, so "no alternative" is distinguishable
	// from "the trace forgot to print one".
	if !strings.Contains(d.Text(), "remedy: (none") {
		t.Errorf("Text() omits the explicit no-remedy line:\n%s", d.Text())
	}
}

// TestAllowCarriesNoRefusalProvenance keeps the fields scoped to refusals: an
// allowed call has no matched deny rule and no remedy to offer.
func TestAllowCarriesNoRefusalProvenance(t *testing.T) {
	ctx := context.Background()
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictAllow, By: "monitor",
		Meta: map[string]string{abi.MetaDenyRule: "rm_rf", "fix": "should not surface"}}}}
	_, d := FoldExplain(ctx, chain, callInline("read", "{}"))

	if d.DenyRule != "" || d.Remedy != "" {
		t.Fatalf("allow carried refusal provenance: rule=%q remedy=%q", d.DenyRule, d.Remedy)
	}
	if strings.Contains(d.Text(), "remedy:") {
		t.Errorf("allow printed a remedy line:\n%s", d.Text())
	}
}
