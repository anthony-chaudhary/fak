package adjudicator

import (
	"context"
	"regexp"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
)

func approvalCall(tool, args string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)}}
}

func approvalFloor(approvals ...string) Policy {
	p := Policy{
		Allow:            map[string]bool{"Bash": true, "Edit": true, "WebFetch": true},
		Deny:             map[string]abi.ReasonCode{"git_push_tool": abi.ReasonPolicyBlock, "exfiltrate": abi.ReasonSecretExfil},
		SelfModifyGlobs:  []string{"internal/kernel/"},
		BlockedPathGlobs: []string{".env"},
		ArgPredicates: []ArgPredicate{{
			Tool: "Bash", Arg: "command", Kind: ArgDenyRegex,
			Re:     regexp.MustCompile(defaultRmRfDenyRegex),
			Reason: abi.ReasonPolicyBlock,
		}},
	}
	if len(approvals) > 0 {
		p.OperatorApprovals = map[string]bool{}
		for _, id := range approvals {
			p.OperatorApprovals[id] = true
		}
	}
	return p
}

// approvalIDOf reads the approval id the floor offers on a refusal.
func approvalIDOf(t *testing.T, p Policy, tool, args string) string {
	t.Helper()
	v := New(p).Adjudicate(context.Background(), approvalCall(tool, args))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("%s %s: kind = %v, want a refusal to approve", tool, args, v.Kind)
	}
	id := v.Meta[MetaOperatorApprovalID]
	if id == "" {
		t.Fatalf("%s %s: refusal %s carries no %s (meta %v)", tool, args, abi.ReasonName(v.Reason), MetaOperatorApprovalID, v.Meta)
	}
	return id
}

func TestOperatorApprovalLiftsExactlyTheApprovedCall(t *testing.T) {
	const rmBuild = `{"command":"rm -rf build"}`
	id := approvalIDOf(t, approvalFloor(), "Bash", rmBuild)

	a := New(approvalFloor(id))
	v := a.Adjudicate(context.Background(), approvalCall("Bash", rmBuild))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("approved call: kind = %v reason %s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v.Meta["posture"] != PostureOperatorApproved || v.Meta["would_deny"] != abi.ReasonName(abi.ReasonPolicyBlock)+","+holdLiftName || v.Meta[MetaOperatorApprovalID] != id {
		t.Fatalf("approved call carries no approval record: meta %v", v.Meta)
	}

	// A re-proposal that only rewords the free-text description keeps its id.
	reworded := `{"command":"rm  -rf build","description":"clean the build dir"}`
	if v := a.Adjudicate(context.Background(), approvalCall("Bash", reworded)); v.Kind != abi.VerdictAllow {
		t.Fatalf("reworded re-proposal: kind = %v, want Allow (same effect-bearing args)", v.Kind)
	}

	// Any other call stays refused.
	other := a.Adjudicate(context.Background(), approvalCall("Bash", `{"command":"rm -rf src"}`))
	if other.Kind != abi.VerdictDeny || other.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("unapproved call: kind = %v reason %s, want Deny/POLICY_BLOCK", other.Kind, abi.ReasonName(other.Reason))
	}
}

func TestOperatorApprovalLiftsDefaultDenyAndPolicyBlockNameDeny(t *testing.T) {
	for _, tc := range []struct{ tool, args string }{
		{"weird_tool", `{"x":1}`},
		{"git_push_tool", `{"remote":"origin"}`},
	} {
		id := approvalIDOf(t, approvalFloor(), tc.tool, tc.args)
		v := New(approvalFloor(id)).Adjudicate(context.Background(), approvalCall(tc.tool, tc.args))
		if v.Kind != abi.VerdictAllow || v.Meta["posture"] != PostureOperatorApproved {
			t.Fatalf("%s approved: kind = %v meta %v, want an operator-approved Allow", tc.tool, v.Kind, v.Meta)
		}
	}
}

// An approval names a call, so the floors it may not lift must still refuse that
// same call. Each probe is approved under its own id to prove the id is not what
// keeps it refused.
func TestOperatorApprovalNeverLiftsHardFloors(t *testing.T) {
	cases := []struct {
		name   string
		tool   string
		args   string
		reason abi.ReasonCode
	}{
		{"credential block", "Bash", `{"command":"echo x > .env"}`, abi.ReasonPolicyBlock},
		{"self-modify", "Edit", `{"file_path":"internal/kernel/x.go"}`, abi.ReasonSelfModify},
		{"secret exfil name deny", "exfiltrate", `{}`, abi.ReasonSecretExfil},
		{"hardwired egress floor", "WebFetch", `{"url":"http://169.254.169.254/latest/meta-data/"}`, egressfloor.ReasonEgressBlock},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var args map[string]any
			args = decodeArgs(context.Background(), approvalCall(c.tool, c.args))
			id := OperatorApprovalID(c.tool, args)
			v := New(approvalFloor(id)).Adjudicate(context.Background(), approvalCall(c.tool, c.args))
			if v.Kind != abi.VerdictDeny || v.Reason != c.reason {
				t.Fatalf("approved %s: kind = %v reason %s, want Deny/%s", c.name, v.Kind, abi.ReasonName(v.Reason), abi.ReasonName(c.reason))
			}
			if v.Meta[MetaOperatorApprovalID] != "" {
				t.Fatalf("%s refusal offers an approval id it cannot honor: meta %v", c.name, v.Meta)
			}
		})
	}
}

func TestOperatorApprovalAnswersReversibilityHold(t *testing.T) {
	p := approvalFloor()
	p.ArgPredicates = append(p.ArgPredicates, ArgPredicate{
		Tool: "Bash", Arg: "command", Kind: ArgDenyRegex,
		Re: regexp.MustCompile(`\bgit\s+push\b`), Reason: abi.ReasonPolicyBlock,
	})
	const push = `{"command":"git push origin main"}`
	id := approvalIDOf(t, p, "Bash", push)
	p.OperatorApprovals = map[string]bool{id: true}
	v := New(p).Adjudicate(context.Background(), approvalCall("Bash", push))
	if v.Kind != abi.VerdictAllow || v.Meta["would_deny"] != "POLICY_BLOCK,"+holdLiftName {
		t.Fatalf("approved git push: kind = %v meta %v, want Allow lifting POLICY_BLOCK then the hold", v.Kind, v.Meta)
	}

	// Without an approval the hold stands and offers the id.
	plain := approvalFloor()
	held := New(plain).Adjudicate(context.Background(), approvalCall("Bash", `{"command":"git push origin main"}`))
	if held.Kind != abi.VerdictRequireWitness || held.Meta[MetaOperatorApprovalID] == "" {
		t.Fatalf("unapproved push: kind = %v meta %v, want the hold carrying an approval id", held.Kind, held.Meta)
	}
}

func TestOperatorApprovalIDStable(t *testing.T) {
	a := OperatorApprovalID("Bash", map[string]any{"command": "rm -rf build"})
	b := OperatorApprovalID("bash", map[string]any{"command": "rm -rf   build", "description": "x", "timeout": 5})
	if a != b {
		t.Fatalf("ids differ for the same effect-bearing call: %s vs %s", a, b)
	}
	if c := OperatorApprovalID("Bash", map[string]any{"command": "rm -rf src"}); c == a {
		t.Fatalf("distinct calls share id %s", a)
	}
	if !regexp.MustCompile(`^ap-[0-9a-f]{12}$`).MatchString(a) {
		t.Fatalf("id %q does not match ap-<12 hex>", a)
	}
}

// Under a default-open posture (the attended guard floor) the dangerous-gotcha
// rung re-checks rm -rf after the arg rule. An approval must lift both, or the
// operator's approval silently fails (witnessed live 2026-10-23: the approved
// call was refused by monitor/gotchas with no approval id).
func TestOperatorApprovalLiftsGotchaFloorUnderDefaultOpen(t *testing.T) {
	p := approvalFloor()
	p.Posture = PostureDefaultOpen
	const rmBuild = `{"command":"rm -rf _dogfood/build"}`
	id := approvalIDOf(t, p, "Bash", rmBuild)

	// Without the arg rule the gotcha rung alone refuses, and it offers the id.
	gotchaOnly := p
	gotchaOnly.ArgPredicates = nil
	v := New(gotchaOnly).Adjudicate(context.Background(), approvalCall("Bash", rmBuild))
	if v.Kind != abi.VerdictDeny || v.By != gotchasBy || v.Meta[MetaOperatorApprovalID] != id {
		t.Fatalf("gotcha refusal: kind = %v by %q meta %v, want a %s deny offering %s", v.Kind, v.By, v.Meta, gotchasBy, id)
	}

	p.OperatorApprovals = map[string]bool{id: true}
	v = New(p).Adjudicate(context.Background(), approvalCall("Bash", rmBuild))
	if v.Kind != abi.VerdictAllow || v.Meta["posture"] != PostureOperatorApproved {
		t.Fatalf("approved call under default-open: kind = %v by %q meta %v, want an operator-approved Allow", v.Kind, v.By, v.Meta)
	}
}
