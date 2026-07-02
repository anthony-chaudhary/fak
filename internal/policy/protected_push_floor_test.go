package policy

import (
	"context"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// The second example floor (issue #449). The dogfood floor guards `git push` with a
// brittle Bash deny_regex on the COMMAND STRING — which only ever inspects the `Bash`
// tool. A structured `git_push` MCP tool (or any non-Bash route) carries its target as
// ARGUMENTS, so a command-string regex never matches it and the destructive call walks
// past the floor: the confused-deputy hole the issue names. The fix is argument-scoped:
// gate the structured tool's argument VALUES so a push to a protected ref, or a
// force-push, is denied AT THE FLOOR regardless of which surface issued the call.
//
// This test pins that contract against the SHIPPED manifest, and proves the gap it
// closes by contrast with a NAIVE floor that allows `git_push` with no arg rule (where
// the same protected-ref push is waved through).
func TestProtectedPushFloorVerdictMatrix(t *testing.T) {
	b, err := os.ReadFile("../../examples/protected-push-floor-policy.json")
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	p, err := Parse(b)
	if err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	a := adjudicator.New(p)

	call := func(tool, jsonArgs string) *abi.ToolCall {
		return &abi.ToolCall{
			Tool: tool,
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(jsonArgs)},
			Meta: map[string]string{},
		}
	}

	cases := []struct {
		name   string
		tool   string
		args   string
		kind   abi.VerdictKind
		reason abi.ReasonCode
	}{
		// A coding agent may legitimately push feature branches; the protected-ref
		// floor does not deny them, but the reversibility rung still requires a
		// preview confirmation before any outward-facing push dispatches.
		{"push feature branch", "git_push", `{"ref":"feature/x"}`, abi.VerdictRequireWitness, abi.ReasonNone},
		{"push fix branch", "git_push", `{"ref":"fix/bug-12"}`, abi.VerdictRequireWitness, abi.ReasonNone},

		// Structured route (the gap the Bash regex misses): push to a protected ref,
		// expressed as a TOOL ARGUMENT, is denied at the floor by value.
		{"push main", "git_push", `{"ref":"main"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},
		{"push master qualified", "git_push", `{"ref":"refs/heads/master"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},
		{"push trunk", "git_push", `{"ref":"trunk"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},
		{"push release line via branch arg", "git_push", `{"branch":"release/1.2"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},

		// Force-push is destructive on ANY ref — denied structurally, JSON boolean form.
		{"force push feature", "git_push", `{"ref":"feature/x","force":true}`, abi.VerdictDeny, abi.ReasonPolicyBlock},
		{"force-with-lease", "git_push", `{"ref":"feature/x","force_with_lease":true}`, abi.VerdictDeny, abi.ReasonPolicyBlock},

		// The Bash route is still covered (defense-in-depth, the first floor's guard).
		{"bash push", "Bash", `{"command":"git push origin main"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},
		{"bash rm -rf", "Bash", `{"command":"rm -rf /tmp/x"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},
		{"bash status ok", "Bash", `{"command":"git status"}`, abi.VerdictAllow, abi.ReasonNone},

		// Read-shaped structured git tools work.
		{"git status tool", "git_status", `{}`, abi.VerdictAllow, abi.ReasonNone},

		// Fail-closed still holds for a tool the floor never named.
		{"unknown tool", "send_email", `{"to":"x@evil.test"}`, abi.VerdictDeny, abi.ReasonDefaultDeny},
	}
	for _, c := range cases {
		v := a.Adjudicate(context.Background(), call(c.tool, c.args))
		if v.Kind != c.kind || v.Reason != c.reason {
			t.Errorf("%s: got Kind=%v Reason=%s, want Kind=%v Reason=%s",
				c.name, v.Kind, abi.ReasonName(v.Reason), c.kind, abi.ReasonName(c.reason))
		}
	}
}

// TestProtectedPushFloorClosesNaiveHole is the load-bearing contrast: a floor that
// allows the structured `git_push` tool WITHOUT an argument rule waves a push to a
// protected ref straight through (the confused-deputy hole), whereas the shipped
// argument-scoped floor refuses the identical call. This is what "structural,
// argument-scoped capability replaces the brittle command-string regex" buys.
func TestProtectedPushFloorClosesNaiveHole(t *testing.T) {
	pushMain := &abi.ToolCall{
		Tool: "git_push",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"ref":"main"}`)},
		Meta: map[string]string{},
	}

	// Naive floor: git_push allowed by NAME, no arg rule. The structured push to a
	// protected ref is not seen by any command-string regex, so the policy floor
	// does NOT return POLICY_BLOCK. The later reversibility rung now pauses the
	// outward-facing push, but that is not the protected-ref floor doing its job.
	naive, err := Parse([]byte(`{"allow":["git_push"]}`))
	if err != nil {
		t.Fatalf("parse naive: %v", err)
	}
	if v := adjudicator.New(naive).Adjudicate(context.Background(), pushMain); v.Kind != abi.VerdictRequireWitness || v.Reason != abi.ReasonNone {
		t.Fatalf("naive floor should not policy-block protected-ref push (reversibility may pause it), got %v/%s",
			v.Kind, abi.ReasonName(v.Reason))
	}

	// Shipped argument-scoped floor: the identical call is refused at the floor.
	b, err := os.ReadFile("../../examples/protected-push-floor-policy.json")
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	shipped, err := Parse(b)
	if err != nil {
		t.Fatalf("parse shipped: %v", err)
	}
	if v := adjudicator.New(shipped).Adjudicate(context.Background(), pushMain); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("shipped floor should deny protected-ref push, got %v/%s",
			v.Kind, abi.ReasonName(v.Reason))
	}
}
