package adjudicator

import (
	"context"
	"regexp"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
)

func TestPostureDefaultOpen(t *testing.T) {
	ctx := context.Background()

	policy := Policy{
		Posture: PostureDefaultOpen,
		Deny: map[string]abi.ReasonCode{
			"explicit_denied_tool": abi.ReasonPolicyBlock,
		},
		SelfModifyGlobs: []string{"internal/abi/"},
		ArgPredicates: []ArgPredicate{
			{
				Tool:   "Bash",
				Arg:    "command",
				Kind:   ArgDenyRegex,
				Re:     regexp.MustCompile(`rm\s+-rf`),
				Reason: abi.ReasonPolicyBlock,
			},
		},
	}
	a := New(policy)

	// 1. Unlisted benign tool is admitted with Kind == VerdictAllow, Meta["posture"] == "default_open".
	v := a.Adjudicate(ctx, inlineCall("unlisted_custom_tool", `{"foo":"bar"}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("unlisted tool: got Kind=%v (%s), want VerdictAllow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v.Meta["posture"] != "default_open" {
		t.Fatalf("unlisted tool: got Meta[posture]=%q, want %q", v.Meta["posture"], "default_open")
	}

	// 2. Explicit Deny rules are still refused.
	v = a.Adjudicate(ctx, inlineCall("explicit_denied_tool", `{}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("explicit Deny: got %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}

	// 3. SelfModifyGlobs (path and command) are still refused.
	v = a.Adjudicate(ctx, inlineCall("write_file", `{"path":"internal/abi/types.go"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Fatalf("self-modify path: got %v/%s, want Deny/SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
	}
	v = a.Adjudicate(ctx, inlineCall("Bash", `{"command":"sed -i s/a/b/ internal/abi/types.go"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Fatalf("self-modify command: got %v/%s, want Deny/SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
	}

	// 4. Arg predicates (e.g. rm -rf) are still refused.
	v = a.Adjudicate(ctx, inlineCall("Bash", `{"command":"rm -rf /tmp/test"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("arg predicate: got %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}

	// 5. SSRF metadata egress is still refused.
	v = a.Adjudicate(ctx, inlineCall("WebFetch", `{"url":"http://169.254.169.254/latest/meta-data"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != egressfloor.ReasonEgressBlock {
		t.Fatalf("ssrf egress: got %v/%s, want Deny/EGRESS_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}

	// 6. PostureFailClosed still returns DEFAULT_DENY for unlisted tools.
	fcPolicy := Policy{Posture: PostureFailClosed}
	fcAdj := New(fcPolicy)
	v = fcAdj.Adjudicate(ctx, inlineCall("unlisted_custom_tool", `{}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("fail_closed unlisted tool: got %v/%s, want Deny/DEFAULT_DENY", v.Kind, abi.ReasonName(v.Reason))
	}

	// 7. NeverAdmits returns false under PostureDefaultOpen for unlisted tools, and true for denied tools.
	if policy.NeverAdmits("unlisted_custom_tool") {
		t.Fatalf("NeverAdmits(unlisted_custom_tool) = true under PostureDefaultOpen, want false")
	}
	if !policy.NeverAdmits("explicit_denied_tool") {
		t.Fatalf("NeverAdmits(explicit_denied_tool) = false under PostureDefaultOpen, want true")
	}
}
