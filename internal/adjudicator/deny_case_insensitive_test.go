package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestDenyCaseInsensitive(t *testing.T) {
	ctx := context.Background()

	// 1. Lowercase p.Deny rules must block mixed/upper-case incoming tool calls.
	a := New(Policy{
		Posture: PostureDefaultOpen,
		Deny: map[string]abi.ReasonCode{
			"exfiltrate":     abi.ReasonSecretExfil,
			"git_push":       abi.ReasonPolicyBlock,
			"bash":           abi.ReasonPolicyBlock,
			"refund_payment": abi.ReasonPolicyBlock,
		},
	})
	for _, tc := range []struct {
		tool       string
		wantReason abi.ReasonCode
	}{
		{"Exfiltrate", abi.ReasonSecretExfil},
		{"EXFILTRATE", abi.ReasonSecretExfil},
		{"GIT_PUSH", abi.ReasonPolicyBlock},
		{"Git_Push", abi.ReasonPolicyBlock},
		{"Bash", abi.ReasonPolicyBlock},
		{"BASH", abi.ReasonPolicyBlock},
		{"REFUND_PAYMENT", abi.ReasonPolicyBlock},
	} {
		v := a.Adjudicate(ctx, inlineCall(tc.tool, `{}`))
		if v.Kind != abi.VerdictDeny || v.Reason != tc.wantReason {
			t.Errorf("Adjudicate(%q): got %v/%v, want Deny/%v", tc.tool, v.Kind, v.Reason, tc.wantReason)
		}
		if !a.NeverAdmits(tc.tool) {
			t.Errorf("NeverAdmits(%q) = false, want true", tc.tool)
		}
	}

	// 2. Mixed-case p.Deny manifest rules must be normalized at New/SetPolicy and block lowercase calls.
	mixed := Policy{
		Posture: PostureDefaultOpen,
		Deny: map[string]abi.ReasonCode{
			"Exfiltrate": abi.ReasonSecretExfil,
			"GIT_PUSH":   abi.ReasonPolicyBlock,
			"Bash":       abi.ReasonPolicyBlock,
		},
	}
	// Raw Policy.NeverAdmits (without New) must also enforce case-insensitive Deny.
	if !mixed.NeverAdmits("exfiltrate") || !mixed.NeverAdmits("git_push") || !mixed.NeverAdmits("bash") {
		t.Fatalf("raw Policy.NeverAdmits failed case-insensitive Deny lookup")
	}

	a.SetPolicy(mixed)
	for _, tc := range []struct {
		tool       string
		wantReason abi.ReasonCode
	}{
		{"exfiltrate", abi.ReasonSecretExfil},
		{"EXFILTRATE", abi.ReasonSecretExfil},
		{"git_push", abi.ReasonPolicyBlock},
		{"bash", abi.ReasonPolicyBlock},
	} {
		v := a.Adjudicate(ctx, inlineCall(tc.tool, `{}`))
		if v.Kind != abi.VerdictDeny || v.Reason != tc.wantReason {
			t.Errorf("SetPolicy mixed-case Adjudicate(%q): got %v/%v, want Deny/%v", tc.tool, v.Kind, v.Reason, tc.wantReason)
		}
	}

	// 3. Unrelated tool under PostureDefaultOpen remains allowed.
	if v := a.Adjudicate(ctx, inlineCall("search_kb", `{}`)); v.Kind != abi.VerdictAllow {
		t.Errorf("unrelated tool under PostureDefaultOpen: got %v, want Allow", v.Kind)
	}
}
