package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // resolver for the args read path
)

// The dev-agent floor DENIES the shared-history git mutations outright: a coding
// agent adapts code, it never moves the branch on its own say-so.
func TestDevAgentDeniesGitMutations(t *testing.T) {
	a := New(DevAgentPolicy())
	ctx := context.Background()
	for _, tool := range []string{"git_push", "git_merge", "git_tag"} {
		v := a.Adjudicate(ctx, inlineCall(tool, `{}`))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("%s: got %v/%s, want Deny/POLICY_BLOCK", tool, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// A write whose target is the kernel/policy spine is a SELF_MODIFY (ESCALATE), with
// bounded disclosure (the witness carries only the offending glob).
func TestDevAgentDeniesSpineSelfModify(t *testing.T) {
	a := New(DevAgentPolicy())
	ctx := context.Background()
	cases := []struct{ path, glob string }{
		{"internal/kernel/kernel.go", "internal/kernel/"},
		{"internal/policy/policy.go", "internal/policy/"},
		{"internal/adjudicator/decide.go", "internal/adjudicator/"},
		{".git/config", ".git/"},
	}
	for _, tc := range cases {
		v := a.Adjudicate(ctx, inlineCall("write_file", `{"path":"`+tc.path+`"}`))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
			t.Fatalf("write %s: got %v/%s, want Deny/SELF_MODIFY", tc.path, v.Kind, abi.ReasonName(v.Reason))
		}
		wp, ok := v.Payload.(abi.WitnessPayload)
		if !ok || wp.Claim != tc.glob {
			t.Fatalf("write %s: bounded witness = %+v, want glob %q", tc.path, v.Payload, tc.glob)
		}
	}
}

// ship_release is AFFIRMATIVELY allowed at the monitor — the require-witness gate
// is a separate rung (shipgate). At the floor alone it is an Allow.
func TestDevAgentAllowsShipReleaseAtFloor(t *testing.T) {
	a := New(DevAgentPolicy())
	v := a.Adjudicate(context.Background(), inlineCall("ship_release", `{}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("ship_release at the floor: got %v, want Allow", v.Kind)
	}
}

// An ordinary write OUTSIDE the spine is not a self-modify (it falls through to the
// fail-closed default-deny, not a SELF_MODIFY escalation) — the floor bounds writes
// to the spine specifically, it does not blanket-deny every write as self-modify.
func TestDevAgentNonSpineWriteIsNotSelfModify(t *testing.T) {
	a := New(DevAgentPolicy())
	v := a.Adjudicate(context.Background(), inlineCall("write_file", `{"path":"./out/report.txt"}`))
	if v.Reason == abi.ReasonSelfModify {
		t.Fatalf("a non-spine write must not be SELF_MODIFY, got %s", abi.ReasonName(v.Reason))
	}
}
