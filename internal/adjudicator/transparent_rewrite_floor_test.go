package adjudicator_test

import (
	"context"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // resolver for the TRANSFORM's re-stored args
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// The in-syscall Read / Grep / Glob → fak_read / fak_grep / fak_glob rewrite
// (#11150, #11499) repairs the dispatch shape of an ADMITTED call; it is never an
// admission. It used to return TRANSFORM before the arg predicates and default-deny
// ran, and the kernel dispatches a TRANSFORM, so a floor that confined Read by
// arg_rules, or never listed it, was bypassed. These pin that the floor judges the
// ORIGINAL call first.

func rewriteFloor(t *testing.T, manifest []byte) *adjudicator.Adjudicator {
	t.Helper()
	rt, err := policy.ParseRuntime(manifest)
	if err != nil {
		t.Fatalf("parse floor: %v", err)
	}
	return adjudicator.New(rt.Adjudicator)
}

func rewriteDecide(a *adjudicator.Adjudicator, tool, args string) abi.Verdict {
	return a.Adjudicate(context.Background(), &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)},
	})
}

func wantDefaultDeny(t *testing.T, a *adjudicator.Adjudicator, tool, args string) {
	t.Helper()
	v := rewriteDecide(a, tool, args)
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
		t.Errorf("%s %s: got %v/%s by=%s, want Deny/DEFAULT_DENY",
			tool, args, v.Kind, abi.ReasonName(v.Reason), v.By)
	}
}

// examples/vm-fs-guard confines Read / Grep / Glob to /workspace/** by arg_rules. A
// call outside that view is the arg rule's DEFAULT_DENY, not a fak_* TRANSFORM; a
// call inside it is admitted and still rewritten.
func TestTransparentRewriteHonorsArgRuleConfinement(t *testing.T) {
	b, err := os.ReadFile("../../examples/vm-fs-guard/vm-fs-floor.json")
	if err != nil {
		t.Fatalf("read vm-fs floor: %v", err)
	}
	a := rewriteFloor(t, b)

	wantDefaultDeny(t, a, "Read", `{"file_path":"/etc/shadow"}`)
	wantDefaultDeny(t, a, "Grep", `{"pattern":"x","path":"/etc"}`)
	wantDefaultDeny(t, a, "Glob", `{"pattern":"*","path":"/etc"}`)

	for _, tc := range []struct{ tool, args, newTool string }{
		{"Read", `{"file_path":"/workspace/src/main.go"}`, "fak_read"},
		{"Grep", `{"pattern":"x","path":"/workspace/src"}`, "fak_grep"},
		{"Glob", `{"pattern":"*","path":"/workspace/src"}`, "fak_glob"},
	} {
		v := rewriteDecide(a, tc.tool, tc.args)
		tp, _ := v.Payload.(abi.TransformPayload)
		if v.Kind != abi.VerdictTransform || tp.NewTool != tc.newTool {
			t.Errorf("%s %s: got %v/%s NewTool=%q, want Transform to %s",
				tc.tool, tc.args, v.Kind, abi.ReasonName(v.Reason), tp.NewTool, tc.newTool)
		}
	}
}

// A floor that never lists Read / Grep / Glob default-denies them exactly as it
// default-denies the fak_* tools they would be rewritten to: the rewrite must not
// admit what neither name is allowed to do.
func TestTransparentRewriteHonorsAllowListOmission(t *testing.T) {
	a := rewriteFloor(t, []byte(`{"version":"fak-policy/v1","allow":["Write"]}`))
	for _, tc := range []struct{ tool, fakTool, args string }{
		{"Read", "fak_read", `{"file_path":"/etc/shadow"}`},
		{"Grep", "fak_grep", `{"pattern":"x","path":"/etc"}`},
		{"Glob", "fak_glob", `{"pattern":"*","path":"/etc"}`},
	} {
		wantDefaultDeny(t, a, tc.tool, tc.args)
		wantDefaultDeny(t, a, tc.fakTool, tc.args)
	}
}

// The rewrite PROMOTES Read's filePath / path alias onto file_path. A deny rule on
// file_path must refuse the promoted call too: judging only the raw keys let
// Read {"filePath":"/etc/shadow"} dodge the rule and dispatch as fak_read.
func TestTransparentRewriteJudgesPromotedAlias(t *testing.T) {
	a := rewriteFloor(t, []byte(`{"version":"fak-policy/v1","allow":["Read"],
		"arg_rules":[{"tool":"Read","arg":"file_path","deny_regex":"^/etc/","reason":"POLICY_BLOCK"}]}`))
	for _, args := range []string{`{"file_path":"/etc/shadow"}`, `{"filePath":"/etc/shadow"}`, `{"path":"/etc/shadow"}`} {
		v := rewriteDecide(a, "Read", args)
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("Read %s: got %v/%s by=%s, want Deny/POLICY_BLOCK", args, v.Kind, abi.ReasonName(v.Reason), v.By)
		}
	}
	v := rewriteDecide(a, "Read", `{"filePath":"/workspace/main.go"}`)
	if tp, _ := v.Payload.(abi.TransformPayload); v.Kind != abi.VerdictTransform || tp.NewTool != "fak_read" {
		t.Errorf("in-bounds Read filePath: got %v NewTool=%q, want Transform to fak_read", v.Kind, tp.NewTool)
	}
}

// A floor that allows Read but denies fak_read BY NAME keeps the host tool: the
// rewrite must not dispatch a tool the operator refused.
func TestTransparentRewriteHonorsTargetNameDeny(t *testing.T) {
	a := rewriteFloor(t, []byte(`{"version":"fak-policy/v1","allow":["Read"],"deny":{"fak_read":"POLICY_BLOCK"}}`))
	if v := rewriteDecide(a, "Read", `{"file_path":"README.md"}`); v.Kind != abi.VerdictAllow {
		t.Errorf("Read with fak_read name-denied: got %v by=%s, want Allow of the host tool", v.Kind, v.By)
	}
}
