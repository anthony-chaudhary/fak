package policy

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

func TestManifestComplainLoadsIntoAdjudicatorPolicy(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{"complain":["custom_tool"," custom_tool "]}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(rt.Adjudicator.Complain) != 1 || !rt.Adjudicator.Complain["custom_tool"] {
		t.Fatalf("complain = %+v", rt.Adjudicator.Complain)
	}
	v := adjudicator.New(rt.Adjudicator).Adjudicate(context.Background(), &abi.ToolCall{Tool: "custom_tool", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}})
	if v.Kind != abi.VerdictAllow || v.Meta["would_deny"] != "DEFAULT_DENY" {
		t.Fatalf("complain trial verdict = %+v", v)
	}
}

func TestManifestComplainRejectsBlankTool(t *testing.T) {
	for _, body := range []string{`{"complain":[""]}`, `{"complain":["  "]}`} {
		if _, err := ParseRuntime([]byte(body)); err == nil {
			t.Fatalf("blank complain entry accepted: %s", body)
		}
	}
}

func TestManifestComplainPreservesUnknownFieldDiscipline(t *testing.T) {
	if _, err := ParseRuntime([]byte(`{"complain":["x"],"complain_typo":["y"]}`)); err == nil {
		t.Fatal("unknown field accepted")
	}
}

func TestManifestComplainEmptyIsBehaviorallyIdentical(t *testing.T) {
	base, err := ParseRuntime([]byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	empty, err := ParseRuntime([]byte(`{"complain":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	calls := []*abi.ToolCall{
		{Tool: "unknown", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}},
		{Tool: "search_kb", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}},
		{Tool: "bash", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"echo ok"}`)}},
	}
	for _, call := range calls {
		gotBase := adjudicator.New(base.Adjudicator).Adjudicate(context.Background(), call)
		gotEmpty := adjudicator.New(empty.Adjudicator).Adjudicate(context.Background(), call)
		if gotBase.Kind != gotEmpty.Kind || gotBase.Reason != gotEmpty.Reason || gotBase.By != gotEmpty.By {
			t.Fatalf("tool %q: absent=%+v empty=%+v", call.Tool, gotBase, gotEmpty)
		}
	}
}

func TestManifestComplainDoesNotBypassHardRefusals(t *testing.T) {
	tests := []struct{ name, manifest, tool, args string }{
		{"explicit deny", `{"complain":["danger"],"deny":{"danger":"POLICY_BLOCK"}}`, "danger", `{}`},
		{"self modify", `{"complain":["write"],"self_modify_globs":["AGENTS.md"]}`, "write", `{"path":"AGENTS.md"}`},
		{"argument rule", `{"complain":["send"],"arg_rules":[{"tool":"send","arg":"amount","allow_glob":"1"}]}`, "send", `{"amount":2}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rt, err := ParseRuntime([]byte(tt.manifest))
			if err != nil {
				t.Fatal(err)
			}
			v := adjudicator.New(rt.Adjudicator).Adjudicate(context.Background(), &abi.ToolCall{Tool: tt.tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(tt.args)}})
			if v.Kind != abi.VerdictDeny {
				t.Fatalf("complain bypassed hard refusal: %+v", v)
			}
			if v.Reason == abi.ReasonDefaultDeny {
				t.Fatalf("hard refusal collapsed to default deny: %+v", v)
			}
		})
	}
}
