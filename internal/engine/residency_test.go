package engine

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestResidencyGateDeniesTenantScopeRemoteRoute(t *testing.T) {
	v := (residencyGate{}).Adjudicate(context.Background(), &abi.ToolCall{
		Tool:   "summarize",
		Engine: "remote",
		Args:   abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeTenant},
	})
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonTrustViolation {
		t.Fatalf("tenant scope to remote route: got %v/%s, want Deny/TRUST_VIOLATION", v.Kind, abi.ReasonName(v.Reason))
	}
	if v.Meta["engine_route"] != "remote" || v.Meta["scope"] != "tenant" {
		t.Fatalf("residency witness metadata = %+v", v.Meta)
	}
}

func TestResidencyGateDeniesSensitiveTagRemoteRoute(t *testing.T) {
	v := (residencyGate{}).Adjudicate(context.Background(), &abi.ToolCall{
		Tool:   "summarize",
		Engine: "openai-primary",
		Args:   abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeFleet},
		Meta:   map[string]string{"sensitivity": "pii"},
	})
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonTrustViolation {
		t.Fatalf("sensitive tag to remote route: got %v/%s, want Deny/TRUST_VIOLATION", v.Kind, abi.ReasonName(v.Reason))
	}
}

func TestResidencyGateDefersLocalAndUnscopedRoutes(t *testing.T) {
	cases := []abi.ToolCall{
		{Tool: "summarize", Engine: "local", Args: abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeTenant}},
		{Tool: "summarize", Engine: "remote", Args: abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeFleet}},
		{Tool: "summarize", Engine: "", Args: abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeTenant}},
	}
	for _, c := range cases {
		if v := (residencyGate{}).Adjudicate(context.Background(), &c); v.Kind != abi.VerdictDefer {
			t.Fatalf("%+v: got %v/%s, want Defer", c, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}
