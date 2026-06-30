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

func TestResidencyGateDeniesSensitiveLLMDRoute(t *testing.T) {
	v := (residencyGate{}).Adjudicate(context.Background(), &abi.ToolCall{
		Tool:   "summarize",
		Engine: LLMDEngineID,
		Args:   abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeFleet},
		Meta:   map[string]string{"sensitivity": "tenant"},
	})
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonTrustViolation {
		t.Fatalf("sensitive tag to llm-d route: got %v/%s, want Deny/TRUST_VIOLATION", v.Kind, abi.ReasonName(v.Reason))
	}
	if v.Meta["engine_route"] != LLMDEngineID {
		t.Fatalf("llm-d residency witness metadata = %+v", v.Meta)
	}
}

func TestResidencyGateDefersLocalAndUnscopedRoutes(t *testing.T) {
	cases := []abi.ToolCall{
		{Tool: "summarize", Engine: "local", Args: abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeTenant}},
		{Tool: "summarize", Engine: "remote", Args: abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeFleet}},
		{Tool: "summarize", Engine: "", Args: abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeTenant}},
		// On-box engine families stay local even under tenant scope: the in-kernel
		// model, an explicit on-device route, and a local-prefixed instance.
		{Tool: "summarize", Engine: "inkernel", Args: abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeTenant}},
		{Tool: "summarize", Engine: "on-device:0", Args: abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeTenant}},
		{Tool: "summarize", Engine: "local/llama", Args: abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeTenant}},
	}
	for _, c := range cases {
		if v := (residencyGate{}).Adjudicate(context.Background(), &c); v.Kind != abi.VerdictDefer {
			t.Fatalf("%+v: got %v/%s, want Defer", c, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestResidencyGateDeniesAggregatorAndCustomRoutes is the integration-floor
// regression: a routing Plan member may bind to a LiteLLM proxy, an OpenRouter
// model, or a user's own gateway. The previous allow-list-of-remote-names form
// matched none of these substrings and so failed OPEN — a tenant-scoped payload
// routed through "litellm/gpt-4o" reached the remote backend unrefused. The
// fail-closed classifier treats every route it cannot prove local as remote, so
// these all deny.
func TestResidencyGateDeniesAggregatorAndCustomRoutes(t *testing.T) {
	for _, engine := range []string{
		"litellm/gpt-4o", "openrouter/anthropic/claude-3.5", "portkey",
		"together/meta-llama", "groq", "fireworks/mixtral", "bedrock/claude",
		"vertex/gemini", "azure/gpt-4o", "my-proxy", "their-thing",
	} {
		v := (residencyGate{}).Adjudicate(context.Background(), &abi.ToolCall{
			Tool:   "summarize",
			Engine: engine,
			Args:   abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeTenant},
		})
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonTrustViolation {
			t.Fatalf("tenant scope to %q: got %v/%s, want Deny/TRUST_VIOLATION", engine, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}
