package contextq

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/capindex"
)

// fakeCapIndexResolver is a minimal capindex.Resolver used to exercise the
// adapter without a live gateway. It serves one capability per kind so the
// adapter's ref/kind round-trip is observable end to end through C2.
type fakeCapIndexResolver struct {
	kind  capindex.CapKind
	name  string
	body  []byte
	fault int // number of Fault calls observed
}

func (f *fakeCapIndexResolver) Index() []capindex.CapCard {
	return []capindex.CapCard{{
		Ref:       capindex.CapRef{Kind: f.kind, Name: f.name},
		Digest:    capindex.Digest([]byte(f.name)),
		Trigger:   "resolve " + f.name,
		Tags:      []string{string(f.kind)},
		CardBytes: []byte("card:" + f.name),
	}}
}

func (f *fakeCapIndexResolver) Fault(ref capindex.CapRef) (capindex.Capability, error) {
	if ref.Kind != f.kind {
		return capindex.Capability{}, capindex.ErrKindMismatch
	}
	if ref.Name != f.name {
		return capindex.Capability{}, capindex.ErrNotFound
	}
	f.fault++
	return capindex.Capability{
		Ref:    ref,
		Digest: capindex.Digest(f.body),
		Body:   f.body,
		Scope:  abi.ScopeFleet,
	}, nil
}

// TestCapIndexResolver_MCPAndA2AFlowThroughC2 is the protocol-blind witness:
// a capindex resolver for an MCP tool and one for an A2A agent are wrapped by
// the adapter and passed to the REAL QueryCapabilities. Both must come back as
// winners with a non-nil Resolve, proving MCP and A2A capabilities flow through
// the C2 query path the same way skills do.
func TestCapIndexResolver_MCPAndA2AFlowThroughC2(t *testing.T) {
	mcp := &fakeCapIndexResolver{kind: capindex.CapKindMCPTool, name: "fak_admit", body: []byte(`{"name":"fak_admit"}`)}
	a2a := &fakeCapIndexResolver{kind: capindex.CapKindA2AAgent, name: "agent.info", body: []byte(`{"method":"agent.info"}`)}

	res := []Resolver{
		NewCapIndexResolver(mcp),
		NewCapIndexResolver(a2a),
	}

	ledger := NewCapabilityLedger()
	got := QueryCapabilities(res, CapQueryRequest{
		Intent:      "resolve fak_admit and agent.info",
		BudgetBytes: 1 << 20,
	}, ledger)

	if len(got.Winners) != 2 {
		t.Fatalf("want 2 winners (one per protocol), got %d: %+v", len(got.Winners), got.Winners)
	}

	kinds := map[CapKind]bool{}
	for _, w := range got.Winners {
		if w.Resolve == nil {
			t.Fatalf("winner %q has nil Resolve; the body was not paged in", w.Ref.Name)
		}
		if len(w.Resolve()) == 0 {
			t.Fatalf("winner %q paged in an empty body", w.Ref.Name)
		}
		kinds[w.Card.Kind] = true
	}
	if !kinds[CapKindMCPTool] {
		t.Fatalf("no MCP-tool capability survived the C2 query: %+v", kinds)
	}
	if !kinds[CapKindA2AAgent] {
		t.Fatalf("no A2A-agent capability survived the C2 query: %+v", kinds)
	}
	if mcp.fault == 0 || a2a.fault == 0 {
		t.Fatalf("C2 did not fault the wrapped resolvers: mcp=%d a2a=%d", mcp.fault, a2a.fault)
	}
}

// TestCapIndexResolver_FaultErrorSkipsCapability proves the adapter maps a
// capindex Fault error to a zero contextq.Capability (nil Resolve) so C2 omits
// the card instead of admitting an unresolvable winner.
func TestCapIndexResolver_FaultErrorSkipsCapability(t *testing.T) {
	r := &fakeCapIndexResolver{kind: capindex.CapKindMCPTool, name: "present", body: []byte("body")}
	adapter := NewCapIndexResolver(r)

	// A foreign kind must not resolve, and must not panic.
	got := adapter.Fault(CapRef{Name: "present", Source: string(capindex.CapKindA2AAgent)})
	if got.Resolve != nil {
		t.Fatalf("foreign-kind fault must yield a zero Capability, got Resolve != nil")
	}

	// A matching ref does resolve.
	ok := adapter.Fault(CapRef{Name: "present", Source: string(capindex.CapKindMCPTool)})
	if ok.Resolve == nil {
		t.Fatalf("matching fault must yield a resolvable Capability")
	}
}

// TestRealGatewayResolversFlowThroughC2 lives in capindex_adapter_external_test.go
// (package contextq_test): importing capindexgw from an IN-PACKAGE test would form
// the cycle contextq -> capindexgw -> gateway -> contextq, which Go rejects.
