// Package contextq_test holds the external protocol-blind witness. It must be
// OUTSIDE package contextq: importing internal/capindexgw (which imports
// internal/gateway, which imports contextq) from an in-package test forms the
// cycle contextq -> capindexgw -> gateway -> contextq. An external test package
// is the sanctioned way to prove the real gateway-backed resolvers flow through
// the unchanged C2 query path.
package contextq_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/capindexgw"
	"github.com/anthony-chaudhary/fak/internal/contextq"
)

// TestRealGatewayResolversFlowThroughC2 is the strongest protocol-blind witness:
// the REAL capindexgw MCP and A2A resolvers (which fold the gateway catalog) are
// wrapped by the adapter and pushed through the REAL QueryCapabilities. If this
// passes, a gateway-backed protocol genuinely flows through the unchanged C2
// query path, not merely through a fake.
func TestRealGatewayResolversFlowThroughC2(t *testing.T) {
	mcp := capindexgw.NewMCPResolver(nil)
	a2a := capindexgw.NewA2AResolver()

	res := []contextq.Resolver{
		contextq.NewCapIndexResolver(mcp),
		contextq.NewCapIndexResolver(a2a),
	}

	mcpCards := mcp.Index()
	a2aCards := a2a.Index()
	if len(mcpCards) == 0 || len(a2aCards) == 0 {
		t.Skipf("gateway catalog empty (mcp=%d a2a=%d); nothing to route", len(mcpCards), len(a2aCards))
	}

	got := contextq.QueryCapabilities(res, contextq.CapQueryRequest{
		Intent:      mcpCards[0].Trigger + " " + a2aCards[0].Trigger,
		BudgetBytes: 1 << 30,
		K:           len(mcpCards) + len(a2aCards),
	}, nil)

	byKind := map[contextq.CapKind]int{}
	for _, w := range got.Winners {
		byKind[w.Card.Kind]++
	}
	if byKind[contextq.CapKindMCPTool] == 0 {
		t.Fatalf("no real MCP-tool capability flowed through C2: %+v", byKind)
	}
	if byKind[contextq.CapKindA2AAgent] == 0 {
		t.Fatalf("no real A2A-agent capability flowed through C2: %+v", byKind)
	}
}
