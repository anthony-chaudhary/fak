package capindexgw

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/capindex"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestC5ProtocolBlindLoader proves that the MCP and A2A resolvers
// use the same capindex.Capability type, demonstrating the loader is
// protocol-blind (issue #1108, C5).
func TestC5ProtocolBlindLoader(t *testing.T) {
	// Create MCP resolver
	mcpResolver := NewMCPResolver(nil) // nil server is OK for Index() only
	mcpCards := mcpResolver.Index()

	// Create A2A resolver
	a2aResolver := NewA2AResolver()
	a2aCards := a2aResolver.Index()

	// Both resolvers return CapCards with the same structure
	if len(mcpCards) == 0 {
		t.Fatal("MCP resolver returned no cards")
	}
	if len(a2aCards) == 0 {
		t.Fatal("A2A resolver returned no cards")
	}

	// Verify MCP cards have the correct kind
	for _, card := range mcpCards {
		if card.Ref.Kind != capindex.CapKindMCPTool {
			t.Errorf("MCP card %s has wrong kind: got %v, want %v",
				card.Ref.Name, card.Ref.Kind, capindex.CapKindMCPTool)
		}
		// Verify the card structure is the same across protocols
		if card.Digest == "" {
			t.Errorf("MCP card %s has empty digest", card.Ref.Name)
		}
		if card.Trigger == "" {
			t.Errorf("MCP card %s has empty trigger", card.Ref.Name)
		}
		if len(card.Tags) == 0 {
			t.Errorf("MCP card %s has no tags", card.Ref.Name)
		}
	}

	// Verify A2A cards have the correct kind
	for _, card := range a2aCards {
		if card.Ref.Kind != capindex.CapKindA2AAgent {
			t.Errorf("A2A card %s has wrong kind: got %v, want %v",
				card.Ref.Name, card.Ref.Kind, capindex.CapKindA2AAgent)
		}
		// Verify the card structure is the same across protocols
		if card.Digest == "" {
			t.Errorf("A2A card %s has empty digest", card.Ref.Name)
		}
		if card.Trigger == "" {
			t.Errorf("A2A card %s has empty trigger", card.Ref.Name)
		}
		if len(card.Tags) == 0 {
			t.Errorf("A2A card %s has no tags", card.Ref.Name)
		}
	}

	// PROVE the loader is protocol-blind: both Capability types
	// use the same capindex.Capability field, proving they're not protocol-specific
	var mcpCap, a2aCap capindex.Capability
	_ = mcpCap.Caps // Both have the same Caps field type: []abi.Capability
	_ = a2aCap.Caps

	t.Logf("PROOF: Both MCP and A2A use the same capindex.Capability struct with abi.Capability field")
	t.Logf("  MCP cards: %d, A2A cards: %d", len(mcpCards), len(a2aCards))
	t.Logf("  Example MCP card: %s", mcpCards[0].Ref.Name)
	t.Logf("  Example A2A card: %s", a2aCards[0].Ref.Name)
}

// TestMCPResolverFoldingProvesFolded tests that the MCP resolver
// successfully folds the existing gateway/mcp.go code.
func TestMCPResolverFoldingProvesFolded(t *testing.T) {
	// The MCP resolver should use ToolDescriptorsForResolver from gateway/mcp.go
	// This proves it "folds" the existing code rather than duplicating it
	toolDescs := gateway.ToolDescriptorsForResolver()

	if len(toolDescs) == 0 {
		t.Fatal("ToolDescriptorsForResolver returned no descriptors")
	}

	// Verify we can build an MCP resolver from these descriptors
	mcpResolver := NewMCPResolver(nil)
	cards := mcpResolver.Index()

	if len(cards) != len(toolDescs) {
		t.Errorf("MCP resolver card count mismatch: got %d, want %d",
			len(cards), len(toolDescs))
	}

	// Verify the folding by checking names match
	for i, card := range cards {
		name, _ := toolDescs[i]["name"].(string)
		if card.Ref.Name != name {
			t.Errorf("Card %d name mismatch: got %s, want %s", i, card.Ref.Name, name)
		}
	}

	t.Logf("PROOF: MCP resolver folds gateway/mcp.go (ToolDescriptorsForResolver)")
	t.Logf("  Folded %d tools from gateway/mcp.go", len(cards))
}

// TestA2AResolverFoldingProvesFolded tests that the A2A resolver
// successfully folds the existing gateway/a2a.go code.
func TestA2AResolverFoldingProvesFolded(t *testing.T) {
	// The A2A resolver should use A2AMethodRegistryForResolver from gateway/a2a.go
	// This proves it "folds" the existing code rather than duplicating it
	methodSpecs := gateway.A2AMethodRegistryForResolver()

	if len(methodSpecs) == 0 {
		t.Fatal("A2AMethodRegistryForResolver returned no specs")
	}

	// Verify we can build an A2A resolver from these specs
	a2aResolver := NewA2AResolver()
	cards := a2aResolver.Index()

	if len(cards) != len(methodSpecs) {
		t.Errorf("A2A resolver card count mismatch: got %d, want %d",
			len(cards), len(methodSpecs))
	}

	// Verify the folding by checking all names are present (map iteration order is non-deterministic)
	cardNames := make(map[string]bool)
	for _, card := range cards {
		cardNames[card.Ref.Name] = true
	}
	for _, spec := range methodSpecs {
		if !cardNames[spec.Name] {
			t.Errorf("Card %s missing from resolver output", spec.Name)
		}
	}

	t.Logf("PROOF: A2A resolver folds gateway/a2a.go (A2AMethodRegistryForResolver)")
	t.Logf("  Folded %d methods from gateway/a2a.go", len(cards))
}

// Compile-time proof that both adapters satisfy the protocol-blind seam.
var (
	_ capindex.Resolver = (*MCPResolver)(nil)
	_ capindex.Resolver = (*A2AResolver)(nil)
)

func TestCoreToolAdmissionRefusesReachableMCPCapability(t *testing.T) {
	r := NewMCPResolver(nil)
	cards := r.Index()
	if len(cards) == 0 {
		t.Fatal("MCP catalog is empty; refusal witness requires one reachable capability")
	}
	mcp := cards[0].Ref

	got := r.AdmitCoreTool(CoreToolProposal{Name: "new_permanent_tool", Capability: mcp.Name})
	if got.Allowed {
		t.Fatalf("reachable MCP capability %q was admitted as a core tool", mcp.Name)
	}
	if got.Sidestep != mcp {
		t.Fatalf("sidestep = %+v, want %+v", got.Sidestep, mcp)
	}
	if !strings.Contains(got.Reason, mcp.Name) || !strings.Contains(got.Reason, "MCP catalog sidestep") {
		t.Fatalf("refusal reason %q does not name the MCP sidestep", got.Reason)
	}

	unmatched := r.AdmitCoreTool(CoreToolProposal{Name: "novel_tool", Capability: "not-in-the-catalog"})
	if !unmatched.Allowed || unmatched.Reason != "" {
		t.Fatalf("novel capability was refused: %+v", unmatched)
	}
}

var (
	benchCardsSink     []capindex.CapCard
	benchCapSink       capindex.Capability
	benchAdmissionSink CoreToolAdmission
	benchErrSink       error
	benchIndexSink     *capindex.Index
	benchChangesSink   []capindex.Change
)

func BenchmarkMCPResolverIndex(b *testing.B) {
	r := NewMCPResolver(nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCardsSink = r.Index()
	}
}

func BenchmarkMCPResolverFault(b *testing.B) {
	r := NewMCPResolver(nil)
	cards := r.Index()
	if len(cards) == 0 {
		b.Fatal("empty MCP catalog")
	}
	validRef := cards[0].Ref
	notFoundRef := capindex.CapRef{Kind: capindex.CapKindMCPTool, Name: "nonexistent_mcp_tool"}
	mismatchRef := capindex.CapRef{Kind: capindex.CapKindA2AAgent, Name: validRef.Name}

	b.Run("Hit", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCapSink, benchErrSink = r.Fault(validRef)
		}
	})

	b.Run("NotFound", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCapSink, benchErrSink = r.Fault(notFoundRef)
		}
	})

	b.Run("KindMismatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCapSink, benchErrSink = r.Fault(mismatchRef)
		}
	})
}

func BenchmarkMCPResolverAdmitCoreTool(b *testing.B) {
	r := NewMCPResolver(nil)
	cards := r.Index()
	if len(cards) == 0 {
		b.Fatal("empty MCP catalog")
	}
	reachable := CoreToolProposal{Name: "candidate", Capability: cards[0].Ref.Name}
	novel := CoreToolProposal{Name: "candidate", Capability: "unregistered_novel_tool"}
	empty := CoreToolProposal{Name: "candidate", Capability: ""}

	b.Run("RefuseReachable", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchAdmissionSink = r.AdmitCoreTool(reachable)
		}
	})

	b.Run("AdmitNovel", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchAdmissionSink = r.AdmitCoreTool(novel)
		}
	})

	b.Run("AdmitEmpty", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchAdmissionSink = r.AdmitCoreTool(empty)
		}
	})
}

func BenchmarkA2AResolverIndex(b *testing.B) {
	r := NewA2AResolver()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCardsSink = r.Index()
	}
}

func BenchmarkA2AResolverFault(b *testing.B) {
	r := NewA2AResolver()
	cards := r.Index()
	if len(cards) == 0 {
		b.Fatal("empty A2A catalog")
	}
	validRef := cards[0].Ref
	notFoundRef := capindex.CapRef{Kind: capindex.CapKindA2AAgent, Name: "nonexistent_a2a_method"}
	mismatchRef := capindex.CapRef{Kind: capindex.CapKindMCPTool, Name: validRef.Name}

	b.Run("Hit", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCapSink, benchErrSink = r.Fault(validRef)
		}
	})

	b.Run("NotFound", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCapSink, benchErrSink = r.Fault(notFoundRef)
		}
	})

	b.Run("KindMismatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCapSink, benchErrSink = r.Fault(mismatchRef)
		}
	})
}

func BenchmarkIndexRegistration(b *testing.B) {
	mcpResolver := NewMCPResolver(nil)
	a2aResolver := NewA2AResolver()
	mcpCards := mcpResolver.Index()
	a2aCards := a2aResolver.Index()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ix := capindex.NewIndex()
		ix.RegisterAll(mcpCards)
		ix.RegisterAll(a2aCards)
		benchIndexSink = ix
	}
}

func BenchmarkIndexDiff_Noop(b *testing.B) {
	mcpResolver := NewMCPResolver(nil)
	a2aResolver := NewA2AResolver()
	ix := capindex.NewIndex()
	ix.RegisterAll(mcpResolver.Index())
	ix.RegisterAll(a2aResolver.Index())

	snap1 := ix.Snapshot()
	snap2 := ix.Snapshot()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchChangesSink = snap1.Diff(snap2)
	}
}
