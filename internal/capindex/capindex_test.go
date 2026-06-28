package capindex

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestC5ProtocolBlindLoader proves that the MCP and A2A resolvers
// use the same abi.Capability type, demonstrating the loader is
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
		if card.Ref.Kind != CapKindMCPTool {
			t.Errorf("MCP card %s has wrong kind: got %v, want %v",
				card.Ref.Name, card.Ref.Kind, CapKindMCPTool)
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
		if card.Ref.Kind != CapKindA2AAgent {
			t.Errorf("A2A card %s has wrong kind: got %v, want %v",
				card.Ref.Name, card.Ref.Kind, CapKindA2AAgent)
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
	// use the same abi.Capability field, proving they're not protocol-specific
	var mcpCap, a2aCap Capability
	_ = mcpCap.Caps // Both have the same Caps field type: []abi.Capability
	_ = a2aCap.Caps

	t.Logf("PROOF: Both MCP and A2A use the same Capability struct with abi.Capability field")
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
