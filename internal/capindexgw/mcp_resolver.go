// Package capindexgw holds the gateway-backed capindex Resolvers (MCP tools,
// A2A methods). They live HERE, above capindex, because they import
// internal/gateway (tier-4): keeping them in capindex pinned the whole keystone
// to tier-4, which blocked the tier-3 skill-loader (ctxresidency/ctxmmu, #1106)
// from importing the core capindex types. The core CapRef/Capability/Index/
// skill-resolver are tier-2; this adapter that couples capindex to gateway sits
// at the higher of the two tiers it bridges.
package capindexgw

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/capindex"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// MCPResolver wraps the existing gateway/mcp.go code as a capindex.Resolver.
// It proves the loader is protocol-blind: this resolver exposes MCP tools
// as generic Capabilities, using the same capindex.Capability type that A2A
// and skills use.
type MCPResolver struct {
	server *gateway.Server // The gateway server (for tool descriptors)
}

// NewMCPResolver creates an MCP resolver from a gateway server.
func NewMCPResolver(server *gateway.Server) *MCPResolver {
	return &MCPResolver{server: server}
}

// Index returns cheap cards only — the at-rest cost.
// For MCP, this is the tools/list descriptor (name + description + inputSchema stub).
func (r *MCPResolver) Index() []capindex.CapCard {
	// Use the existing toolDescriptors from gateway/mcp.go
	toolDescs := gateway.ToolDescriptorsForResolver()

	cards := make([]capindex.CapCard, 0, len(toolDescs))
	for _, td := range toolDescs {
		name, _ := td["name"].(string)
		desc, _ := td["description"].(string)
		inputSchema, _ := td["inputSchema"].(json.RawMessage)

		// Serialize the card for CapCard.CardBytes
		cardBytes, _ := json.Marshal(map[string]any{
			"name":        name,
			"description": desc,
		})

		// Digest is a hash of the full schema (for now, use a placeholder;
		// the real implementation would SHA-256 the full tool definition)
		digest := simpleDigest(string(inputSchema))

		cards = append(cards, capindex.CapCard{
			Ref: capindex.CapRef{
				Kind:    capindex.CapKindMCPTool,
				Name:    name,
				Version: "", // MCP tools don't version in this form
			},
			Digest:       digest,
			Trigger:      desc, // Use description as trigger
			Tags:         []string{"mcp", "tool"},
			CardBytes:    cardBytes,
			RequiredCaps: nil, // MCP tools don't require capabilities in this form
		})
	}
	return cards
}

// Fault pages in the full body for a given reference on demand.
// For MCP, this is the full tool schema including inputSchema.
func (r *MCPResolver) Fault(ref capindex.CapRef) (capindex.Capability, error) {
	if ref.Kind != capindex.CapKindMCPTool {
		return capindex.Capability{}, capindex.ErrKindMismatch
	}

	// Look up the tool by name
	toolDescs := gateway.ToolDescriptorsForResolver()
	for _, td := range toolDescs {
		name, _ := td["name"].(string)
		if name == ref.Name {
			// Full body is the complete tool descriptor
			body, _ := json.Marshal(td)
			inputSchema, _ := td["inputSchema"].(json.RawMessage)
			digest := simpleDigest(string(inputSchema))

			return capindex.Capability{
				Ref:    ref,
				Digest: digest,
				Body:   body,
				Scope:  abi.ScopeFleet, // MCP tools are fleet-wide by default
				Caps:   nil,            // MCP tools don't advertise capabilities in this form
			}, nil
		}
	}

	return capindex.Capability{}, capindex.ErrNotFound
}

// simpleDigest computes the ScaleMCP sync key: a SHA-256 over the input bytes,
// rendered as "sha256:<hex>". A capability whose body changes gets a new digest,
// so a hot-swap is a cheap hash compare (Digest(old) != Digest(new)) rather than
// a re-read of the body. It delegates to capindex.Digest so every resolver and
// the index key on the same hash function.
func simpleDigest(s string) string {
	return capindex.Digest([]byte(s))
}
