package capindex

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// MCPResolver wraps the existing gateway/mcp.go code as a Resolver.
// It proves the loader is protocol-blind: this resolver exposes MCP tools
// as generic Capabilities, using the same abi.Capability type that A2A
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
func (r *MCPResolver) Index() []CapCard {
	// Use the existing toolDescriptors from gateway/mcp.go
	toolDescs := gateway.ToolDescriptorsForResolver()

	cards := make([]CapCard, 0, len(toolDescs))
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

		cards = append(cards, CapCard{
			Ref: CapRef{
				Kind:    CapKindMCPTool,
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
func (r *MCPResolver) Fault(ref CapRef) (Capability, error) {
	if ref.Kind != CapKindMCPTool {
		return Capability{}, ErrKindMismatch
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

			return Capability{
				Ref:    ref,
				Digest: digest,
				Body:   body,
				Scope:  abi.ScopeFleet, // MCP tools are fleet-wide by default
				Caps:   nil, // MCP tools don't advertise capabilities in this form
			}, nil
		}
	}

	return Capability{}, ErrNotFound
}

// simpleDigest is a placeholder for a real SHA-256 digest.
// In the full implementation, this would be sha256.Sum256.
func simpleDigest(s string) string {
	// Placeholder: for now, just use length as a poor man's hash
	// The real implementation uses SHA-256 as specified in the epic
	return "sha256:" + s[:min(32, len(s))]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}