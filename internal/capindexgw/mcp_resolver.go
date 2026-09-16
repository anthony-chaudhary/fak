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
	"fmt"
	"strings"

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

	// testDescriptors is a test-only seam: when non-nil it replaces the
	// gateway catalog Index()/Fault read. It lets a witness hold name and
	// description fixed while varying ONLY inputSchema size, which is the
	// cleanest way to prove the at-rest path never reads a body. Production
	// leaves it nil.
	testDescriptors []map[string]any
}

// descriptors returns the tool catalog this resolver reads, honoring the
// test-only override when set.
func (r *MCPResolver) descriptors() []map[string]any {
	if r.testDescriptors != nil {
		return r.testDescriptors
	}
	return gateway.ToolDescriptorsForResolver()
}

// setDescriptorsForTest installs a synthetic descriptor catalog for a test.
func (r *MCPResolver) setDescriptorsForTest(descs []map[string]any) {
	r.testDescriptors = descs
}

// NewMCPResolver creates an MCP resolver from a gateway server.
func NewMCPResolver(server *gateway.Server) *MCPResolver {
	return &MCPResolver{server: server}
}

// Index returns cheap cards only — the at-rest cost. It is O(cards), NOT
// O(cards × body): the digest it reports is derived from the CHEAP CARD bytes
// (name + description — exactly what CapCard.CardBytes serializes), never from
// the multi-KB inputSchema. This is the "0-for-∞" property the C5 acceptance
// requires: an N-tool MCP catalog costs the same at rest whether each tool's
// body is ten bytes or ten megabytes, because Index() never reads a body. The
// full inputSchema is materialized only by Fault.
func (r *MCPResolver) Index() []capindex.CapCard {
	// Use the existing toolDescriptors from gateway/mcp.go
	toolDescs := r.descriptors()

	cards := make([]capindex.CapCard, 0, len(toolDescs))
	for _, td := range toolDescs {
		name, _ := td["name"].(string)
		desc, _ := td["description"].(string)
		// NOTE: td["inputSchema"] is deliberately NOT read here. Touching it
		// would make the at-rest index O(bodies) and defeat the whole point.

		// Serialize the card for CapCard.CardBytes
		cardBytes, _ := json.Marshal(map[string]any{
			"name":        name,
			"description": desc,
		})

		// The index digest is the CARD-level sync key, not the body content
		// hash: it is capindex.Digest(cardBytes), cheap because cardBytes is
		// the small resident card. This is what makes Index() O(cards).
		//
		// It intentionally DIFFERS from Fault()'s digest by design:
		//   - Index digest = card-level sync key over {name, description};
		//     it moves when the cheap surface moves and costs no body read.
		//   - Fault digest = full-body content hash over the tool's
		//     inputSchema, so a body mutation that leaves the card unchanged
		//     still produces a new ScaleMCP sync key at fault time.
		// Making Index() agree with Fault() would require SHA-256 over every
		// multi-KB inputSchema on the index path — precisely the O(bodies)
		// defect this change removes.
		digest := capindex.Digest(cardBytes)

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
	toolDescs := r.descriptors()
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

// CoreToolProposal is the structured input to the footprint-floor admission
// check. Capability must be the exact catalog capability name; descriptions are
// deliberately not fuzzy-matched because a false refusal would block novel work.
type CoreToolProposal struct {
	Name       string
	Capability string
}

// CoreToolAdmission is the MCP-catalog-over-core-tool decision.
type CoreToolAdmission struct {
	Allowed  bool
	Reason   string
	Sidestep capindex.CapRef
}

// AdmitCoreTool refuses a permanent core-schema addition when its declared
// capability is already reachable through the MCP catalog. The refusal names
// the exact sidestep; unmatched or ambiguous free text remains admissible for
// the rest of the proposal review pipeline.
func (r *MCPResolver) AdmitCoreTool(proposal CoreToolProposal) CoreToolAdmission {
	capability := strings.TrimSpace(proposal.Capability)
	if capability == "" {
		return CoreToolAdmission{Allowed: true}
	}
	for _, card := range r.Index() {
		if card.Ref.Kind != capindex.CapKindMCPTool || card.Ref.Name != capability {
			continue
		}
		return CoreToolAdmission{
			Allowed:  false,
			Sidestep: card.Ref,
			Reason:   fmt.Sprintf("refuse core tool %q: capability is already reachable as MCP tool %q; use the MCP catalog sidestep", strings.TrimSpace(proposal.Name), card.Ref.Name),
		}
	}
	return CoreToolAdmission{Allowed: true}
}
