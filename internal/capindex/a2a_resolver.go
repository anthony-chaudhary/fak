package capindex

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// A2AResolver wraps the existing gateway/a2a.go code as a Resolver.
// It proves the loader is protocol-blind: this resolver exposes A2A methods
// as generic Capabilities, using the same abi.Capability type that MCP
// and skills use.
type A2AResolver struct {
	// No server reference needed; A2A uses a static registry
}

// NewA2AResolver creates an A2A resolver.
func NewA2AResolver() *A2AResolver {
	return &A2AResolver{}
}

// Index returns cheap cards only — the at-rest cost.
// For A2A, this is the method registry (name + description + scope).
func (r *A2AResolver) Index() []CapCard {
	// Use the existing A2A method registry from gateway/a2a.go
	methodSpecs := gateway.A2AMethodRegistryForResolver()

	cards := make([]CapCard, 0, len(methodSpecs))
	for _, spec := range methodSpecs {
		// Serialize the card for CapCard.CardBytes
		cardBytes, _ := json.Marshal(map[string]any{
			"name":        spec.Name,
			"description": spec.Description,
			"scope":       spec.Scope,
		})

		// Digest is a hash of the method definition
		digest := simpleDigest(spec.Name + ":" + spec.Scope)

		cards = append(cards, CapCard{
			Ref: CapRef{
				Kind:    CapKindA2AAgent,
				Name:    spec.Name,
				Version: "", // A2A methods don't version in this form
			},
			Digest:       digest,
			Trigger:      spec.Description, // Use description as trigger
			Tags:         []string{"a2a", "method", spec.Scope},
			CardBytes:    cardBytes,
			RequiredCaps: nil, // A2A methods don't require capabilities in this form
		})
	}
	return cards
}

// Fault pages in the full body for a given reference on demand.
// For A2A, this is the full method spec including inputs/outputs.
func (r *A2AResolver) Fault(ref CapRef) (Capability, error) {
	if ref.Kind != CapKindA2AAgent {
		return Capability{}, ErrKindMismatch
	}

	// Look up the method by name
	methodSpecs := gateway.A2AMethodRegistryForResolver()
	for _, spec := range methodSpecs {
		if spec.Name == ref.Name {
			// Full body is the complete method spec
			body, _ := json.Marshal(spec)
			digest := simpleDigest(spec.Name + ":" + spec.Scope)

			return Capability{
				Ref:    ref,
				Digest: digest,
				Body:   body,
				Scope:  abi.ScopeFleet, // A2A methods are fleet-wide by default
				Caps:   nil,            // A2A methods don't advertise capabilities in this form
			}, nil
		}
	}

	return Capability{}, ErrNotFound
}
