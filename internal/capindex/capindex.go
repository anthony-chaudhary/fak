package capindex

import (
	"github.com/anthony-chaudhary/fak/internal/abi"
)

// CapKind is the kind of capability: skill, mcp-tool, a2a-agent, etc.
type CapKind string

const (
	CapKindSkill    CapKind = "skill"
	CapKindMCPTool  CapKind = "mcp-tool"
	CapKindA2AAgent CapKind = "a2a-agent"
)

// CapRef uniquely identifies a capability across kinds and versions.
type CapRef struct {
	Kind    CapKind // The capability kind (skill, mcp-tool, a2a-agent, ...)
	Name    string  // The capability name
	Version string  // The capability version (optional; empty = latest)
}

// CapCard is the tiny queryable card: trigger clause + tags (layer-1 cost).
// It is the cheap at-rest metadata the index stores.
type CapCard struct {
	Ref       CapRef              // The capability reference
	Digest    string              // Content hash of the body (the ScaleMCP sync key)
	Trigger   string              // The trigger clause (what the model matches)
	Tags      []string            // Tags for filtering/ranking
	CardBytes []byte              // The serialized card (opaque to the index)
	RequiredCaps []abi.Capability // Required capabilities to use this capability
}

// Capability is the full capability object with a lazily paged body.
type Capability struct {
	Ref    CapRef          // {Kind, Name, Version}
	Digest string          // content hash of the body
	Body   []byte          // The full body (nil if not yet faulted in)
	Scope  abi.ShareScope  // Share scope (Agent / Fleet / Tenant)
	Caps   []abi.Capability // Negotiated capabilities this capability requires/offers
}

// Resolver is the protocol-generic interface for resolving capabilities.
// Every protocol (skill, MCP, A2A, ...) implements this one seam.
type Resolver interface {
	// Index returns cheap cards only — the at-rest cost. This is the
	// protocol's "list" operation: tools/list for MCP, skills/* for local skills,
	// etc.
	Index() []CapCard

	// Fault pages in the full body for a given reference on demand.
	// This is the protocol's "get" operation: the full tool schema for MCP,
	// the full SKILL.md for a skill, etc.
	Fault(ref CapRef) (Capability, error)
}