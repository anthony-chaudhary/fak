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
	Ref          CapRef           // The capability reference
	Digest       string           // Content hash of the body (the ScaleMCP sync key)
	Trigger      string           // The trigger clause (what the model matches)
	Tags         []string         // Tags for filtering/ranking
	CardBytes    []byte           // The serialized card (opaque to the index)
	RequiredCaps []abi.Capability // Required capabilities to use this capability
}

// Capability is the protocol-blind capability object: a descriptor plus a
// lazily-paged body. The query, the index, the residency, and the versioning
// are all written once over this type and inherited by every protocol (skill,
// mcp-tool, a2a-agent, ...), which is what makes the loader protocol-blind.
//
// The fields named by the C1 contract:
//   - Ref     — {Kind, Name, Version}, the stable identity.
//   - Digest  — sha256 over the body (the ScaleMCP sync key).
//   - Card    — the tiny queryable card (trigger + tags); the only thing held
//     at rest, so the at-rest cost is O(cards), not O(bodies).
//   - Resolve — pages in the full body on FAULT, lazily and at most once per
//     capability value (the resolver memoizes via Body). nil once resolved.
//   - Scope   — abi.ShareScope: how widely this capability may be shared.
type Capability struct {
	Ref     CapRef         // {Kind, Name, Version}
	Digest  string         // content hash of the body (the ScaleMCP sync key)
	Card    []byte         // the tiny queryable card: trigger clause + tags (at-rest cost)
	Resolve func() []byte  // pages in the full body on FAULT; nil once Body is materialized
	Scope   abi.ShareScope // Share scope (Agent / Fleet / Tenant)

	// Body is the materialized full body once Resolve has faulted it in (nil
	// until then). Resolvers that page eagerly may set it directly. Body is the
	// memoization slot the lazy Resolve writes through, so the body faults at
	// most once.
	Body []byte
	// Caps are the negotiated abi capabilities this capability requires/offers.
	// Shared by every kind, which is part of what proves the loader is
	// protocol-blind.
	Caps []abi.Capability
}

// Materialize faults the body in if it is not already resident. It calls Resolve
// at most once: the first call writes Body and clears Resolve, so a second call
// is a cheap cache hit. This is the lazy-fault discipline the loader relies on —
// a card sits in the index for free until something actually needs the body.
func (c *Capability) Materialize() []byte {
	if c.Body == nil && c.Resolve != nil {
		c.Body = c.Resolve()
		c.Resolve = nil
	}
	return c.Body
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
