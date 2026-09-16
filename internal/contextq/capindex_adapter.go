package contextq

import (
	"github.com/anthony-chaudhary/fak/internal/capindex"
)

// CapIndexResolver adapts a capindex.Resolver (the C1 protocol-blind keystone,
// implemented by the skill loader and by capindexgw for MCP tools / A2A agents)
// onto the C2/C3/C4 contextq.Resolver seam. It is the missing bridge that lets a
// real capindex resolver flow through QueryCapabilities UNCHANGED: the C2 query,
// the C3 residency ledger, and the C4 versioning already operate over the
// contextq Capability shape, so this adapter only translates between the two
// structurally-identical capability vocabularies.
//
// The adapter is deliberately pure translation: it holds no state and adds no
// policy, so any protocol capindex can express (skill, mcp-tool, a2a-agent, and
// future kinds) is queryable through C2 by construction.
type CapIndexResolver struct {
	R capindex.Resolver
}

// CapIndexResolver wraps r so it can be passed to QueryCapabilities. It is a
// convenience constructor for `CapIndexResolver{R: r}`.
func NewCapIndexResolver(r capindex.Resolver) CapIndexResolver {
	return CapIndexResolver{R: r}
}

// Index translates the wrapped resolver's capindex cards into contextq cards.
func (a CapIndexResolver) Index() []CapCard {
	cards := a.R.Index()
	out := make([]CapCard, 0, len(cards))
	for _, c := range cards {
		out = append(out, capIndexCardToContextQ(c))
	}
	return out
}

// Fault translates a contextq CapRef to the capindex identity, faults the
// wrapped resolver, and translates the returned capindex.Capability back into a
// contextq.Capability. A capindex Fault error maps to a zero contextq.Capability
// (Resolve nil) so QueryCapabilities skips it, matching the contract that only
// resolvable winners are admitted.
func (a CapIndexResolver) Fault(ref CapRef) Capability {
	cr := capindex.CapRef{
		Kind:    capindex.CapKind(ref.Source),
		Name:    ref.Name,
		Version: refVersion(ref),
	}
	got, err := a.R.Fault(cr)
	if err != nil {
		return Capability{}
	}
	return capIndexCapabilityToContextQ(got)
}

// capIndexCardToContextQ maps a capindex.CapCard to a contextq.CapCard. Fields
// with no contextq equivalent are dropped; the size hint prefers the serialized
// card length and falls back to the resident intent length so budgeting stays
// meaningful for resolvers that carry only one of the two.
func capIndexCardToContextQ(c capindex.CapCard) CapCard {
	card := CapCard{
		Name:          c.Ref.Name,
		Kind:          CapKind(c.Ref.Kind),
		Version:       c.Ref.Version,
		Trigger:       c.Trigger,
		Tags:          c.Tags,
		EstimateBytes: len(c.CardBytes),
	}
	if card.Trigger == "" {
		card.Trigger = c.Intent
	}
	if card.EstimateBytes == 0 {
		card.EstimateBytes = len(c.Intent)
	}
	return card
}

// capIndexCapabilityToContextQ maps a capindex.Capability to a contextq
// Capability. The body-faulting closure is carried across unchanged (Resolve),
// and Body is a copy of any already-materialized bytes so a resolved capability
// keeps its memoized body. A capability whose Resolve is nil and Body is empty
// has no pageable body and is preserved as such so C2 skips it.
func capIndexCapabilityToContextQ(c capindex.Capability) Capability {
	out := Capability{
		Ref:     capIndexRefToContextQ(c.Ref),
		Digest:  c.Digest,
		Card:    capIndexCardToContextQ(capindex.CapCard{Ref: c.Ref}),
		Resolve: c.Resolve,
		Scope:   c.Scope,
	}
	if out.Resolve == nil && len(c.Body) > 0 {
		body := c.Body
		out.Resolve = func() []byte { return body }
	}
	return out
}

// capIndexRefToContextQ maps a capindex.CapRef {Kind,Name,Version} to a
// contextq.CapRef. contextq's CapRef has no Version field, so the version rides
// on the Source string in the same "kind" encoding C2 uses when it faults.
func capIndexRefToContextQ(r capindex.CapRef) CapRef {
	return CapRef{
		Name:   r.Name,
		Source: string(r.Kind),
	}
}

// refVersion recovers a version from a contextq CapRef. contextq does not carry
// a version on its ref today, so the empty string ("latest") is the honest
// answer; if C4 versioning later widens the contextq ref, this is the single
// seam to thread it through.
func refVersion(ref CapRef) string {
	return ""
}
