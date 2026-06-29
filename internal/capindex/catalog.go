package capindex

import (
	"sort"
	"strings"
)

// Catalog binds one or more Resolvers (one per protocol/kind) to a single Index
// and answers the two resolution questions the loader asks: lookup-by-ref and
// query-by-intent. It is the seam that makes resolution protocol-blind — a
// caller asks for a capability and never has to know whether it came from a
// skill directory, an MCP server, or an A2A endpoint.
//
// At-rest the Catalog holds only cards (via its Index). A Lookup or a Query
// returns a Capability whose body is faulted lazily through the owning
// resolver's Fault, so nothing is paged until the caller materializes it.
type Catalog struct {
	resolvers map[CapKind]Resolver
	index     *Index
}

// NewCatalog builds an empty catalog.
func NewCatalog() *Catalog {
	return &Catalog{
		resolvers: make(map[CapKind]Resolver),
		index:     NewIndex(),
	}
}

// AddResolver registers a resolver for one kind. It does NOT index — call Sync
// to (re)build the at-rest card set. Registering a second resolver for a kind
// replaces the first.
func (c *Catalog) AddResolver(kind CapKind, r Resolver) {
	c.resolvers[kind] = r
}

// Sync rebuilds the index from every registered resolver's Index() and returns
// the CRUD diff against the prior snapshot. Re-syncing an unchanged catalog
// returns no changes (every digest matches); changing one capability returns
// exactly one Change. This is the ScaleMCP hash-diff sync at the catalog level.
func (c *Catalog) Sync() []Change {
	before := c.index.Snapshot()

	next := NewIndex()
	for _, r := range c.resolvers {
		next.RegisterAll(r.Index())
	}
	after := next.Snapshot()
	c.index = next

	return before.Diff(after)
}

// Index exposes the underlying card index (read access for callers that want to
// snapshot or enumerate cards).
func (c *Catalog) Index() *Index { return c.index }

// Lookup resolves a CapRef to a full Capability by faulting through the owning
// resolver. The body is paged lazily (the returned Capability's Resolve is not
// called here). Returns ErrNotFound if no resolver owns the ref's kind or the
// resolver has no such capability.
func (c *Catalog) Lookup(ref CapRef) (Capability, error) {
	r, ok := c.resolvers[ref.Kind]
	if !ok {
		return Capability{}, ErrNotFound
	}
	return r.Fault(ref)
}

// Query resolves an intent string to the best-matching Capability. It ranks the
// at-rest cards by a cheap lexical score over the trigger, tags, and name —
// never paging a body — then faults in only the winner. This is the
// active-discovery move: the model emits an intent, the catalog returns the one
// capability whose card best matches, body still lazy.
//
// Returns ErrNotFound if the index is empty or nothing scores above zero.
func (c *Catalog) Query(intent string) (Capability, error) {
	card, ok := c.bestCard(intent)
	if !ok {
		return Capability{}, ErrNotFound
	}
	return c.Lookup(card.Ref)
}

// RankCards returns every at-rest card scoring above zero for an intent, in
// descending score order (ties broken by name for determinism) — CARDS ONLY, no body
// paged. It is the multi-select form of bestCard: the system-prompt MMU's queried
// overlay (syspromptmmu Rung 3, #1261) faults the top-ranked winners up to a token
// budget instead of taking just the single best. An empty intent, an empty index, or
// no positive match returns nil.
func (c *Catalog) RankCards(intent string) []CapCard {
	terms := tokenize(intent)
	if len(terms) == 0 {
		return nil
	}

	type scored struct {
		card  CapCard
		score int
	}
	var ranked []scored
	for ref := range c.index.cards {
		card := c.index.cards[ref]
		if s := scoreCard(card, terms); s > 0 {
			ranked = append(ranked, scored{card: card, score: s})
		}
	}
	if len(ranked) == 0 {
		return nil
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].score != ranked[j].score {
			return ranked[i].score > ranked[j].score
		}
		return ranked[i].card.Ref.Name < ranked[j].card.Ref.Name // deterministic tiebreak
	})
	out := make([]CapCard, len(ranked))
	for i, r := range ranked {
		out[i] = r.card
	}
	return out
}

// bestCard returns the single highest-scoring card for an intent (cards only —
// no body paged), and whether any card scored above zero.
func (c *Catalog) bestCard(intent string) (CapCard, bool) {
	ranked := c.RankCards(intent)
	if len(ranked) == 0 {
		return CapCard{}, false
	}
	return ranked[0], true
}

// scoreCard is the cheap at-rest relevance score: how many intent terms appear
// in the card's name, trigger, or tags. A name match is weighted higher than a
// trigger/tag match so an exact-name intent wins.
func scoreCard(card CapCard, terms []string) int {
	name := strings.ToLower(card.Ref.Name)
	trigger := strings.ToLower(card.Trigger)
	tagBlob := strings.ToLower(strings.Join(card.Tags, " "))

	score := 0
	for _, t := range terms {
		if strings.Contains(name, t) {
			score += 3
		}
		if strings.Contains(trigger, t) {
			score++
		}
		if strings.Contains(tagBlob, t) {
			score++
		}
	}
	return score
}

// tokenize lowercases and splits an intent string into non-empty word terms.
func tokenize(s string) []string {
	fields := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9')
	})
	return fields
}
