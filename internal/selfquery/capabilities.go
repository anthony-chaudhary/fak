package selfquery

import (
	"errors"
	"strings"
)

// capabilities.go answers #1500 (C2 of the #1494 "fak can answer 'what can I
// do?'" epic): `fak capabilities [<intent>]` / fak_capabilities. Unlike `fak
// feature query` (Query/Cards above), which surfaces the FULL catalog — dev
// facts, context-plan/ask-policy cards, every registered MCP tool, capability
// skills — this surface is deliberately narrower and memory-forward: exactly
// the three families #1500 names (the memq memory algebra, the self-index
// verbs, and the kernel shared-path verbs fak_changes/dos_arbitrate), each
// carrying the exact call to make. It reuses memoryCards/devSurfaceCards
// rather than re-deriving them, so the two surfaces never disagree about what
// a memory driver or a self-index verb IS — they differ only in which cards
// they choose to rank and how a memory-driver card's Request is shaped.
//
// Two request-shape differences from the generic feature-query view matter for
// #1500's proof criterion:
//   - a memory-driver card here carries a ready fak_memory_run call (driver +
//     intent + apply=false), where fak feature query deliberately points at
//     fak_memory_explain first (see memoryCards' Note) — #1500 explicitly asks
//     for "a copy-pasteable fak_memory_run call".
//   - the two "context hygiene" drivers (clean, compact) get synonym tags
//     naming the family they belong to, so an intent like "compact my context"
//     surfaces the driver whose NAME wasn't in the query at all. Without this,
//     rankCards' plain token-overlap scoring only ever matches the driver whose
//     literal name appears in the intent (see score() in selfquery.go); the
//     proof asks for BOTH ranked to the top, not just the one named.

// CapabilitiesRequest is a capabilities query: Query is optional (an empty
// query returns every card in stable source/kind/name order, mirroring
// `fak index verbs` with no filter); Limit optionally caps the result count.
type CapabilitiesRequest struct {
	Query string
	Limit int
}

// CapabilitiesResponse is the ranked (or, for an empty query, stably sorted)
// toolbelt: memory drivers, self-index verbs, and kernel shared-path verbs.
type CapabilitiesResponse struct {
	Query string        `json:"query"`
	Cards []FeatureCard `json:"cards"`
}

// memoryHygieneSynonyms names, for the two memory drivers that reduce context
// bloat, the sibling tags that let an intent naming only ONE of them ("compact
// my context") still surface the other. This is additive to memoryCards'
// existing tags — it does not change what `fak feature query` sees, since that
// surface builds its own card set from c.memoryCards() independently of this
// augmentation.
var memoryHygieneSynonyms = map[string][]string{
	"clean":   {"context", "hygiene", "compact", "cleanup", "trim"},
	"compact": {"context", "hygiene", "clean", "cleanup", "trim", "consolidate"},
}

// Capabilities returns the memory-forward toolbelt view: the memq drivers, the
// `fak index *` self-index verbs, and the kernel shared-path verbs
// (fak_changes, dos_arbitrate), ranked by req.Query when non-empty.
func (c *Catalog) Capabilities(req CapabilitiesRequest) (CapabilitiesResponse, error) {
	if req.Limit < 0 {
		return CapabilitiesResponse{}, errors.New("capabilities limit must be non-negative")
	}
	q := strings.TrimSpace(req.Query)
	var all []FeatureCard
	for _, card := range c.memoryCards() {
		all = append(all, withCapabilitiesMemoryRequest(withHygieneSynonyms(card), firstNonEmpty(q, "the task at hand")))
	}
	all = append(all, c.devSurfaceCards()...)
	all = append(all, kernelVerbCards()...)

	var cards []FeatureCard
	if q == "" {
		cards = append([]FeatureCard(nil), all...)
		sortCards(cards)
	} else {
		cards = rankCards(all, q)
	}
	if req.Limit > 0 && len(cards) > req.Limit {
		cards = cards[:req.Limit]
	}
	return CapabilitiesResponse{Query: q, Cards: cards}, nil
}

// withHygieneSynonyms adds the cross-driver family tags (see
// memoryHygieneSynonyms) to a memory-driver card; cards for drivers with no
// entry in the map are returned unchanged.
func withHygieneSynonyms(c FeatureCard) FeatureCard {
	driver := strings.TrimPrefix(c.Name, "memory-driver:")
	extra, ok := memoryHygieneSynonyms[driver]
	if !ok {
		return c
	}
	c.Tags = cleanTags(append(append([]string(nil), c.Tags...), extra...))
	return c
}

// withCapabilitiesMemoryRequest replaces a memory-driver card's Request with a
// ready-to-run fak_memory_run call (driver + intent + apply=false — a mutation
// stays a proposal until an operator authorizes apply=true), per #1500's proof
// criterion ("a copy-pasteable fak_memory_run call"). fak feature query keeps
// its own explain-first Request untouched; this only affects the card copy
// returned from Capabilities.
func withCapabilitiesMemoryRequest(c FeatureCard, intent string) FeatureCard {
	driver := strings.TrimPrefix(c.Name, "memory-driver:")
	c.Request = RequestShape{
		Route:   "mcp/tools-call",
		MCPTool: "fak_memory_run",
		Command: []string{"fak", "memory", "run", "--driver", driver, "--intent", intent},
		Arguments: map[string]any{
			"driver": driver,
			"intent": intent,
			"apply":  false,
		},
		Note:     "apply=false proposes the effect; pass --apply (or apply=true) to enact it",
		Executed: false,
	}
	return c
}

// kernelVerbCards describes the kernel shared-path verbs #1500 names by name:
// fak_changes (fak's own cross-agent change feed) and dos_arbitrate (the
// external DOS admission kernel's lane-lease verb, which fak's guard floor
// allows through read-only but does not implement itself — so it cannot come
// from toolCards, which only sees fak's own registered MCP tools).
func kernelVerbCards() []FeatureCard {
	return []FeatureCard{
		card("kernel-verb", "fak_changes",
			"drain cross-agent mutation/revocation events since a cursor — reach for this before re-planning a cache over a session shared with other agents",
			[]string{"live", "kernel", "shared-path", "changes", "cache", "coherence"},
			"mcp:fak_changes", EffectRead, "", "gateway.wire", digestOf("kernel-verb:fak_changes"),
			RequestShape{Route: "mcp/tools-call", MCPTool: "fak_changes", Arguments: map[string]any{"since": 0}, Executed: false}),
		card("kernel-verb", "dos_arbitrate",
			"ask the external DOS admission kernel whether two workers may run concurrently on this tree without colliding — reach for this before a multi-agent fan-out",
			[]string{"live", "kernel", "shared-path", "dos", "arbitrate", "lane", "concurrency"},
			"mcp:dos_arbitrate", EffectRead, "", "dos (external MCP server)", digestOf("kernel-verb:dos_arbitrate"),
			RequestShape{
				Route:     "mcp/tools-call",
				MCPTool:   "dos_arbitrate",
				Arguments: map[string]any{"lane": "<lane>"},
				Note:      "external DOS MCP server tool, not implemented by fak — fak's guard floor allows it through as a read-only DOS verb",
				Executed:  false,
			}),
	}
}
