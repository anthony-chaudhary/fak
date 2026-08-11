package disambiguation

import (
	"errors"
	"fmt"
)

// QuerySchemaVersion identifies the canonical-identity query response contract.
const QuerySchemaVersion = "fak-disambiguation-query/1"

// PublicIndexVersion identifies the immutable public seed set used by this reader.
// The later generator may replace the seed while retaining this read contract.
const PublicIndexVersion = "public-seed/1"

// ErrCanonicalTermNotFound reports that an exact canonical-term lookup missed.
var ErrCanonicalTermNotFound = errors.New("canonical term not found")

// QueryResponse is the versioned, machine-readable result of a lookup. Entry
// always exposes the canonical owner. MatchedAlias is populated only when an
// exact declared alias selected that owner and preserves the caller's spelling.
type QueryResponse struct {
	Schema       string `json:"schema"`
	IndexVersion string `json:"index_version"`
	MatchedAlias string `json:"matched_alias,omitempty"`
	Entry        Entry  `json:"entry"`
}

// Query performs an exact, case-sensitive lookup of a canonical term. It stays
// canonical-only so callers that require canonical ownership cannot silently
// broaden their lookup to aliases.
func Query(canonicalTerm string) (QueryResponse, error) {
	entry, ok := publicIndex.queryCanonical(canonicalTerm)
	if !ok {
		return QueryResponse{}, fmt.Errorf("%w: %q", ErrCanonicalTermNotFound, canonicalTerm)
	}
	return queryResponse(entry, ""), nil
}

// Resolve performs an exact, case-sensitive lookup across canonical terms and
// declared aliases. The returned entry always carries the canonical identity;
// MatchedAlias records the exact alias used and is empty for canonical input.
func Resolve(term string) (QueryResponse, error) {
	entry, matchedAlias, ok := publicIndex.resolve(term)
	if !ok {
		return QueryResponse{}, fmt.Errorf("%w: %q", ErrCanonicalTermNotFound, term)
	}
	return queryResponse(entry, matchedAlias), nil
}

func queryResponse(entry Entry, matchedAlias string) QueryResponse {
	return QueryResponse{
		Schema:       QuerySchemaVersion,
		IndexVersion: PublicIndexVersion,
		MatchedAlias: matchedAlias,
		Entry:        entry,
	}
}

var publicEntries = []Entry{
	{
		Schema: EntrySchemaVersion,
		Identity: Identity{
			CanonicalTerm: "agent kernel",
			Aliases:       []string{"fused agent kernel"},
		},
		Definition: "The fak management boundary that governs model traffic, tool effects, context, and recovery.",
		Contrasts: []Contrast{{
			CanonicalTerm:       "compute kernel",
			Explanation:         "An arithmetic routine executed by a processor; it does not govern an agent's tool effects.",
			RequiredPair:        boolPointer(true),
			ForbiddenConflation: boolPointer(true),
		}},
		Scope: Scope{Kind: "product", Value: "fak"},
		Owner: Owner{Leaf: "kernel", Lane: "kernel"},
		Sources: []SourceWitness{{
			Kind:     "document",
			Locator:  "README.md#how-it-works",
			Revision: "692e4b57d0",
		}},
		Freshness: Freshness{
			Verdict:    "fresh",
			ReasonCode: "SOURCE_CURRENT",
			CheckedAt:  "2026-08-11T00:00:00Z",
			Probe:      "public-seed/1",
		},
		Lifecycle: Lifecycle{Class: "current", Rollout: "on"},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "compute kernel", Aliases: []string{}},
		Definition: "An arithmetic routine executed by a processor.",
		Contrasts: []Contrast{{
			CanonicalTerm:       "agent kernel",
			Explanation:         "The fak management boundary governs agent behavior; it is not a processor arithmetic routine.",
			RequiredPair:        boolPointer(true),
			ForbiddenConflation: boolPointer(true),
		}},
		Scope:     Scope{Kind: "computing", Value: "processor"},
		Owner:     Owner{Leaf: "kernel", Lane: "kernel"},
		Sources:   []SourceWitness{{Kind: "document", Locator: "README.md#how-it-works", Revision: "692e4b57d0"}},
		Freshness: Freshness{Verdict: "fresh", ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-11T00:00:00Z", Probe: "public-seed/1"},
		Lifecycle: Lifecycle{Class: "current", Rollout: "on"},
	},
}

var publicIndex = mustNewIndex(publicEntries)

func mustNewIndex(entries []Entry) *Index {
	index, err := NewIndex(entries)
	if err != nil {
		panic(fmt.Sprintf("invalid public disambiguation index: %v", err))
	}
	return index
}

func cloneEntry(entry Entry) Entry {
	entry.Identity.Aliases = append([]string(nil), entry.Identity.Aliases...)
	entry.Contrasts = append([]Contrast(nil), entry.Contrasts...)
	entry.Sources = append([]SourceWitness(nil), entry.Sources...)
	return entry
}

func boolPointer(value bool) *bool { return &value }
