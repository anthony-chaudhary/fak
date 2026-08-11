package disambiguation

import (
	"errors"
	"fmt"
)

// QuerySchemaVersion identifies the exact canonical-term query response contract.
const QuerySchemaVersion = "fak-disambiguation-query/1"

// PublicIndexVersion identifies the immutable public seed set used by this reader.
// The later generator may replace the seed while retaining this read contract.
const PublicIndexVersion = "public-seed/1"

// ErrCanonicalTermNotFound reports that an exact canonical-term lookup missed.
var ErrCanonicalTermNotFound = errors.New("canonical term not found")

// QueryResponse is the versioned, machine-readable result of an exact lookup.
type QueryResponse struct {
	Schema       string `json:"schema"`
	IndexVersion string `json:"index_version"`
	Entry        Entry  `json:"entry"`
}

// Query performs an exact, case-sensitive lookup of a canonical term. Aliases
// are deliberately not accepted by this seam: callers cannot silently change
// which public term owns a meaning.
func Query(canonicalTerm string) (QueryResponse, error) {
	entry, ok := publicSeed[canonicalTerm]
	if !ok {
		return QueryResponse{}, fmt.Errorf("%w: %q", ErrCanonicalTermNotFound, canonicalTerm)
	}
	return QueryResponse{
		Schema:       QuerySchemaVersion,
		IndexVersion: PublicIndexVersion,
		Entry:        cloneEntry(entry),
	}, nil
}

var publicSeed = map[string]Entry{
	"agent kernel": {
		Schema: EntrySchemaVersion,
		Identity: Identity{
			CanonicalTerm: "agent kernel",
			Aliases:       []string{"fused agent kernel"},
		},
		Definition: "The fak management boundary that governs model traffic, tool effects, context, and recovery.",
		Contrasts: []Contrast{{
			CanonicalTerm: "compute kernel",
			Explanation:   "An arithmetic routine executed by a processor; it does not govern an agent's tool effects.",
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
}

func cloneEntry(entry Entry) Entry {
	entry.Identity.Aliases = append([]string(nil), entry.Identity.Aliases...)
	entry.Contrasts = append([]Contrast(nil), entry.Contrasts...)
	entry.Sources = append([]SourceWitness(nil), entry.Sources...)
	return entry
}
