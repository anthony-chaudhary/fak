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

// ErrScopeRequired reports that a token has multiple scoped owners and cannot
// be resolved safely without an exact scope qualifier.
var ErrScopeRequired = errors.New("scope required for overloaded term")

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
	entry, ok, ambiguous := publicIndex.queryCanonical(canonicalTerm)
	if ambiguous {
		return QueryResponse{}, fmt.Errorf("%w: %q", ErrScopeRequired, canonicalTerm)
	}
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
	if publicIndex.ambiguous(term) {
		return QueryResponse{}, fmt.Errorf("%w: %q", ErrScopeRequired, term)
	}
	if !ok {
		return QueryResponse{}, fmt.Errorf("%w: %q", ErrCanonicalTermNotFound, term)
	}
	return queryResponse(entry, matchedAlias), nil
}

// QueryScoped performs an exact canonical lookup constrained by the required
// scope qualifier. The entry returns the stored scope unchanged.
func QueryScoped(canonicalTerm string, scope Scope) (QueryResponse, error) {
	return resolveScoped(canonicalTerm, scope, false)
}

// ResolveScoped performs an exact canonical-or-alias lookup constrained by the
// required scope qualifier. The entry returns the stored scope unchanged.
func ResolveScoped(term string, scope Scope) (QueryResponse, error) {
	return resolveScoped(term, scope, true)
}

func resolveScoped(term string, scope Scope, allowAlias bool) (QueryResponse, error) {
	if err := requireText("scope.kind", scope.Kind); err != nil {
		return QueryResponse{}, err
	}
	if err := requireText("scope.value", scope.Value); err != nil {
		return QueryResponse{}, err
	}
	var entry Entry
	var matchedAlias string
	var ok bool
	if allowAlias {
		entry, matchedAlias, ok = publicIndex.resolveScoped(term, scope)
	} else {
		for _, candidate := range publicIndex.canonical[term] {
			if candidate.Scope == scope {
				entry, ok = cloneEntry(candidate), true
				break
			}
		}
	}
	if !ok {
		return QueryResponse{}, fmt.Errorf("%w: %q in scope %s=%q", ErrCanonicalTermNotFound, term, scope.Kind, scope.Value)
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
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "kernel", Aliases: []string{}},
		Definition: "The internal/disambiguation Go package that validates and queries public terminology records.",
		Contrasts: []Contrast{{
			CanonicalTerm:       "fak CLI kernel",
			Explanation:         "The package API is not the fak command-line product surface.",
			RequiredPair:        boolPointer(false),
			ForbiddenConflation: boolPointer(true),
		}},
		Scope:     Scope{Kind: "package", Value: "internal/disambiguation"},
		Owner:     Owner{Leaf: "disambiguation", Lane: "canon"},
		Sources:   []SourceWitness{{Kind: "document", Locator: "internal/disambiguation/README.md", Revision: "public-seed/1"}},
		Freshness: Freshness{Verdict: "fresh", ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-11T00:00:00Z", Probe: "public-seed/1"},
		Lifecycle: Lifecycle{Class: "current", Rollout: "on"},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "kernel", Aliases: []string{}},
		Definition: "The fak command-line product surface for operating the agent kernel.",
		Contrasts: []Contrast{{
			CanonicalTerm:       "disambiguation package",
			Explanation:         "The command-line product surface is not the internal Go package API.",
			RequiredPair:        boolPointer(false),
			ForbiddenConflation: boolPointer(true),
		}},
		Scope:     Scope{Kind: "cli", Value: "fak"},
		Owner:     Owner{Leaf: "disambiguation", Lane: "canon"},
		Sources:   []SourceWitness{{Kind: "document", Locator: "README.md#how-it-works", Revision: "public-seed/1"}},
		Freshness: Freshness{Verdict: "fresh", ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-11T00:00:00Z", Probe: "public-seed/1"},
		Lifecycle: Lifecycle{Class: "current", Rollout: "on"},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "fak CLI kernel", Aliases: []string{}},
		Definition: "The fak command-line product surface, named as a contrast target for the package-scoped kernel entry.",
		Contrasts:  []Contrast{{CanonicalTerm: "kernel", Explanation: "The CLI surface and package-scoped kernel are distinct.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "cli", Value: "fak"}, Owner: Owner{Leaf: "disambiguation", Lane: "canon"},
		Sources:   []SourceWitness{{Kind: "document", Locator: "README.md#how-it-works", Revision: "public-seed/1"}},
		Freshness: Freshness{Verdict: "fresh", ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-11T00:00:00Z", Probe: "public-seed/1"}, Lifecycle: Lifecycle{Class: "current", Rollout: "on"},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "disambiguation package", Aliases: []string{}},
		Definition: "The internal/disambiguation package, named as a contrast target for the CLI-scoped kernel entry.",
		Contrasts:  []Contrast{{CanonicalTerm: "kernel", Explanation: "The package and CLI-scoped kernel are distinct.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "package", Value: "internal/disambiguation"}, Owner: Owner{Leaf: "disambiguation", Lane: "canon"},
		Sources:   []SourceWitness{{Kind: "document", Locator: "internal/disambiguation/README.md", Revision: "public-seed/1"}},
		Freshness: Freshness{Verdict: "fresh", ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-11T00:00:00Z", Probe: "public-seed/1"}, Lifecycle: Lifecycle{Class: "current", Rollout: "on"},
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
