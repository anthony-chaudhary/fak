package disambiguation

import (
	"sort"
	"strings"
)

// SearchSchemaVersion identifies the ranked terminology-search response contract.
const SearchSchemaVersion = "fak-disambiguation-search/1"

// SearchVerdict tells callers whether search found one safe owner or requires
// them to choose explicitly from multiple candidates.
type SearchVerdict string

const (
	SearchVerdictExact     SearchVerdict = "exact"
	SearchVerdictAlias     SearchVerdict = "alias"
	SearchVerdictPrefix    SearchVerdict = "prefix"
	SearchVerdictAmbiguous SearchVerdict = "ambiguous"
	SearchVerdictNotFound  SearchVerdict = "not_found"
)

// SearchMatch preserves the token and namespace that led to a canonical entry.
type SearchMatch struct {
	MatchedTerm string `json:"matched_term"`
	Entry       Entry  `json:"entry"`
}

// SearchGroups are ordered by authority: exact canonical matches, exact alias
// matches, then canonical-or-alias prefix matches.
type SearchGroups struct {
	Exact  []SearchMatch `json:"exact"`
	Alias  []SearchMatch `json:"alias"`
	Prefix []SearchMatch `json:"prefix"`
}

// SearchResponse returns every ranked candidate and an explicit resolution
// verdict. Ambiguous results never select an owner on the caller's behalf.
type SearchResponse struct {
	Schema       string        `json:"schema"`
	IndexVersion string        `json:"index_version"`
	Query        string        `json:"query"`
	Verdict      SearchVerdict `json:"verdict"`
	Groups       SearchGroups  `json:"groups"`
}

// Search discovers exact canonical, exact alias, and prefix matches in the
// public terminology index.
func Search(term string) SearchResponse {
	return publicIndex.Search(term)
}

// Search discovers terms in this index. Exact matches outrank prefixes, but all
// groups remain visible so callers can explain why a result is ambiguous.
func (i *Index) Search(term string) SearchResponse {
	groups := SearchGroups{
		Exact:  matchesForToken(i.canonical[term], term),
		Alias:  matchesForToken(i.aliases[term], term),
		Prefix: []SearchMatch{},
	}

	canonicalTerms := sortedEntryKeys(i.canonical)
	aliasTerms := sortedEntryKeys(i.aliases)
	for _, candidate := range canonicalTerms {
		if candidate != term && strings.HasPrefix(candidate, term) {
			groups.Prefix = append(groups.Prefix, matchesForToken(i.canonical[candidate], candidate)...)
		}
	}
	for _, candidate := range aliasTerms {
		if candidate != term && strings.HasPrefix(candidate, term) {
			groups.Prefix = append(groups.Prefix, matchesForToken(i.aliases[candidate], candidate)...)
		}
	}

	dominant := groups.Prefix
	verdict := SearchVerdictPrefix
	if len(groups.Exact) != 0 {
		dominant, verdict = groups.Exact, SearchVerdictExact
	} else if len(groups.Alias) != 0 {
		dominant, verdict = groups.Alias, SearchVerdictAlias
	}
	switch len(uniqueOwners(dominant)) {
	case 0:
		verdict = SearchVerdictNotFound
	case 1:
		// Keep the rank-derived verdict.
	default:
		verdict = SearchVerdictAmbiguous
	}

	return SearchResponse{
		Schema:       SearchSchemaVersion,
		IndexVersion: PublicIndexVersion,
		Query:        term,
		Verdict:      verdict,
		Groups:       groups,
	}
}

func matchesForToken(entries []Entry, token string) []SearchMatch {
	matches := make([]SearchMatch, 0, len(entries))
	for _, entry := range entries {
		matches = append(matches, SearchMatch{MatchedTerm: token, Entry: cloneEntry(entry)})
	}
	sort.Slice(matches, func(a, b int) bool {
		left, right := matches[a].Entry, matches[b].Entry
		if left.Identity.CanonicalTerm != right.Identity.CanonicalTerm {
			return left.Identity.CanonicalTerm < right.Identity.CanonicalTerm
		}
		if left.Scope.Kind != right.Scope.Kind {
			return left.Scope.Kind < right.Scope.Kind
		}
		return left.Scope.Value < right.Scope.Value
	})
	return matches
}

func sortedEntryKeys(values map[string][]Entry) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func uniqueOwners(matches []SearchMatch) map[string]struct{} {
	owners := make(map[string]struct{}, len(matches))
	for _, match := range matches {
		entry := match.Entry
		key := entry.Identity.CanonicalTerm + "\x00" + entry.Scope.Kind + "\x00" + entry.Scope.Value
		owners[key] = struct{}{}
	}
	return owners
}
