package disambiguation

import (
	"fmt"
)

// Index is an immutable read index over canonical terms and their declared
// aliases. Construction validates global identity ownership before any query
// can observe the entries.
type Index struct {
	canonical map[string]Entry
	aliases   map[string]string
}

// NewIndex constructs a read-only index. Canonical terms and aliases share one
// exact, case-sensitive namespace: an alias may name exactly one canonical
// owner and may not hide any canonical term.
func NewIndex(entries []Entry) (*Index, error) {
	index := &Index{
		canonical: make(map[string]Entry, len(entries)),
		aliases:   make(map[string]string),
	}
	for i, entry := range entries {
		if err := entry.Validate(); err != nil {
			return nil, fmt.Errorf("entry %d: %w", i, err)
		}
		term := entry.Identity.CanonicalTerm
		if _, exists := index.canonical[term]; exists {
			return nil, fmt.Errorf("duplicate canonical term %q", term)
		}
		index.canonical[term] = cloneEntry(entry)
	}
	for _, sourceEntry := range entries {
		entry := cloneEntry(sourceEntry)
		owner := entry.Identity.CanonicalTerm
		for _, alias := range entry.Identity.Aliases {
			if canonicalOwner, exists := index.canonical[alias]; exists {
				return nil, fmt.Errorf("alias %q for %q conflicts with canonical term owned by %q", alias, owner, canonicalOwner.Identity.CanonicalTerm)
			}
			if priorOwner, exists := index.aliases[alias]; exists {
				return nil, fmt.Errorf("duplicate alias %q claimed by %q and %q", alias, priorOwner, owner)
			}
			index.aliases[alias] = owner
		}
	}
	return index, nil
}

func (i *Index) queryCanonical(term string) (Entry, bool) {
	entry, ok := i.canonical[term]
	return cloneEntry(entry), ok
}

func (i *Index) resolve(term string) (entry Entry, matchedAlias string, ok bool) {
	if entry, ok = i.queryCanonical(term); ok {
		return entry, "", true
	}
	owner, ok := i.aliases[term]
	if !ok {
		return Entry{}, "", false
	}
	entry, ok = i.queryCanonical(owner)
	return entry, term, ok
}
