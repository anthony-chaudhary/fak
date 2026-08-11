package disambiguation

import "fmt"

// Index is an immutable read index over canonical terms and their declared
// aliases. A token may have multiple owners only when every owner has a
// distinct, required scope.
type Index struct {
	canonical map[string][]Entry
	aliases   map[string][]Entry
}

// NewIndex constructs a read-only index. Canonical terms and aliases share one
// exact, case-sensitive namespace. Repeated tokens are accepted only when their
// scope qualifiers are distinct; callers must then use a scoped lookup.
func NewIndex(entries []Entry) (*Index, error) {
	index := &Index{
		canonical: make(map[string][]Entry, len(entries)),
		aliases:   make(map[string][]Entry),
	}
	for i, source := range entries {
		if err := source.Validate(); err != nil {
			return nil, fmt.Errorf("entry %d: %w", i, err)
		}
		entry := cloneEntry(source)
		term := entry.Identity.CanonicalTerm
		if err := appendScopedOwner(index.canonical, term, entry); err != nil {
			return nil, fmt.Errorf("canonical term %q: %w", term, err)
		}
	}
	for _, source := range entries {
		entry := cloneEntry(source)
		owner := entry.Identity.CanonicalTerm
		for _, alias := range entry.Identity.Aliases {
			if canonicalOwners := index.canonical[alias]; len(canonicalOwners) != 0 {
				return nil, fmt.Errorf("alias %q for %q conflicts with canonical term owned by %q", alias, owner, canonicalOwners[0].Identity.CanonicalTerm)
			}
			for _, prior := range index.aliases[alias] {
				if prior.Scope == entry.Scope {
					if prior.Identity.CanonicalTerm == owner {
						return nil, fmt.Errorf("duplicate alias %q for %q", alias, owner)
					}
					return nil, fmt.Errorf("duplicate alias %q claimed by %q and %q", alias, prior.Identity.CanonicalTerm, owner)
				}
			}
			index.aliases[alias] = append(index.aliases[alias], entry)
		}
	}
	for _, sourceEntry := range entries {
		source := sourceEntry.Identity.CanonicalTerm
		for _, contrast := range sourceEntry.Contrasts {
			targets := index.canonical[contrast.CanonicalTerm]
			if len(targets) == 0 {
				return nil, fmt.Errorf("contrast from %q has unknown canonical target %q", source, contrast.CanonicalTerm)
			}
			if !*contrast.RequiredPair {
				continue
			}
			if len(targets) > 1 {
				return nil, fmt.Errorf("required contrast from %q has ambiguous canonical target %q", source, contrast.CanonicalTerm)
			}
			reverse, exists := contrastTo(targets[0], source)
			if !exists || !*reverse.RequiredPair {
				return nil, fmt.Errorf("required contrast pair %q <-> %q is asymmetric", source, contrast.CanonicalTerm)
			}
			if *reverse.ForbiddenConflation != *contrast.ForbiddenConflation {
				return nil, fmt.Errorf("required contrast pair %q <-> %q disagrees on forbidden_conflation", source, contrast.CanonicalTerm)
			}
		}
	}
	return index, nil
}

func appendScopedOwner(dst map[string][]Entry, token string, entry Entry) error {
	for _, prior := range dst[token] {
		if prior.Scope == entry.Scope {
			return fmt.Errorf("duplicate scope %s=%q", entry.Scope.Kind, entry.Scope.Value)
		}
	}
	dst[token] = append(dst[token], entry)
	return nil
}

func contrastTo(entry Entry, target string) (Contrast, bool) {
	for _, contrast := range entry.Contrasts {
		if contrast.CanonicalTerm == target {
			return contrast, true
		}
	}
	return Contrast{}, false
}

func (i *Index) queryCanonical(term string) (Entry, bool, bool) {
	entries := i.canonical[term]
	if len(entries) != 1 {
		return Entry{}, false, len(entries) > 1
	}
	return cloneEntry(entries[0]), true, false
}

func (i *Index) resolve(term string) (entry Entry, matchedAlias string, ok bool) {
	if entry, ok, ambiguous := i.queryCanonical(term); ok || ambiguous {
		return entry, "", ok
	}
	entries := i.aliases[term]
	if len(entries) != 1 {
		return Entry{}, "", false
	}
	return cloneEntry(entries[0]), term, true
}

func (i *Index) ambiguous(term string) bool {
	return len(i.canonical[term]) > 1 || len(i.aliases[term]) > 1
}

func (i *Index) resolveScoped(term string, scope Scope) (entry Entry, matchedAlias string, ok bool) {
	for _, candidate := range i.canonical[term] {
		if candidate.Scope == scope {
			return cloneEntry(candidate), "", true
		}
	}
	for _, candidate := range i.aliases[term] {
		if candidate.Scope == scope {
			return cloneEntry(candidate), term, true
		}
	}
	return Entry{}, "", false
}
