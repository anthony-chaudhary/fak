package disambiguation

import (
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
)

const GeneratedIndexSchemaVersion = "fak-disambiguation-index/1"

type GeneratedIndex struct {
	Schema         string  `json:"schema"`
	SourceRevision string  `json:"source_revision"`
	EntryCount     int     `json:"entry_count"`
	Entries        []Entry `json:"entries"`
}

func GeneratePublicIndex() ([]byte, error) { return GenerateIndex(publicEntries) }

func GenerateIndex(entries []Entry) ([]byte, error) {
	canonical, err := canonicalEntries(entries)
	if err != nil {
		return nil, err
	}
	source, err := json.Marshal(canonical)
	if err != nil {
		return nil, fmt.Errorf("marshal canonical sources: %w", err)
	}
	digest := sha256.Sum256(source)
	document := GeneratedIndex{Schema: GeneratedIndexSchemaVersion, SourceRevision: "sha256:" + hex.EncodeToString(digest[:]), EntryCount: len(canonical), Entries: canonical}
	var out bytes.Buffer
	encoder := json.NewEncoder(&out)
	encoder.SetIndent("", "  ")
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(document); err != nil {
		return nil, fmt.Errorf("encode generated index: %w", err)
	}
	return out.Bytes(), nil
}

func canonicalEntries(entries []Entry) ([]Entry, error) {
	canonical := make([]Entry, len(entries))
	for i, entry := range entries {
		canonical[i] = cloneEntry(entry)
		if canonical[i].Identity.Aliases == nil {
			canonical[i].Identity.Aliases = []string{}
		}
		if err := canonical[i].Validate(); err != nil {
			return nil, fmt.Errorf("entries[%d]: %w", i, err)
		}
		slices.Sort(canonical[i].Identity.Aliases)
		slices.SortFunc(canonical[i].Contrasts, func(a, b Contrast) int {
			return cmp.Compare(a.CanonicalTerm, b.CanonicalTerm)
		})
		slices.SortFunc(canonical[i].Sources, func(a, b SourceWitness) int {
			if order := cmp.Compare(a.Kind, b.Kind); order != 0 {
				return order
			}
			return cmp.Compare(a.Locator, b.Locator)
		})
	}
	slices.SortFunc(canonical, func(a, b Entry) int {
		if order := cmp.Compare(a.Identity.CanonicalTerm, b.Identity.CanonicalTerm); order != 0 {
			return order
		}
		if order := cmp.Compare(a.Scope.Kind, b.Scope.Kind); order != 0 {
			return order
		}
		return cmp.Compare(a.Scope.Value, b.Scope.Value)
	})
	return canonical, nil
}
