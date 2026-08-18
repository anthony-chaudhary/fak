package disambiguation

import (
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
)

// ReverseLookupSchemaVersion identifies the locator-to-canonical-identity contract.
const ReverseLookupSchemaVersion = "fak-disambiguation-reverse/1"

// ReverseLocatorKind names a repository-visible handle carried by an Entry.
type ReverseLocatorKind string

const (
	ReverseSourcePath ReverseLocatorKind = "source-path"
	ReverseGoSymbol   ReverseLocatorKind = "symbol"
	ReverseCLIToken   ReverseLocatorKind = "cli-token"
	ReverseReasonCode ReverseLocatorKind = "reason-code"
)

var (
	ErrReverseKindInvalid = errors.New("invalid reverse locator kind")
	ErrReverseNotFound    = errors.New("reverse locator not found")
)

// ReverseMatch preserves why an entry matched instead of presenting an
// inferred canonical owner without evidence.
type ReverseMatch struct {
	Kind         ReverseLocatorKind `json:"kind"`
	Input        string             `json:"input"`
	MatchedValue string             `json:"matched_value"`
	Entry        Entry              `json:"entry"`
}

// ReverseLookupResponse returns every exact evidenced owner. Multiple matches
// remain visible because a source path or reason code may intentionally support
// several canonical distinctions.
type ReverseLookupResponse struct {
	Schema       string             `json:"schema"`
	IndexVersion string             `json:"index_version"`
	Kind         ReverseLocatorKind `json:"kind"`
	Input        string             `json:"input"`
	Matches      []ReverseMatch     `json:"matches"`
}

// ReverseLookup finds canonical identities from a public source locator.
func ReverseLookup(kind ReverseLocatorKind, value string) (ReverseLookupResponse, error) {
	return publicIndex.ReverseLookup(kind, value)
}

// ReverseLookup finds entries carrying the exact locator. Source paths also
// match the path portion of a locator with a document anchor.
func (i *Index) ReverseLookup(kind ReverseLocatorKind, value string) (ReverseLookupResponse, error) {
	if !validReverseKind(kind) {
		return ReverseLookupResponse{}, fmt.Errorf("%w: %q", ErrReverseKindInvalid, kind)
	}
	value = strings.TrimSpace(value)
	if value == "" {
		return ReverseLookupResponse{}, fmt.Errorf("%w: empty %s", ErrReverseNotFound, kind)
	}
	response := ReverseLookupResponse{
		Schema: ReverseLookupSchemaVersion, IndexVersion: PublicIndexVersion,
		Kind: kind, Input: value, Matches: []ReverseMatch{},
	}
	for _, entries := range i.canonical {
		for _, entry := range entries {
			for _, matched := range reverseValues(entry, kind, value) {
				canonical := cloneEntry(entry)
				if canonical.Identity.Aliases == nil {
					canonical.Identity.Aliases = []string{}
				}
				response.Matches = append(response.Matches, ReverseMatch{Kind: kind, Input: value, MatchedValue: matched, Entry: canonical})
			}
		}
	}
	sort.Slice(response.Matches, func(a, b int) bool {
		left, right := response.Matches[a], response.Matches[b]
		if left.Entry.Identity.CanonicalTerm != right.Entry.Identity.CanonicalTerm {
			return left.Entry.Identity.CanonicalTerm < right.Entry.Identity.CanonicalTerm
		}
		if left.Entry.Scope.Kind != right.Entry.Scope.Kind {
			return left.Entry.Scope.Kind < right.Entry.Scope.Kind
		}
		if left.Entry.Scope.Value != right.Entry.Scope.Value {
			return left.Entry.Scope.Value < right.Entry.Scope.Value
		}
		return left.MatchedValue < right.MatchedValue
	})
	if len(response.Matches) == 0 {
		return response, fmt.Errorf("%w: %s %q", ErrReverseNotFound, kind, value)
	}
	return response, nil
}

func validReverseKind(kind ReverseLocatorKind) bool {
	switch kind {
	case ReverseSourcePath, ReverseGoSymbol, ReverseCLIToken, ReverseReasonCode:
		return true
	default:
		return false
	}
}

func reverseValues(entry Entry, kind ReverseLocatorKind, input string) []string {
	seen := make(map[string]bool)
	var values []string
	add := func(value string) {
		if value == "" || seen[value] {
			return
		}
		seen[value] = true
		values = append(values, value)
	}
	for _, source := range entry.Sources {
		switch kind {
		case ReverseSourcePath:
			locator := filepath.ToSlash(source.Locator)
			query := filepath.ToSlash(input)
			path, _, _ := strings.Cut(locator, "#")
			if query == locator || query == path {
				add(locator)
			}
		case ReverseGoSymbol:
			if source.Reference != nil && source.Reference.Kind == ReferenceKindGoSymbol && source.Reference.Name == input {
				add(source.Reference.Name)
			}
		case ReverseCLIToken:
			if source.Reference != nil && source.Reference.Kind == ReferenceKindCLIVerb && source.Reference.Name == input {
				add(source.Reference.Name)
			}
		case ReverseReasonCode:
			if source.Reference != nil && source.Reference.Kind == ReferenceKindReasonCode && source.Reference.Name == input {
				add(source.Reference.Name)
			}
		}
	}
	if kind == ReverseReasonCode && entry.Freshness.ReasonCode == input {
		add(entry.Freshness.ReasonCode)
	}
	sort.Strings(values)
	return values
}
