package disambiguation

import (
	"errors"
	"testing"
)

func TestReverseLookupSupportsEveryLocatorKind(t *testing.T) {
	entries, err := canonicalEntries(publicEntries)
	if err != nil {
		t.Fatal(err)
	}
	entry := &entries[0]
	entry.Sources = []SourceWitness{
		{Kind: SourceKindGoSource, Locator: "internal/example/example.go", Revision: "test", CheckedAt: "2026-08-17T00:00:00Z", Probe: "test", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "ExampleSymbol"}},
		{Kind: SourceKindGoSource, Locator: "cmd/fak/example.go", Revision: "test", CheckedAt: "2026-08-17T00:00:00Z", Probe: "test", Reference: &PublicReference{Kind: ReferenceKindCLIVerb, Name: "example inspect"}},
		{Kind: SourceKindGoSource, Locator: "internal/example/reasons.go", Revision: "test", CheckedAt: "2026-08-17T00:00:00Z", Probe: "test", Reference: &PublicReference{Kind: ReferenceKindReasonCode, Name: "EXAMPLE_DENIED"}},
		{Kind: SourceKindDocument, Locator: "docs/example.md#canonical", Revision: "test", CheckedAt: "2026-08-17T00:00:00Z", Probe: "test"},
	}
	index, err := NewIndex(entries)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		kind           ReverseLocatorKind
		input, matched string
	}{
		{ReverseSourcePath, "docs/example.md", "docs/example.md#canonical"},
		{ReverseGoSymbol, "ExampleSymbol", "ExampleSymbol"},
		{ReverseCLIToken, "example inspect", "example inspect"},
		{ReverseReasonCode, "EXAMPLE_DENIED", "EXAMPLE_DENIED"},
	}
	for _, tc := range cases {
		t.Run(string(tc.kind), func(t *testing.T) {
			got, err := index.ReverseLookup(tc.kind, tc.input)
			if err != nil {
				t.Fatal(err)
			}
			if len(got.Matches) != 1 || got.Matches[0].MatchedValue != tc.matched || got.Matches[0].Entry.Identity.CanonicalTerm != entry.Identity.CanonicalTerm {
				t.Fatalf("lookup = %#v", got)
			}
		})
	}
}

func TestReverseLookupUnknownNeverFabricatesMatch(t *testing.T) {
	got, err := publicIndex.ReverseLookup(ReverseGoSymbol, "DefinitelyAbsent")
	if !errors.Is(err, ErrReverseNotFound) {
		t.Fatalf("error = %v", err)
	}
	if len(got.Matches) != 0 {
		t.Fatalf("matches = %#v", got.Matches)
	}
	if _, err := publicIndex.ReverseLookup("guess", "anything"); !errors.Is(err, ErrReverseKindInvalid) {
		t.Fatalf("invalid kind error = %v", err)
	}
}

func TestReverseLookupResultsAreDeterministic(t *testing.T) {
	got, err := publicIndex.ReverseLookup(ReverseReasonCode, "SOURCE_CURRENT")
	if err != nil {
		t.Fatal(err)
	}
	for n := 1; n < len(got.Matches); n++ {
		left, right := got.Matches[n-1].Entry, got.Matches[n].Entry
		if left.Identity.CanonicalTerm > right.Identity.CanonicalTerm {
			t.Fatalf("unsorted matches at %d: %q > %q", n, left.Identity.CanonicalTerm, right.Identity.CanonicalTerm)
		}
	}
}
