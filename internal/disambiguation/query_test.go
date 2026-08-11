package disambiguation

import (
	"errors"
	"reflect"
	"testing"
)

func TestQueryCanonicalTermReturnsCompleteVersionedRecord(t *testing.T) {
	got, err := Query("agent kernel")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if got.Schema != QuerySchemaVersion || got.IndexVersion != PublicIndexVersion {
		t.Fatalf("version contract = schema %q index %q", got.Schema, got.IndexVersion)
	}
	if got.Entry.Identity.CanonicalTerm != "agent kernel" {
		t.Fatalf("canonical term = %q", got.Entry.Identity.CanonicalTerm)
	}
	if got.Entry.Definition == "" || len(got.Entry.Contrasts) == 0 || got.Entry.Scope.Value == "" {
		t.Fatalf("meaning/scope/contrasts incomplete: %#v", got.Entry)
	}
	if got.Entry.Owner.Leaf == "" || len(got.Entry.Sources) == 0 || got.Entry.Freshness.Verdict == "" {
		t.Fatalf("ownership/sources/freshness incomplete: %#v", got.Entry)
	}
	if err := got.Entry.Validate(); err != nil {
		t.Fatalf("seed violates entry schema: %v", err)
	}
}

func TestQueryIsExactCanonicalOnly(t *testing.T) {
	for _, query := range []string{"Agent Kernel", "agent kernel ", "fused agent kernel", "unknown"} {
		_, err := Query(query)
		if !errors.Is(err, ErrCanonicalTermNotFound) {
			t.Errorf("Query(%q) error = %v, want ErrCanonicalTermNotFound", query, err)
		}
	}
}

func TestQueryReturnsIndependentRecord(t *testing.T) {
	first, err := Query("agent kernel")
	if err != nil {
		t.Fatal(err)
	}
	first.Entry.Identity.Aliases[0] = "mutated"
	first.Entry.Contrasts[0].Explanation = "mutated"
	first.Entry.Sources[0].Locator = "mutated"
	second, err := Query("agent kernel")
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(first.Entry, second.Entry) {
		t.Fatal("query returned mutable shared seed data")
	}
}
