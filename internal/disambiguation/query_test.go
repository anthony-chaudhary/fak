package disambiguation

import (
	"errors"
	"reflect"
	"strings"
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
	if got.Entry.Identity.CanonicalTerm != "agent kernel" || got.MatchedAlias != "" {
		t.Fatalf("canonical identity = %q, matched alias = %q", got.Entry.Identity.CanonicalTerm, got.MatchedAlias)
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

func TestQueryRemainsExactCanonicalOnly(t *testing.T) {
	for _, query := range []string{"Agent Kernel", "agent kernel ", "fused agent kernel", "unknown"} {
		_, err := Query(query)
		if !errors.Is(err, ErrCanonicalTermNotFound) {
			t.Errorf("Query(%q) error = %v, want ErrCanonicalTermNotFound", query, err)
		}
	}
}

func TestResolveAliasReturnsCanonicalOwnerAndExactMatch(t *testing.T) {
	got, err := Resolve("fused agent kernel")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got.Entry.Identity.CanonicalTerm != "agent kernel" {
		t.Fatalf("canonical term = %q", got.Entry.Identity.CanonicalTerm)
	}
	if got.MatchedAlias != "fused agent kernel" {
		t.Fatalf("matched alias = %q", got.MatchedAlias)
	}
	if got.Entry.Owner.Leaf != "kernel" || got.Entry.Owner.Lane != "kernel" {
		t.Fatalf("canonical ownership hidden: %#v", got.Entry.Owner)
	}
}

func TestResolveCanonicalDoesNotReportAlias(t *testing.T) {
	got, err := Resolve("agent kernel")
	if err != nil {
		t.Fatal(err)
	}
	if got.Entry.Identity.CanonicalTerm != "agent kernel" || got.MatchedAlias != "" {
		t.Fatalf("canonical resolution = %#v", got)
	}
}

func TestResolveIsExact(t *testing.T) {
	for _, query := range []string{"Fused Agent Kernel", "fused agent kernel ", "unknown"} {
		_, err := Resolve(query)
		if !errors.Is(err, ErrCanonicalTermNotFound) {
			t.Errorf("Resolve(%q) error = %v, want ErrCanonicalTermNotFound", query, err)
		}
	}
}

func TestNewIndexRejectsRepeatedAliasUnderOneOwner(t *testing.T) {
	entry := cloneEntry(publicEntries[0])
	entry.Identity.Aliases = []string{"fused agent kernel", "fused agent kernel"}

	_, err := NewIndex([]Entry{entry})
	if err == nil || !strings.Contains(err.Error(), `duplicate alias "fused agent kernel"`) {
		t.Fatalf("repeated alias error = %v", err)
	}
}
func TestNewIndexRejectsDuplicateAliasOwnership(t *testing.T) {
	first := publicEntries[0]
	second := cloneEntry(first)
	second.Identity.CanonicalTerm = "agent runtime"
	second.Identity.Aliases = []string{"fused agent kernel"}
	second.Contrasts[0].CanonicalTerm = "process runtime"

	_, err := NewIndex([]Entry{first, second})
	if err == nil || !strings.Contains(err.Error(), `duplicate alias "fused agent kernel" claimed by`) || !strings.Contains(err.Error(), `"agent kernel"`) || !strings.Contains(err.Error(), `"agent runtime"`) {
		t.Fatalf("duplicate alias error = %v", err)
	}
}

func TestNewIndexRejectsAliasThatHidesCanonicalOwner(t *testing.T) {
	first := publicEntries[0]
	second := cloneEntry(first)
	second.Identity.CanonicalTerm = "fused agent kernel"
	second.Identity.Aliases = []string{}
	second.Contrasts[0].CanonicalTerm = "process runtime"

	_, err := NewIndex([]Entry{first, second})
	if err == nil || !strings.Contains(err.Error(), "conflicts with canonical term owned by") {
		t.Fatalf("alias/canonical collision error = %v", err)
	}
}

func TestQueryReturnsIndependentRecord(t *testing.T) {
	first, err := Resolve("fused agent kernel")
	if err != nil {
		t.Fatal(err)
	}
	first.Entry.Identity.Aliases[0] = "mutated"
	first.Entry.Contrasts[0].Explanation = "mutated"
	first.Entry.Sources[0].Locator = "mutated"
	second, err := Resolve("fused agent kernel")
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(first.Entry, second.Entry) {
		t.Fatal("query returned mutable shared seed data")
	}
}
