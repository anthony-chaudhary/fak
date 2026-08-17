package disambiguation

import "testing"

func TestSearchReturnsRankedGroupsAndTypedAmbiguity(t *testing.T) {
	entries := []Entry{
		searchFixtureEntry("cache", nil, Scope{Kind: "runtime", Value: "local"}),
		searchFixtureEntry("cache metadata", []string{"cachemeta"}, Scope{Kind: "runtime", Value: "metadata"}),
		searchFixtureEntry("cache policy", []string{"cache rules"}, Scope{Kind: "policy", Value: "admission"}),
		searchReferenceEntry("cache"),
	}
	index, err := NewIndex(entries)
	if err != nil {
		t.Fatal(err)
	}

	got := index.Search("cache")
	if got.Verdict != SearchVerdictExact {
		t.Fatalf("exact verdict = %q, want %q", got.Verdict, SearchVerdictExact)
	}
	if len(got.Groups.Exact) != 1 || len(got.Groups.Alias) != 0 || len(got.Groups.Prefix) != 4 {
		t.Fatalf("unexpected ranked groups: %+v", got.Groups)
	}

	got = index.Search("cache ")
	if got.Verdict != SearchVerdictAmbiguous {
		t.Fatalf("prefix verdict = %q, want typed ambiguity", got.Verdict)
	}
	if len(got.Groups.Exact) != 0 || len(got.Groups.Alias) != 0 || len(got.Groups.Prefix) != 3 {
		t.Fatalf("unexpected ambiguous groups: %+v", got.Groups)
	}
	if got.Groups.Prefix[0].MatchedTerm != "cache metadata" || got.Groups.Prefix[1].MatchedTerm != "cache policy" || got.Groups.Prefix[2].MatchedTerm != "cache rules" {
		t.Fatalf("prefix ranking is not deterministic: %+v", got.Groups.Prefix)
	}
}

func TestSearchExactAliasOutranksPrefixAndPreservesCandidates(t *testing.T) {
	entries := []Entry{
		searchFixtureEntry("agent kernel", []string{"kernel"}, Scope{Kind: "product", Value: "fak"}),
		searchFixtureEntry("kernel cache", nil, Scope{Kind: "runtime", Value: "cache"}),
		searchReferenceEntry("agent kernel"),
	}
	index, err := NewIndex(entries)
	if err != nil {
		t.Fatal(err)
	}
	got := index.Search("kernel")
	if got.Verdict != SearchVerdictAlias {
		t.Fatalf("verdict = %q, want %q", got.Verdict, SearchVerdictAlias)
	}
	if len(got.Groups.Alias) != 1 || got.Groups.Alias[0].Entry.Identity.CanonicalTerm != "agent kernel" {
		t.Fatalf("alias group = %+v", got.Groups.Alias)
	}
	if len(got.Groups.Prefix) != 1 || got.Groups.Prefix[0].Entry.Identity.CanonicalTerm != "kernel cache" {
		t.Fatalf("prefix group = %+v", got.Groups.Prefix)
	}
}

func TestSearchReportsOverloadedExactAsAmbiguous(t *testing.T) {
	entries := []Entry{
		searchFixtureEntry("pool", nil, Scope{Kind: "runtime", Value: "workers"}),
		searchFixtureEntry("pool", nil, Scope{Kind: "memory", Value: "buffers"}),
		searchReferenceEntry("pool"),
	}
	index, err := NewIndex(entries)
	if err != nil {
		t.Fatal(err)
	}
	got := index.Search("pool")
	if got.Verdict != SearchVerdictAmbiguous || len(got.Groups.Exact) != 2 {
		t.Fatalf("search = %+v", got)
	}
}

func TestSearchNotFoundKeepsNonNilGroups(t *testing.T) {
	got := Search("does-not-exist")
	if got.Verdict != SearchVerdictNotFound {
		t.Fatalf("verdict = %q", got.Verdict)
	}
	if got.Groups.Exact == nil || got.Groups.Alias == nil || got.Groups.Prefix == nil {
		t.Fatalf("groups must encode as arrays: %+v", got.Groups)
	}
}

func searchFixtureEntry(term string, aliases []string, scope Scope) Entry {
	entry := SelfTestEntry()
	entry.Identity.CanonicalTerm = term
	if aliases == nil {
		aliases = []string{}
	}
	entry.Identity.Aliases = aliases
	entry.Scope = scope
	entry.Contrasts = []Contrast{{
		CanonicalTerm:       "reference distinction",
		Explanation:         "Search fixture keeps identity ownership explicit.",
		RequiredPair:        boolPointer(false),
		ForbiddenConflation: boolPointer(false),
	}}
	return entry
}

func searchReferenceEntry(target string) Entry {
	entry := searchFixtureEntry("reference distinction", nil, Scope{Kind: "fixture", Value: "reference"})
	entry.Contrasts[0].CanonicalTerm = target
	return entry
}
