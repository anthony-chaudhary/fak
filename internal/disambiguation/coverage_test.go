package disambiguation

import (
	"reflect"
	"testing"
	"testing/fstest"
)

func TestInventoryCoverageDetectsAndClearsExportedTerm(t *testing.T) {
	fixture := fstest.MapFS{"api/terms.go": {Data: []byte("package api\n\ntype AgentKernel struct{}\ntype NewlyIntroducedTerm struct{}\n")}}
	index := publicIndex
	surfaces := []PublicTerminologySurface{{Locator: "api", Kind: "go_package"}}
	got, err := InventoryCoverage(fixture, surfaces, index, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []CoverageFinding{{Surface: "api", Term: "NewlyIntroducedTerm", Candidate: "newly introduced term", Reason: CoverageReasonMissingClassification}}
	if !reflect.DeepEqual(got.Findings, want) {
		t.Fatalf("findings = %#v, want %#v", got.Findings, want)
	}
	if got.OK {
		t.Fatal("unclassified exported term reported OK")
	}

	canonicalIndex := &Index{canonical: map[string][]Entry{
		"agent kernel":          {{Identity: Identity{CanonicalTerm: "agent kernel"}}},
		"newly introduced term": {{Identity: Identity{CanonicalTerm: "newly introduced term"}}},
	}}
	canonical, err := InventoryCoverage(fixture, surfaces, canonicalIndex, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !canonical.OK || len(canonical.Findings) != 0 || canonical.Canonical != 2 {
		t.Fatalf("canonical report = %#v", canonical)
	}

	incidental, err := InventoryCoverage(fixture, surfaces, index, []IncidentalTerm{{Term: "NewlyIntroducedTerm", Reason: "API implementation name, not public vocabulary"}})
	if err != nil {
		t.Fatal(err)
	}
	if !incidental.OK || len(incidental.Findings) != 0 || incidental.Incidental != 1 {
		t.Fatalf("incidental report = %#v", incidental)
	}
}

func TestInventoryCoverageIsDeterministicAndIgnoresUndeclaredSurfaces(t *testing.T) {
	fixture := fstest.MapFS{
		"z/z.go":       {Data: []byte("package z\ntype ZebraTerm struct{}\n")},
		"a/b.go":       {Data: []byte("package a\ntype BetaTerm struct{}\n")},
		"a/a.go":       {Data: []byte("package a\ntype AlphaTerm struct{}\n")},
		"private/p.go": {Data: []byte("package private\ntype NoisyPrivateTerm struct{}\n")},
	}
	surfaces := []PublicTerminologySurface{{Locator: "z", Kind: "go_package"}, {Locator: "a", Kind: "go_package"}}
	one, err := InventoryCoverage(fixture, surfaces, mustEmptyIndex(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	two, err := InventoryCoverage(fixture, surfaces, mustEmptyIndex(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(one, two) {
		t.Fatalf("reports differ: %#v != %#v", one, two)
	}
	terms := []string{one.Findings[0].Term, one.Findings[1].Term, one.Findings[2].Term}
	if !reflect.DeepEqual(terms, []string{"AlphaTerm", "BetaTerm", "ZebraTerm"}) {
		t.Fatalf("terms = %v", terms)
	}
}

func TestCoverageSelfCheck(t *testing.T) {
	got := CoverageSelfCheck()
	if !got.Passed {
		t.Fatalf("selfcheck = %#v", got)
	}
}

func mustEmptyIndex(t *testing.T) *Index {
	t.Helper()
	index, err := NewIndex(nil)
	if err != nil {
		t.Fatal(err)
	}
	return index
}
