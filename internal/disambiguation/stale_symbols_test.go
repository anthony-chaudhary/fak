package disambiguation

import (
	"os"
	"path/filepath"
	"testing"
)

func TestProbePublicReferencesDeleteFixtureTransitionsFreshToStale(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "fixture.go")
	write := func(body string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("package fixture\nfunc PublicFixture() {}\n")
	entry := fixtureReferenceEntry(SourceKindGoSource, "fixture.go", PublicReference{Kind: ReferenceKindGoSymbol, Name: "PublicFixture"})

	fresh := ProbePublicReferences(root, entry)
	if fresh.Freshness.Verdict != FreshnessFresh || fresh.Freshness.ReasonCode != FreshnessReasonSourceCurrent {
		t.Fatalf("fresh = %+v", fresh.Freshness)
	}
	write("package fixture\n")
	stale := ProbePublicReferences(root, entry)
	if stale.Freshness.Verdict != FreshnessStale || stale.Freshness.ReasonCode != FreshnessReasonPublicSymbolMissing {
		t.Fatalf("stale = %+v", stale.Freshness)
	}
}

func TestProbePublicReferenceKinds(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"symbols.go": "package fixture\nconst StableReason = \"FIXTURE_REASON\"\n",
		"cli.go":     "package fixture\nfunc dispatch(v string) { switch v { case \"fixture-verb\": } }\n",
		"guide.md":   "# Fixture Anchor\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	tests := []struct {
		locator string
		ref     PublicReference
	}{
		{"symbols.go", PublicReference{ReferenceKindReasonCode, "FIXTURE_REASON"}},
		{"cli.go", PublicReference{ReferenceKindCLIVerb, "fixture-verb"}},
		{"guide.md", PublicReference{ReferenceKindDocAnchor, "fixture-anchor"}},
	}
	for _, tt := range tests {
		got := ProbePublicReferences(root, fixtureReferenceEntry(kindForReference(tt.ref.Kind), tt.locator, tt.ref))
		if got.Freshness.Verdict != FreshnessFresh {
			t.Errorf("%s = %+v", tt.ref.Kind, got.Freshness)
		}
	}
}

func TestProbePublicReferencesPrecedence(t *testing.T) {
	entry := fixtureReferenceEntry(SourceKindGoSource, "fixture.go", PublicReference{Kind: ReferenceKindGoSymbol, Name: "not-exported"})
	if got := ProbePublicReferences("", entry).Freshness; got.Verdict != FreshnessInvalid || got.ReasonCode != FreshnessReasonEvidenceMalformed {
		t.Fatalf("malformed precedence = %+v", got)
	}

	entry = fixtureReferenceEntry(SourceKindGoSource, "fixture.go", PublicReference{Kind: ReferenceKindGoSymbol, Name: "PublicFixture"})
	if got := ProbePublicReferences("", entry).Freshness; got.Verdict != FreshnessUnknown || got.ReasonCode != FreshnessReasonProbeUnavailable {
		t.Fatalf("unavailable = %+v", got)
	}
}

func fixtureReferenceEntry(kind, locator string, reference PublicReference) Entry {
	entry := fixtureEntry()
	entry.Sources = []SourceWitness{{Kind: kind, Locator: locator, Revision: "fixture/1", CheckedAt: "2026-08-11T00:00:00Z", Probe: "fixture-probe", Reference: &reference}}
	entry.Freshness = Freshness{Verdict: FreshnessFresh, ReasonCode: FreshnessReasonSourceCurrent, CheckedAt: "2026-08-11T00:00:00Z", Probe: "fixture-probe"}
	return entry
}

func kindForReference(referenceKind string) string {
	if referenceKind == ReferenceKindDocAnchor {
		return SourceKindDocument
	}
	return SourceKindGoSource
}

func TestStaleSymbolsSelfCheckJSON(t *testing.T) {
	report := StaleSymbolsSelfCheck()
	if !report.Passed || !report.PackagePassed {
		t.Fatalf("report = %+v", report)
	}
	if report.Fresh.Verdict != FreshnessFresh || report.Stale.Verdict != FreshnessStale || report.Stale.ReasonCode != FreshnessReasonPublicSymbolMissing {
		t.Fatalf("transition = %+v -> %+v", report.Fresh, report.Stale)
	}
}
