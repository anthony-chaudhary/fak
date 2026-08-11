package disambiguation

import (
	"os"
	"path/filepath"
)

// StaleSymbolsSelfCheckReport is the package/CLI JSON witness for public
// reference probing and the required fresh-to-stale transition.
type StaleSymbolsSelfCheckReport struct {
	Schema        string    `json:"schema"`
	Fresh         Freshness `json:"fresh"`
	Stale         Freshness `json:"stale"`
	PackagePassed bool      `json:"package_passed"`
	Passed        bool      `json:"passed"`
}

// StaleSymbolsSelfCheck creates and removes a public fixture in an isolated
// directory. It is deterministic apart from the irrelevant temporary path,
// which is never emitted.
func StaleSymbolsSelfCheck() StaleSymbolsSelfCheckReport {
	report := StaleSymbolsSelfCheckReport{Schema: "fak-disambiguation-stale-symbols-self-check/1"}
	root, err := os.MkdirTemp("", "fak-disambiguation-stale-symbols-")
	if err != nil {
		return report
	}
	defer os.RemoveAll(root)

	query, err := QueryScoped("disambiguation package", Scope{Kind: "package", Value: "internal/disambiguation"})
	if err != nil {
		return report
	}
	entry := query.Entry
	if entry.Identity.Aliases == nil {
		entry.Identity.Aliases = []string{}
	}
	ref := PublicReference{Kind: ReferenceKindGoSymbol, Name: "PublicFixture"}
	entry.Sources = []SourceWitness{{Kind: SourceKindGoSource, Locator: "fixture.go", Revision: "self-check/1", CheckedAt: "2026-08-11T00:00:00Z", Probe: "stale-symbols-self-check", Reference: &ref}}
	entry.Freshness = Freshness{Verdict: FreshnessFresh, ReasonCode: FreshnessReasonSourceCurrent, CheckedAt: "2026-08-11T00:00:00Z", Probe: "stale-symbols-self-check"}
	fixture := filepath.Join(root, "fixture.go")
	if err := os.WriteFile(fixture, []byte("package fixture\nfunc PublicFixture() {}\n"), 0o600); err != nil {
		return report
	}
	report.Fresh = ProbePublicReferences(root, entry).Freshness
	if err := os.Remove(fixture); err != nil {
		return report
	}
	report.Stale = ProbePublicReferences(root, entry).Freshness
	report.PackagePassed = report.Fresh.Verdict == FreshnessFresh && report.Stale.Verdict == FreshnessStale && report.Stale.ReasonCode == FreshnessReasonPublicSymbolMissing
	report.Passed = report.PackagePassed
	return report
}
