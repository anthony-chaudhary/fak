package disambiguation

import "testing"

func TestDiffIndexesTypesEveryChange(t *testing.T) {
	base := cloneEntry(publicEntries[0])
	before := []Entry{base, cloneEntry(publicEntries[1])}
	changed := cloneEntry(base)
	changed.Identity.Aliases = append(changed.Identity.Aliases, "new alias")
	changed.Definition += " changed"
	changed.Contrasts = append(changed.Contrasts, Contrast{CanonicalTerm: "removed term", Explanation: "fixture"})
	changed.Owner.Leaf = "moved"
	changed.Freshness.Verdict = FreshnessStale
	added := cloneEntry(publicEntries[2])
	added.Identity.CanonicalTerm = "added fixture term"
	added.Identity.Aliases = []string{}
	report := DiffIndexes(before, []Entry{changed, added})
	got := map[ChangeKind]CompatibilityImpact{}
	for _, change := range report.Changes {
		got[change.Kind] = change.QueryImpact
	}
	for _, kind := range []ChangeKind{ChangeAdditive, ChangeAliasMove, ChangeSemantic, ChangeContrast, ChangeOwnerMove, ChangeStaleTransition, ChangeRemoval} {
		if _, ok := got[kind]; !ok {
			t.Errorf("missing %s: %#v", kind, report.Changes)
		}
	}
	if got[ChangeAdditive] != ImpactCompatible || got[ChangeRemoval] != ImpactBreaking || got[ChangeSemantic] != ImpactReview {
		t.Fatalf("impact=%#v", got)
	}
}
