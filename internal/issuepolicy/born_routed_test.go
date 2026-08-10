package issuepolicy

import "testing"

func TestBornRoutedAdvisoryAndStrict(t *testing.T) {
	base := completeCandidate()
	base.Labels = nil
	advisory := ReviewCandidate(base, Options{})
	if advisory.Dispatchability != Dispatchable || len(advisory.BornRouted.Flags) != 2 {
		t.Fatalf("advisory = %+v, want dispatchable with class+priority flags", advisory)
	}
	strict := ReviewCandidate(base, Options{StrictBornRouted: true})
	if strict.Dispatchability != TriageOnly || !containsString(strict.Reasons, ReasonNotBornRouted) {
		t.Fatalf("strict = %+v, want ISSUE_NOT_BORN_ROUTED triage hold", strict)
	}
	base.Labels = []string{"class:dev", "priority/P1"}
	routed := ReviewCandidate(base, Options{StrictBornRouted: true})
	if routed.Dispatchability != Dispatchable || len(routed.BornRouted.Flags) != 0 {
		t.Fatalf("born-routed = %+v, want dispatchable", routed)
	}
}
