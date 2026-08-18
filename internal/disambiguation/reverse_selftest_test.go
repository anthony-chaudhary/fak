package disambiguation

import "testing"

func TestRunReverseSelfTest(t *testing.T) {
	report, err := RunReverseSelfTest()
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != ReverseLookupSchemaVersion || report.IndexVersion != PublicIndexVersion {
		t.Fatalf("versions = %#v", report)
	}
	if len(report.Cases) != 4 || !report.UnknownRejected {
		t.Fatalf("report = %#v", report)
	}
	seen := map[ReverseLocatorKind]bool{}
	for _, item := range report.Cases {
		seen[item.Kind] = true
		if item.CanonicalTerm == "" || item.MatchCount < 1 {
			t.Errorf("incomplete case = %#v", item)
		}
	}
	for _, kind := range []ReverseLocatorKind{ReverseSourcePath, ReverseGoSymbol, ReverseCLIToken, ReverseReasonCode} {
		if !seen[kind] {
			t.Errorf("missing kind %s", kind)
		}
	}
}
