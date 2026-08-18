package disambiguation

import "testing"

func TestRunClaimsSourceSelfTest(t *testing.T) {
	report, err := RunClaimsSourceSelfTest()
	if err != nil {
		t.Fatal(err)
	}
	if len(report.CanonicalTerms) != 6 || !report.MissingBaselineRejected || !report.MissingProvenanceRejected || !report.MissingScopeRejected {
		t.Fatalf("report=%#v", report)
	}
}
