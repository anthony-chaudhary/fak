package issuepolicy

import (
	"strings"
	"testing"
)

func TestAuthorBatchProjectWorkDefaultsProductionAndPassesStrictReview(t *testing.T) {
	body, err := AuthorBatchProjectWork("## Parent context\nParent: #36\n", BatchProjectWork{ParentIssue: 36, EstimatePoints: 3, ParentBaseline: 20, TargetEnvelope: "real path", WitnessedEnvelope: "not yet"})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"## Work estimate", "Estimate: 3 points", "Contribution: 3/20 points", "## Completion standard", "production"} {
		if !strings.Contains(body, want) {
			t.Fatalf("missing %q in:\n%s", want, body)
		}
	}
}
func TestAuthorBatchProjectWorkPreservesExplicitDemo(t *testing.T) {
	body, err := AuthorBatchProjectWork("", BatchProjectWork{ParentIssue: 36, EstimatePoints: 2, ParentBaseline: 20, CompletionStandard: "demo"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(body, "## Completion standard\n"+"demo") || strings.Contains(body, "## Completion standard\n"+"production") {
		t.Fatalf("maturity changed:\n%s", body)
	}
}
func TestAuthorBatchProjectWorkRefusesUnknownDenominator(t *testing.T) {
	if _, err := AuthorBatchProjectWork("", BatchProjectWork{ParentIssue: 36, EstimatePoints: 2}); err == nil {
		t.Fatal("missing denominator accepted")
	}
}
