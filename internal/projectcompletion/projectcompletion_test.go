package projectcompletion

import (
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
	"testing"
)

func work(s string, c float64) issuepolicy.ProjectWorkReadout {
	return issuepolicy.ProjectWorkReadout{Status: issuepolicy.ProjectWorkValid, EstimatePoints: c, Parent: "#36", ParentBaseline: 20, Contribution: c, CompletionStandard: s, ProductionCredit: s == "production"}
}
func TestToyBringupDoesNotMasqueradeAsProductionComplete(t *testing.T) {
	got := Summarize([]Issue{{1, "toy tokenizer example", "closed", work("demo", 2)}, {2, "single-request bring-up", "closed", work("prototype", 3)}, {3, "production serving path", "open", work("production", 15)}})
	if got.ProductionCompletePoints != 0 || got.ProductionCompletePct != 0 {
		t.Fatalf("toy work received production credit: %+v", got)
	}
	if got.OpenPoints != 15 || got.BaselinePoints != 20 || got.Confidence != "complete" {
		t.Fatalf("unexpected production scope: %+v", got)
	}
	if len(got.ClosedByStandard) != 2 {
		t.Fatalf("closed maturity buckets = %+v", got.ClosedByStandard)
	}
}
func TestUnknownAndDenominatorDriftRefuseFalsePrecision(t *testing.T) {
	bad := work("production", 8)
	bad.ParentBaseline = 10
	got := Summarize([]Issue{{1, "", "closed", work("production", 5)}, {2, "", "closed", bad}, {3, "legacy", "closed", issuepolicy.ProjectWorkReadout{Status: issuepolicy.ProjectWorkUndeclared}}})
	if got.Confidence != "incomplete" || len(got.DenominatorDrift) == 0 || len(got.Unknown) != 1 {
		t.Fatalf("drift/unknown did not lower confidence: %+v", got)
	}
}
