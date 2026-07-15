package metrics

import "testing"

func TestTaxAttributionRanksAndReconciles(t *testing.T) {
	sites := []TaxSite{
		{Key: "session-start", Prose: "Keep working."},
		{Key: "refusal", Prose: "Never delete the audit. Do not repeat the call."},
		{Key: "recovery", Prose: "Do not forget to recover."},
	}
	got := AttributeNegationTax(sites, 0)
	if len(got.Sites) != 3 || got.Sites[0].Site != "refusal" || got.Sites[0].Outstanding != 2 {
		t.Fatalf("ranking=%+v", got)
	}
	var applied, fallback, residual, outstanding int
	for _, row := range got.Sites {
		applied += row.Applied
		fallback += row.VerbatimFallback
		residual += row.Residual
		outstanding += row.Outstanding
	}
	if applied != got.Applied || fallback != got.VerbatimFallback || residual != got.Residual || outstanding != got.Outstanding {
		t.Fatalf("totals do not reconcile: %+v", got)
	}
	if got.Applied != 1 || got.Residual != 2 || got.Outstanding != 2 {
		t.Fatalf("totals=%+v", got)
	}
	top := AttributeNegationTax(sites, 1)
	if len(top.Sites) != 1 || top.Sites[0].Site != "refusal" || top.Outstanding != got.Outstanding {
		t.Fatalf("top=%+v all=%+v", top, got)
	}
}
