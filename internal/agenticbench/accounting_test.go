package agenticbench

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/armbench"
)

func TestAccountingReportRefusesIncomparableCostAndCacheGains(t *testing.T) {
	provider := accountingReceipt(t, armbench.AuthorityProviderAggregate, "agent_turns", 2, 2, 0.40, 80)
	harness := accountingReceipt(t, armbench.AuthorityHarnessAggregate, "agent_turns", 2, 2, 0.30, 90)
	differentCoverage := accountingReceipt(t, armbench.AuthorityProviderAggregate, "provider_requests", 2, 2, 0.35, 100)

	report, err := BuildAccountingReport([]AccountingArm{
		{Arm: "direct", Receipt: provider},
		{Arm: "fak", Receipt: harness},
		{Arm: "other-scope", Receipt: differentCoverage},
	}, []AccountingComparisonRequest{
		{LeftArm: "direct", RightArm: "fak", Metric: armbench.MetricCostUSD},
		{LeftArm: "direct", RightArm: "other-scope", Metric: armbench.MetricCacheReadTokens},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Comparisons) != 2 {
		t.Fatalf("comparisons = %d, want 2", len(report.Comparisons))
	}
	if report.Comparisons[0].State != AccountingClaimRefused || !strings.Contains(report.Comparisons[0].Detail, "authority") {
		t.Fatalf("cost comparison = %+v, want authority refusal", report.Comparisons[0])
	}
	if report.Comparisons[1].State != AccountingClaimRefused || !strings.Contains(report.Comparisons[1].Detail, "coverage") {
		t.Fatalf("cache comparison = %+v, want coverage refusal", report.Comparisons[1])
	}

	b, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"state":"REFUSED"`) || !strings.Contains(string(b), `"authority":"provider_aggregate"`) {
		t.Fatalf("JSON omitted accounting refusal/provenance: %s", b)
	}
	markdown := FormatAccountingMarkdown(report)
	for _, want := range []string{"Accounting Authority Receipt", "REFUSED", "authority", "coverage", "cost_usd", "cache_read_tokens"} {
		if !strings.Contains(markdown, want) {
			t.Fatalf("markdown omitted %q:\n%s", want, markdown)
		}
	}
}

func TestAccountingReportAllowsComparableObservedTotals(t *testing.T) {
	left := accountingReceipt(t, armbench.AuthorityProviderAggregate, "agent_turns", 2, 2, 0.40, 80)
	right := accountingReceipt(t, armbench.AuthorityProviderAggregate, "agent_turns", 2, 2, 0.30, 100)
	report, err := BuildAccountingReport([]AccountingArm{{Arm: "left", Receipt: left}, {Arm: "right", Receipt: right}}, []AccountingComparisonRequest{{LeftArm: "left", RightArm: "right", Metric: armbench.MetricCostUSD}})
	if err != nil {
		t.Fatal(err)
	}
	if got := report.Comparisons[0]; got.State != AccountingClaimAllowed || got.Delta == nil || math.Abs(*got.Delta-(-0.10)) > 1e-12 {
		t.Fatalf("comparable totals = %+v, want allowed delta -0.10", got)
	}
}

func accountingReceipt(t *testing.T, authority armbench.AccountingAuthority, scope string, observed, expected int, cost, cacheRead float64) armbench.AccountingReceipt {
	t.Helper()
	receipt, err := armbench.ReconcileAccounting([]armbench.AccountingSource{{
		Authority: authority,
		Artifact:  armbench.AccountingArtifact{Ref: "fixture://" + string(authority) + "/" + scope, SHA256: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
		Coverage:  armbench.AccountingCoverage{Scope: scope, Observed: observed, Expected: expected},
		Values:    armbench.AccountingValues{CostUSD: &cost, CacheReadTokens: &cacheRead},
	}})
	if err != nil {
		t.Fatal(err)
	}
	return receipt
}
