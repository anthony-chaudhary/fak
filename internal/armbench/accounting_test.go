package armbench

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

type accountingFixtureFile struct {
	Cases []struct {
		Name              string                 `json:"name"`
		Sources           []AccountingSource     `json:"sources"`
		Metric            AccountingMetric       `json:"metric"`
		WantAvailability  AccountingAvailability `json:"want_availability"`
		WantAuthority     AccountingAuthority    `json:"want_authority"`
		WantValue         *float64               `json:"want_value"`
		WantDiscrepancies int                    `json:"want_discrepancies"`
	} `json:"cases"`
}

func TestAccountingReconciliationFixtures(t *testing.T) {
	fixture := loadAccountingFixtures(t)
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			receipt, err := ReconcileAccounting(tc.Sources)
			if err != nil {
				t.Fatal(err)
			}
			field, ok := receipt.Field(tc.Metric)
			if !ok {
				t.Fatalf("fixture names unknown metric %q", tc.Metric)
			}
			if field.Availability != tc.WantAvailability || field.Authority != tc.WantAuthority {
				t.Fatalf("state = %s/%s, want %s/%s", field.Availability, field.Authority, tc.WantAvailability, tc.WantAuthority)
			}
			if !equalFloatPointers(field.Value, tc.WantValue) {
				t.Fatalf("value = %v, want %v", field.Value, tc.WantValue)
			}
			if len(field.Discrepancies) != tc.WantDiscrepancies {
				t.Fatalf("discrepancies = %d, want %d: %+v", len(field.Discrepancies), tc.WantDiscrepancies, field.Discrepancies)
			}
		})
	}
}

func TestAccountingJSONDistinguishesMissingFromObservedZero(t *testing.T) {
	fixture := loadAccountingFixtures(t)
	var missing, zero AccountingField
	for _, tc := range fixture.Cases {
		receipt, err := ReconcileAccounting(tc.Sources)
		if err != nil {
			t.Fatal(err)
		}
		field, _ := receipt.Field(tc.Metric)
		switch tc.Name {
		case "missing_is_null":
			missing = field
		case "observed_zero_is_available":
			zero = field
		}
	}
	b, err := json.Marshal(struct {
		Missing AccountingField `json:"missing"`
		Zero    AccountingField `json:"zero"`
	}{missing, zero})
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)
	if !strings.Contains(got, `"missing":{"availability":"missing","value":null`) {
		t.Fatalf("missing accounting did not remain null: %s", got)
	}
	if !strings.Contains(got, `"zero":{"availability":"available","value":0`) {
		t.Fatalf("observed zero did not remain available: %s", got)
	}
}

func TestAggregateAccountingRefusesPartialAndConflictingTotals(t *testing.T) {
	complete := mustReconcileAccounting(t, []AccountingSource{accountingTestSource(AuthorityProviderAggregate, "provider://complete", 1, 1, AccountingValues{CostUSD: floatPtr(0.25), CacheReadTokens: floatPtr(40)})})
	partial := mustReconcileAccounting(t, []AccountingSource{accountingTestSource(AuthorityStepSum, "transcript://partial", 1, 2, AccountingValues{CostUSD: floatPtr(0.10), CacheReadTokens: floatPtr(20)})})
	rollup := AggregateAccounting([]AccountingReceipt{complete, partial})
	if rollup.CostUSD.Availability != AvailabilityDegraded || rollup.CostUSD.Value == nil {
		t.Fatalf("partial total = %+v, want a visible degraded value", rollup.CostUSD)
	}
	if rollup.CostUSD.Coverage.Observed != 1 || rollup.CostUSD.Coverage.Expected != 2 {
		t.Fatalf("partial total coverage = %+v, want 1/2", rollup.CostUSD.Coverage)
	}

	conflict := mustReconcileAccounting(t, []AccountingSource{
		accountingTestSource(AuthorityHarnessAggregate, "harness://conflict", 1, 1, AccountingValues{CostUSD: floatPtr(0.30)}),
		accountingTestSource(AuthorityProviderAggregate, "provider://conflict", 1, 1, AccountingValues{CostUSD: floatPtr(0.35)}),
	})
	rollup = AggregateAccounting([]AccountingReceipt{complete, conflict})
	if rollup.CostUSD.Availability != AvailabilityConflict || rollup.CostUSD.RefusalReason == "" {
		t.Fatalf("conflicting total = %+v, want conflict with refusal", rollup.CostUSD)
	}
}

func loadAccountingFixtures(t *testing.T) accountingFixtureFile {
	t.Helper()
	b, err := os.ReadFile("testdata/accounting_cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture accountingFixtureFile
	if err := json.Unmarshal(b, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture
}

func mustReconcileAccounting(t *testing.T, sources []AccountingSource) AccountingReceipt {
	t.Helper()
	receipt, err := ReconcileAccounting(sources)
	if err != nil {
		t.Fatal(err)
	}
	return receipt
}

func accountingTestSource(authority AccountingAuthority, ref string, observed, expected int, values AccountingValues) AccountingSource {
	return AccountingSource{
		Authority: authority,
		Artifact:  AccountingArtifact{Ref: ref, SHA256: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
		Coverage:  AccountingCoverage{Scope: "agent_turns", Observed: observed, Expected: expected},
		Values:    values,
	}
}

func equalFloatPointers(a, b *float64) bool {
	return (a == nil && b == nil) || (a != nil && b != nil && *a == *b)
}

func floatPtr(v float64) *float64 { return &v }
