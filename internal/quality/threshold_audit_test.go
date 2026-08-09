package quality

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type thresholdAuditFixture struct {
	Request ThresholdAuditRequest `json:"request"`
	Receipt ThresholdAuditResult  `json:"receipt"`
}

func TestThresholdAuditCommittedReceipts(t *testing.T) {
	for _, name := range []string{"threshold_audit_accepted.json", "threshold_audit_precision_refused.json"} {
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("testdata", name))
			if err != nil {
				t.Fatal(err)
			}
			var fixture thresholdAuditFixture
			if err := json.Unmarshal(data, &fixture); err != nil {
				t.Fatal(err)
			}
			got := AuditThreshold(fixture.Request)
			if !reflect.DeepEqual(got, fixture.Receipt) {
				t.Fatalf("receipt mismatch\n got: %#v\nwant: %#v", got, fixture.Receipt)
			}
		})
	}
}

func TestThresholdAuditRefusesMissingEvidence(t *testing.T) {
	got := AuditThreshold(ThresholdAuditRequest{Comparison: "at_least", Observations: []float64{1}})
	if got.Verdict != "refused" || got.RefusalCode != "evidence_missing" {
		t.Fatalf("got %#v", got)
	}
}

func TestThresholdAuditComparators(t *testing.T) {
	zeroFloat := 0.0
	zeroInt := 0
	for _, comparison := range []string{"at_least", "greater_than", "at_most", "less_than"} {
		got := AuditThreshold(ThresholdAuditRequest{Threshold: 0.5, Comparison: comparison, Observations: []float64{0.2, 0.8}, BoundaryWidth: &zeroFloat, RoundTripDecimalPlaces: &zeroInt, Perturbation: &zeroFloat})
		if got.RefusalCode == "invalid_contract" {
			t.Fatalf("%s rejected: %#v", comparison, got)
		}
	}
}
