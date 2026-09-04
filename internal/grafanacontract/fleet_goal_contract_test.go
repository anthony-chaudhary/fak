package grafanacontract

import (
	"os"
	"testing"
)

func TestFleetOverviewCarriesRunOperationsContract(t *testing.T) {
	path := "../../tools/grafana/dashboards/fak-fleet-overview.json"
	res, err := VerifyFleetOverview(path)
	if err != nil {
		t.Fatalf("VerifyFleetOverview: %v", err)
	}
	if !res.Passed {
		t.Fatalf("contract missing tokens: %v", res.Missing)
	}
}

func TestVerifyDashboardBytesMissingTokens(t *testing.T) {
	sample := []byte(`{
		"title": "My Dashboard",
		"panels": [
			{"title": "P1", "description": "Desc1", "targets": [{"expr": "metric_one"}]}
		]
	}`)
	res, err := VerifyDashboardBytes(sample, []string{"metric_one", "metric_two"})
	if err != nil {
		t.Fatalf("VerifyDashboardBytes failed: %v", err)
	}
	if res.Passed {
		t.Fatalf("expected failure, got pass")
	}
	if len(res.Missing) != 1 || res.Missing[0] != "metric_two" {
		t.Fatalf("unexpected missing: %v", res.Missing)
	}
}

func BenchmarkVerifyDashboardBytes(b *testing.B) {
	content, err := os.ReadFile("../../tools/grafana/dashboards/fak-fleet-overview.json")
	if err != nil {
		b.Fatal(err)
	}
	tokens := DefaultFleetOverviewTokens()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := VerifyDashboardBytes(content, tokens)
		if err != nil || !res.Passed {
			b.Fatalf("benchmark failed: %v, passed: %v", err, res.Passed)
		}
	}
}

func BenchmarkVerifyDashboardFile(b *testing.B) {
	path := "../../tools/grafana/dashboards/fak-fleet-overview.json"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := VerifyFleetOverview(path)
		if err != nil || !res.Passed {
			b.Fatalf("benchmark failed: %v, passed: %v", err, res.Passed)
		}
	}
}
