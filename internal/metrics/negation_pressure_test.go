package metrics

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestNegationPressureCorrelationSchemaAndSign(t *testing.T) {
	report := FoldNegationPressure("MODELED / OFFLINE PROXY", []NegationPressureBucket{
		{Bucket: "low", Pressure: .2, NegativePassRate: .90, PositivePassRate: .94, SamplesPerArm: 8},
		{Bucket: "mid", Pressure: .6, NegativePassRate: .70, PositivePassRate: .90, SamplesPerArm: 8},
		{Bucket: "near-budget", Pressure: .92, NegativePassRate: .40, PositivePassRate: .82, SamplesPerArm: 8},
	})
	if report.Schema != NegationPressureSchema || !report.SignPinned || report.NegativePressureCorrelation >= 0 {
		t.Fatalf("report=%+v", report)
	}
	if len(report.Buckets) != 3 || report.Buckets[2].FramingDelta <= report.Buckets[0].FramingDelta {
		t.Fatalf("buckets=%+v", report.Buckets)
	}
	var envelope map[string]any
	if err := json.Unmarshal(report.JSON(), &envelope); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"schema", "provenance", "buckets", "negative_pressure_correlation", "positive_pressure_correlation", "sign_pinned"} {
		if _, ok := envelope[key]; !ok {
			t.Fatalf("JSON missing %s: %s", key, report.JSON())
		}
	}
	if !strings.Contains(string(report.JSON()), NegationPressureSchema) {
		t.Fatalf("schema missing: %s", report.JSON())
	}
}
