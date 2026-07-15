package bench

import (
	"path/filepath"
	"testing"
)

func TestNegationPressureProbeBucketsAndSign(t *testing.T) {
	items, err := LoadNegatedQAFixture(filepath.Join("testdata", "negated_qa.json"))
	if err != nil {
		t.Fatal(err)
	}
	rows, report := RunNegationPressureProbe(items)
	if len(rows) != len(items)*3*2 || len(report.Buckets) != 3 {
		t.Fatalf("rows=%d report=%+v", len(rows), report)
	}
	for _, bucket := range report.Buckets {
		t.Logf("NEGATION_PRESSURE %s pressure=%.2f negative=%.3f positive=%.3f delta=%.3f", bucket.Bucket, bucket.Pressure, bucket.NegativePassRate, bucket.PositivePassRate, bucket.FramingDelta)
	}
	t.Logf("NEGATION_PRESSURE correlation=%.4f provenance=%s", report.NegativePressureCorrelation, report.Provenance)
	if !report.SignPinned || report.NegativePressureCorrelation >= 0 {
		t.Fatalf("correlation sign flipped: %+v", report)
	}
	if report.Buckets[2].NegativePassRate >= report.Buckets[0].NegativePassRate {
		t.Fatalf("negative adherence did not fall: %+v", report.Buckets)
	}
	for _, bucket := range report.Buckets {
		if bucket.FramingDelta < 0 {
			t.Fatalf("positive framing lost: %+v", bucket)
		}
	}
}
