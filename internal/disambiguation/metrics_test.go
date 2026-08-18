package disambiguation

import "testing"

func TestPublicMetricsSumsToTotal(t *testing.T) {
	report := PublicMetrics()
	if report.Total != len(publicEntries) {
		t.Fatalf("total=%d entries=%d", report.Total, len(publicEntries))
	}
	if got := SumMetrics(report.Freshness); got != report.Total {
		t.Fatalf("freshness sum=%d total=%d", got, report.Total)
	}
	if got := SumMetrics(report.Owners); got != report.Total {
		t.Fatalf("owner sum=%d total=%d", got, report.Total)
	}
	if len(report.SourceFamilies) == 0 {
		t.Fatal("missing source families")
	}
}

func TestMetricsCountsUncoveredCandidateClasses(t *testing.T) {
	report := Metrics(publicEntries[:1], CoverageReport{Findings: []CoverageFinding{{Reason: CoverageReasonMissingClassification}, {Reason: CoverageReasonMissingClassification}}})
	if len(report.UncoveredCandidateClasses) != 1 || report.UncoveredCandidateClasses[0].Count != 2 {
		t.Fatalf("classes=%#v", report.UncoveredCandidateClasses)
	}
}
