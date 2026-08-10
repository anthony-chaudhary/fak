package modelroute

import "testing"

func TestCompareLocalKeepsLiveArmsExplicit(t *testing.T) {
	report, err := CompareLocal(10)
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != ComparisonSchema || report.Complete {
		t.Fatalf("report schema/complete = %q/%v", report.Schema, report.Complete)
	}
	if len(report.Arms) != 6 {
		t.Fatalf("arms=%d, want native, baseline, external, and three integration arms", len(report.Arms))
	}
	if got := report.Arms[0].SelectionAccuracy; got != 1 {
		t.Fatalf("native fixture accuracy=%v, want 1", got)
	}
	if got := report.Arms[1].SelectionAccuracy; got <= 0 || got >= 1 {
		t.Fatalf("fixed-model fixture accuracy=%v, want an honest partial score", got)
	}
	for _, arm := range report.Arms[2:] {
		if arm.Available || arm.UnavailableReason == "" {
			t.Fatalf("external arm is not honestly unavailable: %+v", arm)
		}
	}
}

func TestComparisonCorpusRoutesAsDeclared(t *testing.T) {
	manifest := comparisonManifest()
	for _, c := range ComparisonCorpus() {
		if got := manifest.Route(c.Subject).Plan.Primary(); got != c.ExpectedModel {
			t.Errorf("%s routed to %q, want %q", c.Name, got, c.ExpectedModel)
		}
	}
}

func BenchmarkModelRoutingComparison(b *testing.B) {
	manifest := comparisonManifest()
	corpus := ComparisonCorpus()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, c := range corpus {
			_ = manifest.Route(c.Subject)
		}
	}
}
