package adjudicator

import "testing"

func TestCompareLocalKeepsExternalEnginesExplicit(t *testing.T) {
	report := CompareLocal(10)
	if report.Schema != ComparisonSchema || report.Complete {
		t.Fatalf("schema/complete=%q/%v", report.Schema, report.Complete)
	}
	if len(report.Arms) != 4 {
		t.Fatalf("arms=%d, want native, baseline, OPA, Cedar", len(report.Arms))
	}
	for _, arm := range report.Arms[:2] {
		if !arm.Available || arm.Correctness != 1 {
			t.Fatalf("local arm=%+v", arm)
		}
	}
	for _, arm := range report.Arms[2:] {
		if arm.Available || arm.UnavailableReason == "" {
			t.Fatalf("external arm must remain honestly unavailable: %+v", arm)
		}
	}
}

func TestComparisonCorpusExercisesDifferentPolicyRungs(t *testing.T) {
	report := CompareLocal(1)
	if got, want := report.Arms[0].Calls, len(ComparisonCorpus()); got != want {
		t.Fatalf("calls=%d, want %d", got, want)
	}
}

func BenchmarkPolicyAdjudicationComparison(b *testing.B) {
	a := New(comparisonPolicy())
	corpus := ComparisonCorpus()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range corpus {
			_ = a.Adjudicate(b.Context(), &corpus[j].Call)
		}
	}
}
