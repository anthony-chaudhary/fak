package ctxmmu

import "testing"

func TestCompareLocalKeepsMemorySystemsExplicit(t *testing.T) {
	report := CompareLocal(10)
	if report.Schema != ComparisonSchema || report.Complete {
		t.Fatalf("schema/complete=%q/%v", report.Schema, report.Complete)
	}
	if len(report.Arms) != 7 {
		t.Fatalf("arms=%d, want native, baseline, Letta, and four integration arms", len(report.Arms))
	}
	if !report.Arms[0].Available || report.Arms[0].Correctness != 1 {
		t.Fatalf("native arm=%+v", report.Arms[0])
	}
	if got := report.Arms[1].Correctness; got <= 0 || got >= 1 {
		t.Fatalf("full-history baseline correctness=%v, want honest partial score", got)
	}
	for _, arm := range report.Arms[2:] {
		if arm.Available || arm.UnavailableReason == "" {
			t.Fatalf("live arm must remain explicitly unavailable: %+v", arm)
		}
	}
}

func BenchmarkContextMemoryComparison(b *testing.B) {
	a := NewMemoryWriteAdjudicator()
	corpus := ComparisonCorpus()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, c := range corpus {
			_ = a.AdmitWrite(c.Body)
		}
	}
}
