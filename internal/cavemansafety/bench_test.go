package cavemansafety

import (
	"testing"
)

func BenchmarkCavemanSafety(b *testing.B) {
	arms := DefaultValueArms()
	corpus := DefaultValueCorpus()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := RunValueBenchmark(arms, corpus)
		if err != nil {
			b.Fatalf("RunValueBenchmark failed: %v", err)
		}
		if len(report.Metrics) == 0 {
			b.Fatal("unexpected empty report metrics")
		}
	}
}

func TestBenchmarkCavemanSafetyRuns(t *testing.T) {
	arms := DefaultValueArms()
	corpus := DefaultValueCorpus()
	report, err := RunValueBenchmark(arms, corpus)
	if err != nil {
		t.Fatalf("RunValueBenchmark failed: %v", err)
	}
	if len(report.Metrics) == 0 || len(report.Traces) == 0 {
		t.Fatalf("expected non-empty benchmark results, got metrics=%d traces=%d", len(report.Metrics), len(report.Traces))
	}
}
