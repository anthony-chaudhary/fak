package cavemansafety

import (
	"testing"
)

func BenchmarkCavemanSafetyScan(b *testing.B) {
	corpus := DefaultValueCorpus()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, c := range corpus {
			_, _ = evaluateCase(c)
		}
	}
}

func BenchmarkCavemanSafety(b *testing.B) {
	arms := DefaultValueArms()
	corpus := DefaultValueCorpus()
	b.ReportAllocs()
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

func TestBenchmarkCavemanSafetyScanRuns(t *testing.T) {
	corpus := DefaultValueCorpus()
	for _, c := range corpus {
		allowed, rule := evaluateCase(c)
		if c.Attack && allowed {
			t.Fatalf("expected attack to be blocked: %s", c.ID)
		}
		if rule == "" {
			t.Fatalf("expected non-empty rule for case: %s", c.ID)
		}
	}
}
