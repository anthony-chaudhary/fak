package mtpeval

import (
	"context"
	"testing"
)

// TestBenchmarkMTPEvalSanity verifies that the evaluation logic executed in BenchmarkMTPEval runs cleanly.
func TestBenchmarkMTPEvalSanity(t *testing.T) {
	mock := &deterministicMockGenerator{
		tpsMultiplier: 1.0,
		acceptancePct: 75.0,
	}
	suite := DefaultSmokeSuite()
	gates := DefaultQualityGates()
	report, err := RunEvaluation(context.Background(), mock, suite, gates)
	if err != nil {
		t.Fatalf("RunEvaluation failed: %v", err)
	}
	if !report.Passed {
		t.Fatalf("expected evaluation to pass gates, failed: %v", report.GateFailures)
	}
}

// BenchmarkMTPEval measures execution performance of the MTP speculative evaluation loop.
func BenchmarkMTPEval(b *testing.B) {
	mock := &deterministicMockGenerator{
		tpsMultiplier: 1.0,
		acceptancePct: 75.0,
	}
	suite := DefaultSmokeSuite()
	gates := DefaultQualityGates()
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := RunEvaluation(ctx, mock, suite, gates)
		if err != nil {
			b.Fatalf("RunEvaluation failed: %v", err)
		}
		if !report.Passed {
			b.Fatalf("expected report to pass gates")
		}
	}
}
