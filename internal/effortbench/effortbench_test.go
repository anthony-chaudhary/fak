package effortbench

import (
	"context"
	"testing"
)

func TestEffortBenchmarkRunner(t *testing.T) {
	runner := NewBenchmarkRunner()
	workload := DefaultWorkload()

	comparison := runner.CompareAll(context.Background(), workload)

	t.Logf("\n%s", comparison.String())

	// Dynamic modulation must achieve token savings compared to static thinking
	if comparison.TokenSavingsPct <= 0 {
		t.Errorf("expected positive token savings, got %.2f%%", comparison.TokenSavingsPct)
	}

	// Dynamic modulation must achieve latency speedup compared to static thinking
	if comparison.LatencySpeedupPct <= 0 {
		t.Errorf("expected positive latency speedup, got %.2f%%", comparison.LatencySpeedupPct)
	}

	// Dynamic modulation must preserve cache hit rate compared to cross-model switching
	if comparison.DynamicReport.CacheHitRate <= comparison.CrossModelReport.CacheHitRate {
		t.Errorf("expected dynamic modulation cache hit rate (%f) to exceed cross-model switching (%f)",
			comparison.DynamicReport.CacheHitRate, comparison.CrossModelReport.CacheHitRate)
	}

	// Dynamic modulation cache hit rate should be 80% (4 out of 5 turns)
	if comparison.DynamicReport.CacheHitRate < 0.8 {
		t.Errorf("expected dynamic cache hit rate >= 0.8, got %f", comparison.DynamicReport.CacheHitRate)
	}
}
