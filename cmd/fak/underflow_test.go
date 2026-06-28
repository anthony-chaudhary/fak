package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

func TestCompactionEpisodesUnderflowGuard(t *testing.T) {
	// uint64 underflow case: ON > OFF, which would wrap to a huge positive number
	// without the int64 conversion fix
	rep := CompactionBacktestReport{
		FiredAttempts:     10,
		ShedTokensSum:     10000,
		InputTokensOffSum: 50000,
		InputTokensOnSum:  60000, // ON > OFF - pathological case that would underflow
	}
	ins := compactionEpisodesFromBacktest(rep)
	if len(ins) != 2 {
		t.Fatalf("with ON > OFF, should emit both metrics, got %d", len(ins))
	}
	shed := map[string]dojo.ScoredInput{}
	for _, in := range ins {
		shed[in.Prediction.Metric] = in
	}
	// Verify cache_prefix_preserved is 1.0 with no prefix_mismatch
	if shed["cache_prefix_preserved"].Outcome.Realized != 1.0 {
		t.Fatalf("zero prefix_mismatch should yield 1.0 preserved, got %.2f", shed["cache_prefix_preserved"].Outcome.Realized)
	}
	// The token_shed_ratio should be 0.0 because billedDelta (50000 - 60000 = -10000)
	// gets clamped to 0, and 0 / 10000 = 0
	if shed["token_shed_ratio"].Outcome.Realized != 0.0 {
		t.Fatalf("ON > OFF should clamp billedDelta to 0, giving ratio 0.0, got %.2f", shed["token_shed_ratio"].Outcome.Realized)
	}
}
