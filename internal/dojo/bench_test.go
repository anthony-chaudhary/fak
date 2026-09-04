package dojo

import "testing"

// BenchmarkDojo exercises episode scoring and board rendering in a loop.
func BenchmarkDojo(b *testing.B) {
	band := DefaultCalibBand()
	preds := []Prediction{
		{Lever: "alpha", Metric: "hit_rate", Claimed: 0.85, Unit: "pct", Basis: "model"},
		{Lever: "alpha", Metric: "latency_ms", Claimed: 120.0, Unit: "ms", Basis: "spec", LowerIsBetter: true},
		{Lever: "beta", Metric: "cache_rate", Claimed: 0.50, Unit: "pct", Basis: "estimate"},
		{Lever: "gamma", Metric: "token_cost", Claimed: 1.0, Unit: "tokens", Basis: "prior"},
	}
	outcomes := []Outcome{
		{Realized: 0.82, Measured: true, Sample: 100, Provenance: Observed},
		{Realized: 110.0, Measured: true, Sample: 50, Provenance: Witnessed},
		{Realized: 0.48, Measured: true, Sample: 80, Provenance: Observed},
		{Measured: false},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		episodes := make([]Episode, len(preds))
		for j := range preds {
			episodes[j] = Score("bench_scenario", preds[j], outcomes[j], band)
		}
		board := BoardFromEpisodes(episodes)
		rendered := RenderBoard(board)
		if rendered == "" {
			b.Fatal("RenderBoard produced empty output")
		}
	}
}
