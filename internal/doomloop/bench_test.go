package doomloop

import (
	"testing"
)

// BenchmarkDoomLoop exercises doom loop classification and predictive nudge
// evaluation in a loop to measure detection performance and ensure low overhead.
func BenchmarkDoomLoop(b *testing.B) {
	samples := []Sample{
		{UnixMillis: 0, Effort: 10, Progress: 4, Alive: true},
		{UnixMillis: 60000, Effort: 20, Progress: 4, Alive: true},
		{UnixMillis: 120000, Effort: 30, Progress: 4, Alive: true},
		{UnixMillis: 180000, Effort: 40, Progress: 4, Alive: true},
		{UnixMillis: 240000, Effort: 50, Progress: 4, Alive: true},
		{UnixMillis: 300000, Effort: 60, Progress: 4, Alive: true},
	}
	cfg := DefaultConfig()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Classify(samples, cfg)
		if res.Verdict != VerdictDoomLoop {
			b.Fatalf("expected DOOM_LOOP, got %v", res.Verdict)
		}
	}
}

// BenchmarkPredictiveNudge exercises turn-by-turn predictive nudge evaluation.
func BenchmarkPredictiveNudge(b *testing.B) {
	tracker := NewPredictiveNudgeTracker("bench-objective", 6)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pn := tracker.RecordTurn(i%4 == 0, "simulated compile error")
		_ = pn.Triggered
	}
}
