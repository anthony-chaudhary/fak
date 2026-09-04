package nativeperfslo

import (
	"testing"
	"time"
)

func BenchmarkNativePerfSLO(b *testing.B) {
	now := time.Date(2026, 8, 28, 20, 0, 0, 0, time.UTC)
	envelope := Envelope{
		ModuleRev: "internal/nativeperf@r41+gbbdda8f04",
		Benchmark: "qwen38-4b-in128-out128-b1-quality-v3",
		Model:     "Qwen3.8-4B",
		Backend:   "cuda",
	}
	baseline := observation(now, envelope, 1.0)
	candidate := observation(now, envelope, 1.0)
	evaluator := New(DefaultThresholds())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := evaluator.Observe(now, baseline, candidate)
		if err != nil {
			b.Fatalf("observe failed: %v", err)
		}
	}
}

func TestBenchmarkNativePerfSLO(t *testing.T) {
	res := testing.Benchmark(BenchmarkNativePerfSLO)
	if res.N <= 0 {
		t.Fatalf("benchmark did not run: %+v", res)
	}
}
