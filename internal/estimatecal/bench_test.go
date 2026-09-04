package estimatecal

import "testing"

func BenchmarkEstimateCal(b *testing.B) {
	var s Store
	for range MinSamples {
		s.Observe("provider", "model", 100, 120)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Observe("provider", "model", 100, 125)
		_, _ = s.Ratio("provider", "model")
	}
}

func TestBenchmarkEstimateCalSanity(t *testing.T) {
	var s Store
	for range MinSamples {
		s.Observe("provider", "model", 100, 120)
	}
	ratio, ok := s.Ratio("provider", "model")
	if !ok || ratio <= 0 {
		t.Fatalf("expected valid ratio, got %v, ok=%v", ratio, ok)
	}
}
