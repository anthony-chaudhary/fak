package fleetcap

import "testing"

func BenchmarkFleetCap(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Compute(400, 10)
		_ = Assess(400, 10, 67)
		_ = AvailableFrom(16, 3, 67)
	}
}

func TestBenchmarkFleetCap(t *testing.T) {
	res := testing.Benchmark(BenchmarkFleetCap)
	if res.N <= 0 {
		t.Fatal("expected positive benchmark iterations")
	}
}
