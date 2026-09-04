package dispatchaging

import (
	"testing"
)

// BenchmarkDispatchAging exercises candidate priority boosting and census tracking in a loop.
func BenchmarkDispatchAging(b *testing.B) {
	const now = int64(1_000_000)
	params := DefaultParams(now)

	cands := []Candidate{
		{ID: "p0-fresh", BaseWeight: wP0, ReadySince: now - 30},
		{ID: "p1-fresh", BaseWeight: wP1, ReadySince: now - 60},
		{ID: "p2-aging", BaseWeight: wP2, ReadySince: now - 1800},
		{ID: "def-aging", BaseWeight: wDefault, ReadySince: now - 7200},
		{ID: "def-starved", BaseWeight: wDefault, ReadySince: now - 25000},
		{ID: "cooling", BaseWeight: wP2, ReadySince: now - 3600, CoolingSince: now - 1200, CoolingUntil: now + 600},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Fold(cands, params)
		if res.StarvedCount == 0 || res.AgingCount == 0 || res.FreshCount == 0 {
			b.Fatalf("unexpected census: %+v", res)
		}
		if res.Pick() == "" {
			b.Fatalf("empty pick")
		}
	}
}

// TestBenchmarkDispatchAgingSanity verifies that BenchmarkDispatchAging runs cleanly.
func TestBenchmarkDispatchAgingSanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkDispatchAging)
	if res.N <= 0 {
		t.Fatalf("benchmark failed to execute: %+v", res)
	}
}
