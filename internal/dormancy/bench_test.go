package dormancy

import (
	"testing"
	"time"
)

// BenchmarkDormancy exercises the pure bucket boundary calculation in a loop
// across representative duration thresholds.
func BenchmarkDormancy(b *testing.B) {
	gaps := []time.Duration{
		0,
		30 * time.Second,
		WarmMax,
		30 * time.Minute,
		CoolMax,
		12 * time.Hour,
		ColdMax,
		15 * 24 * time.Hour,
		FrozenMax,
		365 * 24 * time.Hour,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, g := range gaps {
			_ = Bucket(g)
		}
	}
}

// TestBenchmarkDormancySanity verifies that the benchmark inputs evaluate as expected.
func TestBenchmarkDormancySanity(t *testing.T) {
	if Bucket(0) != Warm {
		t.Fatalf("Bucket(0) = %v, want Warm", Bucket(0))
	}
	if Bucket(FrozenMax) != Ancient {
		t.Fatalf("Bucket(FrozenMax) = %v, want Ancient", Bucket(FrozenMax))
	}
}
