package dsparity

import (
	"testing"
)

// TestBenchmarkDSParity verifies that the benchmark loop executes successfully on synthetic fixtures.
func TestBenchmarkDSParity(t *testing.T) {
	rows := Rows()
	for _, r := range rows {
		if err := r.Validate(); err != nil {
			t.Fatalf("row validation failed: %v", err)
		}
	}
}

// BenchmarkDSParity exercises disparity matrix row validation and synthetic candidate ranking in a loop.
func BenchmarkDSParity(b *testing.B) {
	rows := Rows()
	positions := make([]int, 64)
	scores := make([]float64, 64)
	for i := range positions {
		positions[i] = i
		scores[i] = float64(i%7) + 0.05*float64(i%5)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, r := range rows {
			if err := r.Validate(); err != nil {
				b.Fatalf("validation failed: %v", err)
			}
		}
		_ = stableTopK(positions, scores, 8)
	}
}
