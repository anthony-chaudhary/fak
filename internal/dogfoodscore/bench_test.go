package dogfoodscore

import (
	"testing"
	"time"
)

// BenchmarkDogfoodScore benchmarks the calculation and folding of the dogfood scorecard.
func BenchmarkDogfoodScore(b *testing.B) {
	opts := Options{
		Root:        b.TempDir(),
		Now:         time.Date(2026, 6, 25, 12, 0, 0, 0, time.UTC),
		ClaudeHome:  b.TempDir(),
		WindowHours: 72,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		payload := Build(opts)
		if payload.Schema == "" {
			b.Fatal("unexpected empty schema")
		}
	}
}
