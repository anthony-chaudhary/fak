package lightroteval

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkEvaluate(b *testing.B) {
	req := fixture([][]float64{
		{8, .2, -.1, .1},
		{7.5, .1, -.2, .2},
		{-7.8, -.2, .1, -.1},
		{.2, .1, -.2, 7},
	})
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Evaluate(req)
		if res.Outcome != OutcomeSupported {
			b.Fatalf("unexpected outcome: %s", res.Outcome)
		}
	}
}

func BenchmarkEvaluateJSONFixtures(b *testing.B) {
	paths, err := filepath.Glob("testdata/*.json")
	if err != nil || len(paths) == 0 {
		b.Fatalf("fixtures not found: %v", err)
	}
	var fixtures []Request
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			b.Fatal(err)
		}
		var f fixtureFile
		if err := json.Unmarshal(raw, &f); err != nil {
			b.Fatal(err)
		}
		fixtures = append(fixtures, f.Request)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, req := range fixtures {
			res := Evaluate(req)
			if res.Outcome != OutcomeSupported {
				b.Fatalf("unexpected outcome: %s", res.Outcome)
			}
		}
	}
}

func BenchmarkLightRotation(b *testing.B) {
	n := 64
	block := 4
	seed := uint64(6250)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rot := lightRotation(n, block, seed)
		if len(rot) != n {
			b.Fatalf("unexpected dimension: %d", len(rot))
		}
	}
}

func BenchmarkQuantize(b *testing.B) {
	samples := [][]float64{
		{8, .2, -.1, .1},
		{7.5, .1, -.2, .2},
		{-7.8, -.2, .1, -.1},
		{.2, .1, -.2, 7},
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		q := quantize(samples, 4)
		if len(q) != len(samples) {
			b.Fatalf("unexpected row count: %d", len(q))
		}
	}
}
