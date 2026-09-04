package kvquantquality

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkEvaluate(b *testing.B) {
	req := validRequest()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		report := Evaluate(req)
		if report.Outcome != OutcomeSupported {
			b.Fatalf("unexpected outcome: %s", report.Outcome)
		}
	}
}

func BenchmarkEvaluateJSON(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "kv-q4-16384.json"))
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		out, err := EvaluateJSON(raw)
		if err != nil {
			b.Fatal(err)
		}
		if len(out) == 0 {
			b.Fatal("empty output")
		}
	}
}

func BenchmarkMeanJSD(b *testing.B) {
	base := make([][]float64, 16)
	cand := make([][]float64, 16)
	for i := 0; i < 16; i++ {
		base[i] = []float64{0.1, 0.2, 0.3, 0.4}
		cand[i] = []float64{0.12, 0.18, 0.31, 0.39}
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		jsd, err := meanJSD(base, cand)
		if err != nil {
			b.Fatal(err)
		}
		if jsd <= 0 {
			b.Fatalf("expected positive jsd, got %f", jsd)
		}
	}
}

func BenchmarkEvaluateJSONFixtures(b *testing.B) {
	files, err := filepath.Glob("testdata/*.json")
	if err != nil || len(files) == 0 {
		b.Fatalf("fixtures not found: %v", err)
	}
	var fixtures [][]byte
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			b.Fatal(err)
		}
		fixtures = append(fixtures, raw)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, raw := range fixtures {
			out, err := EvaluateJSON(raw)
			if err != nil {
				b.Fatal(err)
			}
			var rep Report
			if err := json.Unmarshal(out, &rep); err != nil {
				b.Fatal(err)
			}
		}
	}
}
