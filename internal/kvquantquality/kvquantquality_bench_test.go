package kvquantquality

import (
	"encoding/json"
	"testing"
)

// BenchmarkEvaluateQuality benchmarks the core deterministic quality evaluation pipeline.
func BenchmarkEvaluateQuality(b *testing.B) {
	req := validRequest()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := Evaluate(req)
		if report.Outcome != OutcomeSupported {
			b.Fatalf("unexpected outcome: %s", report.Outcome)
		}
	}
}

// BenchmarkEvaluateJSON benchmarks the JSON serialization and evaluation adapter.
func BenchmarkEvaluateJSON(b *testing.B) {
	req := validRequest()
	raw, err := json.Marshal(req)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := EvaluateJSON(raw)
		if err != nil {
			b.Fatal(err)
		}
		if len(out) == 0 {
			b.Fatal("empty report output")
		}
	}
}
