package requanteval

import (
	"testing"
)

func BenchmarkRequantEval(b *testing.B) {
	req := fixture(1337, []float64{-0.8, -0.4}, [][]float64{{1, -0.8}, {-0.8, 1}})
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Evaluate(req)
		if res.Outcome != OutcomeSupported {
			b.Fatalf("unexpected outcome: %s", res.Outcome)
		}
	}
}
