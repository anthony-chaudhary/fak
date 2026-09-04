package mixedprecision

import (
	"testing"
)

func BenchmarkMixedPrecision(b *testing.B) {
	d := baseDescriptor()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Evaluate(d, testSupport)
		if res.Outcome != OutcomeSupported {
			b.Fatalf("unexpected outcome: %s", res.Outcome)
		}
	}
}
