package conformance

import (
	"testing"
)

// BenchmarkConformance exercises policy and verdict evaluation across the full conformance suite in a loop.
func BenchmarkConformance(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := Run()
		if !r.Pass {
			b.Fatalf("conformance run failed: %+v", r)
		}
	}
}

// BenchmarkAdjudicationMatrix exercises the dogfood policy verdict matrix evaluation in a loop.
func BenchmarkAdjudicationMatrix(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := checkAdjudication()
		if !res.Pass {
			b.Fatalf("adjudication matrix failed: %+v", res)
		}
	}
}
