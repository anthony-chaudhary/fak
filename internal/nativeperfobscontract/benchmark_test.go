package nativeperfobscontract

import (
	"testing"
)

// Invariant: Performance benchmarks must evaluate real contract instances with zero allocations on validation.
// Precondition: The frozen contract must be valid and conform to the canonical specification.

// BenchmarkContractEvaluate measures the throughput and allocations of validating
// the frozen native performance observability contract.
func BenchmarkContractEvaluate(b *testing.B) {
	c := Frozen()
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if err := Validate(c); err != nil {
			b.Fatalf("Validate(c) failed: %v", err)
		}
	}
}

// BenchmarkContractJSON measures JSON formatting and canonical serialization throughput.
func BenchmarkContractJSON(b *testing.B) {
	c := Frozen()
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		data, err := JSON(c)
		if err != nil || len(data) == 0 {
			b.Fatalf("JSON(c) failed: %v", err)
		}
	}
}
