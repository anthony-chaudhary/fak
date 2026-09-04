package qwensemanticstop

import (
	"testing"
)

func BenchmarkQwenSemanticStop(b *testing.B) {
	r := validReceipt()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rc := r
		if err := Evaluate(&rc); err != nil {
			b.Fatal(err)
		}
	}
}

func TestBenchmarkValidity(t *testing.T) {
	r := validReceipt()
	if err := Evaluate(&r); err != nil {
		t.Fatalf("expected valid receipt in benchmark harness: %v", err)
	}
}
