package lightroteval

import (
	"testing"
)

// BenchmarkEvaluate measures performance of the validation and evaluation routine.
func BenchmarkEvaluate(b *testing.B) {
	req := Request{
		ContractVersion: ContractVersion,
		Bits:            4,
		Evidence:        EvidenceModeled,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Evaluate(req)
	}
}
