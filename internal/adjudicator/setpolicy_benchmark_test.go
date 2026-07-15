package adjudicator

import (
	"fmt"
	"testing"
)

// BenchmarkSetPolicyScaling measures the complete policy-swap path. Its ns/op
// bounds the write-lock stall window paid by concurrent Adjudicate readers.
func BenchmarkSetPolicyScaling(b *testing.B) {
	for _, predicates := range []int{0, 100, 2_000, 10_000} {
		b.Run(fmt.Sprintf("predicates=%d", predicates), func(b *testing.B) {
			p := manyPreds(predicates)
			a := New(Policy{})
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				a.SetPolicy(p)
			}
		})
	}
}
