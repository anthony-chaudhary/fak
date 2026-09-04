package cubicquanteval

import "testing"

func BenchmarkEvaluateReconstructionFixture(b *testing.B) {
	fix := fixture(&testing.T{})
	req := Request{
		Scope:       ScopeReconstruction,
		FixtureJSON: fix,
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Evaluate(req)
		if res.Outcome != Supported {
			b.Fatalf("unexpected outcome: %s", res.Outcome)
		}
	}
}

func BenchmarkGroupErrors(b *testing.B) {
	xs := samples("gaussian", 128, 42)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		c, u, n := groupErrors(xs, 4, 0.25, 17)
		if c <= 0 || u <= 0 || n <= 0 {
			b.Fatalf("invalid group errors: %f, %f, %f", c, u, n)
		}
	}
}
