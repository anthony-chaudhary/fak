package rotationmeta

import "testing"

func BenchmarkRotationMeta(b *testing.B) {
	d := fixture(nil, RecipeQuaRot, "arxiv:2404.00456v2", Transform{
		Name:      "residual",
		Placement: PlacementOffline,
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decision := Validate(d)
		if decision.Outcome != OutcomeSupported {
			b.Fatalf("unexpected validation decision: %+v", decision)
		}
	}
}
