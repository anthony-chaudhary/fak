package composition

import "testing"

func BenchmarkComposition(b *testing.B) {
	s := qwen("bench-qwen3.8")
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		h, r, err := Resolve(s)
		if err != nil {
			b.Fatal(err)
		}
		if h.Snapshot() == nil || r.Outcome != "validated" {
			b.Fatalf("unexpected resolution result: h=%+v r=%+v", h, r)
		}
	}
}
