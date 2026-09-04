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

func BenchmarkResolveForbidden(b *testing.B) {
	s := qwen("bench-forbidden")
	s.Forbidden = [][]string{{"hybrid_attention", "metal", "q4_k", "gdn_state"}}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, err := Resolve(s)
		if err == nil {
			b.Fatal("expected forbidden error")
		}
	}
}
