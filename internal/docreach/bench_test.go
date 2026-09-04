package docreach

import "testing"

func BenchmarkDocReach(b *testing.B) {
	blobs := []Blob{
		{"README.md", "[guide](docs/guide.md) `docs/note.md` [bad](missing.md)"},
		{"docs/guide.md", "[note](note.md) [api](api.md)"},
		{"docs/note.md", "Mentions docs/orphan.md and `docs/api.md`"},
		{"docs/api.md", "[readme](../README.md)"},
		{"docs/orphan.md", ""},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := Census("bench-commit", blobs)
		if r.Documents != 5 {
			b.Fatalf("unexpected document count: %d", r.Documents)
		}
	}
}
