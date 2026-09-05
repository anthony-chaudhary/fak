package simhash

import (
	"fmt"
	"testing"
)

func BenchmarkSimhash(b *testing.B) {
	query := "refund the customer's last payment please"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Simhash(query)
	}
}

func BenchmarkVectorSimhash(b *testing.B) {
	query := "delete all rows from the production table immediately"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = VectorSimhash(query)
	}
}

func BenchmarkHammingDistance(b *testing.B) {
	a := uint64(0x123456789abcdef0)
	c := uint64(0x0fedcba987654321)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = HammingDistance(a, c)
	}
}

func BenchmarkEmbed_Short(b *testing.B) {
	text := "cancel order"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Embed(text)
	}
}

func BenchmarkEmbed_Long(b *testing.B) {
	text := "The quick brown fox jumps over the lazy dog while checking system metrics and verifying database consistency across multi-region replica nodes."
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Embed(text)
	}
}

func BenchmarkCosine(b *testing.B) {
	v1 := Embed("refund customer payment")
	v2 := Embed("return customer payment")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Cosine(v1, v2)
	}
}

func BenchmarkIndexAdd(b *testing.B) {
	v := Embed("sample trajectory step")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var ix Index
		ix.Add("id", v, "meta")
	}
}

func BenchmarkIndexTopK(b *testing.B) {
	var ix Index
	for i := 0; i < 100; i++ {
		ix.AddText(fmt.Sprintf("q%d", i), fmt.Sprintf("sample query text for clustering turn %d", i), "")
	}
	q := Embed("sample query text for search")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ix.TopK(q, 10)
	}
}

func TestBenchmarkSmoke(t *testing.T) {
	v := Simhash("smoke test")
	if len(v) != Dim {
		t.Fatalf("Simhash len = %d, want %d", len(v), Dim)
	}
	if d := HammingDistance(0b101, 0b001); d != 1 {
		t.Fatalf("HammingDistance = %d, want 1", d)
	}
}
