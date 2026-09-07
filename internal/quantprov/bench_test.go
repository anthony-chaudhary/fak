package quantprov

import (
	"os"
	"testing"
)

func BenchmarkVerify(b *testing.B) {
	raw, err := os.ReadFile("testdata/confirmed.json")
	if err != nil {
		b.Fatal(err)
	}
	res, err := ParseAndVerify(raw, support)
	if err != nil {
		b.Fatal(err)
	}
	if res.Record == nil {
		b.Fatal("nil record")
	}
	record := *res.Record

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Verify(record, support)
	}
}

func BenchmarkParseAndVerify(b *testing.B) {
	raw, err := os.ReadFile("testdata/confirmed.json")
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = ParseAndVerify(raw, support)
	}
}
