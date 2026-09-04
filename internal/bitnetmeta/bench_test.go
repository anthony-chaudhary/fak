package bitnetmeta

import (
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkParseAndAdjudicate(b *testing.B) {
	caps := DefaultCapabilities()
	raw, err := os.ReadFile(filepath.Join("testdata", "native-ternary.json"))
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := ParseAndAdjudicate(raw, caps)
		if res.Outcome != OutcomeAccept {
			b.Fatalf("benchmark failed with outcome %s: %s", res.Outcome, res.Detail)
		}
	}
}

func BenchmarkAdjudicate(b *testing.B) {
	caps := DefaultCapabilities()
	raw, err := os.ReadFile(filepath.Join("testdata", "native-ternary.json"))
	if err != nil {
		b.Fatal(err)
	}
	res := ParseAndAdjudicate(raw, caps)
	if res.Descriptor == nil {
		b.Fatal("nil descriptor")
	}
	desc := *res.Descriptor
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		out := Adjudicate(desc, caps)
		if out.Outcome != OutcomeAccept {
			b.Fatalf("benchmark failed with outcome %s", out.Outcome)
		}
	}
}
