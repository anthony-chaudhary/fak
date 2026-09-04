package codebookmeta

import (
	"os"
	"path/filepath"
	"testing"
)

var benchCapability = Capability{
	PackingIDs:     []string{"nibble-lsb@1", "bitpack-lsb@1"},
	DecodeFeatures: []string{"per-block-scale", "explicit-codebook"},
	RoutedRuntimes: []string{"lab-runtime@2.4.1"},
}

func BenchmarkParse(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "nf4.json"))
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res, err := Parse(raw, benchCapability)
		if err != nil || res.Outcome != OutcomeSupported {
			b.Fatalf("Parse failed: res=%#v err=%v", res, err)
		}
	}
}

func BenchmarkAdjudicate(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "learned.json"))
	if err != nil {
		b.Fatal(err)
	}
	res, err := Parse(raw, benchCapability)
	if err != nil || res.Outcome != OutcomeSupported || res.Descriptor == nil {
		b.Fatalf("setup failed: res=%#v err=%v", res, err)
	}
	desc := *res.Descriptor
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		out := Adjudicate(desc, benchCapability)
		if out.Outcome != OutcomeSupported {
			b.Fatalf("Adjudicate failed: out=%#v", out)
		}
	}
}

func BenchmarkMarshalCanonical(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "integer-grid.json"))
	if err != nil {
		b.Fatal(err)
	}
	res, err := Parse(raw, benchCapability)
	if err != nil || res.Outcome != OutcomeSupported || res.Descriptor == nil {
		b.Fatalf("setup failed: res=%#v err=%v", res, err)
	}
	desc := *res.Descriptor
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		data, err := MarshalCanonical(desc)
		if err != nil || len(data) == 0 {
			b.Fatalf("MarshalCanonical failed: len=%d err=%v", len(data), err)
		}
	}
}

func BenchmarkCodebookDigest(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "learned.json"))
	if err != nil {
		b.Fatal(err)
	}
	res, err := Parse(raw, benchCapability)
	if err != nil || res.Outcome != OutcomeSupported || res.Descriptor == nil {
		b.Fatalf("setup failed: res=%#v err=%v", res, err)
	}
	cb := res.Descriptor.Codebook
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		d := CodebookDigest(cb)
		if d.Algorithm != "sha256" || len(d.Value) != 64 {
			b.Fatalf("CodebookDigest failed: %#v", d)
		}
	}
}
