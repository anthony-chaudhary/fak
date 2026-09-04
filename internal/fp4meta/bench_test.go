package fp4meta

import (
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkParse(b *testing.B) {
	caps := DefaultCapabilities()
	raw, err := os.ReadFile(filepath.Join("testdata", "nvfp4.json"))
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, res, err := Parse(raw, caps)
		if err != nil || res.Outcome != OutcomeAccept {
			b.Fatalf("Parse failed: res=%#v err=%v", res, err)
		}
	}
}

func BenchmarkAdjudicate(b *testing.B) {
	caps := DefaultCapabilities()
	raw, err := os.ReadFile(filepath.Join("testdata", "mxfp4.json"))
	if err != nil {
		b.Fatal(err)
	}
	desc, res, err := Parse(raw, caps)
	if err != nil || res.Outcome != OutcomeAccept {
		b.Fatalf("setup Parse failed: res=%#v err=%v", res, err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		out := Adjudicate(desc, caps)
		if out.Outcome != OutcomeAccept {
			b.Fatalf("Adjudicate failed: %#v", out)
		}
	}
}

func BenchmarkDecodeE2M1(b *testing.B) {
	patterns := []byte{0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, bits := range patterns {
			_, err := DecodeE2M1(bits)
			if err != nil {
				b.Fatalf("DecodeE2M1 failed: %v", err)
			}
		}
	}
}

func BenchmarkMarshalCanonical(b *testing.B) {
	caps := DefaultCapabilities()
	raw, err := os.ReadFile(filepath.Join("testdata", "e2m1.json"))
	if err != nil {
		b.Fatal(err)
	}
	desc, res, err := Parse(raw, caps)
	if err != nil || res.Outcome != OutcomeAccept {
		b.Fatalf("setup Parse failed: res=%#v err=%v", res, err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		data, err := MarshalCanonical(desc)
		if err != nil || len(data) == 0 {
			b.Fatalf("MarshalCanonical failed: len=%d err=%v", len(data), err)
		}
	}
}
