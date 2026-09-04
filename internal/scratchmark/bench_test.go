package scratchmark

import (
	"testing"
)

func BenchmarkScratchMark(b *testing.B) {
	sample := []byte("// This helper is scratch code.\npackage probe\n\nfunc Run() {}\n")
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Detect(sample)
		if !res.Marked {
			b.Fatal("expected sample to be marked as scratch")
		}
	}
}

func TestBenchmarkScratchMark(t *testing.T) {
	sample := []byte("// This helper is scratch code.\npackage probe\n\nfunc Run() {}\n")
	res := Detect(sample)
	if !res.Marked || res.Marker != "scratch" {
		t.Fatalf("Detect() = %+v, want scratch marker", res)
	}
}
