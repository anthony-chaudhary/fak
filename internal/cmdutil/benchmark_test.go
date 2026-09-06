package cmdutil

import (
	"testing"
	"time"
)

var (
	benchIntSink    int
	benchFloatSink  float64
	benchSliceSink  []int
	benchStringSink string
)

// TestBenchmarkSanity ensures that all benchmarked code paths execute cleanly.
func TestBenchmarkSanity(t *testing.T) {
	logits := make([]float32, 32000)
	logits[17500] = 999.0
	if got := Argmax(logits); got != 17500 {
		t.Fatalf("Argmax failed: got %d, want 17500", got)
	}

	samples := []time.Duration{10 * time.Millisecond, 20 * time.Millisecond, 30 * time.Millisecond}
	if got := MedianMS(samples); got != 20.0 {
		t.Fatalf("MedianMS failed: got %f, want 20.0", got)
	}

	ids := LCGIDs(512, 32000, 42)
	if len(ids) != 512 {
		t.Fatalf("LCGIDs failed: len=%d, want 512", len(ids))
	}

	a := []float32{1.0, 2.0}
	bVec := []float32{1.5, 2.0}
	if diff := MaxAbsDiffF32(a, bVec); diff < 0.49 || diff > 0.51 {
		t.Fatalf("MaxAbsDiffF32 failed: got %f, want 0.5", diff)
	}

	if got := MarkdownCell("a|b\n"); got != `a\|b ` {
		t.Fatalf("MarkdownCell failed: got %q", got)
	}

	if got := CapPositive(10, 5); got != 5 {
		t.Fatalf("CapPositive failed: got %d, want 5", got)
	}

	if got := Ms(5 * time.Millisecond); got != 5.0 {
		t.Fatalf("Ms failed: got %f, want 5.0", got)
	}
}

// BenchmarkArgmax measures finding the maximum element across a production-sized logits vector.
func BenchmarkArgmax(b *testing.B) {
	logits := make([]float32, 32000)
	for i := range logits {
		logits[i] = float32(i % 100)
	}
	logits[17500] = 999.0

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchIntSink = Argmax(logits)
	}
}

// BenchmarkMedianMS measures sorting and computing the duration median.
func BenchmarkMedianMS(b *testing.B) {
	samples := make([]time.Duration, 100)
	for i := range samples {
		samples[i] = time.Duration(((i*37)%100)+1) * time.Millisecond
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchFloatSink = MedianMS(samples)
	}
}

// BenchmarkLCGIDs measures pseudo-random token id generation.
func BenchmarkLCGIDs(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSliceSink = LCGIDs(512, 32000, 42)
	}
}

// BenchmarkMaxAbsDiffF32 measures computing the max absolute difference over float slices.
func BenchmarkMaxAbsDiffF32(b *testing.B) {
	a := make([]float32, 1024)
	bVec := make([]float32, 1024)
	for i := range a {
		a[i] = float32(i) * 0.001
		bVec[i] = float32(i)*0.001 + 0.0001
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchFloatSink = MaxAbsDiffF32(a, bVec)
	}
}

// BenchmarkMarkdownCell measures escaping pipe characters and flattening newlines.
func BenchmarkMarkdownCell(b *testing.B) {
	raw := "benchmark | result: 142.5 tok/s\r\nlatency | 12.3 ms\nnotes: clean"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = MarkdownCell(raw)
	}
}

// BenchmarkCapPositive measures upper and lower bounds clamping.
func BenchmarkCapPositive(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchIntSink = CapPositive(i%256, 128)
	}
}

// BenchmarkMs measures duration to fractional millisecond conversion.
func BenchmarkMs(b *testing.B) {
	d := 123456789 * time.Nanosecond

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchFloatSink = Ms(d)
	}
}
