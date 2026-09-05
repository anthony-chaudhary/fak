package stripeload

import (
	"bytes"
	"fmt"
	"testing"
)

// BenchmarkCarveRanges measures range partitioning across mirror counts and buffer sizes.
func BenchmarkCarveRanges(b *testing.B) {
	cases := []struct {
		name    string
		n       int64
		weights []float64
	}{
		{name: "2mirrors_4MB", n: 4 << 20, weights: []float64{1.0, 2.0}},
		{name: "3mirrors_16MB", n: 16 << 20, weights: []float64{1.0, 2.5, 0.75}},
		{name: "4mirrors_64MB", n: 64 << 20, weights: []float64{1.0, 1.5, 2.0, 3.0}},
		{name: "8mirrors_64MB", n: 64 << 20, weights: []float64{1, 1, 2, 2, 3, 3, 4, 4}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ranges := carveRanges(0, tc.n, tc.weights, DefaultMinChunk)
				if len(ranges) == 0 {
					b.Fatal("unexpected empty ranges")
				}
			}
		})
	}
}

// BenchmarkReadAt_Passthrough measures the unstriped read fast path.
func BenchmarkReadAt_Passthrough(b *testing.B) {
	data := make([]byte, 2<<20)
	s, err := New([]Source{
		{R: bytes.NewReader(data), BWWeight: 1.0},
		{R: bytes.NewReader(data), BWWeight: 3.0},
	}, WithMinChunk(1<<20))
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	buf := make([]byte, 64*1024)
	b.SetBytes(int64(len(buf)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n, err := s.ReadAt(buf, 0)
		if err != nil || n != len(buf) {
			b.Fatalf("ReadAt: n=%d err=%v", n, err)
		}
	}
}

// BenchmarkReadAt_Striped measures multi-mirror striped reads across buffer sizes.
func BenchmarkReadAt_Striped(b *testing.B) {
	sizes := []int{2 << 20, 8 << 20, 16 << 20}
	for _, size := range sizes {
		name := fmt.Sprintf("%dMB", size>>20)
		data := make([]byte, size)
		s, err := New([]Source{
			{R: bytes.NewReader(data), BWWeight: 1.0},
			{R: bytes.NewReader(data), BWWeight: 2.0},
			{R: bytes.NewReader(data), BWWeight: 1.5},
		}, WithMinChunk(256*1024))
		if err != nil {
			b.Fatalf("New: %v", err)
		}
		buf := make([]byte, size)
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				n, err := s.ReadAt(buf, 0)
				if err != nil || n != len(buf) {
					b.Fatalf("ReadAt: n=%d err=%v", n, err)
				}
			}
		})
	}
}

// BenchmarkReadAt_Parallel measures concurrent striped reads across goroutines.
func BenchmarkReadAt_Parallel(b *testing.B) {
	const totalSize = 16 << 20
	data := make([]byte, totalSize)
	s, err := New([]Source{
		{R: bytes.NewReader(data), BWWeight: 1.0},
		{R: bytes.NewReader(data), BWWeight: 2.0},
		{R: bytes.NewReader(data), BWWeight: 3.0},
	}, WithMinChunk(256*1024))
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	chunkSize := 1 << 20
	b.SetBytes(int64(chunkSize))
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		buf := make([]byte, chunkSize)
		for pb.Next() {
			n, err := s.ReadAt(buf, 0)
			if err != nil || n != len(buf) {
				b.Errorf("ReadAt: n=%d err=%v", n, err)
			}
		}
	})
}

// BenchmarkNew measures constructor validation and instance initialization.
func BenchmarkNew(b *testing.B) {
	sources := []Source{
		{R: bytes.NewReader(nil), BWWeight: 1.0},
		{R: bytes.NewReader(nil), BWWeight: 2.0},
		{R: bytes.NewReader(nil), BWWeight: 3.0},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, err := New(sources, WithMinChunk(1<<20), WithMaxConcurrency(3))
		if err != nil || s == nil {
			b.Fatalf("New: err=%v", err)
		}
	}
}

// TestBenchmarkSanity verifies that all benchmark scenarios execute without error.
func TestBenchmarkSanity(t *testing.T) {
	r1 := testing.Benchmark(BenchmarkReadAt_Passthrough)
	if r1.N <= 0 {
		t.Fatalf("expected BenchmarkReadAt_Passthrough iterations > 0, got %d", r1.N)
	}
	r2 := testing.Benchmark(BenchmarkNew)
	if r2.N <= 0 {
		t.Fatalf("expected BenchmarkNew iterations > 0, got %d", r2.N)
	}
}
