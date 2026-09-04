package orientation

import (
	"testing"
	"time"
)

// BenchmarkOrientation measures end-to-end loading, validation, assessment,
// and text rendering of the product orientation snapshot in a b.N loop.
func BenchmarkOrientation(b *testing.B) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, err := Current()
		if err != nil {
			b.Fatalf("Current failed: %v", err)
		}
		v := Assess(s, now)
		text := v.Text()
		if len(text) == 0 {
			b.Fatal("expected non-empty rendered text")
		}
	}
}

// BenchmarkCurrent measures unmarshaling and embedded snapshot validation.
func BenchmarkCurrent(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, err := Current()
		if err != nil {
			b.Fatalf("Current failed: %v", err)
		}
		if len(s.Items) == 0 {
			b.Fatal("expected items in snapshot")
		}
	}
}

// BenchmarkValidate measures schema, field, and transition rule validation on an in-memory snapshot.
func BenchmarkValidate(b *testing.B) {
	s, err := Current()
	if err != nil {
		b.Fatalf("Current failed: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Validate(s); err != nil {
			b.Fatalf("Validate failed: %v", err)
		}
	}
}

// BenchmarkAssess measures review date calculation, freshness categorization, and view synthesis.
func BenchmarkAssess(b *testing.B) {
	s, err := Current()
	if err != nil {
		b.Fatalf("Current failed: %v", err)
	}
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := Assess(s, now)
		if v.Freshness == "" {
			b.Fatal("expected non-empty freshness")
		}
	}
}

// BenchmarkViewText measures string formatting and text report construction.
func BenchmarkViewText(b *testing.B) {
	s, err := Current()
	if err != nil {
		b.Fatalf("Current failed: %v", err)
	}
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	v := Assess(s, now)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		text := v.Text()
		if len(text) == 0 {
			b.Fatal("expected non-empty rendered text")
		}
	}
}

// TestBenchmarkOrientationExecution verifies that BenchmarkOrientation executes cleanly.
func TestBenchmarkOrientationExecution(t *testing.T) {
	res := testing.Benchmark(BenchmarkOrientation)
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
