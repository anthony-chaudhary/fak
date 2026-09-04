package nativeperfbackend

import (
	"testing"
)

// BenchmarkNativePerfBackend exercises backend metrics snapshotting and validation in a loop.
func BenchmarkNativePerfBackend(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := populatedSnapshot()
		if err := Validate(s); err != nil {
			b.Fatalf("validation failed: %v", err)
		}
	}
}

// TestBenchmarkNativePerfBackendSanity verifies that the benchmarked snapshot passes validation.
func TestBenchmarkNativePerfBackendSanity(t *testing.T) {
	s := populatedSnapshot()
	if err := Validate(s); err != nil {
		t.Fatalf("snapshot validation failed: %v", err)
	}
}
