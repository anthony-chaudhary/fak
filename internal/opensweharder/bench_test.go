package opensweharder

import (
	"path/filepath"
	"testing"
)

// TestBenchmarkPreflight verifies that the benchmark fixture loads and executes cleanly.
func TestBenchmarkPreflight(t *testing.T) {
	fixture, err := LoadFixture(filepath.Join("testdata", "frozen_tasks.json"))
	if err != nil {
		t.Fatalf("failed to load fixture: %v", err)
	}
	report, err := Run(fixture)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if !report.ReversalOK {
		t.Fatalf("expected reversal to succeed, got false")
	}
}

// BenchmarkOpenSWEHarder measures the closed-loop evaluation throughput over frozen tasks.
func BenchmarkOpenSWEHarder(b *testing.B) {
	fixture, err := LoadFixture(filepath.Join("testdata", "frozen_tasks.json"))
	if err != nil {
		b.Fatalf("failed to load fixture: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		report, err := Run(fixture)
		if err != nil {
			b.Fatalf("Run failed: %v", err)
		}
		if !report.ReversalOK {
			b.Fatalf("expected reversal to succeed, got false")
		}
	}
}
