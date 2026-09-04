package findingsink

import (
	"io"
	"path/filepath"
	"testing"
)

// TestBenchmarkFindingSinkVerifiesOutput verifies that the benchmarked sink operations succeed cleanly.
func TestBenchmarkFindingSinkVerifiesOutput(t *testing.T) {
	sink := StdoutSink{W: io.Discard}
	findings := []Finding{
		{Key: "bench-key-1", Title: "Benchmark Finding 1", Summary: "Summary 1"},
		{Key: "bench-key-2", Title: "Benchmark Finding 2", Summary: "Summary 2"},
	}
	rep, err := sink.Emit(findings, EmitOptions{})
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	if rep.Planned != 2 {
		t.Fatalf("planned = %d, want 2", rep.Planned)
	}
}

// BenchmarkFindingSink measures throughput of finding planning and local-db upserts.
func BenchmarkFindingSink(b *testing.B) {
	findings := []Finding{
		{Key: "bench-k1", Title: "benchmark debt 1", Summary: "action 1"},
		{Key: "bench-k2", Title: "benchmark debt 2", Summary: "action 2"},
		{Key: "bench-k3", Title: "benchmark debt 3", Summary: "action 3"},
	}

	b.Run("StdoutSink_Plan", func(b *testing.B) {
		sink := StdoutSink{W: io.Discard}
		opt := EmitOptions{}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := sink.Emit(findings, opt); err != nil {
				b.Fatalf("Emit: %v", err)
			}
		}
	})

	b.Run("LocalDBSink_DryRunPlan", func(b *testing.B) {
		dir := b.TempDir()
		sink := LocalDBSink{}
		opt := EmitOptions{Live: false, Dir: dir}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := sink.Emit(findings, opt); err != nil {
				b.Fatalf("Emit: %v", err)
			}
		}
	})

	b.Run("LocalDBSink_LiveUpsert", func(b *testing.B) {
		dir := b.TempDir()
		sink := LocalDBSink{File: filepath.Join(dir, "bench.jsonl")}
		opt := EmitOptions{Live: true, Dir: dir}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := sink.Emit(findings, opt); err != nil {
				b.Fatalf("Emit: %v", err)
			}
		}
	})
}
