package guardcorpus

import (
	"path/filepath"
	"testing"
)

// BenchmarkCorpusScan benchmarks the throughput of folding sequential journal
// rows into SessionRecord aggregations and redacted Example dataset rows.
func BenchmarkCorpusScan(b *testing.B) {
	meta := SessionMeta{
		TraceID:       "bench-trace-session-001",
		Agent:         "claude-code",
		HostClass:     "benchmark-runner",
		PolicyDigest:  "sha256:4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
		ChainVerified: true,
	}

	rows := readJournalFixture(b, filepath.Join("testdata", "session.journal.jsonl"))
	if len(rows) == 0 {
		rows = planted()
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rec, ex := Fold(meta, rows)
		if rec.ToolCalls == 0 || len(ex) == 0 {
			b.Fatal("unexpected empty fold result during benchmark")
		}
	}
}

// BenchmarkFoldPlanted measures pure fold performance across synthetic rows
// exercising all decision, crash, rate-limit, and honesty hole code paths.
func BenchmarkFoldPlanted(b *testing.B) {
	meta := SessionMeta{
		TraceID:       "bench-planted-session-002",
		Agent:         "codex",
		HostClass:     "synthetic-runner",
		PolicyDigest:  "sha256:8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4",
		ChainVerified: true,
	}
	rows := planted()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rec, ex := Fold(meta, rows)
		if rec.ToolCalls == 0 || len(ex) == 0 {
			b.Fatal("unexpected empty fold result during benchmark")
		}
	}
}

// TestBenchmarkCorpusScanSanity ensures the benchmarked Fold execution path
// functions accurately with expected non-empty output records.
func TestBenchmarkCorpusScanSanity(t *testing.T) {
	meta := SessionMeta{
		TraceID:       "sanity-trace",
		Agent:         "claude-code",
		HostClass:     "test",
		PolicyDigest:  "sha256:sanity",
		ChainVerified: true,
	}
	rows := readJournalFixture(t, filepath.Join("testdata", "session.journal.jsonl"))
	rec, ex := Fold(meta, rows)
	if rec.ToolCalls == 0 || len(ex) == 0 {
		t.Fatalf("expected non-empty fold result, got tool_calls=%d examples=%d", rec.ToolCalls, len(ex))
	}
}
