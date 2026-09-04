package eveimport_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/eveimport"
)

// BenchmarkEVEImportParse benchmarks end-to-end parsing of real Eve session stream NDJSON data.
func BenchmarkEVEImportParse(b *testing.B) {
	data, err := os.ReadFile(filepath.Join("testdata", "eve-session-stream.ndjson"))
	if err != nil {
		b.Fatalf("failed reading benchmark fixture: %v", err)
	}
	opts := eveimport.Options{}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		run := eveimport.ImportNDJSON("benchmark.ndjson", data, opts)
		if run.Root == nil {
			b.Fatal("expected non-nil root session in benchmark")
		}
	}
}

// BenchmarkEVEImportOTelSpans benchmarks parsing and tree reconstruction from OpenTelemetry spans.
func BenchmarkEVEImportOTelSpans(b *testing.B) {
	data, err := os.ReadFile(filepath.Join("testdata", "eve-otel-spans.json"))
	if err != nil {
		b.Fatalf("failed reading benchmark fixture: %v", err)
	}
	opts := eveimport.Options{}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		run := eveimport.ImportOTelSpans("benchmark.json", data, opts)
		if run.Root == nil {
			b.Fatal("expected non-nil root session in benchmark")
		}
	}
}

// BenchmarkEVEImportJoinLedger benchmarks session projection into cost ledger rows.
func BenchmarkEVEImportJoinLedger(b *testing.B) {
	data, err := os.ReadFile(filepath.Join("testdata", "eve-session-stream.ndjson"))
	if err != nil {
		b.Fatalf("failed reading benchmark fixture: %v", err)
	}
	run := eveimport.ImportNDJSON("benchmark.ndjson", data, eveimport.Options{})
	if run.Root == nil {
		b.Fatal("expected non-nil root session in benchmark setup")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows := eveimport.JoinLedger(run)
		if len(rows) == 0 {
			b.Fatal("expected non-empty ledger rows in benchmark")
		}
	}
}
