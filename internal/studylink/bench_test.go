package studylink

import (
	"path/filepath"
	"testing"
)

// BenchmarkReadLedger measures reading and decoding the checked study join ledger.
func BenchmarkReadLedger(b *testing.B) {
	ledgerPath := filepath.Join("..", "..", "docs", "research", "vllm-fak-join-2026-08-27", "ledger.json")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ledger, err := ReadLedger(ledgerPath)
		if err != nil {
			b.Fatalf("ReadLedger failed: %v", err)
		}
		if len(ledger.Joins) == 0 {
			b.Fatal("ReadLedger returned empty joins")
		}
	}
}

// BenchmarkSummarize measures computing summary metrics and disposition counts over a ledger.
func BenchmarkSummarize(b *testing.B) {
	ledgerPath := filepath.Join("..", "..", "docs", "research", "vllm-fak-join-2026-08-27", "ledger.json")
	ledger, err := ReadLedger(ledgerPath)
	if err != nil {
		b.Fatalf("setup ReadLedger failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		summary, err := Summarize(ledger)
		if err != nil {
			b.Fatalf("Summarize failed: %v", err)
		}
		if summary.Total == 0 {
			b.Fatal("Summarize returned zero total")
		}
	}
}

// BenchmarkRenderSummary measures serializing and rendering the summary table.
func BenchmarkRenderSummary(b *testing.B) {
	ledgerPath := filepath.Join("..", "..", "docs", "research", "vllm-fak-join-2026-08-27", "ledger.json")
	ledger, err := ReadLedger(ledgerPath)
	if err != nil {
		b.Fatalf("setup ReadLedger failed: %v", err)
	}
	summary, err := Summarize(ledger)
	if err != nil {
		b.Fatalf("setup Summarize failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := RenderSummary(summary)
		if len(out) == 0 {
			b.Fatal("RenderSummary produced empty output")
		}
	}
}

// BenchmarkValidateStructure measures structural schema validation over the study ledger.
func BenchmarkValidateStructure(b *testing.B) {
	ledgerPath := filepath.Join("..", "..", "docs", "research", "vllm-fak-join-2026-08-27", "ledger.json")
	ledger, err := ReadLedger(ledgerPath)
	if err != nil {
		b.Fatalf("setup ReadLedger failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ValidateStructure(ledger, nil, ""); err != nil {
			b.Fatalf("ValidateStructure failed: %v", err)
		}
	}
}
