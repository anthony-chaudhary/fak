package studyprio

import (
	"path/filepath"
	"testing"
)

// BenchmarkBuild measures deterministically building candidates and queue from a source ledger.
func BenchmarkBuild(b *testing.B) {
	source := filepath.Join("..", "..", "docs", "research", "vllm-fak-join-2026-08-27", "ledger.json")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ledger, summary, err := Build(BuildOptions{SourceLedgerPath: source})
		if err != nil {
			b.Fatalf("Build failed: %v", err)
		}
		if len(ledger.Candidates) == 0 || summary.CandidateCount == 0 {
			b.Fatal("Build returned empty candidates")
		}
	}
}

// BenchmarkValidate measures schema and invariant validation on a study priority ledger.
func BenchmarkValidate(b *testing.B) {
	source := filepath.Join("..", "..", "docs", "research", "vllm-fak-join-2026-08-27", "ledger.json")
	ledger, _, err := Build(BuildOptions{SourceLedgerPath: source})
	if err != nil {
		b.Fatalf("setup Build failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Validate(ledger); err != nil {
			b.Fatalf("Validate failed: %v", err)
		}
	}
}

// BenchmarkScoreDimensions measures weighted scoring of candidate dimensions against the rubric.
func BenchmarkScoreDimensions(b *testing.B) {
	dims := Dimensions{
		ProductCentrality:          4,
		FakNativeQwen38Impact:      5,
		EndToEndValue:              4,
		EvidenceStrength:           3,
		Recurrence:                 2,
		DependencyUnlock:           3,
		ImplementationCost:         1,
		HardwareWitnessCost:        1,
		CompatibilityRisk:          1,
		DuplicationConflictPenalty: 0,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := score(dims)
		if s == 0 {
			b.Fatal("unexpected zero score")
		}
	}
}
