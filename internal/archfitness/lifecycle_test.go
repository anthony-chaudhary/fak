package archfitness

import (
	"testing"
)

// Invariant: Architectural fitness analysis must accurately aggregate hard debt across all architectural dimensions.
// Guard: Ratchet prevents regressions in hard debt counts.

func TestArchFitnessLifecycle(t *testing.T) {
	t.Parallel()

	clean := Analyze(Input{})
	if clean.HardDebt != 0 || clean.Score != 100 {
		t.Fatalf("expected clean report with 0 hard debt, got: %+v", clean)
	}

	finding := Analyze(Input{
		ForbiddenImports: []Finding{f("pkg.go", "Sym", "illegal import")},
	})
	if finding.HardDebt != 1 {
		t.Fatalf("expected 1 hard debt, got: %d", finding.HardDebt)
	}
	if Ratchet(Report{HardDebt: 0}, finding) {
		t.Fatal("ratchet must fail when hard debt increases")
	}
}
