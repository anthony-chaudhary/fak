package commitsubject

import "testing"

func TestBenchmarkSubjectValidation(t *testing.T) {
	cov := Fold([]string{"fix(tools): add a noun-form rung (fak tools)"})
	if cov.Total != 1 || cov.Gradeable != 1 {
		t.Fatalf("unexpected coverage: %+v", cov)
	}
}

func BenchmarkCommitSubject(b *testing.B) {
	subjects := []string{
		"fix(tools): add a noun-form rung to the closure auditor (fak tools)",
		"feat(model): MiniMax-M3 witness oracle (fak model)",
		"Merge branch 'main' of origin",
		"v0.32.0: cut the release",
		"ci: add the dispatch tool-test cluster to the gate (fak ci)",
		"refactor(gateway): decouple session state from transport (fak gateway)",
		"docs: update architecture overview",
		"invalid subject lacking conventional structure",
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		cov := Fold(subjects)
		if cov.Total == 0 {
			b.Fatal("unexpected zero total in benchmark")
		}
	}
}
