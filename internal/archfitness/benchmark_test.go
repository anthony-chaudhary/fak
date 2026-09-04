package archfitness

import "testing"

// BenchmarkArchFitnessEvaluate measures architecture fitness evaluation and hard debt aggregation throughput.
func BenchmarkArchFitnessEvaluate(b *testing.B) {
	in := Input{
		ForbiddenImports:           []Finding{f("a.go", "Import", "lateral import")},
		FrozenSeamChurn:            []Finding{f("b.go", "Submit", "frozen seam bypass")},
		FamilySwitches:             []Finding{f("c.go", "switchModel", "family switch outside leaf")},
		CrossPlaneAmplification:    []Finding{f("d.go", "Feature", "touch set exceeds budget")},
		BespokeBranches:            []Finding{f("e.go", "ifQwen", "descriptor bypass")},
		AmbiguousResources:         []Finding{f("f.go", "State", "missing owner/lifetime")},
		MissingCompositionFixtures: []Finding{f("g.go", "Fixture", "forbidden interaction untested")},
		MissingCausalProjection:    []Finding{f("h.go", "Node", "no receipt projection")},
		UnversionedSchemas:         []Finding{f("i.go", "Schema", "migration absent")},
		DynamicHotPath:             []Finding{f("j.go", "Resolve", "dynamic lookup after resolve")},
		PrivacyCardinality:         []Finding{f("k.go", "Labels", "causal ID metric label")},
		StaleExceptions:            []Finding{f("l.go", "Waiver", "missing owner/expiry/issue")},
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r := Analyze(in)
		if r.HardDebt != 12 {
			b.Fatalf("unexpected hard debt: %d", r.HardDebt)
		}
	}
}

// BenchmarkArchFitnessClean measures architecture fitness evaluation throughput with zero violations.
func BenchmarkArchFitnessClean(b *testing.B) {
	in := Input{}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r := Analyze(in)
		if r.HardDebt != 0 {
			b.Fatalf("unexpected hard debt: %d", r.HardDebt)
		}
	}
}
