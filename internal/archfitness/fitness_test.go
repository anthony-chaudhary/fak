package archfitness

import (
	"bytes"
	"testing"
)

func f(file, symbol, reason string) Finding {
	return Finding{File: file, Symbol: symbol, Reason: reason}
}
func TestEveryHardDimensionAndStableOutput(t *testing.T) {
	in := Input{ForbiddenImports: []Finding{f("a.go", "Import", "lateral import")}, FrozenSeamChurn: []Finding{f("b.go", "Submit", "frozen seam bypass")}, FamilySwitches: []Finding{f("c.go", "switchModel", "family switch outside leaf")}, CrossPlaneAmplification: []Finding{f("d.go", "Feature", "touch set exceeds budget")}, BespokeBranches: []Finding{f("e.go", "ifQwen", "descriptor bypass")}, AmbiguousResources: []Finding{f("f.go", "State", "missing owner/lifetime")}, MissingCompositionFixtures: []Finding{f("g.go", "Fixture", "forbidden interaction untested")}, MissingCausalProjection: []Finding{f("h.go", "Node", "no receipt projection")}, UnversionedSchemas: []Finding{f("i.go", "Schema", "migration absent")}, DynamicHotPath: []Finding{f("j.go", "Resolve", "dynamic lookup after resolve")}, PrivacyCardinality: []Finding{f("k.go", "Labels", "causal ID metric label")}, StaleExceptions: []Finding{f("l.go", "Waiver", "missing owner/expiry/issue")}}
	r := Analyze(in)
	if r.HardDebt != 12 || r.Score != 40 || len(r.Dimensions) != 12 { //boundarylint:ignore CHANGE_DETECTOR_TEST the fixture supplies one finding for each of the 12 declared hard architecture dimensions
		t.Fatalf("report=%+v", r)
	}
	a, _ := JSON(r)
	b, _ := JSON(Analyze(in))
	if !bytes.Equal(a, b) {
		t.Fatal("output not byte stable")
	}
	if WorkList(r) == "" {
		t.Fatal("missing human work list")
	}
}
func TestFalseGreenAndMetricGamingStayDebt(t *testing.T) {
	falseGreen := Analyze(Input{MissingCompositionFixtures: []Finding{f("fixture.go", "AllFlagsPresent", "forbidden combination passes individually")}})
	movedSwitch := Analyze(Input{FamilySwitches: []Finding{f("renamed.go", "selectVariation", "family branch moved")}})
	if falseGreen.HardDebt != 1 || movedSwitch.HardDebt != 1 {
		t.Fatal("hard debt hidden")
	}
	if Ratchet(Report{HardDebt: 0}, falseGreen) {
		t.Fatal("new debt passed ratchet")
	}
	clean := Analyze(Input{})
	if clean.Score != 100 || !Ratchet(falseGreen, clean) {
		t.Fatal("retiring debt did not improve")
	}
}

func BenchmarkAnalyze(b *testing.B) {
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
	for i := 0; i < b.N; i++ {
		_ = Analyze(in)
	}
}
