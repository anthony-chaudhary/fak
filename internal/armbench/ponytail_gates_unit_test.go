package armbench

import "testing"

func TestPonytailGateUnknownFailsClosed(t *testing.T) {
	ok, reason := runPinnedGate(t.Context(), "", GateScenario{ID: "up.unknown"}, "anything")
	if ok || reason != "unknown gate" {
		t.Fatalf("got %v %q", ok, reason)
	}
}
func TestPonytailGateSummaryDoesNotHideCategoryRegression(t *testing.T) {
	sc := []GateScenario{{ID: "b", Category: "behavior", RequiresProvider: true}, {ID: "c", Category: "correctness", RequiresProvider: true}}
	cells := []GateCell{{ScenarioID: "b", Arm: "ponytail", Category: "behavior", Pass: true}, {ScenarioID: "c", Arm: "ponytail", Category: "correctness", Pass: false}, {ScenarioID: "r", Arm: "deterministic", Category: "correctness-regression", Pass: true}}
	sums, overall := summarizeGates(sc, cells, true, 1)
	if overall {
		t.Fatal("aggregate concealed regression")
	}
	found := false
	for _, s := range sums {
		if s.Arm == "ponytail" && s.Category == "correctness" {
			found = true
			if s.GatePass || s.Failed != 1 {
				t.Fatalf("bad %+v", s)
			}
		}
	}
	if !found {
		t.Fatal("missing category")
	}
}
func TestExtensionFixturesAreSeparateDetectorPasses(t *testing.T) {
	for _, c := range extensionFixtureCells() {
		if c.Category != "extension" || !c.Pass || c.Arm != "detector" {
			t.Fatalf("bad extension %+v", c)
		}
	}
}
