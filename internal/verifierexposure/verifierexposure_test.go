package verifierexposure

import "testing"

func TestFoldPinsDebtAndWorstFirstOrdering(t *testing.T) {
	gates := []Gate{
		{Name: "pinned-diff", Kind: Deterministic, CheckerBytesPinned: true, IndependentlyRemeasured: true},
		{Name: "injectable-judge", Kind: LLMJudge, FailsOpen: true, InjectableProse: true},
		{Name: "hardened-judge", Kind: LLMJudge, SchemaPinned: true, TemperatureZero: true, IndependentlyRemeasured: true},
		{Name: "self-report", Kind: SelfReport},
	}
	r := Fold(gates, nil)
	if r.VerifierExposureDebt != 2 || r.Grade != "C" {
		t.Fatalf("report debt/grade = %d/%s, want 2/C", r.VerifierExposureDebt, r.Grade)
	}
	want := []string{"injectable-judge", "self-report", "hardened-judge", "pinned-diff"}
	// The first two tie at 0.85 and therefore use stable name ordering.
	for i, name := range want {
		if r.Worklist[i].Name != name {
			t.Fatalf("worklist[%d] = %q, want %q (%+v)", i, r.Worklist[i].Name, name, r.Worklist)
		}
	}
	if r.Worklist[0].Exposure <= r.Worklist[len(r.Worklist)-1].Exposure {
		t.Fatal("gameable judge must rank above pinned deterministic gate")
	}
}

func TestFoldFailsClosedOnInventoryErrors(t *testing.T) {
	r := Fold([]Gate{{Name: "gate", Kind: Deterministic}}, []string{"missing source"})
	if r.Grade != "F" || len(r.InventoryErrors) != 1 {
		t.Fatalf("report = %+v", r)
	}
}

func TestGatherFailsClosedWhenDeclaredSignalDrifts(t *testing.T) {
	r := Gather(t.TempDir())
	if r.Grade != "F" || len(r.InventoryErrors) < len(DeclaredInventory()) {
		t.Fatalf("Gather(missing tree) = grade %s errors %d", r.Grade, len(r.InventoryErrors))
	}
}
