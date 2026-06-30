package milestonereport

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
)

// climbCorpus builds a payload corpus from an explicit cell list so the two
// ratcheted KPIs (matured_cells / milestone_progress) are deterministic.
func climbCorpus(t *testing.T, cells []covmatrix.Cell) map[string]any {
	t.Helper()
	m := InterpretMaturity(cells)
	p := BuildScorecard(m, fixtureEpics())
	if _, ok := p.Corpus["matured_cells"]; !ok {
		t.Fatalf("payload corpus missing matured_cells (KPI not surfaced)")
	}
	if _, ok := p.Corpus["milestone_progress"]; !ok {
		t.Fatalf("payload corpus missing milestone_progress (KPI not surfaced)")
	}
	return p.Corpus
}

// climbHigh has 2 matured cells (M4+); climbLow has 1 — a rung regression.
func climbHigh() []covmatrix.Cell {
	return []covmatrix.Cell{
		{Family: "a", Backend: "cpu", Support: covmatrix.Supported},   // M4 matured
		{Family: "b", Backend: "cpu", Support: covmatrix.Supported},   // M4 matured
		{Family: "c", Backend: "cpu", Support: covmatrix.ProofPathOnly}, // M3
	}
}

func climbLow() []covmatrix.Cell {
	return []covmatrix.Cell{
		{Family: "a", Backend: "cpu", Support: covmatrix.Supported},   // M4 matured
		{Family: "b", Backend: "cpu", Support: covmatrix.Fenced},      // M1 — DROPPED a rung
		{Family: "c", Backend: "cpu", Support: covmatrix.ProofPathOnly}, // M3
	}
}

// TestRatchetFirstRunIsCleanAndPins: with no baseline pinned, the gate is a clean
// first run, and PinFrom captures the current KPIs.
func TestRatchetFirstRunIsCleanAndPins(t *testing.T) {
	corpus := climbCorpus(t, climbHigh())

	v, err := CheckRatchet(corpus, nil, false)
	if err != nil {
		t.Fatalf("CheckRatchet: %v", err)
	}
	if !v.OK || !v.FirstRun {
		t.Fatalf("first run want OK+FirstRun, got OK=%v FirstRun=%v", v.OK, v.FirstRun)
	}

	pinned, err := PinFrom(corpus, "deadbeef")
	if err != nil {
		t.Fatalf("PinFrom: %v", err)
	}
	if pinned.MaturedCells != 2 {
		t.Fatalf("pinned matured_cells = %d, want 2", pinned.MaturedCells)
	}
	if pinned.Schema != BaselineSchema {
		t.Fatalf("pinned schema = %q, want %q", pinned.Schema, BaselineSchema)
	}
}

// TestRatchetRegressionRedsGate: a fixture where a cell drops a rung (matured 2 -> 1)
// fails the gate, and the regression names matured_cells.
func TestRatchetRegressionRedsGate(t *testing.T) {
	base, err := PinFrom(climbCorpus(t, climbHigh()), "base")
	if err != nil {
		t.Fatalf("PinFrom base: %v", err)
	}

	cur := climbCorpus(t, climbLow())
	v, err := CheckRatchet(cur, base, false)
	if err != nil {
		t.Fatalf("CheckRatchet: %v", err)
	}
	if v.OK {
		t.Fatalf("regression must red the gate, got OK=true (regressions=%v)", v.Regressions)
	}
	if len(v.Regressions) == 0 {
		t.Fatalf("expected a named regression, got none")
	}
	foundMatured := false
	for _, r := range v.Regressions {
		if len(r) >= len("matured_cells") && r[:len("matured_cells")] == "matured_cells" {
			foundMatured = true
		}
	}
	if !foundMatured {
		t.Fatalf("expected a matured_cells regression, got %v", v.Regressions)
	}
}

// TestRatchetOverrideHonored: the same regression passes when --allow-regress is set,
// but the drop is still recorded in Regressions (honored, not hidden).
func TestRatchetOverrideHonored(t *testing.T) {
	base, _ := PinFrom(climbCorpus(t, climbHigh()), "base")
	cur := climbCorpus(t, climbLow())

	v, err := CheckRatchet(cur, base, true)
	if err != nil {
		t.Fatalf("CheckRatchet: %v", err)
	}
	if !v.OK {
		t.Fatalf("override must keep the gate green, got OK=false")
	}
	if len(v.Regressions) == 0 {
		t.Fatalf("override must still RECORD the regression, got none")
	}
}

// TestRatchetImprovementPassesAndRepins: climbing from 1 -> 2 matured cells passes,
// flags Improved, and PinFrom re-pins the higher floor.
func TestRatchetImprovementPassesAndRepins(t *testing.T) {
	base, _ := PinFrom(climbCorpus(t, climbLow()), "base") // 1 matured
	cur := climbCorpus(t, climbHigh())                     // 2 matured

	v, err := CheckRatchet(cur, base, false)
	if err != nil {
		t.Fatalf("CheckRatchet: %v", err)
	}
	if !v.OK {
		t.Fatalf("improvement must pass, got OK=false (%v)", v.Regressions)
	}
	if !v.Improved {
		t.Fatalf("improvement must set Improved=true")
	}

	repin, _ := PinFrom(cur, "newer")
	if repin.MaturedCells <= base.MaturedCells {
		t.Fatalf("re-pin must ratchet UP: %d -> %d", base.MaturedCells, repin.MaturedCells)
	}
}

// TestRatchetProgressRegressionRedsGate: holding matured_cells constant but lowering
// the pinned progress floor reds the gate on milestone_progress alone.
func TestRatchetProgressRegressionRedsGate(t *testing.T) {
	cur := climbCorpus(t, climbHigh())
	curKPIs, err := climbFromPayload(cur)
	if err != nil {
		t.Fatalf("climbFromPayload: %v", err)
	}
	// Pin progress ABOVE the current value (matured equal), so current regresses on it.
	base := &Baseline{
		Schema:       BaselineSchema,
		MaturedCells: curKPIs.MaturedCells,
		ProgressPct:  curKPIs.ProgressPct + 5.0,
	}
	v, err := CheckRatchet(cur, base, false)
	if err != nil {
		t.Fatalf("CheckRatchet: %v", err)
	}
	if v.OK {
		t.Fatalf("progress regression must red the gate, got OK=true")
	}
}

// TestRatchetMissingKPIIsAWiringError: a corpus that does not carry the climb KPIs is
// a wiring bug, not a silent pass.
func TestRatchetMissingKPIIsAWiringError(t *testing.T) {
	if _, err := CheckRatchet(map[string]any{"unrelated": 1}, &Baseline{}, false); err == nil {
		t.Fatalf("expected an error for a corpus missing the climb KPIs")
	}
}

// TestRatchetScorecardFoldsControlPaneDebt: the control-pane payload the python pane
// reads carries climb_ratchet_debt 1 on an un-overridden regression and 0 when held —
// so tools/scorecard_control_pane.py's find_int fold reds CI exactly on a climb drop.
func TestRatchetScorecardFoldsControlPaneDebt(t *testing.T) {
	base, _ := PinFrom(climbCorpus(t, climbHigh()), "base") // 2 matured

	// Regression: current is climbLow (1 matured) vs the 2-matured pin.
	reg, err := CheckRatchet(climbCorpus(t, climbLow()), base, false)
	if err != nil {
		t.Fatalf("CheckRatchet regression: %v", err)
	}
	regPayload := BuildRatchetScorecard(reg, false)
	if d, ok := ratchetInt(regPayload.Corpus[RatchetDebtKey]); !ok || d != 1 {
		t.Fatalf("regressed climb_ratchet_debt = %v (ok=%v), want 1", regPayload.Corpus[RatchetDebtKey], ok)
	}
	if regPayload.OK {
		t.Fatalf("regressed ratchet payload must be OK=false")
	}

	// Held: current still at the pinned floor -> debt 0, OK.
	held, _ := CheckRatchet(climbCorpus(t, climbHigh()), base, false)
	heldPayload := BuildRatchetScorecard(held, false)
	if d, _ := ratchetInt(heldPayload.Corpus[RatchetDebtKey]); d != 0 {
		t.Fatalf("held climb_ratchet_debt = %v, want 0", heldPayload.Corpus[RatchetDebtKey])
	}
	if !heldPayload.OK {
		t.Fatalf("held ratchet payload must be OK=true")
	}

	// Override: a regression with override set folds debt 0 (the gate stays green).
	ovr, _ := CheckRatchet(climbCorpus(t, climbLow()), base, true)
	ovrPayload := BuildRatchetScorecard(ovr, true)
	if d, _ := ratchetInt(ovrPayload.Corpus[RatchetDebtKey]); d != 0 {
		t.Fatalf("overridden climb_ratchet_debt = %v, want 0", ovrPayload.Corpus[RatchetDebtKey])
	}
}

// TestBaselineRoundTrips: WriteBaseline + LoadBaseline preserve the pinned floor.
func TestBaselineRoundTrips(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	want := &Baseline{Schema: BaselineSchema, MaturedCells: 7, ProgressPct: 42.5, Commit: "abc"}
	if err := WriteBaseline(path, want); err != nil {
		t.Fatalf("WriteBaseline: %v", err)
	}
	got, err := LoadBaseline(path)
	if err != nil {
		t.Fatalf("LoadBaseline: %v", err)
	}
	if got.MaturedCells != 7 || got.ProgressPct != 42.5 || got.Commit != "abc" {
		t.Fatalf("round-trip mismatch: %+v", got)
	}
}
