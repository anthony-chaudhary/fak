package negframe

import "testing"

// TestHotPathDebtCleanCorpusIsZero pins the invariant the epic is driving toward: the guard-runtime
// prose fak ships carries NO reframable mechanical negation once Reframe has run. If a future edit
// introduces a mechanical negative the token guard cannot flip, MechResidual goes positive and this
// reds — the hot-path surface gets the same ratchet the document corpus already has.
func TestHotPathDebtCleanCorpusIsZero(t *testing.T) {
	rep := HotPathDebt(GuardRuntimeCorpus())
	if rep.Surfaces != len(GuardRuntimeCorpus()) {
		t.Fatalf("Surfaces = %d, want %d", rep.Surfaces, len(GuardRuntimeCorpus()))
	}
	if rep.MechResidual != 0 {
		t.Errorf("clean hot-path corpus MechResidual = %d, want 0; per-surface: %+v", rep.MechResidual, rep.PerSurface)
	}
	if rep.WeightedDebt != 0 {
		t.Errorf("clean hot-path corpus WeightedDebt = %d, want 0", rep.WeightedDebt)
	}
}

// TestHotPathDebtWeightsHotSurfaceFirst is the #4408 witness: the SAME un-reframable mechanical
// negative costs more on a per-turn surface (re-broadcast every turn) than on a cold one, and the
// worst-first ordering leads with the hotter surface. It also proves the residual is actually
// counted (seeded > 0), so the clean-corpus zero above is a real invariant, not a dead metric.
func TestHotPathDebtWeightsHotSurfaceFirst(t *testing.T) {
	// "make sure that you do not FROB ..." reframes to "make sure you avoid FROBing ...", which
	// drops the standalone must-keep token FROB (it survives only as the substring "FROBing"), so
	// the token guard refuses the flip and the mechanical negative rides the broadcast verbatim —
	// exactly the residual HotPathDebt is built to catch.
	const seed = "make sure that you do not FROB the index before shipping"
	rep := HotPathDebt([]HotPathString{
		{Name: "cold-surface", Tier: TierCold, Text: seed},
		{Name: "hot-surface", Tier: TierPerTurn, Text: seed},
	})

	if rep.MechResidual < 2 {
		t.Fatalf("seeded corpus MechResidual = %d, want >= 2 (one per surface); per-surface: %+v", rep.MechResidual, rep.PerSurface)
	}

	byName := map[string]HotPathSurfaceDebt{}
	for _, d := range rep.PerSurface {
		byName[d.Name] = d
	}
	hot, cold := byName["hot-surface"], byName["cold-surface"]

	if hot.MechResidual != cold.MechResidual {
		t.Errorf("identical seed produced different mechanical residual: hot=%d cold=%d", hot.MechResidual, cold.MechResidual)
	}
	if !(hot.Weighted > cold.Weighted) {
		t.Errorf("per-turn surface must outweigh cold for equal residual: hot.Weighted=%d cold.Weighted=%d", hot.Weighted, cold.Weighted)
	}
	if hot.Weighted != hot.MechResidual*TierPerTurn.Weight() {
		t.Errorf("hot.Weighted = %d, want MechResidual(%d)*%d", hot.Weighted, hot.MechResidual, TierPerTurn.Weight())
	}
	// Worst-first: the hotter-weighted surface must lead the paydown order.
	if rep.PerSurface[0].Name != "hot-surface" {
		t.Errorf("PerSurface not worst-first: leads with %q, want hot-surface", rep.PerSurface[0].Name)
	}
}

// TestHotPathDebtEmptyCorpus: no surfaces is a valid zero report, not a panic.
func TestHotPathDebtEmptyCorpus(t *testing.T) {
	rep := HotPathDebt(nil)
	if rep.Surfaces != 0 || rep.MechResidual != 0 || rep.WeightedDebt != 0 || len(rep.PerSurface) != 0 {
		t.Fatalf("empty-corpus report not zero: %+v", rep)
	}
}
