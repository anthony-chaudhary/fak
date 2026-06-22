package turnbench

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// The generic mask must reproduce the existing vDSO ablation result: with the vDSO lever
// flipped via the SAME generic lever-flip loop (NOT a hard-coded SetVDSO path in the
// attribution), every baseline VDSOHit disappears — so the vdso_hits delta equals the
// baseline VDSOHits, exactly the net(on) − net(off) == VDSOHits relation
// TestRun_VDSOAblationIsARealPathSwap asserts. This is the acceptance gate that the vDSO
// path is not special-cased: it is one named lever among L.
func TestRunLeverFlip_VDSOLeverReproducesPathSwap(t *testing.T) {
	rep, err := RunLeverFlip(context.Background(), mustLoad(t, airlineTrace))
	if err != nil {
		t.Fatalf("RunLeverFlip: %v", err)
	}
	if rep.Baseline.VDSOHits != wantVDSOTotal {
		t.Fatalf("baseline VDSOHits = %d, want %d", rep.Baseline.VDSOHits, wantVDSOTotal)
	}

	vdso, ok := rep.LeverByName(VDSOLever)
	if !ok {
		t.Fatal("lever-flip report has no vdso lever — the generic sweep must include it")
	}
	if !vdso.Present {
		t.Fatal("vdso lever reported not present")
	}
	// Masked (vDSO off) VDSOHits must be 0 — the fast path served nothing.
	if vdso.Masked.VDSOHits != 0 {
		t.Errorf("vdso-off VDSOHits = %d, want 0", vdso.Masked.VDSOHits)
	}
	// The delta is negative and its magnitude is exactly the baseline VDSOHits — the
	// generic mask reproduces net(on) − net(off) == VDSOHits.
	if got := -vdso.Delta.VDSOHits; got != rep.Baseline.VDSOHits {
		t.Errorf("vdso lever |vdso_hits delta| = %d, want baseline VDSOHits = %d", got, rep.Baseline.VDSOHits)
	}
	if !vdso.Changed {
		t.Error("vdso lever changed_outcome = false, want true")
	}
}

// Cross-check the generic lever-flip against the legacy special-case path swap: the vdso
// lever's delta (generic mask) must equal the Run report's net(on) − net(off) turns-saved
// difference, which equals VDSOHits. Same number, two independent mechanisms — proves the
// generic mask is not a reimplementation that drifted.
func TestRunLeverFlip_VDSOLeverMatchesLegacyAblation(t *testing.T) {
	ctx := context.Background()
	tr := mustLoad(t, airlineTrace)

	legacy, err := Run(ctx, tr, DefaultCostModel())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	legacyDelta := legacy.Net.TurnsSaved - legacy.VDSOOffNet.TurnsSaved // == VDSOHits

	flip, err := RunLeverFlip(ctx, tr)
	if err != nil {
		t.Fatalf("RunLeverFlip: %v", err)
	}
	vdso, _ := flip.LeverByName(VDSOLever)
	if got := int(-vdso.Delta.VDSOHits); got != legacyDelta {
		t.Errorf("generic vdso lever delta = %d, legacy net(on)-net(off) = %d (== VDSOHits %d) — must match",
			got, legacyDelta, legacy.Counters.VDSOHits)
	}
	if int(-vdso.Delta.VDSOHits) != int(legacy.Counters.VDSOHits) {
		t.Errorf("generic vdso lever delta %d != legacy VDSOHits %d", -vdso.Delta.VDSOHits, legacy.Counters.VDSOHits)
	}
}

// A CHAIN rung (not the fast-path) must ablate cleanly too: masking "grammar" removes
// exactly the grammar TRANSFORMs and nothing else — Transforms drops to 0 while every
// other counter is unchanged. This proves the generic mask works on a real adjudicator
// rung, so the vDSO reproduction above is not a one-off fast-path trick.
func TestRunLeverFlip_GrammarChainRungAblatesCleanly(t *testing.T) {
	rep, err := RunLeverFlip(context.Background(), mustLoad(t, airlineTrace))
	if err != nil {
		t.Fatalf("RunLeverFlip: %v", err)
	}
	if rep.Baseline.Transforms != wantGrammar {
		t.Fatalf("baseline Transforms = %d, want %d", rep.Baseline.Transforms, wantGrammar)
	}
	g, ok := rep.LeverByName("grammar")
	if !ok {
		t.Fatal("no grammar lever in the sweep")
	}
	if g.Realization != "chain-mask" {
		t.Errorf("grammar realization = %q, want chain-mask (a real chain rung)", g.Realization)
	}
	if !g.Present {
		t.Fatal("grammar lever reported not present")
	}
	if g.Masked.Transforms != 0 {
		t.Errorf("grammar-off Transforms = %d, want 0 (the grammar repairs are gone)", g.Masked.Transforms)
	}
	if g.Delta.Transforms != -int64(wantGrammar) {
		t.Errorf("grammar lever transforms delta = %d, want %d", g.Delta.Transforms, -wantGrammar)
	}
	// Removing grammar must not touch the OTHER outcome counters on this trace.
	if g.Delta.VDSOHits != 0 || g.Delta.Quarantines != 0 || g.Delta.Denies != 0 {
		t.Errorf("grammar lever leaked into other counters: vdso %d quar %d deny %d",
			g.Delta.VDSOHits, g.Delta.Quarantines, g.Delta.Denies)
	}
}

// The attribution table is the deliverable: a per-rung row for every registered chain
// rung plus vdso, each with the live-counter delta and a JSON() rendering that mirrors the
// other turnbench reports. The table must cover every named chain rung (no rung silently
// missing) and round-trip through JSON.
func TestRunLeverFlip_AttributionTableShapeAndJSON(t *testing.T) {
	ctx := context.Background()
	tr := mustLoad(t, airlineTrace)
	rep, err := RunLeverFlip(ctx, tr)
	if err != nil {
		t.Fatalf("RunLeverFlip: %v", err)
	}

	// Every named chain rung must appear as a lever row, plus the vdso lever.
	wantNames := map[string]bool{VDSOLever: false}
	for _, n := range abi.RungNames(abi.Adjudicators()) {
		if n != "" {
			wantNames[n] = false
		}
	}
	for _, l := range rep.Levers {
		if _, ok := wantNames[l.Lever]; ok {
			wantNames[l.Lever] = true
		}
		if l.Witness == "" {
			t.Errorf("lever %q has an empty witness", l.Lever)
		}
	}
	for name, covered := range wantNames {
		if !covered {
			t.Errorf("attribution table missing lever %q", name)
		}
	}

	// L == levers replayed == attribution-per-run multiple (the honest L× coverage).
	if rep.LeversReplayed != len(rep.Levers) || rep.AttributionPerRun != len(rep.Levers) {
		t.Errorf("L bookkeeping: replayed=%d per_run=%d rows=%d (must all equal)",
			rep.LeversReplayed, rep.AttributionPerRun, len(rep.Levers))
	}
	if rep.AttributionPerRun < 2 {
		t.Errorf("attribution_per_run = %d, want >= 2 (a real sweep)", rep.AttributionPerRun)
	}

	// JSON() round-trips.
	var back LeverFlipReport
	if err := json.Unmarshal(rep.JSON(), &back); err != nil {
		t.Fatalf("report JSON does not round-trip: %v", err)
	}
	if back.Calls != rep.Calls || len(back.Levers) != len(rep.Levers) {
		t.Errorf("JSON round-trip lost data: calls %d->%d levers %d->%d",
			rep.Calls, back.Calls, len(rep.Levers), len(back.Levers))
	}
	if back.Provenance.AppVersion == "" {
		t.Error("report provenance app_version is empty")
	}
}

// An explicitly-named lever set is honored — and a lever that names no registered rung is
// reported Present=false with an all-zero delta (requested-but-absent, never silently
// dropped). This also pins the generic naming: "grammar" and "vdso" are addressable by
// the SAME name interface, no per-lever code path.
func TestRunLeverFlip_ExplicitLeversAndAbsentLever(t *testing.T) {
	rep, err := RunLeverFlip(context.Background(), mustLoad(t, airlineTrace),
		"grammar", VDSOLever, "no-such-rung")
	if err != nil {
		t.Fatalf("RunLeverFlip: %v", err)
	}
	if len(rep.Levers) != 3 {
		t.Fatalf("explicit levers = %d rows, want 3", len(rep.Levers))
	}
	absent, ok := rep.LeverByName("no-such-rung")
	if !ok {
		t.Fatal("absent lever row missing")
	}
	if absent.Present {
		t.Error("no-such-rung reported Present=true, want false")
	}
	if absent.Delta.any() {
		t.Errorf("absent lever has a non-zero delta: %+v", absent.Delta)
	}
}

// The driver must reject an empty trace rather than produce a hollow report.
func TestRunLeverFlip_EmptyTraceErrors(t *testing.T) {
	if _, err := RunLeverFlip(context.Background(), &Trace{SliceID: "empty"}); err == nil {
		t.Fatal("RunLeverFlip on an empty trace should error")
	}
}
