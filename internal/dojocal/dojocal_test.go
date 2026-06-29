package dojocal

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

// scoreEp builds one real scored episode the way the live builders do — through
// dojo.Score with the default band — so the test never hand-asserts a calib_err
// or a verdict. claimed/realized drive the gap; floor flips IntentionalFloor on.
func scoreEp(t *testing.T, lever, metric string, claimed, realized float64, sample int, floor, lowerIsBetter bool) dojo.Episode {
	t.Helper()
	p := dojo.Prediction{
		Lever:            lever,
		Metric:           metric,
		Claimed:          claimed,
		Unit:             "ratio",
		Basis:            "test",
		LowerIsBetter:    lowerIsBetter,
		IntentionalFloor: floor,
	}
	o := dojo.Outcome{Realized: realized, Provenance: dojo.Witnessed, Source: "test", Measured: true, Sample: sample}
	return dojo.Score(lever, p, o, dojo.DefaultCalibBand())
}

// unmeasuredEp builds an UNMEASURED episode (no ground truth) for a lever.
func unmeasuredEp(t *testing.T, lever, metric string) dojo.Episode {
	t.Helper()
	p := dojo.Prediction{Lever: lever, Metric: metric, Claimed: 1.0, Unit: "ratio", Basis: "test"}
	o := dojo.Outcome{Measured: false}
	return dojo.Score(lever, p, o, dojo.DefaultCalibBand())
}

func report(eps []dojo.Episode) dojo.Report {
	return dojo.Fold(eps, dojo.FoldOpts{Workspace: "test"})
}

// TestProposeRecalsRanksWorstEstimateFirst proves the proposer is the twin of
// guardrsi.WorstBucket: a measured, mis-calibrated ESTIMATE becomes a RECALIBRATE
// candidate pointed at the corpus mean, worst calib_err first.
func TestProposeRecalsRanksWorstEstimateFirst(t *testing.T) {
	eps := []dojo.Episode{
		// big over-claim: claim 0.85, corpus mean ~0.50 -> calib_err ~0.41
		scoreEp(t, "resume-posture", "cold_write_share", 0.85, 0.40, 5, false, false),
		scoreEp(t, "resume-posture", "cold_write_share", 0.85, 0.60, 5, false, false),
		// near-calibrated estimate on another lever -> smaller gap
		scoreEp(t, "vcache-warmth", "warm_recall", 1.0, 0.95, 4, false, false),
	}
	pl := ProposeRecals(report(eps))

	if len(pl.Candidates) != 2 {
		t.Fatalf("candidates = %d, want 2 (%+v)", len(pl.Candidates), pl.Candidates)
	}
	worst := pl.Worst
	if worst.Lever != "resume-posture" || worst.Metric != "cold_write_share" {
		t.Fatalf("worst = %s/%s, want resume-posture/cold_write_share", worst.Lever, worst.Metric)
	}
	if worst.Kind != RecalibrateKind {
		t.Fatalf("worst kind = %s, want RECALIBRATE", worst.Kind)
	}
	// the proposed claim must be the measured mean (0.40+0.60)/2 = 0.50, not the old claim
	if worst.NewClaimed != 0.5 {
		t.Fatalf("new claimed = %v, want 0.5 (corpus mean)", worst.NewClaimed)
	}
	if worst.OldClaimed != 0.85 {
		t.Fatalf("old claimed = %v, want 0.85", worst.OldClaimed)
	}
	if worst.Sample != 2 {
		t.Fatalf("sample = %d, want 2", worst.Sample)
	}
	// worst-first ordering: the mis-calibrated cell sorts ahead of the near-one
	if pl.Candidates[0].CalibErr <= pl.Candidates[1].CalibErr {
		t.Fatalf("candidates not worst-first: %v then %v", pl.Candidates[0].CalibErr, pl.Candidates[1].CalibErr)
	}
	// the board rides alongside (reuses dojo.BoardFromEpisodes)
	if len(pl.Board.Rows) != 2 {
		t.Fatalf("board rows = %d, want 2", len(pl.Board.Rows))
	}
}

// TestProposeRoutesAgentArms proves Phase 3's split: a pinned projection target
// routes to REPROJECT with a declared path set, while an under-claimed saving
// routes to HARVEST rather than being auto-recalibrated.
func TestProposeRoutesAgentArms(t *testing.T) {
	eps := []dojo.Episode{
		scoreEp(t, "resume-posture", "posture_accuracy", 1.0, 0.70, 10, false, false),
		scoreEp(t, "compaction", "token_shed_ratio", 1.0, 1.40, 10, false, false),
	}
	pl := ProposeRecals(report(eps))
	byMetric := map[string]Recal{}
	for _, c := range pl.Candidates {
		byMetric[c.Metric] = c
	}
	reproject := byMetric["posture_accuracy"]
	if reproject.Kind != ReprojectKind {
		t.Fatalf("posture_accuracy kind = %s, want REPROJECT (%+v)", reproject.Kind, reproject)
	}
	if len(reproject.DeclaredPaths) == 0 {
		t.Fatalf("REPROJECT candidate must declare path allow-list: %+v", reproject)
	}
	if reproject.NewClaimed != reproject.OldClaimed {
		t.Fatalf("REPROJECT must not rewrite the claim, got old=%v new=%v", reproject.OldClaimed, reproject.NewClaimed)
	}

	harvest := byMetric["token_shed_ratio"]
	if harvest.Kind != HarvestKind {
		t.Fatalf("token_shed_ratio kind = %s, want HARVEST (%+v)", harvest.Kind, harvest)
	}
	if harvest.NewClaimed != harvest.OldClaimed {
		t.Fatalf("HARVEST must not rewrite the claim, got old=%v new=%v", harvest.OldClaimed, harvest.NewClaimed)
	}
}

// TestKeepOnEstimateGain — the positive arm: a RECALIBRATE that strictly lowers
// the folded calibrable, with enough samples and a green witness, KEEPs; and the
// keep bit survives CheckIteration.
func TestKeepOnEstimateGain(t *testing.T) {
	eps := []dojo.Episode{
		scoreEp(t, "resume-posture", "cold_write_share", 0.85, 0.40, 5, false, false),
		scoreEp(t, "resume-posture", "cold_write_share", 0.85, 0.60, 5, false, false),
		scoreEp(t, "resume-posture", "cold_write_share", 0.85, 0.50, 5, false, false),
	}
	r := report(eps)
	pl := ProposeRecals(r)
	it := RunIteration(r, pl.Worst, DefaultMinSample, map[string]any{"ok": true, "suite": "go test ./... PASS"})

	if !it.Kept {
		t.Fatalf("iteration not kept: %s", it.Reason)
	}
	if it.MeasuredDelta <= 0 {
		t.Fatalf("measured delta = %v, want strict positive drop", it.MeasuredDelta)
	}
	if it.ReplayedValue >= it.BaselineValue {
		t.Fatalf("replayed value %v not below baseline %v", it.ReplayedValue, it.BaselineValue)
	}
	if v := CheckIteration(it); len(v) != 0 {
		t.Fatalf("check violations on a legitimate keep: %v", v)
	}
}

// TestRevertOnNoGain — swapping an already-calibrated estimate to its own mean
// closes no gap, so the fold does not strictly drop and the loop REVERTs even
// with a green witness.
func TestRevertOnNoGain(t *testing.T) {
	// claim already equals the realized mean -> swap is a no-op, no strict gain
	eps := []dojo.Episode{
		scoreEp(t, "resume-posture", "cold_write_share", 0.50, 0.50, 5, false, false),
		scoreEp(t, "resume-posture", "cold_write_share", 0.50, 0.50, 5, false, false),
	}
	r := report(eps)
	pl := ProposeRecals(r)
	it := RunIteration(r, pl.Worst, DefaultMinSample, map[string]any{"ok": true})
	if it.Kept {
		t.Fatalf("iteration kept on no gain: %+v", it)
	}
	if it.MeasuredDelta > 0 {
		t.Fatalf("measured delta = %v, want no strict gain", it.MeasuredDelta)
	}
}

// TestRevertOnFloorTarget — THE HEADLINE. A floor (false_warm_rate must stay 0.0)
// that reality breaches must NEVER be kept: the proposer routes it (ROUTE_FLOOR,
// no claim swap), and even if a caller forces a recalibrate-to-the-mean claim
// swap, FoldCalibrable folds the floor by its BREACH so closing the gap RAISES the
// fold and the loop REVERTs. The auto-erasure is mechanically impossible.
func TestRevertOnFloorTarget(t *testing.T) {
	// false_warm_rate: claim 0.0 (floor, lower-is-better), reality 0.30 — a breach.
	floorEps := []dojo.Episode{
		scoreEp(t, "vcache-warmth", "false_warm_rate", 0.0, 0.30, 6, true, true),
		scoreEp(t, "vcache-warmth", "false_warm_rate", 0.0, 0.30, 6, true, true),
	}
	r := report(floorEps)
	pl := ProposeRecals(r)

	// the proposer routes the floor, never proposing a claim swap
	if pl.Worst.Kind != RouteFloor {
		t.Fatalf("floor candidate kind = %s, want ROUTE_FLOOR", pl.Worst.Kind)
	}
	if pl.Worst.NewClaimed != 0.0 {
		t.Fatalf("floor new claimed = %v, want 0.0 (floor claim never swapped)", pl.Worst.NewClaimed)
	}
	routed := RunIteration(r, pl.Worst, DefaultMinSample, map[string]any{"ok": true})
	if routed.Kept {
		t.Fatalf("routed floor candidate kept: %+v", routed)
	}

	// Now the structural guarantee: even an ADVERSARIAL forced RECALIBRATE swapping
	// the floor claim up to its empirical 0.30 must REVERT, because FoldCalibrable
	// folds the floor by its breach — closing the gap RAISES the fold.
	forced := pl.Worst
	forced.Kind = RecalibrateKind
	forced.NewClaimed = 0.30
	forced.Sample = 2
	it := RunIteration(r, forced, 1, map[string]any{"ok": true})
	if it.Kept {
		t.Fatalf("forced floor recalibration KEPT — auto-erasure not blocked: %+v", it)
	}
	if it.MeasuredDelta > 0 {
		t.Fatalf("forced floor recal delta = %v, want <= 0 (closing the gap must RAISE the fold)", it.MeasuredDelta)
	}
	// the replayed floor-breach term must not be below baseline (it rises or holds)
	if it.ReplayedFold.FloorBreachErr < it.BaselineFold.FloorBreachErr {
		t.Fatalf("floor breach term DROPPED %v -> %v on a floor recalibration — guard erased",
			it.BaselineFold.FloorBreachErr, it.ReplayedFold.FloorBreachErr)
	}
}

// TestRefuseOnUnmeasuredCorpus — a corpus with no ground truth is uncandidatable:
// the proposer surfaces a ROUTE_UNMEASURED floor and RunIteration refuses it.
func TestRefuseOnUnmeasuredCorpus(t *testing.T) {
	eps := []dojo.Episode{
		unmeasuredEp(t, "compaction", "token_shed_ratio"),
		unmeasuredEp(t, "compaction", "cache_prefix_preserved"),
	}
	r := report(eps)
	pl := ProposeRecals(r)
	if len(pl.Candidates) == 0 {
		t.Fatal("no candidate surfaced for an unmeasured corpus")
	}
	if pl.Worst.Kind != RouteUnmeasured {
		t.Fatalf("unmeasured candidate kind = %s, want ROUTE_UNMEASURED", pl.Worst.Kind)
	}
	it := RunIteration(r, pl.Worst, DefaultMinSample, map[string]any{"ok": true})
	if it.Kept {
		t.Fatalf("unmeasured corpus kept: %+v", it)
	}
	if it.BaselineFold.Measured != 0 {
		t.Fatalf("baseline measured = %d, want 0 for an unmeasured corpus", it.BaselineFold.Measured)
	}
}

// TestRevertWithoutWitness — a real estimate gain with too-few-samples or no
// witness must REVERT, mirroring guard-verdict-rsi's witness gate.
func TestRevertWithoutWitness(t *testing.T) {
	eps := []dojo.Episode{
		scoreEp(t, "resume-posture", "cold_write_share", 0.85, 0.40, 5, false, false),
		scoreEp(t, "resume-posture", "cold_write_share", 0.85, 0.60, 5, false, false),
		scoreEp(t, "resume-posture", "cold_write_share", 0.85, 0.50, 5, false, false),
	}
	r := report(eps)
	pl := ProposeRecals(r)

	noWitness := RunIteration(r, pl.Worst, DefaultMinSample, nil)
	if noWitness.Kept {
		t.Fatalf("kept without a witness: %+v", noWitness)
	}
	if noWitness.MeasuredDelta <= 0 {
		t.Fatalf("expected a real gain to exist (delta %v) — the revert must be witness-driven, not gain-driven", noWitness.MeasuredDelta)
	}

	// same gain, but a too-small sample floor also REVERTs
	tooFew := RunIteration(r, pl.Worst, 99, map[string]any{"ok": true})
	if tooFew.Kept {
		t.Fatalf("kept with sample %d below min 99: %+v", pl.Worst.Sample, tooFew)
	}
}

// TestCheckRejectsFabricatedKeptIteration — the guard's honesty gate: a kept
// iteration that cannot defend its keep bit is caught by CheckIteration.
func TestCheckRejectsFabricatedKeptIteration(t *testing.T) {
	it := Iteration{
		Schema:        IterationSchema,
		Kept:          true,
		MeasuredDelta: 0,
		Witness:       nil,
		Candidate:     Recal{Kind: RouteFloor, Sample: 0},
		BaselineFold:  dojo.FoldResult{Measured: 0},
		ReplayedFold:  dojo.FoldResult{Measured: 0},
		MinSample:     DefaultMinSample,
	}
	v := CheckIteration(it)
	if len(v) < 4 {
		t.Fatalf("violations = %v, want kind/rows/delta/sample/witness failures", v)
	}
}
