package dojo

import (
	"math"
	"testing"
)

// TestRegistryPreservesPinnedClaims is the extraction-fidelity witness: every
// (lever, metric) cell the cmd/fak builders used to inline must resolve from the
// registry with the SAME Claimed number, so moving the literals into claims.go
// changed no theory. The cmd/fak pinned-claim tests prove the builders still emit
// these; this proves the registry is their single source.
func TestRegistryPreservesPinnedClaims(t *testing.T) {
	want := []struct {
		lever, metric string
		claimed       float64
		floor         bool
		lowerIsBetter bool
	}{
		{"vdso-ablation", "engine_call_elision", 1.0, false, false},
		{"resume-posture", "posture_accuracy", 1.0, false, false},
		{"resume-posture", "cold_write_share", 0.85, false, false},
		{"resume-posture", "cross_session_warm_hit_rate", 0.0, true, false},
		{"vcache-warmth", "false_warm_rate", 0.0, true, true},
		{"vcache-warmth", "warm_recall", 1.0, false, false},
		{"compaction", "token_shed_ratio", 1.0, false, false},
		{"compaction", "cache_prefix_preserved", 1.0, false, false},
	}
	if len(Registry) != len(want) {
		t.Fatalf("registry has %d cells, want %d — a cell was added or dropped without updating the witness", len(Registry), len(want))
	}
	for _, w := range want {
		c, ok := Registry.Lookup(w.lever, w.metric)
		if !ok {
			t.Fatalf("registry missing cell %s/%s", w.lever, w.metric)
		}
		if c.Claimed != w.claimed {
			t.Errorf("%s/%s claimed = %g, want %g (extraction changed a value)", w.lever, w.metric, c.Claimed, w.claimed)
		}
		if c.IntentionalFloor != w.floor {
			t.Errorf("%s/%s IntentionalFloor = %v, want %v", w.lever, w.metric, c.IntentionalFloor, w.floor)
		}
		if c.LowerIsBetter != w.lowerIsBetter {
			t.Errorf("%s/%s LowerIsBetter = %v, want %v", w.lever, w.metric, c.LowerIsBetter, w.lowerIsBetter)
		}
		// Every cell carries a real basis — an empty basis is an extraction slip.
		if c.Basis == "" {
			t.Errorf("%s/%s has an empty basis", w.lever, w.metric)
		}
	}
}

// TestPredictFillsPredictionFromCell proves Predict copies the whole cell onto the
// Prediction (Claimed, IntentionalFloor, LowerIsBetter, Basis) plus the caller's
// unit — the builders rely on this so the floor bit and direction reach Score.
func TestPredictFillsPredictionFromCell(t *testing.T) {
	p, ok := Registry.Predict("vcache-warmth", "false_warm_rate", "fraction")
	if !ok {
		t.Fatal("Predict ok=false for a registered floor cell")
	}
	if p.Claimed != 0.0 || !p.IntentionalFloor || !p.LowerIsBetter || p.Unit != "fraction" || p.Basis == "" {
		t.Fatalf("floor cell predicted wrong: %+v", p)
	}
	est, ok := Registry.Predict("resume-posture", "cold_write_share", "fraction")
	if !ok || est.Claimed != 0.85 || est.IntentionalFloor || est.LowerIsBetter {
		t.Fatalf("estimate cell predicted wrong: %+v (ok=%v)", est, ok)
	}
}

// TestPredictUnknownCellFailsLoud proves an unregistered cell is ok=false (Predict)
// and panics (MustPredict) — a missing claim is a programming error, never a
// silent zero claim scored as a real theory.
func TestPredictUnknownCellFailsLoud(t *testing.T) {
	if _, ok := Registry.Predict("no-such-lever", "no-such-metric", "fraction"); ok {
		t.Fatal("Predict returned ok=true for an unregistered cell")
	}
	defer func() {
		if recover() == nil {
			t.Fatal("MustPredict did not panic on an unregistered cell")
		}
	}()
	Registry.MustPredict("no-such-lever", "no-such-metric", "fraction")
}

// floorEp scores one INTENTIONAL-FLOOR episode the way a builder would: a floor
// claim (0.0) with the metric's direction, against a realized value.
func floorEp(metric string, claimed, realized float64, lowerIsBetter bool) Episode {
	p := predDir(metric, claimed, lowerIsBetter)
	p.IntentionalFloor = true
	return Score("s", p, obs(realized, true), DefaultCalibBand())
}

// TestFloorRespectErrZeroWhileHeldRisesOnBreach proves the floor fold term is 0
// while the floor holds and rises with the breach — the property that flips the
// loop's incentive sign so closing a floor's gap can never look like a gain.
func TestFloorRespectErrZeroWhileHeldRisesOnBreach(t *testing.T) {
	// false_warm_rate is an UPPER-bound floor (lower-is-better, claim 0.0). Reality
	// at the floor contributes 0; reality above it (a real false-warm) contributes
	// the breach.
	held := floorEp("false_warm_rate", 0.0, 0.0, true)
	if got := FloorRespectErr(held); got != 0 {
		t.Fatalf("a held floor (realized 0.0) must contribute 0, got %g", got)
	}
	small := floorEp("false_warm_rate", 0.0, 0.05, true)
	big := floorEp("false_warm_rate", 0.0, 0.40, true)
	if FloorRespectErr(small) <= 0 {
		t.Fatalf("a breached floor must contribute > 0, got %g", FloorRespectErr(small))
	}
	if FloorRespectErr(big) <= FloorRespectErr(small) {
		t.Fatalf("a larger breach must contribute more: 0.40 gave %g, 0.05 gave %g", FloorRespectErr(big), FloorRespectErr(small))
	}
	// The breach magnitude is the absolute residual (0.05 here), capped at MaxCalibErr.
	if math.Abs(FloorRespectErr(small)-0.05) > 1e-9 {
		t.Fatalf("breach term should be the absolute residual 0.05, got %g", FloorRespectErr(small))
	}
	// A lower-bound floor (higher-is-better, claim 0.0): reality ABOVE it is the
	// safe (under-claim) side, so it holds and contributes 0 — the bimodal
	// cross-session warm-hit default.
	safe := floorEp("cross_session_warm_hit_rate", 0.0, 0.30, false)
	if got := FloorRespectErr(safe); got != 0 {
		t.Fatalf("a lower-bound floor with reality on the safe side must contribute 0, got %g", got)
	}
}

// TestFoldCalibrableExcludesFloorsFromEstimateFold is the issue's core witness:
// FoldCalibrable folds estimates by calib_err and floors only by their breach, in
// two disjoint populations, and UNMEASURED episodes never fold.
func TestFoldCalibrableExcludesFloorsFromEstimateFold(t *testing.T) {
	eps := []Episode{
		// estimate, miss: calib_err = |0.68-0.85|/0.85 = 0.20
		Score("s", pred("cold_write_share", 0.85), obs(0.68, true), DefaultCalibBand()),
		// estimate, calibrated: calib_err = |0.99-1.0|/1.0 = 0.01
		Score("s", pred("warm_recall", 1.0), obs(0.99, true), DefaultCalibBand()),
		// floor, held: contributes 0 to the floor population, NOTHING to the estimate fold
		floorEp("false_warm_rate", 0.0, 0.0, true),
		// unmeasured: never folds
		Score("s", pred("token_shed_ratio", 1.0), obs(0, false), DefaultCalibBand()),
	}
	fr := FoldCalibrable(eps)
	if fr.EstimateCount != 2 {
		t.Fatalf("estimate count = %d, want 2 (floor and unmeasured excluded)", fr.EstimateCount)
	}
	if fr.FloorCount != 1 {
		t.Fatalf("floor count = %d, want 1", fr.FloorCount)
	}
	if fr.Measured != 3 {
		t.Fatalf("measured = %d, want 3 (the unmeasured episode never folds)", fr.Measured)
	}
	// estimate mean = (0.20 + 0.01)/2 = 0.105
	if math.Abs(fr.EstimateMeanCalibErr-0.105) > 1e-9 {
		t.Fatalf("estimate mean calib_err = %g, want 0.105", fr.EstimateMeanCalibErr)
	}
	if fr.FloorBreachErr != 0 {
		t.Fatalf("a held floor must leave FloorBreachErr 0, got %g", fr.FloorBreachErr)
	}
	if math.Abs(fr.Value-0.105) > 1e-9 {
		t.Fatalf("value = estimate mean + floor breach = 0.105, got %g", fr.Value)
	}
}

// TestFloorCalibGapIsNotFoldableEstimateGapIs is the anti-gaming witness at the
// FOLD layer — the asymmetry the whole design rests on. A genuine ESTIMATE whose
// claim sits far from reality folds its full calib_err, so re-pointing the claim
// at reality LOWERS FoldCalibrable (the admissible RECALIBRATE win). A FLOOR whose
// claim sits far from reality folds NOT its calib_err but only its breach, so the
// large calibration gap a "recalibrate the floor to its empirical rate" rewrite
// would close is INVISIBLE to the fold — there is no content-free gain to harvest
// by erasing the guard. (The proposer additionally refuses to rewrite a floor cell
// at all — Phase 1's IntentionalFloor routing — but the fold already denies the
// incentive.)
func TestFloorCalibGapIsNotFoldableEstimateGapIs(t *testing.T) {
	const claim, realized = 0.0, 0.20 // same numbers, scored two ways

	// As an ESTIMATE (claim 0.0 vs reality 0.20): calib_err is the absolute residual
	// 0.20, fully folded — a real, closeable gap.
	estFold := FoldCalibrable([]Episode{
		Score("s", predDir("m", claim, true), obs(realized, true), DefaultCalibBand()),
	})
	if estFold.EstimateCount != 1 || math.Abs(estFold.Value-0.20) > 1e-9 {
		t.Fatalf("estimate 0.0-vs-0.20 should fold calib_err 0.20, got %+v", estFold)
	}

	// As a FLOOR (the same claim 0.0 vs reality 0.20): the breach folds, but the cell
	// is in the floor population, so its calib_err never reaches the estimate fold —
	// the recalibration target is gone.
	floorFold := FoldCalibrable([]Episode{floorEp("m", claim, realized, true)})
	if floorFold.EstimateCount != 0 {
		t.Fatalf("a floor must never enter the estimate fold, got estimate count %d", floorFold.EstimateCount)
	}
	if floorFold.FloorCount != 1 {
		t.Fatalf("floor count = %d, want 1", floorFold.FloorCount)
	}
	// The breach (0.20 here) keeps the floor fold POSITIVE — a real false-warm cannot
	// be calibrated away to a content-free 0 the way the estimate gap can be closed.
	if floorFold.Value <= 0 {
		t.Fatalf("a breached floor must leave a positive fold, got %g", floorFold.Value)
	}
	if floorFold.EstimateMeanCalibErr != 0 {
		t.Fatalf("a floor contributes nothing to the estimate fold, got %g", floorFold.EstimateMeanCalibErr)
	}
}

// TestRecalibratingEstimateTowardRealityLowersFold proves the admissible win the
// loop is allowed to keep: re-pointing a genuine estimate's claim at its realized
// central tendency strictly LOWERS FoldCalibrable.
func TestRecalibratingEstimateTowardRealityLowersFold(t *testing.T) {
	before := FoldCalibrable([]Episode{
		Score("s", pred("cold_write_share", 0.85), obs(0.68, true), DefaultCalibBand()),
	}).Value
	after := FoldCalibrable([]Episode{
		Score("s", pred("cold_write_share", 0.68), obs(0.68, true), DefaultCalibBand()),
	}).Value
	if !(after < before) {
		t.Fatalf("recalibrating an estimate toward reality must LOWER the fold: before %g, after %g", before, after)
	}
	if after != 0 {
		t.Fatalf("claim == realized should fold to calib_err 0, got %g", after)
	}
}
