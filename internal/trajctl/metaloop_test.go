package trajctl

import (
	"errors"
	"strings"
	"testing"
)

// TestMetaObjective_DeclaresScorerObjective pins the declare-scorer-objective sugar
// (#2567): a measured base-scorer calibration reading becomes a first-class objective
// "raise scorer X calibration", carrying the base scorer as its meta target, the meta
// scorer as its scoring method, and the reading's coefficient as the baseline to raise.
func TestMetaObjective_DeclaresScorerObjective(t *testing.T) {
	cal := ScorerCalibration{Method: "judge", Version: "2", Measured: true, Coefficient: 0.1}
	obj, err := MetaObjective(cal)
	if err != nil {
		t.Fatalf("MetaObjective on a measured scorer must succeed, got %v", err)
	}
	if !obj.IsMeta() {
		t.Fatal("a scorer-improvement objective must report IsMeta()==true")
	}
	if obj.ID != MetaObjectiveID("judge", "2") {
		t.Errorf("objective id = %q, want deterministic %q", obj.ID, MetaObjectiveID("judge", "2"))
	}
	if obj.Meta.Method != "judge" || obj.Meta.Version != "2" || obj.Meta.Baseline != 0.1 {
		t.Errorf("meta target = %+v, want judge@2 baseline 0.1", *obj.Meta)
	}
	if len(obj.Scorers) != 1 || obj.Scorers[0] != MetaScorerMethod {
		t.Errorf("scorers = %v, want [%s]", obj.Scorers, MetaScorerMethod)
	}
	if obj.Status != StatusActive {
		t.Errorf("status = %q, want active", obj.Status)
	}
	// The sugar's statement must name the scorer it raises so the curve is legible.
	if !strings.Contains(obj.Statement, "judge@2") {
		t.Errorf("statement must name the target scorer; got %q", obj.Statement)
	}
	// A declared meta objective must survive the ledger validator.
	if err := Validate(ObjectiveRecord(obj)); err != nil {
		t.Errorf("declared meta objective must validate, got %v", err)
	}
}

// TestMetaObjective_FenceRefusesMetaScorer is the one-level fence at the constructor
// (#2567/#2364): you cannot declare a scorer-improvement objective that targets the meta
// scorer itself — that would be a level-2 objective, the unbounded-scoring regress.
func TestMetaObjective_FenceRefusesMetaScorer(t *testing.T) {
	cal := ScorerCalibration{Method: MetaScorerMethod, Version: MetaScorerVersion, Measured: true, Coefficient: 0.2}
	_, err := MetaObjective(cal)
	if !errors.Is(err, ErrMetaFence) {
		t.Fatalf("targeting the meta scorer must return ErrMetaFence, got %v", err)
	}
}

// TestMetaObjective_RefusesUnmeasured: an INSUFFICIENT scorer has no baseline coefficient
// to raise, so there is nothing to score progress against — the sugar refuses it rather
// than declaring an objective that can never move.
func TestMetaObjective_RefusesUnmeasured(t *testing.T) {
	cal := ScorerCalibration{Method: "judge", Version: "1", Measured: false}
	if _, err := MetaObjective(cal); err == nil {
		t.Fatal("MetaObjective on an unmeasured scorer must error (no baseline to raise)")
	}
}

// TestValidateObjective_MetaFenceAtLedgerBoundary: the fence is not only in the
// constructor — a hand-authored objective row targeting the meta scorer, or one with an
// out-of-range baseline, is refused at Validate so it can never reach the ledger.
func TestValidateObjective_MetaFenceAtLedgerBoundary(t *testing.T) {
	level2 := Objective{ID: "x", Statement: "s", Status: StatusActive,
		Meta: &MetaTarget{Method: MetaScorerMethod, Version: "1", Baseline: 0.1}}
	if err := Validate(ObjectiveRecord(level2)); !errors.Is(err, ErrMetaFence) {
		t.Errorf("a level-2 meta objective must be refused at the ledger boundary, got %v", err)
	}

	badBaseline := Objective{ID: "y", Statement: "s", Status: StatusActive,
		Meta: &MetaTarget{Method: "judge", Version: "1", Baseline: 1.5}}
	if err := Validate(ObjectiveRecord(badBaseline)); err == nil {
		t.Error("a meta target baseline outside [-1,1] must be refused")
	}

	ok := Objective{ID: "z", Statement: "s", Status: StatusActive,
		Meta: &MetaTarget{Method: "judge", Version: "1", Baseline: -0.2}}
	if err := Validate(ObjectiveRecord(ok)); err != nil {
		t.Errorf("a well-formed meta objective must validate, got %v", err)
	}
}

// TestCalibrate_ExcludesMetaScorer is the third fence point: the meta scorer's own rows
// never enter the calibration leaderboard, so it can never be surfaced as its own repair
// target — WorstCalibrated can only ever hand the loop a BASE scorer to raise.
func TestCalibrate_ExcludesMetaScorer(t *testing.T) {
	st := calibState(
		[]string{"objMet", "objFail"},
		[]calibScore{
			{"objMet", GroundTruthMethod, "1", 1.0, W3},
			{"objFail", GroundTruthMethod, "1", 0.0, W3},
			{"objMet", "judge", "1", 0.9, W1},
			{"objFail", "judge", "1", 0.1, W1},
			// Meta scorer rows exist in the ledger, but must not be calibrated.
			{"objMet", MetaScorerMethod, MetaScorerVersion, 0.5, W3},
			{"objFail", MetaScorerMethod, MetaScorerVersion, 0.5, W3},
		},
	)
	for _, sc := range st.Calibrate().Scorers {
		if sc.Method == MetaScorerMethod {
			t.Fatalf("the meta scorer must never appear in the calibration leaderboard, got %+v", sc)
		}
	}
}

// TestMetaProgress pins the calibration-delta math: progress is the fraction of the gap
// from baseline to the well-calibrated threshold that the current coefficient has closed,
// clamped, with an already-well-calibrated baseline reading fully met.
func TestMetaProgress(t *testing.T) {
	cases := []struct {
		name              string
		baseline, current float64
		want              float64
	}{
		{"no movement", -1.0, -1.0, 0},
		{"half the gap closed", 0.0, 0.25, 0.5},
		{"overshoot clamps to met", 0.0, 1.0, 1},
		{"regression clamps to zero", 0.0, -0.3, 0},
		{"miscalibrated halfway to threshold", -0.5, 0.0, 0.5},
		{"already well-calibrated at declaration", 0.6, 0.1, 1},
		{"exactly at threshold is met", 0.5, 0.5, 1},
	}
	for _, c := range cases {
		if got := metaProgress(c.baseline, c.current); got != c.want {
			t.Errorf("%s: metaProgress(%v, %v) = %v, want %v", c.name, c.baseline, c.current, got, c.want)
		}
	}
}

// TestMetaScore_FromCalibrationDelta is the end-to-end witness (#2567): a scorer-improvement
// objective scores FROM the calibration meter. The base "judge" scorer's value/outcome
// pairs (0.6,1),(0.5,0),(0.4,1) give a Pearson coefficient of exactly 0.0; a meta objective
// declared at baseline -0.5 therefore reads exactly half the gap to the well-calibrated
// threshold closed, as a deterministic W3 row the ledger accepts.
func TestMetaScore_FromCalibrationDelta(t *testing.T) {
	st := calibState(
		[]string{"objA", "objB", "objC"},
		[]calibScore{
			{"objA", GroundTruthMethod, "1", 1.0, W3},
			{"objB", GroundTruthMethod, "1", 0.0, W3},
			{"objC", GroundTruthMethod, "1", 1.0, W3},
			{"objA", "judge", "1", 0.6, W1},
			{"objB", "judge", "1", 0.5, W1},
			{"objC", "judge", "1", 0.4, W1},
		},
	)
	// Sanity: the meter reads the judge at coefficient 0.0 (weak but measured).
	if cur, measured := st.Calibrate().coefficientOf("judge", "1"); !measured || cur != 0.0 {
		t.Fatalf("fixture precondition: judge coefficient = %v measured=%v, want 0 true", cur, measured)
	}

	obj, err := MetaObjective(ScorerCalibration{Method: "judge", Version: "1", Measured: true, Coefficient: -0.5})
	if err != nil {
		t.Fatalf("declare meta objective: %v", err)
	}
	row, ok := st.MetaScore(obj)
	if !ok {
		t.Fatal("MetaScore must score a meta objective")
	}
	if row.Witness != W3 {
		t.Errorf("the calibration delta is deterministic evidence; witness = %q, want W3", row.Witness)
	}
	if row.Method != MetaScorerMethod || row.Version != MetaScorerVersion {
		t.Errorf("row method/version = %s/%s, want %s/%s", row.Method, row.Version, MetaScorerMethod, MetaScorerVersion)
	}
	if row.ObjectiveID != obj.ID {
		t.Errorf("row objective = %q, want %q", row.ObjectiveID, obj.ID)
	}
	if row.Value != 0.5 {
		t.Errorf("baseline -0.5 to current 0.0 is half the gap to threshold 0.5; value = %v, want 0.5", row.Value)
	}
	if len(row.Evidence) != 1 || row.Evidence[0].Ref != "judge@1" {
		t.Errorf("evidence must point at the target scorer; got %+v", row.Evidence)
	}
	// The produced row must be a valid ledger score.
	if err := Validate(ScoreRecord(row)); err != nil {
		t.Errorf("meta score row must validate, got %v", err)
	}
}

// TestMetaScore_TargetNowInsufficient: a target scorer that has lost measurability witnesses
// no delta and reads 0 progress — fail closed, never a fabricated improvement.
func TestMetaScore_TargetNowInsufficient(t *testing.T) {
	st := calibState([]string{"objA"}, []calibScore{
		{"objA", GroundTruthMethod, "1", 1.0, W3},
	})
	obj := Objective{ID: MetaObjectiveID("ghost", "1"), Statement: "raise ghost@1", Status: StatusActive,
		Meta: &MetaTarget{Method: "ghost", Version: "1", Baseline: 0.1}}
	row, ok := st.MetaScore(obj)
	if !ok {
		t.Fatal("MetaScore must score a meta objective even when the target is unmeasured now")
	}
	if row.Value != 0 {
		t.Errorf("an unmeasurable target witnesses no delta; value = %v, want 0", row.Value)
	}
	if !strings.Contains(row.Evidence[0].Detail, "INSUFFICIENT") {
		t.Errorf("the detail must record the lost measurability; got %q", row.Evidence[0].Detail)
	}
}

// TestMetaScore_NotMetaObjective: a plain objective is not scored by the meta scorer.
func TestMetaScore_NotMetaObjective(t *testing.T) {
	plain := Objective{ID: "plain", Statement: "ship the thing", Status: StatusActive}
	if _, ok := (State{}).MetaScore(plain); ok {
		t.Error("MetaScore must return false for a non-meta objective")
	}
}
