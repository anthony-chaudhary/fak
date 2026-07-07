package trajctl

import "testing"

// calibFixture builds a State from objective + score rows for the calibration fold. Each
// score is (objectiveID, method, version, value, witness).
type calibScore struct {
	obj     string
	method  string
	version string
	value   float64
	witness WitnessRung
}

func calibState(objIDs []string, scores []calibScore) State {
	st := State{Objectives: map[string]Objective{}}
	for _, id := range objIDs {
		st.Objectives[id] = Objective{ID: id, Status: StatusActive}
	}
	for _, s := range scores {
		st.Scores = append(st.Scores, ScoreRow{
			ObjectiveID: s.obj, Method: s.method, Version: s.version, Value: s.value, Witness: s.witness,
		})
	}
	return st
}

// TestCalibrateBrokenScorerRanksLast is the done-condition witness (#2566): a leaderboard
// folds from fixture rows; the well-calibrated scorer (its numbers track the W3 outcome)
// ranks first and the deliberately broken scorer (anti-correlated with the outcome) ranks
// last, annotated MISCALIBRATED — never dropped.
func TestCalibrateBrokenScorerRanksLast(t *testing.T) {
	// Two objectives with OPPOSITE witnessed outcomes: objMet ended at 1.0, objFail at 0.0.
	// A GOOD judge scored them 0.9 / 0.1 (tracks outcome); a BROKEN judge scored 0.1 / 0.9.
	st := calibState(
		[]string{"objMet", "objFail"},
		[]calibScore{
			{"objMet", GroundTruthMethod, "1", 1.0, W3},
			{"objFail", GroundTruthMethod, "1", 0.0, W3},
			{"objMet", "good-judge", "1", 0.9, W1},
			{"objFail", "good-judge", "1", 0.1, W1},
			{"objMet", "broken-judge", "1", 0.1, W1},
			{"objFail", "broken-judge", "1", 0.9, W1},
		},
	)
	rep := st.Calibrate()

	if rep.Schema != CalibrationSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, CalibrationSchema)
	}
	if rep.GroundTruth != GroundTruthMethod {
		t.Errorf("ground truth = %q, want %q", rep.GroundTruth, GroundTruthMethod)
	}
	// The ground-truth method never calibrates against itself, so only the two judges rank.
	if len(rep.Scorers) != 2 {
		t.Fatalf("want exactly the two judge scorers ranked, got %d: %+v", len(rep.Scorers), rep.Scorers)
	}
	first, last := rep.Scorers[0], rep.Scorers[len(rep.Scorers)-1]
	if first.Method != "good-judge" || first.Verdict != CalibrationWell {
		t.Errorf("best-first: rank 1 should be the well-calibrated judge, got %s/%s", first.Method, first.Verdict)
	}
	if last.Method != "broken-judge" || last.Verdict != CalibrationMiscalibrated {
		t.Errorf("the deliberately broken judge must rank LAST, annotated MISCALIBRATED, got %s/%s", last.Method, last.Verdict)
	}
	if first.Coefficient <= last.Coefficient {
		t.Errorf("well-calibrated coefficient (%.2f) must exceed the broken one (%.2f)", first.Coefficient, last.Coefficient)
	}
	if !first.Measured || !last.Measured {
		t.Errorf("both judges have varying outcomes, so both must be measured, got %v/%v", first.Measured, last.Measured)
	}
	// The worst-first repair target the meta loop enters is the broken judge.
	worst, ok := rep.WorstCalibrated()
	if !ok || worst.Method != "broken-judge" {
		t.Errorf("WorstCalibrated must surface the broken judge, got ok=%v %s", ok, worst.Method)
	}
}

// TestCalibrateInsufficientWhenSingleOutcome pins the honesty fence: a scorer that has
// only ever scored objectives with the SAME outcome has no variance to correlate, so it
// reads INSUFFICIENT (unmeasured) — never scored as if calibrated — and sorts below the
// measured rows.
func TestCalibrateInsufficientWhenSingleOutcome(t *testing.T) {
	st := calibState(
		[]string{"objA", "objB"},
		[]calibScore{
			// Both objectives ended at the SAME outcome → zero outcome variance.
			{"objA", GroundTruthMethod, "1", 1.0, W3},
			{"objB", GroundTruthMethod, "1", 1.0, W3},
			{"objA", "lonely-judge", "1", 0.4, W1},
			{"objB", "lonely-judge", "1", 0.8, W1},
		},
	)
	rep := st.Calibrate()
	if len(rep.Scorers) != 1 {
		t.Fatalf("want the one judge ranked, got %d", len(rep.Scorers))
	}
	sc := rep.Scorers[0]
	if sc.Measured || sc.Verdict != CalibrationInsufficient {
		t.Errorf("a scorer with no outcome variance must read INSUFFICIENT/unmeasured, got measured=%v verdict=%s", sc.Measured, sc.Verdict)
	}
	if _, ok := rep.WorstCalibrated(); ok {
		t.Error("no measured scorer exists, so WorstCalibrated must report none")
	}
}

// TestCalibrateEmptyLedger pins the empty case: no scores → an empty, well-formed report.
func TestCalibrateEmptyLedger(t *testing.T) {
	rep := State{Objectives: map[string]Objective{}}.Calibrate()
	if rep.Schema != CalibrationSchema || len(rep.Scorers) != 0 {
		t.Errorf("empty ledger → empty schema-pinned report, got %+v", rep)
	}
}
