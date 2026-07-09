package trajctl

// metaloop.go — issue #2567, epic #2533: the META LOOP, fenced at exactly one level.
//
// The doctrine ("declare an objective, score anything, steer by the curve") eats itself
// here: a SCORING METHOD you want to improve becomes an objective with its own score. The
// calibration meter (#2566, calibration.go) already measures how well a base scorer's
// numbers track the W3 witnessed outcome; this leaf closes the loop by declaring "raise
// scorer X's calibration" as a first-class objective whose OWN W3 score is the meter's
// calibration DELTA — the base scorer's coefficient at declaration (the baseline) advanced
// toward well-calibrated. A scorer that was steering sessions wrong becomes a worst-first
// repair target the loop can enter and measure like any other objective.
//
// The fence: EXACTLY ONE meta level. A scorer-improvement objective may target a BASE
// scorer, but NOTHING may target the meta scorer itself. Without that stop the doctrine
// recurses — score the scorer that scores the scorer that scores the scorer — which is the
// #2364 "unbounded scoring loop" this epic names as an enemy, not a risk it accepts. The
// fence is structural, enforced in three places that cannot drift apart:
//   1. MetaObjective refuses to construct a meta-objective targeting the meta scorer.
//   2. validateMetaTarget refuses the same at the ledger boundary (a hand-authored row
//      cannot smuggle a level-2 objective past Append).
//   3. Calibrate skips the meta scorer method, so it never appears as its own repair
//      target and WorstCalibrated can never hand the loop a level-2 target to declare.

import (
	"errors"
	"fmt"
)

const (
	// MetaScorerMethod is the method name of the calibration-delta score — the W3 value a
	// scorer-improvement objective is measured by. It is itself a scorer method, so the
	// one-level fence must forbid a meta-objective from targeting it.
	MetaScorerMethod = "calibration-delta"
	// MetaScorerVersion is the meta scorer's implementation version; it travels in every
	// row so a re-scored delta can be told apart from an older one.
	MetaScorerVersion = "1"
)

// ErrMetaFence is returned when a scorer-improvement objective would target the meta
// scorer itself — the one-level fence. It is a sentinel so callers can distinguish the
// fence from ordinary validation failures.
var ErrMetaFence = errors.New("trajctl: meta-loop fence: a scorer-improvement objective may not target the meta scorer (one level only, #2567/#2364)")

// MetaTarget is the base scorer a meta-objective drives toward well-calibrated, plus the
// baseline coefficient captured at declaration. Progress is the calibration meter's delta
// from Baseline to the scorer's current coefficient (see State.MetaScore).
type MetaTarget struct {
	Method   string  `json:"method"`
	Version  string  `json:"version"`
	Baseline float64 `json:"baseline"`
}

// validateMetaTarget checks a meta target is well-formed and does not breach the one-level
// fence. Baseline is a Pearson coefficient, so it must sit in [-1, 1].
func validateMetaTarget(m MetaTarget) error {
	if m.Method == "" {
		return errors.New("trajctl: meta target method is required")
	}
	if m.Version == "" {
		return errors.New("trajctl: meta target version is required")
	}
	if m.Method == MetaScorerMethod {
		return ErrMetaFence
	}
	if m.Baseline < -1 || m.Baseline > 1 {
		return fmt.Errorf("trajctl: meta target baseline %v outside [-1,1]", m.Baseline)
	}
	return nil
}

// MetaObjective is the declare-scorer-objective sugar: it turns a measured base-scorer
// calibration reading into a first-class objective "raise scorer X calibration", whose own
// W3 score is the calibration-meter delta (see State.MetaScore). The target's current
// coefficient is captured as the baseline the delta is measured from.
//
// It refuses two inputs so the loop stays honest and bounded: an UNMEASURED scorer (a
// calibration reading with no coefficient has no baseline to raise — nothing to score
// against), and the meta scorer itself (the one-level fence, returning ErrMetaFence).
func MetaObjective(cal ScorerCalibration) (Objective, error) {
	if cal.Method == "" || cal.Version == "" {
		return Objective{}, errors.New("trajctl: meta objective needs a scorer method and version")
	}
	if cal.Method == MetaScorerMethod {
		return Objective{}, ErrMetaFence
	}
	if !cal.Measured {
		return Objective{}, fmt.Errorf("trajctl: cannot raise calibration of unmeasured scorer %s@%s (no baseline coefficient to improve)", cal.Method, cal.Version)
	}
	obj := Objective{
		ID:        MetaObjectiveID(cal.Method, cal.Version),
		Statement: fmt.Sprintf("raise scorer %s@%s calibration (baseline r=%+.2f) toward well-calibrated (r>=%.2f)", cal.Method, cal.Version, cal.Coefficient, calibrationWellThreshold),
		Scorers:   []string{MetaScorerMethod},
		Status:    StatusActive,
		Meta: &MetaTarget{
			Method:   cal.Method,
			Version:  cal.Version,
			Baseline: cal.Coefficient,
		},
	}
	return obj, nil
}

// MetaObjectiveID is the deterministic id of the scorer-improvement objective for one base
// scorer method+version, so re-declaring the same target folds onto the same objective.
func MetaObjectiveID(method, version string) string {
	return "meta:calibration:" + method + "@" + version
}

// IsMeta reports whether o is a scorer-improvement (meta-loop) objective.
func (o Objective) IsMeta() bool { return o.Meta != nil }

// MetaScore computes the W3 calibration-delta score for a scorer-improvement objective. It
// re-runs the calibration meter over the folded state, reads the target base scorer's
// CURRENT coefficient, and returns progress from the objective's baseline toward
// well-calibrated. The witness is W3 because the meter is a deterministic, zero-model-call
// fold over the W3 witnessed outcome — the same rung #2567 pins for this score.
//
// Returns (row, false) when o is not a meta objective. A target that has since lost
// measurability (INSUFFICIENT signal now) witnesses no delta and reads 0 progress — fail
// closed, never a fabricated improvement.
func (s State) MetaScore(o Objective) (ScoreRow, bool) {
	if o.Meta == nil {
		return ScoreRow{}, false
	}
	current, measured := s.Calibrate().coefficientOf(o.Meta.Method, o.Meta.Version)
	detail := fmt.Sprintf("baseline r=%+.2f -> current r=%+.2f (target well-calibrated r>=%.2f)", o.Meta.Baseline, current, calibrationWellThreshold)
	if !measured {
		current = o.Meta.Baseline // no current coefficient to witness: fail closed to zero delta
		detail = fmt.Sprintf("baseline r=%+.2f -> target now INSUFFICIENT signal: no witnessed delta", o.Meta.Baseline)
	}
	row := ScoreRow{
		ObjectiveID: o.ID,
		Value:       metaProgress(o.Meta.Baseline, current),
		Method:      MetaScorerMethod,
		Version:     MetaScorerVersion,
		Witness:     W3,
		Evidence: []EvidenceRef{{
			Kind:   "calibration",
			Ref:    o.Meta.Method + "@" + o.Meta.Version,
			Detail: detail,
		}},
	}
	return row, true
}

// metaProgress maps a base scorer's baseline and current calibration coefficients to a
// [0,1] progress value toward well-calibrated. A scorer already at or above the
// well-calibrated threshold when the objective was declared reads fully met; otherwise the
// value is the fraction of the gap from baseline to the threshold that the current
// coefficient has closed, clamped so a regression reads 0 and an overshoot reads 1.
func metaProgress(baseline, current float64) float64 {
	if baseline >= calibrationWellThreshold {
		return 1
	}
	v := (current - baseline) / (calibrationWellThreshold - baseline)
	switch {
	case v < 0:
		return 0
	case v > 1:
		return 1
	default:
		return v
	}
}

// coefficientOf returns the calibration coefficient the report measured for one scorer
// method+version and whether it was measured (an INSUFFICIENT or absent scorer returns
// false). Used by MetaScore to read a target's current calibration.
func (r CalibrationReport) coefficientOf(method, version string) (coeff float64, measured bool) {
	for _, sc := range r.Scorers {
		if sc.Method == method && sc.Version == version {
			return sc.Coefficient, sc.Measured
		}
	}
	return 0, false
}
