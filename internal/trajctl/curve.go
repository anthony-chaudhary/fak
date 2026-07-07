package trajctl

// curve.go — issue #2538, spine step 5 of the trajectory-control epic (#2533):
// the curve fold + the closed signal vocabulary. After the scorers accumulate
// ScoreRows, nothing folds them into a readable trend; every consumer would
// re-derive stall/drift ad hoc and disagree. This is the one fold that turns the
// accumulated rows into a per-(objective, method) time-ordered curve plus a
// single closed-vocabulary signal, so declare -> score -> curve runs end to end.
//
// The fold is pure and tier-1: it reads the already-folded State (objectives +
// score history) and derives the signal from the witnessed curve alone. It never
// steers or acts on a signal — that is the controller rungs' job (out of scope
// here). The signal vocabulary is closed and the thresholds are pinned so the
// four shapes are golden-testable and every consumer agrees on the verdict.

import (
	"fmt"
	"math"
	"sort"
)

// CurveSchema is the pinned schema id for a curve report. Downstream consumers
// (steering, metrics, status) pin to this string.
const CurveSchema = "fak-trajctl-curve/1"

// Signal is the closed trajectory-control vocabulary derived from a curve. It is
// exactly four values — a consumer that sees any other string is reading a
// foreign or newer schema.
type Signal string

const (
	// SignalHealthy: the witnessed progress curve is rising or steady with no
	// competing stall/drift/overrun evidence. The controller's default action on
	// HEALTHY is nothing (the regime gate).
	SignalHealthy Signal = "HEALTHY"
	// SignalStall: a flat progress curve while a W2 activity-divergence signal
	// fired — busy but not moving.
	SignalStall Signal = "STALL"
	// SignalDrift: the witnessed progress curve is declining — alignment is
	// falling, e.g. a re-score after a commit went dangling.
	SignalDrift Signal = "DRIFT"
	// SignalDetourOverrun: a detour (child objective) has run past its declared
	// turn budget while its parent is paused — the return-to-main signal.
	SignalDetourOverrun Signal = "DETOUR_OVERRUN"
)

// curveFlatEpsilon is the pinned progress-delta band that counts a curve as flat.
// A step within ±epsilon is flat (candidate STALL); a step below -epsilon is a
// decline (DRIFT). It mirrors the activity-divergence scorer's progress epsilon so
// the STALL fold and the scorer that feeds it agree on "flat".
const curveFlatEpsilon = activityDivergenceProgressEpsilon

// CurvePoint is one value in a method's time-ordered curve.
type CurvePoint struct {
	Value      float64     `json:"value"`
	Witness    WitnessRung `json:"witness"`
	UnixMillis int64       `json:"unix_millis,omitempty"`
}

// MethodCurve is one scoring method's time-ordered fold for an objective. Version
// is the latest version seen for the method, so a method change is visible.
type MethodCurve struct {
	Method  string       `json:"method"`
	Version string       `json:"version"`
	Points  []CurvePoint `json:"points"`
}

// ObjectiveCurve is one objective's folded curves plus its derived signal. Latest
// and Delta summarize the witnessed progress curve (the W3 commit-progress
// method) so a consumer need not re-walk the points to get the headline.
type ObjectiveCurve struct {
	ObjectiveID string          `json:"objective_id"`
	ParentID    string          `json:"parent_id,omitempty"`
	Status      ObjectiveStatus `json:"status"`
	Signal      Signal          `json:"signal"`
	Latest      float64         `json:"latest"`
	Delta       float64         `json:"delta"`
	Detail      string          `json:"detail"`
	Methods     []MethodCurve   `json:"methods"`
}

// CurveReport is the schema-pinned envelope. For a single objective it holds one
// ObjectiveCurve; for the worst-first listing it holds every open objective in
// severity order. Always exactly one schema so downstream consumers pin once.
type CurveReport struct {
	Schema     string           `json:"schema"`
	Objectives []ObjectiveCurve `json:"objectives"`
}

// curveMethods folds an objective's score rows into per-method, time-ordered
// curves. Methods are listed in lexical order and points within a method in
// UnixMillis order (stable, so an unstamped or tied point keeps append order).
func (s State) curveMethods(objectiveID string) []MethodCurve {
	byMethod := map[string]*MethodCurve{}
	order := make([]string, 0)
	for _, row := range s.Scores {
		if row.ObjectiveID != objectiveID {
			continue
		}
		mc, ok := byMethod[row.Method]
		if !ok {
			mc = &MethodCurve{Method: row.Method, Version: row.Version}
			byMethod[row.Method] = mc
			order = append(order, row.Method)
		}
		mc.Version = row.Version // latest version wins; a rescored curve shows it
		mc.Points = append(mc.Points, CurvePoint{
			Value:      row.Value,
			Witness:    row.Witness,
			UnixMillis: row.UnixMillis,
		})
	}
	sort.Strings(order)
	out := make([]MethodCurve, 0, len(order))
	for _, m := range order {
		mc := byMethod[m]
		sort.SliceStable(mc.Points, func(i, j int) bool {
			return mc.Points[i].UnixMillis < mc.Points[j].UnixMillis
		})
		out = append(out, *mc)
	}
	return out
}

// progressPoints returns the witnessed-commit-progress curve — the primary W3
// progress signal that HEALTHY/STALL/DRIFT read.
func progressPoints(methods []MethodCurve) []CurvePoint {
	for _, mc := range methods {
		if mc.Method == CommitScorerMethod {
			return mc.Points
		}
	}
	return nil
}

// hasMethod reports whether the folded methods carry any row for method.
func hasMethod(methods []MethodCurve, method string) bool {
	for _, mc := range methods {
		if mc.Method == method && len(mc.Points) > 0 {
			return true
		}
	}
	return false
}

// latestDelta returns the last progress value and the step from the prior point
// (0 for a single point or empty curve).
func latestDelta(progress []CurvePoint) (latest, delta float64) {
	if len(progress) == 0 {
		return 0, 0
	}
	latest = progress[len(progress)-1].Value
	if len(progress) >= 2 {
		delta = latest - progress[len(progress)-2].Value
	}
	return latest, delta
}

// objectiveOpen reports whether a status is a live/steerable one (active|paused).
func objectiveOpen(st ObjectiveStatus) bool {
	return st == StatusActive || st == StatusPaused
}

// CurveFor folds one objective into its per-method curves and derived signal. ok
// is false if the objective was never declared.
func (s State) CurveFor(objectiveID string) (ObjectiveCurve, bool) {
	obj, ok := s.Objectives[objectiveID]
	if !ok {
		return ObjectiveCurve{}, false
	}
	methods := s.curveMethods(objectiveID)
	progress := progressPoints(methods)
	latest, delta := latestDelta(progress)
	sig, detail := s.classify(obj, methods, progress, latest, delta)
	return ObjectiveCurve{
		ObjectiveID: obj.ID,
		ParentID:    obj.ParentID,
		Status:      obj.Status,
		Signal:      sig,
		Latest:      latest,
		Delta:       delta,
		Detail:      detail,
		Methods:     methods,
	}, true
}

// classify derives the closed-vocabulary signal from the witnessed curve. The
// order is priority order — the most actionable, most specific signal wins:
// DETOUR_OVERRUN (a paused-parent detour past budget) > DRIFT (declining) >
// STALL (flat × active) > HEALTHY.
func (s State) classify(obj Objective, methods []MethodCurve, progress []CurvePoint, latest, delta float64) (Signal, string) {
	turns := len(progress)
	// DETOUR_OVERRUN: an open child whose parent is paused and which has scored
	// more turns than its declared turn budget. One W3 progress point per scored
	// turn (the stop-hook writes one per turn), so the point count is the turns
	// spent on the detour.
	if obj.ParentID != "" && objectiveOpen(obj.Status) && obj.Budget.Turns > 0 && turns > obj.Budget.Turns {
		if parent, ok := s.Objectives[obj.ParentID]; ok && parent.Status == StatusPaused {
			return SignalDetourOverrun, fmt.Sprintf(
				"detour ran %d turns past a %d-turn budget while parent %q is paused",
				turns-obj.Budget.Turns, obj.Budget.Turns, obj.ParentID)
		}
	}
	// DRIFT: the witnessed progress curve is declining.
	if len(progress) >= 2 && delta < -curveFlatEpsilon {
		return SignalDrift, fmt.Sprintf("progress declined %.2f -> %.2f (delta %+.2f)", latest-delta, latest, delta)
	}
	// STALL: a flat (or not-yet-moving) progress curve while the W2
	// activity-divergence scorer fired — high activity, no witnessed movement.
	if curveFlat(progress) && hasMethod(methods, ActivityDivergenceScorerMethod) {
		return SignalStall, fmt.Sprintf("flat progress (delta %+.2f) with an active divergence signal", delta)
	}
	// HEALTHY: rising or steady with nothing pulling against it.
	return SignalHealthy, fmt.Sprintf("progress %.2f (delta %+.2f)", latest, delta)
}

// curveFlat reports whether the progress curve is flat: a single/empty curve, or
// a last step within ±curveFlatEpsilon. A declining step is not flat (that is
// DRIFT), so STALL and DRIFT never both fire.
func curveFlat(progress []CurvePoint) bool {
	if len(progress) < 2 {
		return true
	}
	d := progress[len(progress)-1].Value - progress[len(progress)-2].Value
	return math.Abs(d) <= curveFlatEpsilon
}

// CurveReportFor wraps one objective's curve in a schema-pinned report — the
// `curve --objective <id>` shape.
func (s State) CurveReportFor(objectiveID string) (CurveReport, bool) {
	oc, ok := s.CurveFor(objectiveID)
	if !ok {
		return CurveReport{}, false
	}
	return CurveReport{Schema: CurveSchema, Objectives: []ObjectiveCurve{oc}}, true
}

// OpenCurves folds every open objective (active|paused) into a worst-first
// report — the `curve` (no id) shape. Worst-first ranks by signal severity
// (DETOUR_OVERRUN > DRIFT > STALL > HEALTHY), then by lower latest progress, then
// by id, so the objective most in need of attention lists first.
func (s State) OpenCurves() CurveReport {
	rep := CurveReport{Schema: CurveSchema, Objectives: make([]ObjectiveCurve, 0)}
	for _, id := range s.ObjectiveIDs() {
		if !objectiveOpen(s.Objectives[id].Status) {
			continue
		}
		oc, _ := s.CurveFor(id)
		rep.Objectives = append(rep.Objectives, oc)
	}
	sort.SliceStable(rep.Objectives, func(i, j int) bool {
		a, b := rep.Objectives[i], rep.Objectives[j]
		if sa, sb := signalSeverity(a.Signal), signalSeverity(b.Signal); sa != sb {
			return sa > sb
		}
		if a.Latest != b.Latest {
			return a.Latest < b.Latest
		}
		return a.ObjectiveID < b.ObjectiveID
	})
	return rep
}

// SignalDebt maps a curve signal to a worst-first debt weight for EXTERNAL folds —
// the superloop trajectory member (issue #2563) weighs an open objective by this so
// its walk orders objectives the same way OpenCurves does: HEALTHY 0 < STALL 1 <
// DRIFT 2 < DETOUR_OVERRUN 3. A HEALTHY objective carries zero debt (an on-course
// objective is nothing to enter); it reuses the internal severity ranking so the two
// orderings can never drift apart.
func SignalDebt(sig Signal) int { return signalSeverity(sig) }

// signalSeverity ranks the vocabulary so the worst-first listing orders the most
// actionable signal first.
func signalSeverity(sig Signal) int {
	switch sig {
	case SignalDetourOverrun:
		return 3
	case SignalDrift:
		return 2
	case SignalStall:
		return 1
	default:
		return 0
	}
}
