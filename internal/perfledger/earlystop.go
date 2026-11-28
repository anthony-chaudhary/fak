// src/aiperf/config/sweep/multi_run.py and the improvement-patience early-stop it
// drives. Apache-2.0.
//
// The source's MultiRunConfig.convergence enables adaptive trial-stopping for a
// multi-run sweep. The patience/best-so-far bookkeeping that decides "stop, no
// improvement" lives in the search planner's scalar-improvement tracker
// (optuna_planner._track_scalar_improvement) and the three-signal convergence
// evaluator (_sla_helpers.evaluate_three_signal_convergence). This port keeps
// that exact bookkeeping — a best-so-far value plus a runs-since-improvement
// counter — and the two stop signals (budget exhausted, patience exhausted),
// re-expressed in idiomatic Go over a plain slice of per-run metric values.
package perfledger

import "math"

// Early-stop reason strings. They mirror the source's stable
// convergence_reason contract: ReasonMaxRuns is the budget signal (source
// "max_iterations") and ReasonNoImprovement is the patience signal (source
// "improvement_patience"). The empty string means "continue".
const (
	ReasonMaxRuns       = "max_runs"
	ReasonNoImprovement = "no_improvement"
)

// EarlyStopTracker tracks the best-so-far metric value and the number of runs
// since the last improvement for a multi-run sweep. It mirrors aiperf's
// _track_scalar_improvement: a run that strictly beats the best-so-far resets
// the patience counter to zero; any other run increments it. The first run
// always establishes the best and leaves the counter at zero.
type EarlyStopTracker struct {
	lowerIsBetter bool
	best          float64
	hasBest       bool
	sinceImprove  int
	runs          int
}

// NewEarlyStopTracker returns a tracker for a sweep whose metric improves in
// the given direction. lowerIsBetter=true means smaller values are better
// (e.g. latency); false means larger values are better (e.g. throughput).
func NewEarlyStopTracker(lowerIsBetter bool) *EarlyStopTracker {
	return &EarlyStopTracker{lowerIsBetter: lowerIsBetter}
}

// Observe records one run's metric value, updating the best-so-far and the
// runs-since-improvement counter. A non-finite value (NaN/Inf) represents a
// failed or unmeasured run: it is counted but never becomes the best, matching
// the source's handling of a failed iteration (objective=None).
func (t *EarlyStopTracker) Observe(value float64) {
	t.runs++
	if math.IsNaN(value) || math.IsInf(value, 0) {
		t.sinceImprove++
		return
	}
	if !t.hasBest || t.improves(value, t.best) {
		t.best = value
		t.hasBest = true
		t.sinceImprove = 0
		return
	}
	t.sinceImprove++
}

// improves reports whether value strictly beats best in the tracker's
// direction.
func (t *EarlyStopTracker) improves(value, best float64) bool {
	if t.lowerIsBetter {
		return value < best
	}
	return value > best
}

// Best returns the best-so-far value and whether any finite run has been
// observed.
func (t *EarlyStopTracker) Best() (float64, bool) {
	return t.best, t.hasBest
}

// SinceImprovement returns the number of runs since the last improvement.
func (t *EarlyStopTracker) SinceImprovement() int {
	return t.sinceImprove
}

// Runs returns the total number of runs observed.
func (t *EarlyStopTracker) Runs() int {
	return t.runs
}

// ShouldEarlyStop decides whether a multi-run sweep should stop early,
// mirroring aiperf's improvement-patience early-stop criterion. values are the
// per-run metric values in run order; lowerIsBetter selects the improvement
// direction; patience is the number of consecutive runs without a new
// best-so-far after which the sweep stops; maxRuns is the hard budget ceiling
// on the number of runs.
//
// It returns (stop, reason). The budget signal is checked first, matching the
// source's ordering: stop with ReasonMaxRuns once maxRuns runs have been
// observed, else stop with ReasonNoImprovement once patience consecutive runs
// have failed to improve on the best-so-far. A patience or maxRuns of zero
// disables that signal. When neither fires it returns (false, "").
func ShouldEarlyStop(values []float64, lowerIsBetter bool, patience, maxRuns int) (bool, string) {
	tr := NewEarlyStopTracker(lowerIsBetter)
	for i := range values {
		tr.Observe(values[i])
	}
	if maxRuns > 0 && tr.Runs() >= maxRuns {
		return true, ReasonMaxRuns
	}
	if patience > 0 && tr.SinceImprovement() >= patience {
		return true, ReasonNoImprovement
	}
	return false, ""
}
