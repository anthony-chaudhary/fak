// Adapted from ai-dynamo/aiperf
// modified from the original): the adaptive-search sweep's three-signal
// convergence knobs and
// the validators that guard them. The pinned file declares the knobs, their
// defaults and the validators; the stop rule here is built from those field
// contracts on top of this package's EarlyStopTracker, which already carries
// the budget and improvement-patience signals. The upstream evaluator is not
// in the pinned file, so the plateau form (population standard deviation over
// |mean| of the trailing window), the order the signals are checked in and the
// handling of failed iterations are this port's choices.
package perfledger

import (
	"fmt"
	"math"
	"strings"
)

// ReasonPlateau is the stop reason of the plateau signal. The budget and
// patience signals reuse ReasonMaxRuns and ReasonNoImprovement.
const ReasonPlateau = "plateau"

// Planner names a ConvergenceConfig accepts. The N-D planners consume the
// three-signal knobs; the 1-D SLA planners stop on their own signals plus the
// iteration budget.
const (
	PlannerBayesian       = "bayesian"
	PlannerOptuna         = "optuna"
	PlannerMonotonicSLA   = "monotonic_sla"
	PlannerSmoothIsotonic = "smooth_isotonic"
)

// Defaults for ConvergenceConfig fields left at zero. The plateau window is
// deliberately larger than the initial-point count, so the plateau test always
// sees some model-driven iterations and cannot trip on a flat random prefix.
const (
	DefaultNInitialPoints      = 5
	DefaultPlateauWindow       = 8
	DefaultPlateauThreshold    = 0.01
	DefaultImprovementPatience = 10
)

const (
	minConvergenceIterations = 2
	maxConvergenceIterations = 200
)

// ConvergenceConfig is the stopping contract of one adaptive search. A zero
// NInitialPoints, PlateauWindow, PlateauThreshold or ImprovementPatience means
// "not set" and takes its default.
type ConvergenceConfig struct {
	Planner             string
	SearchSpacePaths    []string
	MaxIterations       int
	NInitialPoints      int
	PlateauWindow       int
	PlateauThreshold    float64
	ImprovementPatience int
}

func (c ConvergenceConfig) usesThreeSignals() bool {
	return c.Planner == PlannerBayesian || c.Planner == PlannerOptuna
}

func (c ConvergenceConfig) withDefaults() ConvergenceConfig {
	if c.NInitialPoints == 0 {
		c.NInitialPoints = DefaultNInitialPoints
	}
	if c.PlateauWindow == 0 {
		c.PlateauWindow = DefaultPlateauWindow
	}
	if c.PlateauThreshold == 0 {
		c.PlateauThreshold = DefaultPlateauThreshold
	}
	if c.ImprovementPatience == 0 {
		c.ImprovementPatience = DefaultImprovementPatience
	}
	return c
}

// DuplicateSearchSpacePaths returns each path that appears more than once, in
// order of first repeat.
func DuplicateSearchSpacePaths(paths []string) []string {
	seen := make(map[string]int, len(paths))
	var dups []string
	for _, path := range paths {
		seen[path]++
		if seen[path] == 2 {
			dups = append(dups, path)
		}
	}
	return dups
}

// Validate refuses a config that cannot run as written and returns warnings
// for settings that are legal but will not do what they appear to.
//
// Two dimensions on one path are refused: the planner would explore both
// against the same field, the second write of each trial would clobber the
// first, and iterations would be spent on a degree of freedom that does not
// exist. In-range three-signal knobs set alongside a 1-D SLA planner are only
// warned about, because that planner ignores them rather than misbehaving. The warning
// for a plateau window no larger than the initial-point count is added here;
// upstream states that rule only in the field's description.
func (c ConvergenceConfig) Validate() (warnings []string, err error) {
	switch c.Planner {
	case PlannerBayesian, PlannerOptuna, PlannerMonotonicSLA, PlannerSmoothIsotonic:
	default:
		return nil, fmt.Errorf("perfledger: convergence: unknown planner %q", c.Planner)
	}
	if len(c.SearchSpacePaths) == 0 {
		return nil, fmt.Errorf("perfledger: convergence: search space must have at least one dimension")
	}
	if dups := DuplicateSearchSpacePaths(c.SearchSpacePaths); len(dups) > 0 {
		return nil, fmt.Errorf("perfledger: convergence: search space dimensions must have unique paths; duplicates: %q", dups)
	}
	if c.MaxIterations < minConvergenceIterations || c.MaxIterations > maxConvergenceIterations {
		return nil, fmt.Errorf("perfledger: convergence: max iterations %d outside [%d, %d]", c.MaxIterations, minConvergenceIterations, maxConvergenceIterations)
	}

	set := c
	c = c.withDefaults()
	switch {
	case c.NInitialPoints < 1:
		return nil, fmt.Errorf("perfledger: convergence: initial points %d must be at least 1", c.NInitialPoints)
	case c.PlateauWindow < 2:
		return nil, fmt.Errorf("perfledger: convergence: plateau window %d must be at least 2", c.PlateauWindow)
	case !(c.PlateauThreshold > 0) || math.IsInf(c.PlateauThreshold, 0):
		return nil, fmt.Errorf("perfledger: convergence: plateau threshold %v must be positive and finite", c.PlateauThreshold)
	case c.ImprovementPatience < 2:
		return nil, fmt.Errorf("perfledger: convergence: improvement patience %d must be at least 2", c.ImprovementPatience)
	}

	if !c.usesThreeSignals() {
		// The initial-point gate is skipped here on purpose: a 1-D planner
		// drives its own probe sequence, so a short budget under the default
		// initial-point count is a valid config.
		var ignored []string
		if set.ImprovementPatience != 0 {
			ignored = append(ignored, "improvement_patience")
		}
		if set.PlateauWindow != 0 {
			ignored = append(ignored, "plateau_window")
		}
		if set.PlateauThreshold != 0 {
			ignored = append(ignored, "plateau_threshold")
		}
		if len(ignored) > 0 {
			warnings = append(warnings, fmt.Sprintf(
				"three-signal convergence keys (%s) only apply to N-D planners (%s, %s); planner %q stops on its own signals and the iteration budget, so they will be ignored",
				strings.Join(ignored, ", "), PlannerBayesian, PlannerOptuna, c.Planner))
		}
		return warnings, nil
	}

	if c.NInitialPoints >= c.MaxIterations {
		return nil, fmt.Errorf("perfledger: convergence: initial points (%d) must be < max iterations (%d); otherwise the model never fits", c.NInitialPoints, c.MaxIterations)
	}
	if c.PlateauWindow <= c.NInitialPoints {
		warnings = append(warnings, fmt.Sprintf(
			"plateau window (%d) is not larger than initial points (%d): the plateau test can trip on the random prefix alone",
			c.PlateauWindow, c.NInitialPoints))
	}
	return warnings, nil
}

// Evaluate decides whether the search should stop after the given per-iteration
// objective values, and why. It assumes a config Validate accepted.
//
// Signals are checked in order: the iteration budget (ReasonMaxRuns), then the
// plateau (ReasonPlateau), then improvement patience (ReasonNoImprovement). A
// 1-D SLA planner honours only the budget.
func (c ConvergenceConfig) Evaluate(objectives []float64, lowerIsBetter bool) (stop bool, reason string) {
	c = c.withDefaults()
	tracker := NewEarlyStopTracker(lowerIsBetter)
	for _, value := range objectives {
		tracker.Observe(value)
	}
	if c.MaxIterations > 0 && tracker.Runs() >= c.MaxIterations {
		return true, ReasonMaxRuns
	}
	if !c.usesThreeSignals() {
		return false, ""
	}
	if plateaued(objectives, c.PlateauWindow, c.PlateauThreshold) {
		return true, ReasonPlateau
	}
	if tracker.SinceImprovement() >= c.ImprovementPatience {
		return true, ReasonNoImprovement
	}
	return false, ""
}

// plateaued reports whether the last window values vary by less than threshold,
// as population standard deviation over |mean|. A window holding a failed
// (non-finite) iteration is never a plateau, and a zero-mean window is one
// only when it is exactly flat.
func plateaued(values []float64, window int, threshold float64) bool {
	if window < 2 || len(values) < window {
		return false
	}
	tail := values[len(values)-window:]
	var sum float64
	for _, v := range tail {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
		sum += v
	}
	mean := sum / float64(window)
	var sq float64
	for _, v := range tail {
		sq += (v - mean) * (v - mean)
	}
	std := math.Sqrt(sq / float64(window))
	if mean == 0 {
		return std == 0
	}
	return std/math.Abs(mean) < threshold
}
