// Adapted from bentoml/llm-optimizer src/llm_optimizer/performance.py
// (estimate_performance_under_constraints and run_performance_estimation) at
// revision 7be87d776e60a978516a2ae53e4a4ff745d04c20, Apache-2.0:
// https://github.com/bentoml/llm-optimizer/blob/7be87d776e60a978516a2ae53e4a4ff745d04c20/src/llm_optimizer/performance.py
// The idea is re-expressed in idiomatic Go; no upstream code is copied.
// Changed: typed sentinel errors instead of raised ValueErrors, an explicit
// memory-infeasible flag instead of an infinite-TTFT sentinel, and a
// caller-supplied estimator instead of a built-in roofline model.
package modelperfobs

import (
	"errors"
	"fmt"
	"math"
)

// SLOFeasibilitySweepSchema identifies the serialized sweep result schema.
const SLOFeasibilitySweepSchema = "fak-slo-feasibility-sweep/1"

// Closed-contract sentinels for the SLO feasibility sweep.
var (
	ErrSLOConstraintInvalid = errors.New("modelperfobs: invalid SLO constraint")
	ErrSLOEstimateInvalid   = errors.New("modelperfobs: invalid performance estimate")
	ErrSLOLadderInvalid     = errors.New("modelperfobs: invalid concurrency ladder")
)

// SLOMetric names the performance quantity a constraint bounds.
type SLOMetric string

const (
	SLOMetricTTFT       SLOMetric = "ttft"
	SLOMetricITL        SLOMetric = "itl"
	SLOMetricE2ELatency SLOMetric = "e2e_latency"
)

// SLOStatType names the statistic the constraint applies to.
type SLOStatType string

const (
	SLOStatMean   SLOStatType = "mean"
	SLOStatMedian SLOStatType = "median"
	SLOStatP95    SLOStatType = "p95"
	SLOStatP99    SLOStatType = "p99"
)

// SLOOperator is the comparison between the estimated statistic and the bound.
type SLOOperator string

const (
	SLOOpLessThan           SLOOperator = "<"
	SLOOpLessThanOrEqual    SLOOperator = "<="
	SLOOpGreaterThan        SLOOperator = ">"
	SLOOpGreaterThanOrEqual SLOOperator = ">="
	SLOOpEquals             SLOOperator = "=="
	SLOOpNotEquals          SLOOperator = "!="
)

// SLOUnit is the unit of a constraint value.
type SLOUnit string

const (
	SLOUnitMilliseconds SLOUnit = "ms"
	SLOUnitSeconds      SLOUnit = "s"
)

// SLOConstraint bounds one performance metric of a serving configuration.
// Value carries Unit; Metric selects the estimate field it compares.
type SLOConstraint struct {
	Metric   SLOMetric   `json:"metric"`
	StatType SLOStatType `json:"stat_type"`
	Operator SLOOperator `json:"operator"`
	Value    float64     `json:"value"`
	Unit     SLOUnit     `json:"unit"`
}

// PerformanceEstimate is one adapter-neutral estimate at a concurrency level.
// It may come from a theoretical roofline model or from a measured benchmark;
// the sweep logic is identical either way. When MemoryInfeasible is set the
// numeric fields are ignored and must not be read.
type PerformanceEstimate struct {
	Concurrency         int     `json:"concurrency"`
	TTFTMS              float64 `json:"ttft_ms"`
	ITLMS               float64 `json:"itl_ms"`
	E2ELatencySeconds   float64 `json:"e2e_latency_seconds"`
	OutputThroughputTPS float64 `json:"output_throughput_tps"`
	MemoryInfeasible    bool    `json:"memory_infeasible,omitempty"`
}

// sloStatAdjustmentFactor scales a theoretical mean estimate up to the
// requested statistic. The factors are heuristic approximations of tail
// inflation in serving latency distributions, so percentile SLOs stay
// conservative when only one theoretical estimate is available. Callers whose
// Estimate callback already returns measured percentiles should constrain on
// SLOStatMean to avoid double scaling.
var sloStatAdjustmentFactor = map[SLOStatType]float64{
	SLOStatMean:   1.0,
	SLOStatMedian: 1.1,
	SLOStatP95:    1.3,
	SLOStatP99:    1.6,
}

// sloEqualityEpsilonMS bounds the absolute difference that counts as equal
// for the == and != operators.
const sloEqualityEpsilonMS = 1e-6

// PowerOfTwoConcurrencyLevels returns the ascending powers-of-two ladder
// ending at the largest power of two not exceeding max.
func PowerOfTwoConcurrencyLevels(max int) ([]int, error) {
	if max < 1 {
		return nil, fmt.Errorf("%w: maximum concurrency must be positive", ErrSLOLadderInvalid)
	}
	levels := make([]int, 0, 8)
	for level := 1; level <= max; level <<= 1 {
		levels = append(levels, level)
	}
	return levels, nil
}

func validateSLOConstraint(c SLOConstraint) error {
	switch c.Metric {
	case SLOMetricTTFT, SLOMetricITL, SLOMetricE2ELatency:
	default:
		return fmt.Errorf("%w: metric %q", ErrSLOConstraintInvalid, c.Metric)
	}
	if _, ok := sloStatAdjustmentFactor[c.StatType]; !ok {
		return fmt.Errorf("%w: stat type %q", ErrSLOConstraintInvalid, c.StatType)
	}
	switch c.Operator {
	case SLOOpLessThan, SLOOpLessThanOrEqual, SLOOpGreaterThan, SLOOpGreaterThanOrEqual, SLOOpEquals, SLOOpNotEquals:
	default:
		return fmt.Errorf("%w: operator %q", ErrSLOConstraintInvalid, c.Operator)
	}
	if !finite(c.Value) || c.Value < 0 {
		return fmt.Errorf("%w: value must be finite and non-negative", ErrSLOConstraintInvalid)
	}
	switch c.Unit {
	case SLOUnitMilliseconds, SLOUnitSeconds:
	default:
		return fmt.Errorf("%w: unit %q", ErrSLOConstraintInvalid, c.Unit)
	}
	return nil
}

func validatePerformanceEstimate(est PerformanceEstimate) error {
	if est.Concurrency <= 0 {
		return fmt.Errorf("%w: concurrency must be positive", ErrSLOEstimateInvalid)
	}
	if est.MemoryInfeasible {
		return nil
	}
	fields := []struct {
		name string
		v    float64
	}{
		{"ttft_ms", est.TTFTMS},
		{"itl_ms", est.ITLMS},
		{"e2e_latency_seconds", est.E2ELatencySeconds},
		{"output_throughput_tps", est.OutputThroughputTPS},
	}
	for _, f := range fields {
		if !finite(f.v) || f.v < 0 {
			return fmt.Errorf("%w: %s must be finite and non-negative", ErrSLOEstimateInvalid, f.name)
		}
	}
	return nil
}

// sloMetricValueMS returns the constrained metric of est in milliseconds.
func sloMetricValueMS(est PerformanceEstimate, metric SLOMetric) float64 {
	switch metric {
	case SLOMetricTTFT:
		return est.TTFTMS
	case SLOMetricITL:
		return est.ITLMS
	default:
		return est.E2ELatencySeconds * 1000
	}
}

// sloConstraintValueMS returns the constraint bound in milliseconds.
func sloConstraintValueMS(c SLOConstraint) float64 {
	if c.Unit == SLOUnitSeconds {
		return c.Value * 1000
	}
	return c.Value
}

func sloConstraintHolds(valueMS, boundMS float64, op SLOOperator) bool {
	switch op {
	case SLOOpLessThan:
		return valueMS < boundMS
	case SLOOpLessThanOrEqual:
		return valueMS <= boundMS
	case SLOOpGreaterThan:
		return valueMS > boundMS
	case SLOOpGreaterThanOrEqual:
		return valueMS >= boundMS
	case SLOOpEquals:
		return math.Abs(valueMS-boundMS) < sloEqualityEpsilonMS
	default: // SLOOpNotEquals
		return math.Abs(valueMS-boundMS) >= sloEqualityEpsilonMS
	}
}

func sloAllConstraintsHold(est PerformanceEstimate, constraints []SLOConstraint) bool {
	for _, c := range constraints {
		value := sloMetricValueMS(est, c.Metric) * sloStatAdjustmentFactor[c.StatType]
		if !sloConstraintHolds(value, sloConstraintValueMS(c), c.Operator) {
			return false
		}
	}
	return true
}

// SatisfiesSLO reports whether est meets every constraint. Each constraint
// compares its metric against the estimate after scaling by the stat-type
// adjustment factor and converting both sides to one unit.
func SatisfiesSLO(est PerformanceEstimate, constraints []SLOConstraint) (bool, error) {
	if err := validatePerformanceEstimate(est); err != nil {
		return false, err
	}
	for _, c := range constraints {
		if err := validateSLOConstraint(c); err != nil {
			return false, err
		}
	}
	return sloAllConstraintsHold(est, constraints), nil
}

// SLOLevelStatus is the disposition of one ladder level.
type SLOLevelStatus string

const (
	SLOLevelEvaluated        SLOLevelStatus = "evaluated"
	SLOLevelMemoryInfeasible SLOLevelStatus = "memory-infeasible"
	SLOLevelEstimatorError   SLOLevelStatus = "estimator-error"
)

// SLOLevelOutcome records what happened at one concurrency level.
type SLOLevelOutcome struct {
	Concurrency int                  `json:"concurrency"`
	Status      SLOLevelStatus       `json:"status"`
	Estimate    *PerformanceEstimate `json:"estimate,omitempty"`
	Feasible    bool                 `json:"feasible,omitempty"`
	Detail      string               `json:"detail,omitempty"`
}

// SLOFeasibilitySweepInput describes one sweep. Levels must ascend strictly.
// Estimate produces one estimate per level and may be theoretical or measured.
type SLOFeasibilitySweepInput struct {
	Levels      []int
	Estimate    func(concurrency int) (PerformanceEstimate, error)
	Constraints []SLOConstraint
}

// SLOFeasibilitySweepResult is the closed sweep record. Best is the feasible
// level with the highest output throughput; ties keep the lower concurrency.
type SLOFeasibilitySweepResult struct {
	Schema              string               `json:"schema"`
	Levels              []SLOLevelOutcome    `json:"levels"`
	Best                *PerformanceEstimate `json:"best,omitempty"`
	PrunedAtConcurrency int                  `json:"pruned_at_concurrency,omitempty"`
	Feasible            bool                 `json:"feasible"`
	Reason              string               `json:"reason"`
}

// SweepSLOFeasibility walks the concurrency ladder in ascending order:
// estimate each level, stop when resident memory becomes infeasible (higher
// levels need strictly more memory, so the ladder prunes there), evaluate the
// SLO constraints on every surviving level, and keep the feasible point with
// the highest output throughput. Estimator errors skip one level without
// aborting the walk.
func SweepSLOFeasibility(in SLOFeasibilitySweepInput) (SLOFeasibilitySweepResult, error) {
	result := SLOFeasibilitySweepResult{Schema: SLOFeasibilitySweepSchema, Levels: []SLOLevelOutcome{}}
	if in.Estimate == nil {
		return result, fmt.Errorf("%w: estimate function is required", ErrSLOLadderInvalid)
	}
	if len(in.Levels) == 0 {
		return result, fmt.Errorf("%w: at least one concurrency level is required", ErrSLOLadderInvalid)
	}
	for i, level := range in.Levels {
		if level <= 0 {
			return result, fmt.Errorf("%w: level %d must be positive", ErrSLOLadderInvalid, level)
		}
		if i > 0 && level <= in.Levels[i-1] {
			return result, fmt.Errorf("%w: levels must ascend strictly", ErrSLOLadderInvalid)
		}
	}
	for _, c := range in.Constraints {
		if err := validateSLOConstraint(c); err != nil {
			return result, err
		}
	}

	var best *PerformanceEstimate
	evaluated := 0
	for _, level := range in.Levels {
		est, err := in.Estimate(level)
		if err != nil {
			result.Levels = append(result.Levels, SLOLevelOutcome{Concurrency: level, Status: SLOLevelEstimatorError, Detail: err.Error()})
			continue
		}
		if est.Concurrency != level {
			result.Levels = append(result.Levels, SLOLevelOutcome{
				Concurrency: level, Status: SLOLevelEstimatorError,
				Detail: fmt.Sprintf("estimate reports concurrency %d, want %d", est.Concurrency, level),
			})
			continue
		}
		if err := validatePerformanceEstimate(est); err != nil {
			result.Levels = append(result.Levels, SLOLevelOutcome{Concurrency: level, Status: SLOLevelEstimatorError, Detail: err.Error()})
			continue
		}
		if est.MemoryInfeasible {
			result.PrunedAtConcurrency = level
			result.Levels = append(result.Levels, SLOLevelOutcome{Concurrency: level, Status: SLOLevelMemoryInfeasible})
			break
		}
		feasible := sloAllConstraintsHold(est, in.Constraints)
		evaluated++
		result.Levels = append(result.Levels, SLOLevelOutcome{Concurrency: level, Status: SLOLevelEvaluated, Estimate: &est, Feasible: feasible})
		if feasible && (best == nil || est.OutputThroughputTPS > best.OutputThroughputTPS) {
			chosen := est
			best = &chosen
		}
	}

	switch {
	case best != nil:
		result.Best = best
		result.Feasible = true
		result.Reason = fmt.Sprintf("highest output throughput among %d evaluated feasible level(s)", evaluated)
	case result.PrunedAtConcurrency > 0:
		result.Reason = fmt.Sprintf("no feasible level before memory pruning at concurrency %d", result.PrunedAtConcurrency)
	case evaluated == 0:
		result.Reason = "no concurrency level produced a usable estimate"
	default:
		result.Reason = "no concurrency level met the SLO constraints"
	}
	return result, nil
}
