// SLO constraint mini-grammar: parse and evaluate latency SLO constraints
// written as "metric[:stat]<op><value>[unit]".
//
// Adapted from bentoml/llm-optimizer, src/llm_optimizer/performance.py
// (SLOConstraint + parse_slo_constraints + the constraint check inside
// estimate_performance_under_constraints), pinned at
// 7be87d776e60a978516a2ae53e4a4ff745d04c20 (Apache-2.0). Re-expressed in
// idiomatic Go; no upstream code is copied. Study note:
// docs/research/STUDY-PRODUCTION-SCALING-ECON-2026-11-07.md, candidate
// llm-optimizer-1.
package benchscore

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// Sentinel errors for the SLO constraint grammar. Callers match these with
// errors.Is, never error prose.
var (
	// ErrSLONoOperator reports a constraint part without a recognized operator.
	ErrSLONoOperator = errors.New("slo: no operator")
	// ErrSLOEmptyMetric reports a constraint whose metric name is empty.
	ErrSLOEmptyMetric = errors.New("slo: empty metric")
	// ErrSLOUnknownStat reports a :stat qualifier outside mean|median|p95|p99.
	ErrSLOUnknownStat = errors.New("slo: unknown stat")
	// ErrSLOInvalidValue reports a constraint value that is not a finite number.
	ErrSLOInvalidValue = errors.New("slo: invalid value")
	// ErrSLOMissingSamples reports evaluating a constraint whose metric has no samples.
	ErrSLOMissingSamples = errors.New("slo: missing samples")
)

// SLOConstraint is one parsed SLO constraint: metric[:stat]<op><value>[unit].
type SLOConstraint struct {
	Metric string // e.g. "ttft", "itl", "e2e_latency"
	Stat   string // "mean" (default), "median", "p95", or "p99"
	Op     string // "<", "<=", ">", ">=", "==", or "!="
	Value  float64
	Unit   string // "ms" or "s"
}

// SLOVerdict is the outcome of evaluating one constraint against observed samples.
type SLOVerdict struct {
	Constraint SLOConstraint
	Observed   float64 // expressed in the constraint's unit
	Pass       bool
}

var sloOperators = []string{"<=", ">=", "==", "!=", "<", ">"}

// ParseSLOConstraints parses a semicolon-separated list of SLO constraints in
// the mini-grammar metric[:stat]<op><value>[unit].
//
// Closed contract:
//   - Constraints are separated by ';'; surrounding whitespace is ignored and
//     empty parts are skipped. Empty input yields zero constraints, no error.
//   - The operator is the earliest occurrence of <=, >=, ==, !=, <, or >; at
//     equal positions the two-character operator wins.
//   - An optional ":stat" qualifier selects median, p95, or p99 (case
//     insensitive); the default stat is mean.
//   - The value is a finite decimal number followed by an optional unit
//     suffix "ms" or "s"; a bare number defaults to milliseconds.
func ParseSLOConstraints(s string) ([]SLOConstraint, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	var out []SLOConstraint
	for _, part := range strings.Split(s, ";") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		c, err := parseSLOPart(part)
		if err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, nil
}

func parseSLOPart(part string) (SLOConstraint, error) {
	op, pos := findSLOOperator(part)
	if op == "" {
		return SLOConstraint{}, fmt.Errorf("%w: %q", ErrSLONoOperator, part)
	}
	metricPart := strings.TrimSpace(part[:pos])
	valuePart := strings.TrimSpace(part[pos+len(op):])

	metric := metricPart
	stat := "mean"
	if i := strings.Index(metricPart, ":"); i >= 0 {
		metric = strings.TrimSpace(metricPart[:i])
		stat = strings.ToLower(strings.TrimSpace(metricPart[i+1:]))
	}
	if metric == "" {
		return SLOConstraint{}, fmt.Errorf("%w: %q", ErrSLOEmptyMetric, part)
	}
	switch stat {
	case "mean", "median", "p95", "p99":
	default:
		return SLOConstraint{}, fmt.Errorf("%w %q: want mean, median, p95, or p99", ErrSLOUnknownStat, stat)
	}

	unit := "ms"
	numeral := valuePart
	switch {
	case strings.HasSuffix(valuePart, "ms"):
		numeral = valuePart[:len(valuePart)-2]
	case strings.HasSuffix(valuePart, "s"):
		unit = "s"
		numeral = valuePart[:len(valuePart)-1]
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(numeral), 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		return SLOConstraint{}, fmt.Errorf("%w: value %q in %q", ErrSLOInvalidValue, valuePart, part)
	}
	return SLOConstraint{Metric: metric, Stat: stat, Op: op, Value: v, Unit: unit}, nil
}

// findSLOOperator returns the operator at its earliest position; ties prefer
// the longer operator so "<=" is never read as "<" plus a stray "=".
func findSLOOperator(part string) (string, int) {
	best, bestPos := "", -1
	for _, op := range sloOperators {
		i := strings.Index(part, op)
		if i < 0 {
			continue
		}
		if bestPos == -1 || i < bestPos || (i == bestPos && len(op) > len(best)) {
			best, bestPos = op, i
		}
	}
	return best, bestPos
}

// EvaluateSLO evaluates each constraint against per-metric sample values
// recorded in milliseconds (the canonical latency unit). It returns one
// verdict per constraint, in input order.
func EvaluateSLO(constraints []SLOConstraint, samples map[string][]float64) ([]SLOVerdict, error) {
	verdicts := make([]SLOVerdict, 0, len(constraints))
	for _, c := range constraints {
		raw, ok := samples[c.Metric]
		if !ok || len(raw) == 0 {
			return nil, fmt.Errorf("%w: metric %q", ErrSLOMissingSamples, c.Metric)
		}
		ms := sloStat(raw, c.Stat)
		observed := ms
		if c.Unit == "s" {
			observed = ms / 1000
		}
		verdicts = append(verdicts, SLOVerdict{Constraint: c, Observed: observed, Pass: sloCompare(observed, c.Op, c.Value)})
	}
	return verdicts, nil
}

// SLOFeasible reports whether every constraint holds against the samples.
func SLOFeasible(constraints []SLOConstraint, samples map[string][]float64) (bool, error) {
	verdicts, err := EvaluateSLO(constraints, samples)
	if err != nil {
		return false, err
	}
	for _, v := range verdicts {
		if !v.Pass {
			return false, nil
		}
	}
	return true, nil
}

// sloStat computes the named statistic over raw millisecond samples.
// Percentiles use the nearest-rank definition: p95 of n sorted samples is
// the ceil(0.95*n)-th smallest value.
func sloStat(samples []float64, stat string) float64 {
	switch stat {
	case "mean":
		sum := 0.0
		for _, v := range samples {
			sum += v
		}
		return sum / float64(len(samples))
	default:
		sorted := append([]float64(nil), samples...)
		sort.Float64s(sorted)
		n := len(sorted)
		switch stat {
		case "median":
			if n%2 == 1 {
				return sorted[n/2]
			}
			return (sorted[n/2-1] + sorted[n/2]) / 2
		case "p95":
			return sorted[sloRank(n, 95)-1]
		case "p99":
			return sorted[sloRank(n, 99)-1]
		}
	}
	panic("unreachable: stat validated by parseSLOPart")
}

// sloRank is the nearest-rank percentile index (1-based, clamped to [1,n]).
func sloRank(n, p int) int {
	r := (p*n + 99) / 100 // ceil(p*n/100) in integer arithmetic
	if r < 1 {
		r = 1
	}
	if r > n {
		r = n
	}
	return r
}

// sloCompare applies the constraint operator. Equality operators compare
// exactly; callers pin exact values.
func sloCompare(observed float64, op string, threshold float64) bool {
	switch op {
	case "<":
		return observed < threshold
	case "<=":
		return observed <= threshold
	case ">":
		return observed > threshold
	case ">=":
		return observed >= threshold
	case "==":
		return observed == threshold
	case "!=":
		return observed != threshold
	}
	return false
}
