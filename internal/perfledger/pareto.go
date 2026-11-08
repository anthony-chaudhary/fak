// Adapted from bentoml/llm-optimizer, src/llm_optimizer/visualization/template.html
// (getParetoPoints at :79, calculateBestConfig at :191) at
// 7be87d776e60a978516a2ae53e4a4ff745d04c20. Apache-2.0; re-expressed in idiomatic
// Go over typed (latency, throughput) config points instead of dashboard rows.
// The dominance rule keeps equal points on the front, as upstream does; best-config
// selection adds deterministic tie-breaks the dashboard left to chart order.
package perfledger

import (
	"errors"
	"math"
	"sort"
)

// ConfigPoint is one measured serving configuration in the two-axis trade-off
// space: request latency (lower is better) against aggregate throughput
// (higher is better). Config names the point's stable identity (an args digest
// or sweep label) so ties resolve deterministically.
type ConfigPoint struct {
	Config     string
	LatencyMS  float64
	Throughput float64 // tokens/s
}

// valid reports whether the point carries measurable axes. Latency must be
// finite and positive: zero means unobserved, matching Record's convention.
// Throughput must be finite and non-negative: a deadlocked config measures
// zero and stays a legal worst point. Negative or non-finite values are
// unmeasured and never become frontier points.
func (p ConfigPoint) valid() bool {
	if p.LatencyMS <= 0 || math.IsNaN(p.LatencyMS) || math.IsInf(p.LatencyMS, 0) {
		return false
	}
	return !math.IsNaN(p.Throughput) && !math.IsInf(p.Throughput, 0) && p.Throughput >= 0
}

// dominates reports whether b beats a: at least as good on both axes and
// strictly better on one. Equal points dominate neither, so duplicates both
// stay on the front.
func (b ConfigPoint) dominates(a ConfigPoint) bool {
	return b.LatencyMS <= a.LatencyMS && b.Throughput >= a.Throughput &&
		(b.LatencyMS < a.LatencyMS || b.Throughput > a.Throughput)
}

// ParetoFrontier returns the non-dominated subset of points: the configs no
// other config beats on both axes at once. Invalid points are dropped. The
// result is sorted latency ascending, then throughput descending, then
// Config ascending, so repeated calls on equal input agree and the front
// reads best-latency first.
func ParetoFrontier(points []ConfigPoint) []ConfigPoint {
	valid := make([]ConfigPoint, 0, len(points))
	for _, p := range points {
		if p.valid() {
			valid = append(valid, p)
		}
	}
	front := make([]ConfigPoint, 0, len(valid))
	for i := range valid {
		dominated := false
		for j := range valid {
			if i == j {
				continue
			}
			if valid[j].dominates(valid[i]) {
				dominated = true
				break
			}
		}
		if !dominated {
			front = append(front, valid[i])
		}
	}
	sort.SliceStable(front, func(i, j int) bool {
		a, b := front[i], front[j]
		if a.LatencyMS != b.LatencyMS {
			return a.LatencyMS < b.LatencyMS
		}
		if a.Throughput != b.Throughput {
			return a.Throughput > b.Throughput
		}
		return a.Config < b.Config
	})
	return front
}

// ErrNoFeasibleConfig is returned by BestConfigUnderConstraints when no valid
// point satisfies the bounds.
var ErrNoFeasibleConfig = errors.New("perfledger: no feasible config")

// BestConfigUnderConstraints returns the feasible point with the highest
// throughput. A point is feasible when its latency is at most *maxLatencyMS
// (when given) and its throughput at least *minThroughput (when given); a nil
// bound leaves that axis unconstrained. Bounds are inclusive. Ties break
// toward lower latency, then lexicographically smaller Config. It returns
// ErrNoFeasibleConfig when no valid point satisfies the bounds.
func BestConfigUnderConstraints(points []ConfigPoint, maxLatencyMS, minThroughput *float64) (ConfigPoint, error) {
	var best ConfigPoint
	haveBest := false
	for _, p := range points {
		if !p.valid() {
			continue
		}
		if maxLatencyMS != nil && p.LatencyMS > *maxLatencyMS {
			continue
		}
		if minThroughput != nil && p.Throughput < *minThroughput {
			continue
		}
		if !haveBest || betterConfig(p, best) {
			best, haveBest = p, true
		}
	}
	if !haveBest {
		return ConfigPoint{}, ErrNoFeasibleConfig
	}
	return best, nil
}

// betterConfig reports whether a beats b for best-config selection: higher
// throughput, else lower latency, else lexicographically smaller Config.
func betterConfig(a, b ConfigPoint) bool {
	if a.Throughput != b.Throughput {
		return a.Throughput > b.Throughput
	}
	if a.LatencyMS != b.LatencyMS {
		return a.LatencyMS < b.LatencyMS
	}
	return a.Config < b.Config
}
