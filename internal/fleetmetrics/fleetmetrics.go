// Package fleetmetrics is a pure duration-percentile fold over worker-session
// records. It exists so an operator can see the TAIL of worker-session
// duration (p95) beside the median (p50) before deciding to raise the worker
// count: a fleet whose p50 looks healthy can still be starved by a heavy tail,
// and only the tail tells you whether adding workers will help or just pile up
// behind the slow ones.
//
// # Percentile method
//
// This package uses the NEAREST-RANK method (the "C = 1" variant in the
// Wikipedia "Percentile" article). For a percentile p in [0, 100] over a
// slice of N durations sorted ascending:
//
//	rank = ceil( (p / 100) * N )          // 1-indexed
//	value = sorted[rank - 1]
//
// with rank clamped to [1, N]. Nearest-rank always returns a value that is
// actually present in the input (no interpolation), which makes hand-verifying
// a fixture trivial: pick the element at the computed rank and that is the
// answer. Edge cases:
//
//   - empty input   -> every percentile is 0.
//   - single value  -> that value for every percentile.
//   - p == 0         -> rank clamps to 1 (the minimum), so P0 == the min.
//
// Everything here is stdlib-only, imports nothing internal, and is off the hot
// path.
package fleetmetrics

import (
	"fmt"
	"math"
	"sort"
)

// Session is one worker-session record: which worker ran it and how long it
// took. DurationSec is the wall-clock duration of the session in seconds.
type Session struct {
	WorkerID    string
	DurationSec float64
}

// Percentiles computes, for each requested percentile p (expressed on a 0..100
// scale), the nearest-rank value over durations. The returned map is keyed by
// the requested percentile so a caller can ask for several in one pass:
//
//	m := Percentiles(ds, 50, 95)
//	p50, p95 := m[50], m[95]
//
// durations is not required to be sorted; Percentiles sorts a copy and leaves
// the caller's slice untouched. An empty durations slice yields 0 for every
// requested percentile. A percentile outside [0, 100] is clamped into range.
func Percentiles(durations []float64, ps ...float64) map[float64]float64 {
	out := make(map[float64]float64, len(ps))
	if len(durations) == 0 {
		for _, p := range ps {
			out[p] = 0
		}
		return out
	}

	sorted := make([]float64, len(durations))
	copy(sorted, durations)
	sort.Float64s(sorted)

	for _, p := range ps {
		out[p] = nearestRank(sorted, p)
	}
	return out
}

// nearestRank returns the nearest-rank percentile value from an ascending
// sorted slice. It assumes len(sorted) > 0.
func nearestRank(sorted []float64, p float64) float64 {
	// Clamp the percentile into [0, 100] so a stray 120 or -5 cannot index
	// out of bounds.
	if p < 0 {
		p = 0
	}
	if p > 100 {
		p = 100
	}
	n := len(sorted)
	// rank = ceil( (p/100) * n ), 1-indexed.
	rank := int(math.Ceil((p / 100) * float64(n)))
	if rank < 1 {
		rank = 1
	}
	if rank > n {
		rank = n
	}
	return sorted[rank-1]
}

// P50P95 is the convenience the progress output wants: the median and the 95th
// percentile of the session durations, in seconds. On empty input both are 0.
// On a single session both equal that session's duration.
func P50P95(sessions []Session) (p50, p95 float64) {
	ds := durations(sessions)
	m := Percentiles(ds, 50, 95)
	return m[50], m[95]
}

// durations projects the DurationSec field out of a slice of sessions.
func durations(sessions []Session) []float64 {
	ds := make([]float64, len(sessions))
	for i, s := range sessions {
		ds[i] = s.DurationSec
	}
	return ds
}

// Render formats p50 and p95 worker-session duration over a window for operator
// output. window is a human label for the selection (e.g. "last 100 sessions"
// or "24h"); n is how many sessions the percentiles were computed over. The
// line is a single operator-readable string, e.g.:
//
//	worker-session duration (last 100 sessions, n=100): p50=42.0s p95=310.5s
//
// An empty fleet (n == 0) is reported explicitly rather than as a misleading
// 0s/0s pair.
func Render(window string, sessions []Session) string {
	n := len(sessions)
	if n == 0 {
		return fmt.Sprintf("worker-session duration (%s, n=0): no sessions", window)
	}
	p50, p95 := P50P95(sessions)
	return fmt.Sprintf("worker-session duration (%s, n=%d): p50=%.1fs p95=%.1fs",
		window, n, p50, p95)
}
