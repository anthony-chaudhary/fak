// Time-weighted statistics over sweep-line step functions, ported DIRECT from
// ai-dynamo/aiperf src/aiperf/analysis/sweepline_stats.py:55 @ ab610add4e8876fca223197596c728e17365f381
// (aiperf 0.14.0, Apache-2.0), with the SweepLineStats shape, the ZERO stats
// value, and _step_lookup taken from src/aiperf/analysis/sweepline.py. The
// numpy vector kernels are re-expressed as plain loops over []float64
// (stdlib only); the searchsorted side conventions, the clipped-segment
// tiling of the window, and the duration-weighted percentile CDF are
// preserved exactly. Only the stats core is ported: the curve builders,
// MetricResult plumbing, and SWEEP_LINE_METRIC_SPECS map to aiperf reporting
// scaffolding fak does not have.
//
// A caller feeds step functions built from perfledger Record rows: sorted
// UnixMS timestamps as float64, each row carrying the value held until the
// next row — PrefillTPS/DecodeTPS rates sampled at request boundaries, or a
// mask curve such as decode-phase concurrency (records currently in decode).
// ComputeTimeWeightedStats averages a rate curve over the whole window;
// ComputeActiveWeightedStats restricts the same curve to segments where the
// mask is strictly positive, so the result reflects intensity while the phase
// is happening rather than diluted by idle gaps.
//
// NaN note: Avg/Std propagate a caller-supplied NaN exactly as upstream
// numpy does; Min/Max scan with < / > comparisons, so a NaN mid-curve is
// skipped unless it is the first value (upstream np.min/np.max propagate
// any NaN). Upstream scrubs at the MetricResult layer, which is out of
// scope here.
package perfledger

import (
	"math"
	"sort"
)

// SweepLineStats holds time-weighted statistics over a sweep-line step
// function (upstream aiperf SweepLineStats): a duration-weighted average and
// standard deviation plus duration-weighted percentiles of the segment
// values, all in the curve's own units.
type SweepLineStats struct {
	Avg, Min, Max, P50, P90, P95, P99, Std float64
}

// ZeroSweepLineStats is the all-zero result returned for degenerate inputs —
// an empty step function, a non-positive window, or no active segments —
// mirroring upstream ZERO_SWEEP_LINE_STATS.
var ZeroSweepLineStats = SweepLineStats{}

// searchsortedRight returns the first index i with a[i] > v (numpy
// searchsorted side="right"): the insertion point after any entries equal to
// v. a must be sorted ascending.
func searchsortedRight(a []float64, v float64) int {
	return sort.Search(len(a), func(i int) bool { return a[i] > v })
}

// searchsortedLeft returns the first index i with a[i] >= v (numpy
// searchsorted default side="left"). a must be sorted ascending.
func searchsortedLeft(a []float64, v float64) int {
	return sort.Search(len(a), func(i int) bool { return a[i] >= v })
}

// stepLookup resolves step-function values at query timestamps: for each
// query, the value held from the last event at or before it (searchsorted
// right, minus one), or 0 before the first event. Ported from sweepline.py
// _step_lookup.
func stepLookup(eventTS, eventVals, queryTS []float64) []float64 {
	out := make([]float64, len(queryTS))
	for i, q := range queryTS {
		idx := searchsortedRight(eventTS, q) - 1
		switch {
		case idx < 0 || len(eventVals) == 0:
			out[i] = 0 // the step function is 0 before its first event
		case idx >= len(eventVals):
			out[i] = eventVals[len(eventVals)-1] // upstream np.clip guard
		default:
			out[i] = eventVals[idx]
		}
	}
	return out
}

// buildClippedSegments slices the step function to [windowStart, windowEnd]
// and returns the positive-duration segments of the resulting tiling as
// matched (durations, values) pairs. Ported from sweepline_stats.py
// _build_clipped_segments.
func buildClippedSegments(sortedTS, values []float64, windowStart, windowEnd float64) (durations, segValues []float64) {
	lo := max(0, searchsortedRight(sortedTS, windowStart)-1)
	hi := min(len(sortedTS), searchsortedLeft(sortedTS, windowEnd)+1)

	n := hi - lo
	segStarts := make([]float64, n+1)
	allValues := make([]float64, n+1)
	segStarts[0] = windowStart
	if lo > 0 {
		allValues[0] = values[lo-1] // value held from the last pre-slice event
	}
	copy(segStarts[1:], sortedTS[lo:hi])
	copy(allValues[1:], values[lo:hi])

	segEnds := make([]float64, n+1)
	copy(segEnds, segStarts[1:])
	segEnds[n] = windowEnd

	durations = make([]float64, 0, n+1)
	segValues = make([]float64, 0, n+1)
	for i := range segStarts {
		start := max(segStarts[i], windowStart)
		end := min(segEnds[i], windowEnd)
		if d := end - start; d > 0 {
			durations = append(durations, d)
			segValues = append(segValues, allValues[i])
		}
	}
	return durations, segValues
}

// ComputeTimeWeightedStats computes time-weighted statistics over a
// step-function within a window.
//
// The sweep-line output defines a step function: values[i] is held from
// sortedTS[i] to sortedTS[i+1], and the value is 0 before the first event.
// This function clips the step function to [windowStart, windowEnd] and
// computes time-weighted stats: Avg and Std weight each segment's value by
// its duration in the window, Min/Max range over the in-window segment
// values, and the percentiles read off the duration-weighted CDF. Returns
// ZeroSweepLineStats for an empty step function or a non-positive window.
func ComputeTimeWeightedStats(sortedTS, values []float64, windowStart, windowEnd float64) SweepLineStats {
	totalDur := windowEnd - windowStart
	// A step function pairs one value per event; a mismatched pair is
	// degenerate (upstream numpy raises on the shape mismatch — here the
	// all-zero contract applies).
	if len(sortedTS) == 0 || len(values) != len(sortedTS) || totalDur <= 0 {
		return ZeroSweepLineStats
	}

	dur, val := buildClippedSegments(sortedTS, values, windowStart, windowEnd)
	if len(dur) == 0 {
		return ZeroSweepLineStats
	}
	return sweepLineStatsFromSegments(val, dur, totalDur)
}

// buildActiveWindowGrid merges the window edges with the in-window event
// timestamps of both curves into the ascending, deduplicated boundary grid
// the active-only sweep evaluates on (upstream np.unique of the
// concatenation). Ported from sweepline_stats.py _build_active_window_grid.
func buildActiveWindowGrid(rateTS, maskTS []float64, windowStart, windowEnd float64) []float64 {
	rateLo := searchsortedRight(rateTS, windowStart)
	rateHi := searchsortedLeft(rateTS, windowEnd)
	maskLo := searchsortedRight(maskTS, windowStart)
	maskHi := searchsortedLeft(maskTS, windowEnd)

	grid := make([]float64, 0, 2+(rateHi-rateLo)+(maskHi-maskLo))
	grid = append(grid, windowStart, windowEnd)
	if rateLo < rateHi {
		grid = append(grid, rateTS[rateLo:rateHi]...)
	}
	if maskLo < maskHi {
		grid = append(grid, maskTS[maskLo:maskHi]...)
	}
	sort.Float64s(grid)

	j := 1
	for i := 1; i < len(grid); i++ {
		if grid[i] != grid[j-1] {
			grid[j] = grid[i]
			j++
		}
	}
	return grid[:j]
}

// ComputeActiveWeightedStats computes time-weighted stats over a rate curve,
// restricted to the segments where a mask curve is strictly positive.
//
// Useful for "phase-aware" throughput metrics: e.g. average decode throughput
// restricted to time periods when at least one record is in decode. Inactive
// segments (mask <= 0) are excluded from the weighted average and from the
// duration-weighted percentile CDF, so the result reflects intensity while
// the phase is happening rather than averaged over the whole run window.
// rateTS/maskTS are sorted event timestamps whose rateVals/maskVals are held
// until the next event; returns ZeroSweepLineStats when no active segments
// overlap the window.
func ComputeActiveWeightedStats(rateTS, rateVals, maskTS, maskVals []float64, windowStart, windowEnd float64) SweepLineStats {
	totalDur := windowEnd - windowStart
	if totalDur <= 0 || len(rateTS) == 0 || len(rateVals) != len(rateTS) ||
		len(maskTS) == 0 || len(maskVals) != len(maskTS) {
		return ZeroSweepLineStats
	}

	// Window edges cover exact-boundary events; stepLookup resolves the value
	// held from any predecessor outside the window.
	grid := buildActiveWindowGrid(rateTS, maskTS, windowStart, windowEnd)
	if len(grid) < 2 {
		return ZeroSweepLineStats
	}

	segStarts := grid[:len(grid)-1]
	rateAt := stepLookup(rateTS, rateVals, segStarts)
	maskAt := stepLookup(maskTS, maskVals, segStarts)

	val := make([]float64, 0, len(segStarts))
	dur := make([]float64, 0, len(segStarts))
	for i, start := range segStarts {
		segDur := grid[i+1] - start
		if maskAt[i] > 0 && segDur > 0 {
			val = append(val, rateAt[i])
			dur = append(dur, segDur)
		}
	}
	if len(val) == 0 {
		return ZeroSweepLineStats
	}

	activeDur := 0.0
	for _, d := range dur {
		activeDur += d
	}
	if activeDur <= 0 {
		return ZeroSweepLineStats
	}
	return sweepLineStatsFromSegments(val, dur, activeDur)
}

// sweepLineStatsFromSegments finishes duration-weighted stats over matched
// (value, duration) segments: Avg and Std normalized by normDur, and
// percentiles read off the duration-weighted CDF. Upstream duplicates this
// block in compute_time_weighted_stats and compute_active_weighted_stats; it
// is factored here with identical semantics — normDur is the window length
// for the time-weighted variant and the active duration for the active-only
// variant.
func sweepLineStatsFromSegments(vals, durs []float64, normDur float64) SweepLineStats {
	weighted := 0.0
	mn := vals[0]
	mx := vals[0]
	for i, v := range vals {
		weighted += durs[i] * v
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	avg := weighted / normDur

	residual := 0.0
	for i, v := range vals {
		d := v - avg
		residual += durs[i] * d * d
	}
	std := math.Sqrt(residual / normDur)

	// Duration-weighted percentile CDF: order values ascending (carrying
	// durations), walk the cumulative-duration fractions, and take the value
	// at the first fraction reaching each quantile (searchsorted left).
	order := make([]int, len(vals))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return vals[order[a]] < vals[order[b]] })
	sortedVals := make([]float64, len(vals))
	sortedDurs := make([]float64, len(vals))
	for i, o := range order {
		sortedVals[i] = vals[o]
		sortedDurs[i] = durs[o]
	}

	cumDur := make([]float64, len(vals))
	run := 0.0
	for i, d := range sortedDurs {
		run += d
		cumDur[i] = run
	}
	cumFrac := make([]float64, len(vals))
	for i := range cumDur {
		cumFrac[i] = cumDur[i] / cumDur[len(cumDur)-1]
	}

	percentile := func(q float64) float64 {
		idx := min(searchsortedLeft(cumFrac, q), len(sortedVals)-1)
		return sortedVals[idx]
	}
	return SweepLineStats{
		Avg: avg,
		Min: mn,
		Max: mx,
		P50: percentile(0.50),
		P90: percentile(0.90),
		P95: percentile(0.95),
		P99: percentile(0.99),
		Std: std,
	}
}
