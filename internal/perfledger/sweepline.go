// Adapted from ai-dynamo/aiperf, src/aiperf/analysis/sweepline.py
// (+ sweepline_kv_cache.py) at ab610add4e8876fca223197596c728e17365f381.
// Copyright 2025-2026 NVIDIA CORPORATION & AFFILIATES. Apache-2.0.
// Changed for Fak: re-expressed over Go []float64 slices (epoch-nanosecond
// timestamps) with a stdlib stable sort instead of numpy lexsort; the
// MetricResult/SweepLineStats plumbing and the SSE ICL variants are omitted.
//
// The sweep-line seam turns per-request wall-clock records into vectorized
// step-function curves — concurrency, decode/prefill/total throughput,
// per-user throughput, and KV-cache tokens in flight — each returned as a
// (sorted timestamps, values) pair with one entry per event. Every input
// column is an equal-length []float64 over epoch nanoseconds with NaN marking
// a missing record, exactly as in the source's session_num-indexed numpy
// columns; mismatched column lengths panic on index where numpy would raise
// a broadcast error. Throughput curves are in tokens/ns (the source's
// NANOS_PER_SECOND spec scale converts to tokens/sec at render time).
package perfledger

import (
	"math"
	"sort"
)

// sweepLineCumsum sorts (timestamp, delta) events by timestamp — ends before
// starts at ties, mirroring numpy lexsort on (event_type, timestamp) where
// event_type is 0 for non-positive deltas (ends) and 1 for positive ones
// (starts) — then cumsums the deltas in that order and snaps floating-point
// roundoff to zero. The sort is stable, so events equal on both keys keep
// their input order, as numpy's indirect stable lexsort does.
//
// The snap matters: all sweep curves represent physically non-negative
// quantities (concurrency, tokens, throughput), and imperfect cancellation of
// large +/- pairs leaves residuals of relative size ~1e-12 at peak magnitudes
// that render as "-0.00" in formatted output. A real ordering bug would
// produce a magnitude orders larger than the 1e-9*max|v| threshold and remain
// visible. Timestamps are epoch nanoseconds.
func sweepLineCumsum(timestamps, deltas []float64) ([]float64, []float64) {
	n := len(timestamps)
	if n == 0 {
		return []float64{}, []float64{}
	}
	types := make([]int8, n) // 0 = end (delta <= 0), 1 = start (delta > 0)
	for i, d := range deltas {
		if d > 0 {
			types[i] = 1
		}
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		ia, ib := order[a], order[b]
		if timestamps[ia] != timestamps[ib] {
			return timestamps[ia] < timestamps[ib]
		}
		return types[ia] < types[ib]
	})
	vals := make([]float64, n)
	run := 0.0
	for i, oi := range order {
		run += deltas[oi]
		vals[i] = run
	}
	maxAbs := 0.0
	for _, v := range vals {
		if a := math.Abs(v); a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs > 0 {
		for i, v := range vals {
			if math.Abs(v) < 1e-9*maxAbs {
				vals[i] = 0
			}
		}
	}
	sortedTS := make([]float64, n)
	for i, oi := range order {
		sortedTS[i] = timestamps[oi]
	}
	return sortedTS, vals
}

// sweepLineStepLookup evaluates a step function at query timestamps: each
// query gets the value of the last event at or before it (numpy searchsorted
// side="right", minus one) and 0 before the first event. eventTS must be
// sorted ascending; timestamps are epoch nanoseconds. Named with the
// sweepLine prefix because the package already carries a sibling stepLookup
// from the time-weighted stats port.
func sweepLineStepLookup(eventTS, eventVals, queryTS []float64) []float64 {
	out := make([]float64, len(queryTS))
	for i, q := range queryTS {
		hi := sort.Search(len(eventTS), func(j int) bool { return eventTS[j] > q })
		if hi == 0 {
			continue // before the first event: value 0
		}
		out[i] = eventVals[hi-1]
	}
	return out
}

// copySlice returns an independent copy of s, mirroring numpy's .copy() so
// returned curves never alias caller memory; an empty or nil input yields an
// empty (len 0) slice.
func copySlice(s []float64) []float64 {
	out := make([]float64, len(s))
	copy(out, s)
	return out
}

// mergeUniqueTimestamps returns the sorted unique union of two timestamp
// slices — numpy np.unique(np.concatenate([a, b])). Timestamps are epoch
// nanoseconds.
func mergeUniqueTimestamps(a, b []float64) []float64 {
	merged := make([]float64, 0, len(a)+len(b))
	merged = append(merged, a...)
	merged = append(merged, b...)
	if len(merged) == 0 {
		return merged
	}
	sort.Float64s(merged)
	j := 1
	for i := 1; i < len(merged); i++ {
		if merged[i] != merged[j-1] {
			merged[j] = merged[i]
			j++
		}
	}
	return merged[:j]
}

// AddStepFunctions adds two step functions, returning a new step function on
// the merged unique timestamps of both. Each input is a sorted (timestamps,
// values) pair over epoch nanoseconds; at every merged timestamp each side
// contributes the value of its last event at or before it (0 before that
// side's first event). Empty inputs return a copy of the other side, so the
// result never aliases caller memory.
func AddStepFunctions(aTS, aVals, bTS, bVals []float64) ([]float64, []float64) {
	if len(aTS) == 0 {
		return copySlice(bTS), copySlice(bVals)
	}
	if len(bTS) == 0 {
		return copySlice(aTS), copySlice(aVals)
	}
	merged := mergeUniqueTimestamps(aTS, bTS)
	aAt := sweepLineStepLookup(aTS, aVals, merged)
	bAt := sweepLineStepLookup(bTS, bVals, merged)
	sum := make([]float64, len(merged))
	for i := range sum {
		sum[i] = aAt[i] + bAt[i]
	}
	return merged, sum
}

// DivideStepFunctions divides two step functions, returning a new step
// function on the merged unique timestamps of both. Division is safe: where
// the denominator's value is <= 0 — including NaN, which fails the > 0 test —
// the quotient is 0. If either input is empty the result is two empty slices.
// Timestamps are epoch nanoseconds; with a numerator in tokens/ns and a
// denominator in requests the quotient is in tokens/ns/user.
func DivideStepFunctions(numTS, numVals, denTS, denVals []float64) ([]float64, []float64) {
	if len(numTS) == 0 || len(denTS) == 0 {
		return []float64{}, []float64{}
	}
	merged := mergeUniqueTimestamps(numTS, denTS)
	numAt := sweepLineStepLookup(numTS, numVals, merged)
	denAt := sweepLineStepLookup(denTS, denVals, merged)
	out := make([]float64, len(merged))
	for i := range out {
		if denAt[i] > 0 {
			out[i] = numAt[i] / denAt[i]
		}
	}
	return merged, out
}

// ThroughputPerUserSweepLine divides an aggregate decode-throughput curve by
// the generation-phase concurrency curve (concurrency over
// [generationStartNS, endNS)), yielding tokens/ns/user. tputTS/tputVals are
// the sorted output of ThroughputSweepLine (or an ICL variant upstream);
// generationStartNS/endNS are per-record first-token and end wall-clocks in
// epoch nanoseconds, NaN for missing records. Idle gaps with zero concurrency
// force the quotient to 0 through the safe division.
func ThroughputPerUserSweepLine(generationStartNS, endNS, tputTS, tputVals []float64) ([]float64, []float64) {
	concTS, concVals := ConcurrencySweepLine(generationStartNS, endNS)
	return DivideStepFunctions(tputTS, tputVals, concTS, concVals)
}

// PrefillThroughputPerUserSweepLine divides an aggregate prefill-throughput
// curve by the prefill-phase concurrency curve (concurrency over
// [startNS, generationStartNS)), yielding tokens/ns/user. ptputTS/ptputVals
// are the sorted output of PrefillThroughputSweepLine; startNS and
// generationStartNS are per-record request-start and first-token wall-clocks
// in epoch nanoseconds, NaN for missing records. Idle gaps with zero
// concurrency force the quotient to 0 through the safe division.
func PrefillThroughputPerUserSweepLine(startNS, generationStartNS, ptputTS, ptputVals []float64) ([]float64, []float64) {
	concTS, concVals := ConcurrencySweepLine(startNS, generationStartNS)
	return DivideStepFunctions(ptputTS, ptputVals, concTS, concVals)
}

// ConcurrencySweepLine computes exact instantaneous concurrency at every
// event boundary: each valid record contributes +1 at startNS and -1 at
// endNS, and the sorted cumsum is the number of requests in flight. A record
// is valid when both startNS and endNS are non-NaN. Returns (sorted
// timestamps, values) with 2K events for K valid records, or two empty
// slices when no record is valid; timestamps are epoch nanoseconds.
func ConcurrencySweepLine(startNS, endNS []float64) ([]float64, []float64) {
	n := len(startNS)
	valid := make([]bool, n)
	k := 0
	for i := 0; i < n; i++ {
		valid[i] = !math.IsNaN(startNS[i]) && !math.IsNaN(endNS[i])
		if valid[i] {
			k++
		}
	}
	if k == 0 {
		return []float64{}, []float64{}
	}
	timestamps := make([]float64, 0, 2*k)
	deltas := make([]float64, 0, 2*k)
	for i := 0; i < n; i++ {
		if valid[i] {
			timestamps = append(timestamps, startNS[i])
			deltas = append(deltas, 1)
		}
	}
	for i := 0; i < n; i++ {
		if valid[i] {
			timestamps = append(timestamps, endNS[i])
			deltas = append(deltas, -1)
		}
	}
	return sweepLineCumsum(timestamps, deltas)
}

// WeightedConcurrencySweepLine is the concurrency sweep where each interval
// contributes weights[i] instead of 1: the step function is the sum of every
// active interval's weight at each event boundary — e.g. tokens in flight
// when weights holds the per-request token counts held over
// [startNS, endNS). A record is valid when startNS, endNS, and weights are
// all non-NaN. Returns (sorted timestamps, values), or two empty slices when
// no record is valid; timestamps are epoch nanoseconds.
func WeightedConcurrencySweepLine(startNS, endNS, weights []float64) ([]float64, []float64) {
	n := len(startNS)
	valid := make([]bool, n)
	k := 0
	for i := 0; i < n; i++ {
		valid[i] = !math.IsNaN(startNS[i]) && !math.IsNaN(endNS[i]) && !math.IsNaN(weights[i])
		if valid[i] {
			k++
		}
	}
	if k == 0 {
		return []float64{}, []float64{}
	}
	timestamps := make([]float64, 0, 2*k)
	deltas := make([]float64, 0, 2*k)
	for i := 0; i < n; i++ {
		if valid[i] {
			timestamps = append(timestamps, startNS[i])
			deltas = append(deltas, weights[i])
		}
	}
	for i := 0; i < n; i++ {
		if valid[i] {
			timestamps = append(timestamps, endNS[i])
			deltas = append(deltas, -weights[i])
		}
	}
	return sweepLineCumsum(timestamps, deltas)
}

// ThroughputSweepLine computes exact instantaneous decode throughput
// (tokens/ns) at every event boundary with the uniform per-request rate
// (output_tokens - 1) / generation_duration: the -1 discounts the first
// token, which arrives instantaneously at generationStartNS and cannot be
// modeled as a rate over the generation interval. A record is valid when
// generationStartNS and outputTokens are non-NaN, endNS - generationStartNS
// > 0, and outputTokens >= 1. Returns (sorted timestamps, values) in
// tokens/ns over epoch-nanosecond timestamps, or two empty slices when no
// record is valid.
func ThroughputSweepLine(generationStartNS, endNS, outputTokens []float64) ([]float64, []float64) {
	n := len(generationStartNS)
	valid := make([]bool, n)
	rates := make([]float64, n)
	k := 0
	for i := 0; i < n; i++ {
		genDur := endNS[i] - generationStartNS[i]
		valid[i] = !math.IsNaN(generationStartNS[i]) && !math.IsNaN(outputTokens[i]) &&
			genDur > 0 && outputTokens[i] >= 1
		if valid[i] {
			rates[i] = (outputTokens[i] - 1) / genDur
			k++
		}
	}
	if k == 0 {
		return []float64{}, []float64{}
	}
	timestamps := make([]float64, 0, 2*k)
	deltas := make([]float64, 0, 2*k)
	for i := 0; i < n; i++ {
		if valid[i] {
			timestamps = append(timestamps, generationStartNS[i])
			deltas = append(deltas, rates[i])
		}
	}
	for i := 0; i < n; i++ {
		if valid[i] {
			timestamps = append(timestamps, endNS[i])
			deltas = append(deltas, -rates[i])
		}
	}
	return sweepLineCumsum(timestamps, deltas)
}

// PrefillThroughputSweepLine computes exact instantaneous prefill throughput
// (tokens/ns) at every event boundary. During prefill
// [startNS, generationStartNS) the model processes inputTokens tokens, so
// the per-request prefill rate is input_tokens / prefill_duration. A record
// is valid when startNS, generationStartNS, and inputTokens are non-NaN and
// generationStartNS - startNS > 0. Returns (sorted timestamps, values) in
// tokens/ns over epoch-nanosecond timestamps, or two empty slices when no
// record is valid.
func PrefillThroughputSweepLine(startNS, generationStartNS, inputTokens []float64) ([]float64, []float64) {
	n := len(startNS)
	valid := make([]bool, n)
	rates := make([]float64, n)
	k := 0
	for i := 0; i < n; i++ {
		prefillDur := generationStartNS[i] - startNS[i]
		valid[i] = !math.IsNaN(startNS[i]) && !math.IsNaN(generationStartNS[i]) &&
			!math.IsNaN(inputTokens[i]) && prefillDur > 0
		if valid[i] {
			rates[i] = inputTokens[i] / prefillDur
			k++
		}
	}
	if k == 0 {
		return []float64{}, []float64{}
	}
	timestamps := make([]float64, 0, 2*k)
	deltas := make([]float64, 0, 2*k)
	for i := 0; i < n; i++ {
		if valid[i] {
			timestamps = append(timestamps, startNS[i])
			deltas = append(deltas, rates[i])
		}
	}
	for i := 0; i < n; i++ {
		if valid[i] {
			timestamps = append(timestamps, generationStartNS[i])
			deltas = append(deltas, -rates[i])
		}
	}
	return sweepLineCumsum(timestamps, deltas)
}

// TotalThroughputSweepLine computes total throughput (prefill + generation)
// in a single sweep pass: prefill rate events over [startNS, generationStartNS)
// and generation rate events over [generationStartNS, endNS) combined into
// one sorted cumsum, avoiding two separate sweeps plus a grid merge and
// searchsorted lookups. Prefill validity: startNS, generationStartNS, and
// inputTokens non-NaN with generationStartNS - startNS > 0. Generation
// validity: generationStartNS and outputTokens non-NaN with
// endNS - generationStartNS > 0 and outputTokens >= 1. Returns (sorted
// timestamps, values) in tokens/ns over epoch-nanosecond timestamps, or two
// empty slices when no record is valid for either phase.
func TotalThroughputSweepLine(startNS, generationStartNS, endNS, inputTokens, outputTokens []float64) ([]float64, []float64) {
	n := len(startNS)
	pfValid := make([]bool, n)
	pfRates := make([]float64, n)
	gnValid := make([]bool, n)
	gnRates := make([]float64, n)
	pfK, gnK := 0, 0
	for i := 0; i < n; i++ {
		// Prefill: input_tokens / prefill_duration during [start, gen_start).
		prefillDur := generationStartNS[i] - startNS[i]
		pfValid[i] = !math.IsNaN(startNS[i]) && !math.IsNaN(generationStartNS[i]) &&
			!math.IsNaN(inputTokens[i]) && prefillDur > 0
		if pfValid[i] {
			pfRates[i] = inputTokens[i] / prefillDur
			pfK++
		}
		// Generation: (output_tokens - 1) / gen_duration during [gen_start, end).
		genDur := endNS[i] - generationStartNS[i]
		gnValid[i] = !math.IsNaN(generationStartNS[i]) && !math.IsNaN(outputTokens[i]) &&
			genDur > 0 && outputTokens[i] >= 1
		if gnValid[i] {
			gnRates[i] = (outputTokens[i] - 1) / genDur
			gnK++
		}
	}
	if pfK == 0 && gnK == 0 {
		return []float64{}, []float64{}
	}
	timestamps := make([]float64, 0, 2*(pfK+gnK))
	deltas := make([]float64, 0, 2*(pfK+gnK))
	for i := 0; i < n; i++ {
		if pfValid[i] {
			timestamps = append(timestamps, startNS[i])
			deltas = append(deltas, pfRates[i])
		}
	}
	for i := 0; i < n; i++ {
		if pfValid[i] {
			timestamps = append(timestamps, generationStartNS[i])
			deltas = append(deltas, -pfRates[i])
		}
	}
	for i := 0; i < n; i++ {
		if gnValid[i] {
			timestamps = append(timestamps, generationStartNS[i])
			deltas = append(deltas, gnRates[i])
		}
	}
	for i := 0; i < n; i++ {
		if gnValid[i] {
			timestamps = append(timestamps, endNS[i])
			deltas = append(deltas, -gnRates[i])
		}
	}
	return sweepLineCumsum(timestamps, deltas)
}

// kvCacheEvents collects the (timestamp, token-delta) events for input +
// output tokens held in KV cache per request, appended in the source's part
// order (prefill additions, generation additions, then end frees decomposed
// by which phases fired) so that stable-sort tie behavior matches the
// vectorized concatenation order exactly:
//
//   - +input_tokens at start_ns, only when generation starts strictly after
//     the request start (a distinct prefill phase actually ran);
//   - +output_tokens at generation_start_ns, when the generation duration
//     end_ns - generation_start_ns is > 0;
//   - free at end_ns: -(input + output) when both phases fired, -input when
//     only the input phase fired, -output when only the generation phase
//     fired.
//
// Timestamps are epoch nanoseconds; NaN marks a missing record.
func kvCacheEvents(startNS, generationStartNS, endNS, inputTokens, outputTokens []float64) ([]float64, []float64) {
	n := len(startNS)
	pfValid := make([]bool, n)
	hasGen := make([]bool, n)
	hasEnd := make([]bool, n)
	for i := 0; i < n; i++ {
		hasStart := !math.IsNaN(startNS[i]) && !math.IsNaN(inputTokens[i])
		genDur := endNS[i] - generationStartNS[i]
		hasGen[i] = !math.IsNaN(generationStartNS[i]) && !math.IsNaN(outputTokens[i]) && genDur > 0
		hasEnd[i] = !math.IsNaN(endNS[i])
		pfValid[i] = hasStart && !math.IsNaN(generationStartNS[i]) && generationStartNS[i] > startNS[i]
	}

	ts := make([]float64, 0, 3*n)
	deltas := make([]float64, 0, 3*n)

	// Event 1: +input_tokens at start_ns (prefill begins).
	for i := 0; i < n; i++ {
		if pfValid[i] {
			ts = append(ts, startNS[i])
			deltas = append(deltas, inputTokens[i])
		}
	}
	// Event 2: +output_tokens at generation_start_ns.
	for i := 0; i < n; i++ {
		if hasGen[i] {
			ts = append(ts, generationStartNS[i])
			deltas = append(deltas, outputTokens[i])
		}
	}
	// Event 3: free tokens at end_ns, with the source's exact
	// both/input_only/gen_only branch decomposition.
	for i := 0; i < n; i++ {
		if pfValid[i] && hasEnd[i] && hasGen[i] {
			ts = append(ts, endNS[i])
			deltas = append(deltas, -(inputTokens[i] + outputTokens[i]))
		}
	}
	for i := 0; i < n; i++ {
		if pfValid[i] && hasEnd[i] && !hasGen[i] {
			ts = append(ts, endNS[i])
			deltas = append(deltas, -inputTokens[i])
		}
	}
	for i := 0; i < n; i++ {
		if hasGen[i] && hasEnd[i] && !pfValid[i] {
			ts = append(ts, endNS[i])
			deltas = append(deltas, -outputTokens[i])
		}
	}
	return ts, deltas
}

// TokensInFlightSweepLine computes instantaneous KV-cache token load at every
// event boundary. It models the total tokens held in server memory per
// request: input_tokens during prefill [startNS, generationStartNS), then
// input + output during generation [generationStartNS, endNS) — input tokens
// stay in the KV cache throughout the request lifetime and output tokens
// accumulate on top during generation. This reveals GPU memory pressure that
// concurrency alone hides: two concurrent 4K-token requests look identical to
// two 128-token requests in concurrency but wildly different here. A record
// contributes events only for its non-NaN phases per the kvCacheEvents rules;
// returns (sorted timestamps, values) in tokens over epoch-nanosecond
// timestamps, or two empty slices when no record contributes an event.
func TokensInFlightSweepLine(startNS, generationStartNS, endNS, inputTokens, outputTokens []float64) ([]float64, []float64) {
	ts, deltas := kvCacheEvents(startNS, generationStartNS, endNS, inputTokens, outputTokens)
	if len(ts) == 0 {
		return []float64{}, []float64{}
	}
	return sweepLineCumsum(ts, deltas)
}
