// the percentile-band metric dict (metric_result_from_array's quantile core).
// Given a sample set, computes the p1/p5/p10/p25/p50/p75/p90/p95/p99 band with
// the source's linear-interpolation method (numpy default "linear") and returns
// a structured band. The numpy vector kernel is re-expressed as a single sort
// plus a loop over the nine quantiles (stdlib only). Empty input returns an
// error: the source would index clean[0] on an empty array and panic, so the
// Go behavior is defined explicitly as a refusal.
package perfledger

import (
	"errors"
	"math"
	"sort"
)

// percentileQS is the standard percentile band shared by the source's
// metric_result_from_array and its derived-latency builders, kept here (vs
// inlined) so the band layout matches across all callers.
var percentileQS = []float64{1, 5, 10, 25, 50, 75, 90, 95, 99}

// PercentileBand is the p1/p5/p10/p25/p50/p75/p90/p95/p99 quantile band of a
// sample set, in the samples' own units, mirroring the upstream MetricResult
// percentile fields. Count is the number of samples the band was computed from.
type PercentileBand struct {
	Count int
	P1    float64
	P5    float64
	P10   float64
	P25   float64
	P50   float64
	P75   float64
	P90   float64
	P95   float64
	P99   float64
}

// ComputePercentileBand reports the percentile band of samples using the
// source's linear-interpolation method (numpy default "linear"): for sorted
// samples and rank r = q/100*(n-1), the value is the linear interpolation
// between floor(r) and min(floor(r)+1, n-1). The input is copied before
// sorting — the caller's slice is never mutated.
//
// Empty input (n==0) returns an error rather than a zero band: the source
// would index clean[0] on an empty array and panic, so the Go behavior is
// defined explicitly as a refusal.
func ComputePercentileBand(samples []float64) (PercentileBand, error) {
	n := len(samples)
	if n == 0 {
		return PercentileBand{}, errors.New("perfledger: ComputePercentileBand: empty sample set")
	}
	sorted := append([]float64(nil), samples...)
	sort.Float64s(sorted)

	pcts := make([]float64, len(percentileQS))
	for i, q := range percentileQS {
		v := q / 100.0 * float64(n-1)
		lo := int(math.Floor(v))
		hi := lo + 1
		if hi > n-1 {
			hi = n - 1
		}
		frac := v - float64(lo)
		pcts[i] = sorted[lo] + frac*(sorted[hi]-sorted[lo])
	}

	return PercentileBand{
		Count: n,
		P1:    pcts[0],
		P5:    pcts[1],
		P10:   pcts[2],
		P25:   pcts[3],
		P50:   pcts[4],
		P75:   pcts[5],
		P90:   pcts[6],
		P95:   pcts[7],
		P99:   pcts[8],
	}, nil
}
