// E2E Normalized Interactivity, the per-request rate at which a user receives
// output tokens INCLUDING the prefill wait: output_sequence_length /
// request_latency. Source: src/aiperf/metrics/types/e2e_normalized_interactivity_metrics.py
// (the e2e_normalized_interactivity_{p75,p90} family) and the summarize-time
// analyzer src/aiperf/metrics/e2e_normalized_interactivity_analyzer.py
// (_compute_finite_ratios, inject_e2e_normalized_interactivity_metrics) that owns
// the columnar computation. To reproduce InferenceX's number exactly, the
// percentile is taken over the seconds-per-output-token ratio
// request_latency_s / OSL and then inverted:
//
//	p90 interactivity = 1 / p90(request_latency_s / OSL)
//
// This is the slow-tail convention InferenceX enforces for interactivity
// (1 / p(x), NOT p(1 / x)): a higher percentile names a slower, more pessimistic
// tail, so p90 is the effective token rate of the 90th-percentile worst request.
// Inverting the ratio's percentile (rather than percentiling the per-request
// rate directly) is required because 1 / p(x) != p(1 / x). Ported from the
// analyzer: one shared sampling pass feeds both percentiles, the base metric
// reports the success-only distribution, and the zero/missing guards (ratio_p
// <= 0, non-finite reciprocal) are preserved. Re-expressed in idiomatic Go over
// InteractivitySample values instead of numpy column slices; percentiles come
// from the unexported percentileLinear helper (linear on the clean array,
// matching numpy's default method="linear").
package perfledger

import (
	"math"
)

// nanosPerSecond is the NANOS_PER_SECOND spec scale (upstream
// aiperf.common.constants) that converts request latency from ns to seconds.
const nanosPerSecond = 1_000_000_000.0

// isFinite reports whether x is a finite float64 (not NaN and not ±Inf), the
// Go analog of numpy's np.isfinite used by the upstream filter.
func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// InteractivitySample is one request's contribution to the E2E normalized
// interactivity family: its end-to-end latency and how many output tokens it
// produced. LatencyNS and OutputTokens are the request_latency and
// output_sequence_length columns. A non-positive or non-finite value means the
// request is excluded from the sample (upstream "all fields > 0" filter); NaN is
// the "not recorded" encoding for a failed request (mirroring upstream NaN,
// where a failed row's metric columns are absent).
type InteractivitySample struct {
	LatencyNS    float64 // request latency, ns; NaN = not recorded
	OutputTokens float64 // output sequence length; NaN = not recorded
}

// Interactivity is the summary of the E2E normalized interactivity family in
// tokens/sec/user, mirroring the upstream e2e_normalized_interactivity_{p75,p90}
// metrics. P75 and P90 are the slow-tail percentiles: 1 / p(request_latency_s /
// OSL). Count is the number of requests that passed the positive-and-finite
// filter (the sample size the percentiles are computed over).
type Interactivity struct {
	P75   float64 `json:"p75"`
	P90   float64 `json:"p90"`
	Count int     `json:"count"`
}

// interactivityRatios returns the finite per-request seconds-per-output-token
// ratios that pass the positive/finite filter, or nil when nothing qualifies.
// Matches upstream _compute_finite_ratios: request_latency and OSL are required
// (no-op without them); a request is kept only when both are finite and > 0.
// Because count is dropped on export for the derived scalar, the caller reports
// the kept sample size so it is not silently invisible.
func interactivityRatios(samples []InteractivitySample) []float64 {
	var ratios []float64
	for _, s := range samples {
		if !isFinite(s.LatencyNS) || s.LatencyNS <= 0 ||
			!isFinite(s.OutputTokens) || s.OutputTokens <= 0 {
			continue
		}
		// Seconds per output token, per surviving request. Drop any non-finite
		// ratio (an extreme latency/OSL can overflow the division) so the
		// percentile and its reciprocal stay finite -- injected metric values
		// must be finite per the NaN/Inf discipline.
		ratio := s.LatencyNS / nanosPerSecond / s.OutputTokens
		if !isFinite(ratio) {
			continue
		}
		ratios = append(ratios, ratio)
	}
	return ratios
}

// invertedPercentile returns 1 / p(vals, q) with the upstream guards: a
// non-positive percentile (no positive ratios) or a non-finite reciprocal (a
// subnormal ratio can overflow to inf) yields math.NaN(), signaling "no value."
// The percentile is taken over the ratio and then inverted (the slow-tail
// convention 1 / p(x), NOT p(1 / x)).
func invertedPercentile(vals []float64, q float64) float64 {
	p := percentileLinear(vals, q)
	if p <= 0 {
		return math.NaN()
	}
	v := 1.0 / p
	if !isFinite(v) {
		return math.NaN()
	}
	return v
}

// ComputeInteractivity reports the E2E normalized interactivity family over
// per-request samples: p75 and p90 of 1 / p(request_latency_s / OSL). The
// percentile is taken over the seconds-per-output-token ratio and then inverted
// (the slow-tail convention 1 / p(x), NOT p(1 / x)), so a higher percentile
// names a slower, more pessimistic tail. Returns nil when no request passes the
// positive-and-finite filter (nothing to report).
func ComputeInteractivity(samples []InteractivitySample) *Interactivity {
	ratios := interactivityRatios(samples)
	if len(ratios) == 0 {
		return nil
	}
	return &Interactivity{
		P75:   invertedPercentile(ratios, 0.75),
		P90:   invertedPercentile(ratios, 0.90),
		Count: len(ratios),
	}
}
