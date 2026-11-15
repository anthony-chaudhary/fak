// DIRECT-PORT ai-dynamo/aiperf src/aiperf/metrics/derived_latency.py@ab610add4e8876fca223197596c728e17365f381 (Apache-2.0):
// the generic failure-inflated "adj_<tag>" percentile band (upstream
// _inflated_metric_result / inject_adjusted_latency_metrics, the
// PERCENTILE_INCLUDES_FAILED_REQUESTS treatment) and the credit-to-start
// derived latency (compute_credit_to_start_latency + _delta_ms). Re-expressed
// in idiomatic Go: AdjustedPercentiles appends one +Inf sample per errored
// record to a success-only ms distribution and reads the p1..p99 band by
// nearest rank (numpy method="nearest", round-half-to-even), which avoids the
// linear-interpolation NaN at the inf-inf boundaries upstream cites;
// CreditToStartLatency reduces request_start_ns - credit_issued_ns to ms,
// dropping unissued credits and clamping near-zero negative deltas to 0. The
// success-only effective-latency pair and the percentile helpers already live
// in effective_latency.go and are reused here.
package perfledger

import "math"

// AdjustedPercentileBand is the failure-inflated summary of a success-only
// latency distribution: the count, sum, mean, min and max of the
// Inf-inflated population plus the p1..p99 band read by nearest rank. It is
// the Go analog of the upstream adj_<tag> MetricResult. StdMS is always NaN:
// a standard deviation is mathematically undefined in a distribution that
// contains +Inf (the variance computation yields NaN), so it is reported as
// NaN rather than a misleading finite number.
type AdjustedPercentileBand struct {
	Count int     `json:"count"`
	SumMS float64 `json:"sum_ms"`
	AvgMS float64 `json:"avg_ms"`
	StdMS float64 `json:"std_ms"`
	MinMS float64 `json:"min_ms"`
	MaxMS float64 `json:"max_ms"`
	P1MS  float64 `json:"p1_ms"`
	P5MS  float64 `json:"p5_ms"`
	P10MS float64 `json:"p10_ms"`
	P25MS float64 `json:"p25_ms"`
	P50MS float64 `json:"p50_ms"`
	P75MS float64 `json:"p75_ms"`
	P90MS float64 `json:"p90_ms"`
	P95MS float64 `json:"p95_ms"`
	P99MS float64 `json:"p99_ms"`
}

// AdjustedPercentiles computes the failure-inflated percentile band for a
// success-only sample set, mirroring upstream _inflated_metric_result: one
// +Inf sample is appended per errored record and the p1/p5/p10/p25/p50/p75/
// p90/p95/p99 band is read by nearest rank (numpy method="nearest"), which
// avoids the linear-interpolation NaN at the inf-inf boundaries. SumMS, AvgMS
// and MaxMS are +Inf whenever any error is present — that is the intended
// user-perceived-latency-under-failure reading the metric is meant to
// surface — and StdMS is NaN (undefined with +Inf). The input need not be
// pre-sorted: the percentile helpers sort a copy. Returns nil when
// errorCount <= 0 (nothing to inflate, so the adjusted band would equal the
// success-only one) or clean is empty (no distribution to inflate), matching
// the upstream no-op.
func AdjustedPercentiles(clean []float64, errorCount int) *AdjustedPercentileBand {
	if errorCount <= 0 || len(clean) == 0 {
		return nil
	}
	inflated := make([]float64, 0, len(clean)+errorCount)
	inflated = append(inflated, clean...)
	for i := 0; i < errorCount; i++ {
		inflated = append(inflated, math.Inf(1))
	}
	sum, mn := 0.0, math.Inf(1)
	for _, v := range inflated {
		sum += v
		if v < mn {
			mn = v
		}
	}
	n := len(inflated)
	return &AdjustedPercentileBand{
		Count: n,
		SumMS: sum,
		AvgMS: sum / float64(n),
		StdMS: math.NaN(),
		MinMS: mn,
		MaxMS: math.Inf(1),
		P1MS:  percentileNearest(inflated, 0.01),
		P5MS:  percentileNearest(inflated, 0.05),
		P10MS: percentileNearest(inflated, 0.10),
		P25MS: percentileNearest(inflated, 0.25),
		P50MS: percentileNearest(inflated, 0.50),
		P75MS: percentileNearest(inflated, 0.75),
		P90MS: percentileNearest(inflated, 0.90),
		P95MS: percentileNearest(inflated, 0.95),
		P99MS: percentileNearest(inflated, 0.99),
	}
}

// CreditToStartLatency computes the per-record credit-queue wait
// (request_start_ns - credit_issued_ns) in ms over the records that carry a
// credit instant, mirroring upstream compute_credit_to_start_latency +
// _delta_ms. A credit instant of 0 means "not issued" (the upstream NaN
// encoding) and such records are dropped. Near-zero negative deltas are
// clamped to 0: float64 epoch-ns quantization (~256 ns ULP at ~1.75e18) plus
// cross-process wall-clock skew can push a near-zero delta slightly negative,
// and latencies are physically non-negative. The two slices are parallel
// columns and are read up to the shorter length. The result is the success-only
// ms-distribution summary (linear-interpolation percentiles, population std),
// reported through the shared EffectiveLatency type. Returns nil when no
// record has a credit instant.
func CreditToStartLatency(startNS, creditIssuedNS []int64) *EffectiveLatency {
	n := len(startNS)
	if len(creditIssuedNS) < n {
		n = len(creditIssuedNS)
	}
	clean := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		if creditIssuedNS[i] == 0 {
			continue
		}
		delta := startNS[i] - creditIssuedNS[i]
		if delta < 0 {
			delta = 0
		}
		clean = append(clean, float64(delta)/nsPerMS)
	}
	if len(clean) == 0 {
		return nil
	}
	return cleanResult(clean)
}
