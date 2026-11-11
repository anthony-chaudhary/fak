// DIRECT-PORT ai-dynamo/aiperf src/aiperf/metrics/derived_latency.py@ab610add4e8876fca223197596c728e17365f381 (Apache-2.0):
// CO-aware effective_latency = end_ns − credit_issued_ns over successful records, plus the
// failure-inflated adj_effective_latency companion. Ported from the summarize-time derived
// metrics (_effective_samples, _inflated_metric_result): one shared sampling pass feeds both
// views, the base metric reports the success-only distribution as-is and the adjusted one
// appends one +Inf sample per errored record that carries a credit instant. Re-expressed in
// idiomatic Go over EffectiveSample values instead of numpy column slices; percentiles come
// from the two unexported helpers (linear on the clean array, nearest-rank on the Inf-inflated
// one, which avoids the linear-interpolation NaN at the inf−inf boundaries upstream cites).
package perfledger

import (
	"math"
	"sort"
	"time"
)

// nsPerMS converts epoch-ns deltas to the display unit (milliseconds), matching
// upstream _NS_PER_MS.
const nsPerMS = 1_000_000.0

// EffectiveSample is one ledger row reduced to the instants the CO-aware
// effective-latency metrics need: when the request finished and when its
// admission credit was issued. CreditIssuedNS == 0 is the "not issued" encoding
// (mirroring upstream NaN): such a sample contributes to neither metric. Errored
// is the has_error analog.
type EffectiveSample struct {
	EndNS          int64 // request end instant, ns since epoch
	CreditIssuedNS int64 // admission/credit-issued instant, ns since epoch; 0 = not issued
	Errored        bool  // has_error analog
}

// EffectiveLatency is the summary of an effective-latency distribution in
// display units (milliseconds), mirroring the upstream MetricResult fields the
// derived metrics emit. The adjusted view carries +Inf in SumMS/AvgMS/MaxMS and
// NaN in StdMS whenever a failure inflated it — see
// ComputeAdjustedEffectiveLatency before marshaling one raw.
type EffectiveLatency struct {
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

// effectiveSamples is the single sampling pass both effective-latency views are
// built from (upstream _effective_samples): the success-only ms distribution
// plus the errored-with-credit count, drawn from the same population so the two
// views never disagree. A sample without a credit instant contributes nothing —
// not a success sample, not an error sample. Deltas clamp to 0 before the ms
// conversion: float64 epoch-ns quantization plus cross-process wall-clock skew
// can push a near-zero delta slightly negative, and latencies are physically
// non-negative. ok is false when no sample carries a credit instant (e.g. a
// workload that bypasses the credit issuer) — there is nothing either metric
// can report.
func effectiveSamples(samples []EffectiveSample) (cleanMS []float64, errorCount int, ok bool) {
	for _, s := range samples {
		if s.CreditIssuedNS == 0 {
			continue
		}
		if s.Errored {
			errorCount++
			continue
		}
		delta := s.EndNS - s.CreditIssuedNS
		if delta < 0 {
			delta = 0
		}
		cleanMS = append(cleanMS, float64(delta)/nsPerMS)
	}
	return cleanMS, errorCount, len(cleanMS) > 0 || errorCount > 0
}

// cleanResult summarizes a success-only ms distribution the way upstream
// _array_to_metric_result does: population std (ddof=0) and linear-interpolation
// percentiles, nothing rounded.
func cleanResult(clean []float64) *EffectiveLatency {
	sum, mn, mx := 0.0, math.Inf(1), math.Inf(-1)
	for _, v := range clean {
		sum += v
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	mean := sum / float64(len(clean))
	variance := 0.0
	for _, v := range clean {
		d := v - mean
		variance += d * d
	}
	variance /= float64(len(clean))
	return &EffectiveLatency{
		Count: len(clean),
		SumMS: sum,
		AvgMS: mean,
		StdMS: math.Sqrt(variance),
		MinMS: mn,
		MaxMS: mx,
		P1MS:  percentileLinear(clean, 0.01),
		P5MS:  percentileLinear(clean, 0.05),
		P10MS: percentileLinear(clean, 0.10),
		P25MS: percentileLinear(clean, 0.25),
		P50MS: percentileLinear(clean, 0.50),
		P75MS: percentileLinear(clean, 0.75),
		P90MS: percentileLinear(clean, 0.90),
		P95MS: percentileLinear(clean, 0.95),
		P99MS: percentileLinear(clean, 0.99),
	}
}

// ComputeEffectiveLatency reports the CO-aware effective latency over successful
// records: end_ns − credit_issued_ns. The request finishes at EndNS but the user
// issued it at CreditIssuedNS, so the credit-queue wait is part of the perceived
// latency; compare against the registry's request latency to see how much of it
// is queue-induced vs server-induced. Errored samples are excluded entirely —
// the failure-aware view is ComputeAdjustedEffectiveLatency. Returns nil when no
// sample carries a credit instant or no successful sample does.
func ComputeEffectiveLatency(samples []EffectiveSample) *EffectiveLatency {
	clean, _, ok := effectiveSamples(samples)
	if !ok || len(clean) == 0 {
		return nil
	}
	return cleanResult(clean)
}

// ComputeAdjustedEffectiveLatency is the failure-inflated companion to
// ComputeEffectiveLatency, mirroring the adj_<tag> treatment
// PERCENTILE_INCLUDES_FAILED_REQUESTS gives the registry latency metrics: the
// success-only distribution plus one +Inf sample per errored record that carries
// a credit instant. A CO-aware metric with no failure-aware companion is the one
// gap the success-only view would otherwise open.
//
// AvgMS and SumMS are +Inf when any error is present — that is the intended
// user-perceived-latency-under-failure reading — and StdMS is mathematically
// undefined in a distribution containing Inf, so it is math.NaN(): callers must
// not JSON-marshal the adjusted result's std/avg/sum raw (encoding/json rejects
// Inf/NaN; render through the string fold instead). Percentiles use nearest rank
// on the inflated array to avoid the linear-interpolation NaN at the inf−inf
// boundaries. Returns nil when there are no errors (nothing to inflate) or no
// successful samples (no distribution to inflate).
func ComputeAdjustedEffectiveLatency(samples []EffectiveSample) *EffectiveLatency {
	clean, errorCount, ok := effectiveSamples(samples)
	if !ok || errorCount == 0 || len(clean) == 0 {
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
	return &EffectiveLatency{
		Count: len(inflated),
		SumMS: sum,
		AvgMS: sum / float64(len(inflated)),
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

// EffectiveSamplesFromRecords reduces ledger rows to effective-latency samples.
// EndNS comes from the row's unix_ms and Errored from a non-empty Error. Probe
// rows are left out of every quantile (package law), and rows whose
// creditIssuedNS accessor reports no admission instant are skipped entirely —
// no credit, no sample, for either metric. The accessor is caller-supplied
// because fak does not persist admission yet; the gateway wires it later. A nil
// accessor or empty input yields nil.
func EffectiveSamplesFromRecords(recs []Record, creditIssuedNS func(Record) (int64, bool)) []EffectiveSample {
	if creditIssuedNS == nil || len(recs) == 0 {
		return nil
	}
	out := make([]EffectiveSample, 0, len(recs))
	for _, r := range recs {
		if r.IsProbe() {
			continue
		}
		issued, ok := creditIssuedNS(r)
		if !ok {
			continue
		}
		out = append(out, EffectiveSample{
			EndNS:          r.UnixMS * int64(time.Millisecond),
			CreditIssuedNS: issued,
			Errored:        r.Error != "",
		})
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// percentileLinear interpolates the q-quantile of vals the way numpy's default
// percentile (method="linear") does: for sorted values and rank r = q*(n-1), the
// result is the linear interpolation between floor(r) and ceil(r). The input is
// copied before sorting — the caller's slice is never mutated.
func percentileLinear(vals []float64, q float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sorted := append([]float64(nil), vals...)
	sort.Float64s(sorted)
	r := q * float64(len(sorted)-1)
	lo := int(math.Floor(r))
	if lo < 0 {
		lo = 0
	}
	if lo > len(sorted)-1 {
		lo = len(sorted) - 1
	}
	hi := lo + 1
	if hi > len(sorted)-1 {
		hi = len(sorted) - 1
	}
	frac := r - float64(lo)
	if frac < 0 {
		frac = 0
	}
	return sorted[lo] + frac*(sorted[hi]-sorted[lo])
}

// percentileNearest reports the q-quantile of vals by nearest rank the way
// numpy's method="nearest" does: the sorted index is round-half-to-even of
// q*(n-1), clamped to [0, n-1]. The adjusted effective-latency view uses it
// because linear interpolation produces a NaN at the inf−inf boundaries of an
// Inf-inflated array. The input is copied before sorting — the caller's slice is
// never mutated.
func percentileNearest(vals []float64, q float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sorted := append([]float64(nil), vals...)
	sort.Float64s(sorted)
	idx := int(math.RoundToEven(q * float64(len(sorted)-1)))
	if idx < 0 {
		idx = 0
	}
	if idx > len(sorted)-1 {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}
