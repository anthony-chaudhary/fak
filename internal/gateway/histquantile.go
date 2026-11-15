package gateway

import (
	"math"
	"sort"
)

// NOTICE / attribution:
//
// This file is a direct port of the histogram quantile estimator from
// ai-dynamo/aiperf, pinned at commit ab610add4e8876fca223197596c728e17365f381
// (short: ab610ad), file src/aiperf/server_metrics/histogram_percentiles.py.
//
//   - Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   - SPDX-License-Identifier: Apache-2.0
//
// The ported functions mirror the source's Prometheus linear-interpolation
// path:
//
//   - _sort_bucket_keys      -> sortHistogramBuckets
//   - total_count selection  -> histogramTotalCount
//   - _prometheus_quantile   -> prometheusQuantile
//   - compute_prometheus_percentiles (empty-input handling + dispatch)
//                              -> HistQuantile
//
// The source's polynomial-histogram / learned-statistics path
// (compute_estimated_percentiles and friends) is intentionally NOT ported:
// it requires NumPy and per-bucket learned means/variance accumulated across
// scrape intervals, which is outside the "buckets + counts -> q-quantile"
// contract of this estimator.

// HistogramBucket is one cumulative bucket of a Prometheus-style histogram.
//
// Bound is the bucket's upper edge (the "le" value). Use math.Inf(1) for the
// +Inf bucket. Count is the CUMULATIVE number of observations with value
// <= Bound (Prometheus "le" semantics), not the count within this bucket
// alone.
type HistogramBucket struct {
	Bound float64
	Count float64
}

// sortHistogramBuckets returns the buckets sorted by numeric bound with the
// +Inf bucket last, mirroring aiperf's _sort_bucket_keys. The input slice is
// not mutated.
func sortHistogramBuckets(buckets []HistogramBucket) []HistogramBucket {
	out := make([]HistogramBucket, len(buckets))
	copy(out, buckets)
	sort.SliceStable(out, func(i, j int) bool {
		bi, bj := out[i].Bound, out[j].Bound
		if math.IsInf(bi, 1) {
			return false // +Inf sorts last
		}
		if math.IsInf(bj, 1) {
			return true
		}
		return bi < bj
	})
	return out
}

// histogramTotalCount returns the total observation count for a sorted
// bucket list: the +Inf bucket's count when present, otherwise the last
// (largest-bound) bucket's count. This mirrors the source's
// bucket_cumulative.get("+Inf", bucket_cumulative[sorted_keys[-1]]).
func histogramTotalCount(sorted []HistogramBucket) float64 {
	for i := len(sorted) - 1; i >= 0; i-- {
		if math.IsInf(sorted[i].Bound, 1) {
			return sorted[i].Count
		}
	}
	return sorted[len(sorted)-1].Count
}

// HistQuantile estimates the q-quantile (0 <= q <= 1) of a histogram given
// as cumulative buckets, using the source's bucket-selection and linear
// in-bucket interpolation.
//
// The target rank is q * total_count. The first bucket whose cumulative count
// reaches the target rank is selected, and the value is linearly interpolated
// within that bucket assuming a uniform distribution. When the target rank
// falls in the +Inf bucket, the upper bound of the last finite bucket is
// returned (interpolation within +Inf is undefined).
//
// ok is false for an empty histogram or a zero total count, mirroring the
// source's empty-estimate handling (which returns an all-None estimate).
func HistQuantile(buckets []HistogramBucket, q float64) (value float64, ok bool) {
	if len(buckets) == 0 {
		return 0, false
	}
	sorted := sortHistogramBuckets(buckets)
	if len(sorted) == 0 {
		return 0, false
	}
	total := histogramTotalCount(sorted)
	if total == 0 {
		return 0, false
	}
	return prometheusQuantile(q, sorted, total), true
}

// prometheusQuantile is a direct port of aiperf's _prometheus_quantile.
//
// The algorithm:
//  1. Find the bucket where the quantile rank falls.
//  2. Linearly interpolate within that bucket assuming a uniform distribution.
//
// sorted must be the output of sortHistogramBuckets. totalCount must be the
// output of histogramTotalCount for the same buckets and must be > 0.
func prometheusQuantile(quantile float64, sorted []HistogramBucket, totalCount float64) float64 {
	if totalCount == 0 {
		// Source returns None here; the caller guards via the ok return.
		return 0
	}

	// Target rank for the quantile.
	targetRank := quantile * totalCount

	// Find the bucket containing the target rank.
	prevBound := 0.0
	prevCount := 0.0

	for _, b := range sorted {
		if math.IsInf(b.Bound, 1) {
			// Can't interpolate within +Inf bucket:
			// return the upper bound of the last finite bucket.
			return prevBound
		}

		currentBound := b.Bound

		if b.Count >= targetRank {
			// Target is in this bucket - interpolate.
			bucketCount := b.Count - prevCount
			if bucketCount == 0 {
				return prevBound
			}
			// Linear interpolation within bucket.
			bucketFraction := (targetRank - prevCount) / bucketCount
			return prevBound + (currentBound-prevBound)*bucketFraction
		}

		prevBound = currentBound
		prevCount = b.Count
	}

	// Fallback: return last finite bound.
	return prevBound
}
