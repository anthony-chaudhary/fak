package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
)

func computeDistribution(samples []float64) DistributionStats {
	if len(samples) == 0 {
		return DistributionStats{}
	}
	sorted := make([]float64, len(samples))
	copy(sorted, samples)
	sort.Float64s(sorted)

	count := len(sorted)
	minVal := sorted[0]
	maxVal := sorted[count-1]

	var sum float64
	for _, v := range sorted {
		sum += v
	}
	mean := sum / float64(count)

	var varianceSum float64
	for _, v := range sorted {
		diff := v - mean
		varianceSum += diff * diff
	}
	stdDev := math.Sqrt(varianceSum / float64(count))

	return DistributionStats{
		Count:  count,
		Mean:   mean,
		Min:    minVal,
		Max:    maxVal,
		P50:    calcPercentile(sorted, 50.0),
		P95:    calcPercentile(sorted, 95.0),
		P99:    calcPercentile(sorted, 99.0),
		StdDev: stdDev,
	}
}

func computeITLStats(samples []float64) ITLStats {
	if len(samples) == 0 {
		return ITLStats{}
	}
	sorted := make([]float64, len(samples))
	copy(sorted, samples)
	sort.Float64s(sorted)

	count := len(sorted)
	var sum float64
	for _, v := range sorted {
		sum += v
	}
	mean := sum / float64(count)

	var varianceSum float64
	for _, v := range sorted {
		diff := v - mean
		varianceSum += diff * diff
	}
	stdDev := math.Sqrt(varianceSum / float64(count))

	p50 := calcPercentile(sorted, 50.0)
	p95 := calcPercentile(sorted, 95.0)
	p99 := calcPercentile(sorted, 99.0)

	return ITLStats{
		Count:    count,
		MeanMs:   mean,
		P50Ms:    p50,
		P95Ms:    p95,
		P99Ms:    p99,
		StdDevMs: stdDev,
		JitterMs: p99 - p50,
	}
}

func calcPercentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 100 {
		return sorted[len(sorted)-1]
	}
	rank := (p / 100.0) * float64(len(sorted)-1)
	low := int(math.Floor(rank))
	high := int(math.Ceil(rank))
	if low == high {
		return sorted[low]
	}
	weight := rank - float64(low)
	return sorted[low]*(1.0-weight) + sorted[high]*weight
}

func deterministicOutputHash(arm string, n, p, s, d int) string {
	h := sha256.New()
	fmt.Fprintf(h, "arm=%s;n=%d;p=%d;s=%d;d=%d;contract=6036", arm, n, p, s, d)
	return hex.EncodeToString(h.Sum(nil))[:16]
}
