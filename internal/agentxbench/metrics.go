package agentxbench

import (
	"math"
	"sort"
)

// Quantile computes the p-th quantile (0.0 <= p <= 1.0) of a sorted slice using linear interpolation.
func Quantile(sortedValues []float64, p float64) float64 {
	n := len(sortedValues)
	if n == 0 {
		return 0.0
	}
	if n == 1 || p <= 0.0 {
		return sortedValues[0]
	}
	if p >= 1.0 {
		return sortedValues[n-1]
	}
	idx := p * float64(n-1)
	lower := int(math.Floor(idx))
	upper := int(math.Ceil(idx))
	fraction := idx - float64(lower)
	if lower == upper {
		return sortedValues[lower]
	}
	return sortedValues[lower] + fraction*(sortedValues[upper]-sortedValues[lower])
}

// ComputeInteractivity computes ITL distribution and interactivity metrics from token arrival timestamps.
func ComputeInteractivity(timestamps []int64, ttftMS float64, executionMS float64) InteractivityMetrics {
	metrics := InteractivityMetrics{
		TTFTMS: ttftMS,
	}
	numTokens := len(timestamps)
	if numTokens <= 1 {
		if executionMS > 0 && numTokens > 0 {
			metrics.NormalizedInteractivity = (float64(numTokens) / executionMS) * 1000.0
		}
		return metrics
	}

	itls := make([]float64, numTokens-1)
	for i := 1; i < numTokens; i++ {
		diffNanos := timestamps[i] - timestamps[i-1]
		if diffNanos < 0 {
			diffNanos = 0
		}
		itls[i-1] = float64(diffNanos) / 1e6
	}

	sortedITLs := make([]float64, len(itls))
	copy(sortedITLs, itls)
	sort.Float64s(sortedITLs)

	metrics.ITLMedianMS = Quantile(sortedITLs, 0.50)
	metrics.ITLP90MS = Quantile(sortedITLs, 0.90)
	metrics.ITLP95MS = Quantile(sortedITLs, 0.95)
	metrics.ITLP99MS = Quantile(sortedITLs, 0.99)
	metrics.ITLMaxMS = sortedITLs[len(sortedITLs)-1]

	// Active generation time is time spent generating tokens after TTFT.
	genTimeMS := executionMS - ttftMS
	if genTimeMS <= 0 {
		genTimeMS = float64(timestamps[numTokens-1]-timestamps[0]) / 1e6
	}
	if genTimeMS > 0 {
		// Output tokens per second during generation phase
		metrics.NormalizedInteractivity = (float64(numTokens-1) / genTimeMS) * 1000.0
	} else if executionMS > 0 {
		metrics.NormalizedInteractivity = (float64(numTokens) / executionMS) * 1000.0
	}

	return metrics
}

// Aggregate compiles high-level benchmark statistics from a collection of request records.
func Aggregate(requests []RequestRecord, totalWallMS float64) AggregatedMetrics {
	agg := AggregatedMetrics{
		TotalRequests:   len(requests),
		TotalWallTimeMS: totalWallMS,
	}
	if len(requests) == 0 {
		return agg
	}

	var allTTFTs []float64
	var allITLs []float64
	var coldTTFTs []float64
	var warmTTFTs []float64
	var interactivities []float64

	for _, req := range requests {
		if req.Success {
			agg.SuccessfulRequests++
		} else {
			agg.FailedRequests++
		}
		agg.TotalPromptTokens += req.PromptTokens
		agg.TotalCompletionTokens += req.CompletionTokens
		agg.TotalCachedTokens += req.CachedTokens

		if req.Success {
			allTTFTs = append(allTTFTs, req.Interactivity.TTFTMS)
			if req.Interactivity.NormalizedInteractivity > 0 {
				interactivities = append(interactivities, req.Interactivity.NormalizedInteractivity)
			}
			if req.TurnIndex == 0 || req.CachedTokens == 0 {
				coldTTFTs = append(coldTTFTs, req.Interactivity.TTFTMS)
			} else {
				warmTTFTs = append(warmTTFTs, req.Interactivity.TTFTMS)
			}
			if len(req.TokenTimestampsUnixNano) > 1 {
				for i := 1; i < len(req.TokenTimestampsUnixNano); i++ {
					diffNanos := req.TokenTimestampsUnixNano[i] - req.TokenTimestampsUnixNano[i-1]
					if diffNanos >= 0 {
						allITLs = append(allITLs, float64(diffNanos)/1e6)
					}
				}
			}
		}
	}

	if agg.TotalRequests > 0 {
		agg.SuccessRate = float64(agg.SuccessfulRequests) / float64(agg.TotalRequests)
	}
	if agg.TotalPromptTokens > 0 {
		agg.AggregateCacheHitRatio = float64(agg.TotalCachedTokens) / float64(agg.TotalPromptTokens)
	}

	if len(coldTTFTs) > 0 {
		var sum float64
		for _, v := range coldTTFTs {
			sum += v
		}
		agg.ColdTTFTMeanMS = sum / float64(len(coldTTFTs))
	}
	if len(warmTTFTs) > 0 {
		var sum float64
		for _, v := range warmTTFTs {
			sum += v
		}
		agg.WarmTTFTMeanMS = sum / float64(len(warmTTFTs))
	}

	if agg.WarmTTFTMeanMS > 0 && agg.ColdTTFTMeanMS > 0 {
		agg.PrefixSpeedupRatio = agg.ColdTTFTMeanMS / agg.WarmTTFTMeanMS
	} else {
		agg.PrefixSpeedupRatio = 1.0
	}

	if len(allTTFTs) > 0 {
		sort.Float64s(allTTFTs)
		agg.TTFTP50MS = Quantile(allTTFTs, 0.50)
		agg.TTFTP95MS = Quantile(allTTFTs, 0.95)
	}

	if len(allITLs) > 0 {
		sort.Float64s(allITLs)
		agg.ITLP50MS = Quantile(allITLs, 0.50)
		agg.ITLP95MS = Quantile(allITLs, 0.95)
	}

	if len(interactivities) > 0 {
		sort.Float64s(interactivities)
		agg.NormalizedInteractivity = Quantile(interactivities, 0.50)
	}

	if totalWallMS > 0 {
		wallSec := totalWallMS / 1000.0
		agg.RequestThroughputPerSec = float64(agg.SuccessfulRequests) / wallSec
		agg.OutputTokenThroughputPerSec = float64(agg.TotalCompletionTokens) / wallSec
		totalTokens := agg.TotalPromptTokens + agg.TotalCompletionTokens
		agg.ClusterTokenThroughputPerSec = float64(totalTokens) / wallSec
	}

	return agg
}
