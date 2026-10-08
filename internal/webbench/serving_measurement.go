package webbench

import (
	"math"
	"sync"
	"time"
)

type servingInFlightTracker struct {
	mu      sync.Mutex
	current int
	max     int
}

func (t *servingInFlightTracker) begin() func() {
	if t == nil {
		return func() {}
	}
	t.mu.Lock()
	t.current++
	if t.current > t.max {
		t.max = t.current
	}
	t.mu.Unlock()
	return func() {
		t.mu.Lock()
		t.current--
		t.mu.Unlock()
	}
}

func (t *servingInFlightTracker) maxObserved() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.max
}

func setServingOverlapEvidence(stats *ServingStats, observedMax *int) {
	if stats == nil || observedMax == nil {
		return
	}
	stats.ObservedMaxInFlight = observedMax
	stats.ObservedInFlightBasis = "client_http"
}

func setServingExactEvidence(stats *ServingStats, samples []ServingSample, wallSeconds float64, slo time.Duration) {
	if stats == nil {
		return
	}
	stats.GoodputTokensS = notMeasuredScalar("usage.completion_tokens/s", "exact SLO-qualified output tokens were not measured")
	if servingFinitePositive(wallSeconds) {
		stats.WallSeconds = floatPtr(wallSeconds)
	}
	if slo > 0 {
		sloSeconds := slo.Seconds()
		stats.GoodputSLOSeconds = floatPtr(sloSeconds)
	}

	okCount := 0
	exactTotal, sloExactTotal := int64(0), int64(0)
	exactComplete, exactOverflow := true, false
	latenciesValid := true
	for _, sample := range samples {
		if sample.Status != "ok" {
			continue
		}
		okCount++
		if !servingFiniteNonnegative(sample.EndToEndMillis) {
			latenciesValid = false
		}
		if sample.OutputTokensExact == nil || *sample.OutputTokensExact < 0 {
			exactComplete = false
			continue
		}
		tokens := int64(*sample.OutputTokensExact)
		if exactTotal > math.MaxInt64-tokens {
			exactOverflow = true
			continue
		}
		exactTotal += tokens
		if servingWithinSLO(sample.EndToEndMillis, slo) {
			if sloExactTotal > math.MaxInt64-tokens {
				exactOverflow = true
				continue
			}
			sloExactTotal += tokens
		}
	}

	if okCount == 0 {
		stats.GoodputTokensS = notMeasuredScalar("usage.completion_tokens/s", "no successful requests measured")
		return
	}
	if !exactComplete {
		stats.GoodputTokensS = notMeasuredScalar("usage.completion_tokens/s", "successful requests lack exact output token counts")
		return
	}
	if exactOverflow {
		stats.GoodputTokensS = notMeasuredScalar("usage.completion_tokens/s", "exact output token count overflow")
		return
	}
	stats.SuccessfulOutputTokensExact = int64Ptr(exactTotal)
	if slo <= 0 {
		stats.GoodputTokensS = notMeasuredScalar("usage.completion_tokens/s", "no positive end-to-end SLO configured")
		return
	}
	if !latenciesValid {
		stats.GoodputTokensS = notMeasuredScalar("usage.completion_tokens/s", "successful request latency is not finite and nonnegative")
		return
	}
	stats.SLOSuccessfulOutputTokensExact = int64Ptr(sloExactTotal)
	if stats.WallSeconds == nil {
		stats.GoodputTokensS = notMeasuredScalar("usage.completion_tokens/s", "no finite positive common wall window measured")
		return
	}
	rate := float64(sloExactTotal) / wallSeconds
	if !servingFiniteNonnegative(rate) {
		stats.GoodputTokensS = notMeasuredScalar("usage.completion_tokens/s", "derived token goodput is not finite")
		return
	}
	stats.GoodputTokensS = measuredScalar(rate, "usage.completion_tokens/s", "")
}

func servingFinitePositive(value float64) bool {
	return value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

func servingFiniteNonnegative(value float64) bool {
	return value >= 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

func servingWithinSLO(milliseconds float64, slo time.Duration) bool {
	return slo > 0 && servingFiniteNonnegative(milliseconds) && milliseconds <= float64(slo)/float64(time.Millisecond)
}

func int64Ptr(value int64) *int64 { return &value }
