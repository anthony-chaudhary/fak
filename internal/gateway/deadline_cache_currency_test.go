package gateway

import (
	"testing"
	"time"
)

// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime fast est=10ms lane=default
func TestDeadlineAdmissionUsesFullResidentPromptCurrency(t *testing.T) {
	// Each turn is a 2000-token resident prompt with a 10 s time to first token.
	// Only the uncached part was prefilled in those 10 s, so a cache read makes
	// the measured prefill rate slower (500 t / 10 s), and a later admission of
	// the same prompt is fast only when the same prefix is predicted resident.
	tests := []struct {
		name                         string
		uncached, cached, cacheWrite int
		kvCacheN                     int
		promptTok                    int
	}{
		{name: "uncached", uncached: 2000, promptTok: 2000},
		{name: "cache read", uncached: 500, cached: 1500, promptTok: 500},
		{name: "cache creation", uncached: 500, cacheWrite: 1500, promptTok: 500},
		{name: "llama timings cache_n", uncached: 500, kvCacheN: 1500, promptTok: 2000},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := newGatewayMetrics(time.Now())
			m.observeInferenceTimedDetail(localityUnknown, tc.promptTok, 20, tc.cached, tc.cacheWrite, "stop",
				12*time.Second, 10*time.Second, perfDetail{kvCacheN: tc.kvCacheN})

			readTok := tc.cached + tc.kvCacheN
			verdict := m.deadlineEstimator().AdmitCached(2000, readTok, 20, 20*time.Second, true)
			if !verdict.Admit || !verdict.Measured {
				t.Fatalf("20s client budget verdict: admit=%v measured=%v reason=%q estimate=%v; want measured admission",
					verdict.Admit, verdict.Measured, verdict.Reason, verdict.Estimate)
			}
			if verdict.Estimate < 11*time.Second || verdict.Estimate > 13*time.Second {
				t.Fatalf("2000-token prompt with %d resident estimate = %v, want about 12s", readTok, verdict.Estimate)
			}
			if readTok > 0 {
				cold, _ := m.deadlineEstimator().Estimate(2000, 20)
				if cold < 41*time.Second || cold > 43*time.Second {
					t.Fatalf("cold 2000-token estimate = %v, want about 42s at the 50 t/s uncached prefill rate", cold)
				}
			}

			sum := m.adjudicationSummary()
			if sum.InputTokens != uint64(tc.promptTok) ||
				sum.CachedPromptTokens != uint64(tc.cached) ||
				sum.CacheCreationTokens != uint64(tc.cacheWrite) {
				t.Fatalf("token counters = input %d cached %d creation %d, want %d/%d/%d",
					sum.InputTokens, sum.CachedPromptTokens, sum.CacheCreationTokens,
					tc.promptTok, tc.cached, tc.cacheWrite)
			}
		})
	}
}

// fak-test:justify why=invariant when=changed:internal/gateway/**
// fak-test:runtime fast est=10ms lane=default
func TestDeadlineAdmissionResidentTokenArithmeticStaysPositive(t *testing.T) {
	const outputTokens = 20
	maxInt := int(^uint(0) >> 1)

	t.Run("negative counts", func(t *testing.T) {
		m := newGatewayMetrics(time.Now())
		m.observeInferenceTimed(-1, outputTokens, -1, -1, "stop", 12*time.Second, 10*time.Second)

		estimate, measured := m.deadlineEstimator().Estimate(2000, outputTokens)
		if !measured || estimate <= 0 {
			t.Fatalf("estimate after negative counts = (%v, %v), want a positive measured estimate", estimate, measured)
		}
	})

	t.Run("extreme resident sum", func(t *testing.T) {
		m := newGatewayMetrics(time.Now())
		m.observeInferenceTimed(1, outputTokens, maxInt, maxInt, "stop", 12*time.Second, 10*time.Second)

		estimate, measured := m.deadlineEstimator().Estimate(maxInt, outputTokens)
		if !measured || estimate < 11*time.Second || estimate > 13*time.Second {
			t.Fatalf("estimate after extreme resident sum = (%v, %v), want a positive estimate near 12s", estimate, measured)
		}
	})
}
