package gateway

import (
	"testing"
	"time"
)

// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime fast est=10ms lane=default
func TestDeadlineAdmissionUsesFullResidentPromptCurrency(t *testing.T) {
	tests := []struct {
		name                         string
		uncached, cached, cacheWrite int
	}{
		{name: "uncached", uncached: 2000},
		{name: "cache read", uncached: 500, cached: 1500},
		{name: "cache creation", uncached: 500, cacheWrite: 1500},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := newGatewayMetrics(time.Now())
			m.observeInferenceTimed(tc.uncached, 20, tc.cached, tc.cacheWrite, "stop", 12*time.Second, 10*time.Second)

			verdict := m.deadlineEstimator().Admit(2000, 20, 20*time.Second, true)
			if !verdict.Admit || !verdict.Measured {
				t.Fatalf("20s client budget verdict: admit=%v measured=%v reason=%q estimate=%v; want measured admission",
					verdict.Admit, verdict.Measured, verdict.Reason, verdict.Estimate)
			}
			if verdict.Estimate < 11*time.Second || verdict.Estimate > 13*time.Second {
				t.Fatalf("full 2000-token resident prompt estimate = %v, want about 12s", verdict.Estimate)
			}

			sum := m.adjudicationSummary()
			if sum.InputTokens != uint64(tc.uncached) ||
				sum.CachedPromptTokens != uint64(tc.cached) ||
				sum.CacheCreationTokens != uint64(tc.cacheWrite) {
				t.Fatalf("token counters = input %d cached %d creation %d, want %d/%d/%d",
					sum.InputTokens, sum.CachedPromptTokens, sum.CacheCreationTokens,
					tc.uncached, tc.cached, tc.cacheWrite)
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
