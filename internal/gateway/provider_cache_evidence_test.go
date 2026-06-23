package gateway

import (
	"strings"
	"testing"
	"time"
)

// TestProviderCacheTelemetryIsPerformanceNotTrust witnesses issue #432 acceptance #3
// on the LIVE gateway path: the gateway records the upstream provider's prompt-cache
// reuse (CachedPromptTokens / the cached-prompt-tokens counter), but that telemetry is
// PERFORMANCE evidence only — never local trust. The classification is delegated to the
// proven cachemeta provider_prefix materialization gate, so a cache_read can never be
// promoted to a serveable local-trust hit, and recording it never inflates the kernel's
// trust-verdict tallies.
func TestProviderCacheTelemetryIsPerformanceNotTrust(t *testing.T) {
	m := newGatewayMetrics(time.Now())

	// One real adjudication verdict (TRUST evidence) and two provider cache reads
	// (PERFORMANCE evidence) recorded on the same gateway.
	m.observeOperation("proxy_admit", WireVerdict{Kind: "ALLOW"}, nil, time.Millisecond)
	m.observeInference(10, 5, 4096, "end_turn", time.Second)
	m.observeInference(10, 5, 1024, "end_turn", time.Second)

	s := m.adjudicationSummary()

	// The provider cache reuse is recorded as performance evidence...
	if s.CachedPromptTokens != 5120 || s.CachedTurns != 2 {
		t.Fatalf("provider cache telemetry not recorded: %d tok / %d turns", s.CachedPromptTokens, s.CachedTurns)
	}
	// ...but the #432 bridge classifies it as cost/latency-only, NEVER local trust.
	ev := s.ProviderCacheEvidence()
	if ev.CanServe() {
		t.Fatalf("provider cache must never serve as local trust: %+v", ev)
	}
	if ev.Meta["provider_cache"] != "cost_latency_only" {
		t.Fatalf("provider cache must be marked cost/latency-only telemetry: %+v", ev.Meta)
	}

	// And it must NOT inflate the trust-verdict tallies — only the one real ALLOW counts.
	if s.Total != 1 || s.Allowed != 1 {
		t.Fatalf("provider cache reuse must not count as a trust verdict: total=%d allowed=%d", s.Total, s.Allowed)
	}

	// The invariant is exported live in the scrape, derived from the same bridge.
	scrape := renderInference(m)
	if !strings.Contains(scrape, "fak_gateway_provider_cache_local_trust 0\n") {
		t.Fatalf("scrape must export the provider-cache-not-trust invariant as 0:\n%s", scrape)
	}
}

// TestProviderCacheEvidenceHoldsWhenIdle proves the trust/performance separation is an
// INVARIANT, not a function of traffic: with no served turns the bridge still refuses to
// treat provider cache as local trust, so the exported gauge reads 0 from the first scrape.
func TestProviderCacheEvidenceHoldsWhenIdle(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	if m.adjudicationSummary().ProviderCacheEvidence().CanServe() {
		t.Fatal("an idle gateway must still classify provider cache as non-serveable")
	}
	if !strings.Contains(renderInference(m), "fak_gateway_provider_cache_local_trust 0\n") {
		t.Fatalf("idle scrape must export the invariant as 0:\n%s", renderInference(m))
	}
}
