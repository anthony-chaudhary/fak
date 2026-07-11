package gateway

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// The per-session cache-break family (#2916) is the observability half of the
// cache-integrity invariant: a warm prompt prefix that breaks mid-conversation is
// otherwise invisible until the invoice. This asserts the gateway lowers the
// internal/metrics counter onto its live Prometheus surface — labeled by the closed
// cause vocabulary, with the cold-rebuild token cost each break caused — so a rise is
// caught by a gate rather than an eyeball.
//
// The witnesses recorded here stand in for sibling #2915's mid-conversation
// prefix-mutation detector (this counter's live producer); the deliberate breaks are a
// toolset change that invalidated a 900-token warm prefix and a provider-side eviction.
func TestCacheBreakMetricsWitnessOnDeliberateBreak(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	// Deliberately break the prefix: a mid-conversation toolset change cost a 900-token
	// cold rebuild; a second toolset change cost 300 more; a provider quirk cost 120.
	m.recordCacheBreak(metrics.CacheBreakToolsetChange, 900)
	m.recordCacheBreak(metrics.CacheBreakToolsetChange, 300)
	m.recordCacheBreak(metrics.CacheBreakProviderQuirk, 120)

	var b strings.Builder
	m.writeCacheBreakMetrics(&b)
	got := b.String()
	for _, want := range []string{
		"# TYPE fak_cache_break_events_total counter",
		"# TYPE fak_cache_break_cost_tokens_total counter",
		`fak_cache_break_events_total{cause="toolset_change"} 2`,
		`fak_cache_break_cost_tokens_total{cause="toolset_change"} 1200`,
		`fak_cache_break_events_total{cause="provider_quirk"} 1`,
		`fak_cache_break_cost_tokens_total{cause="provider_quirk"} 120`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("cache-break metrics missing %q:\n%s", want, got)
		}
	}

	// The gate the family feeds: a defined budget fails once the witnessed session
	// regresses past it (not "any nonzero count fails"). Three breaks / 1320 tokens is
	// over a 1-event / 1000-token budget.
	if err := m.cacheBreakReport().CheckBudget(metrics.CacheBreakBudget{MaxEvents: 1, MaxCostTokens: 1000}); err == nil {
		t.Fatal("a regressed session did not fail the cache-break budget gate")
	}
}

// A session with no cache breaks still declares both families (HELP/TYPE) on the surface
// — a clean zero — so a regression gate and a dashboard panel exist from the first scrape
// rather than appearing only after the first break. No per-cause sample is emitted, which
// is exactly the empty state a gate reads as "no regression".
func TestCacheBreakMetricsCleanZero(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	var b strings.Builder
	m.writeCacheBreakMetrics(&b)
	got := b.String()
	for _, want := range []string{
		"# TYPE fak_cache_break_events_total counter",
		"# TYPE fak_cache_break_cost_tokens_total counter",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("clean-zero session dropped a family declaration %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, `fak_cache_break_events_total{cause=`) {
		t.Fatalf("clean-zero session emitted a phantom per-cause sample:\n%s", got)
	}
	if err := m.cacheBreakReport().CheckBudget(metrics.CacheBreakBudget{MaxEvents: 0, MaxCostTokens: 0}); err != nil {
		t.Fatalf("clean-zero session failed a zero budget: %v", err)
	}
}

// The family is wired into the live /metrics render, not merely callable in isolation:
// srv.renderMetrics() declares it on a fresh server, and a witnessed break surfaces its
// labeled series with the per-event cost on the same live surface — the "labeled metric
// emitted per session" acceptance, on the Prometheus surface the issue names.
func TestCacheBreakMetricsWiredIntoRenderMetrics(t *testing.T) {
	srv := newTestServer(t)
	if srv.metrics == nil {
		t.Fatal("test server has no metrics sink")
	}
	pre := srv.renderMetrics()
	if !strings.Contains(pre, "# TYPE fak_cache_break_events_total counter") {
		t.Fatalf("cache-break family not declared on the live /metrics surface:\n%s", pre)
	}
	if strings.Contains(pre, `fak_cache_break_events_total{cause=`) {
		t.Fatalf("cache-break family emitted a sample before any break was witnessed:\n%s", pre)
	}

	srv.metrics.recordCacheBreak(metrics.CacheBreakRebuiltPrompt, 512)
	post := srv.renderMetrics()
	for _, want := range []string{
		`fak_cache_break_events_total{cause="rebuilt_prompt"} 1`,
		`fak_cache_break_cost_tokens_total{cause="rebuilt_prompt"} 512`,
	} {
		if !strings.Contains(post, want) {
			t.Fatalf("witnessed break missing from live /metrics %q:\n%s", want, post)
		}
	}
}
