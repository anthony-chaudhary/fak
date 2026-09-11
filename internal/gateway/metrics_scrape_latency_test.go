package gateway

import (
	"context"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// slowKVReporterPlanner is a Planner + KVMemoryReporter whose KVMemoryStats
// burns wall-clock (standing in for the ROCm hipMemGetInfo probe) and counts
// invocations, so a scrape can be proven to hit the reporter at most once.
type slowKVReporterPlanner struct {
	delay time.Duration
	calls int64
	stats agent.KVMemoryStats
}

func (p *slowKVReporterPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

func (p *slowKVReporterPlanner) Model() string { return "slow-kv-reporter-test" }

func (p *slowKVReporterPlanner) KVMemoryStats() agent.KVMemoryStats {
	atomic.AddInt64(&p.calls, 1)
	time.Sleep(p.delay)
	return p.stats
}

func testKVStats() agent.KVMemoryStats {
	return agent.KVMemoryStats{
		Enabled:            true,
		Backend:            "test-backend",
		MemoryClass:        "kv_cache",
		Scope:              "device",
		DType:              "f32",
		BytesPerToken:      128,
		ResidentTokens:     64,
		ResidentBytes:      8192,
		CapacityKnown:      true,
		CapacityFreeKnown:  true,
		CapacityTotalBytes: 1 << 20,
		CapacityFreeBytes:  1 << 19,
		FitBudgetBytes:     1 << 20,
		FitMarginBytes:     1 << 19,
	}
}

// TestMetricsScrapeCallsKVReporterOnce is the core regression: renderMetrics
// used to call KVMemoryStats() twice per scrape (writeKVMemoryMetrics and
// writeServingMetrics). The scrape-scoped memo must make that exactly one call.
// The latency bound is delay-relative so it actually catches a 2x regression
// even when the reporter is fast relative to the absolute 1s budget.
func TestMetricsScrapeCallsKVReporterOnce(t *testing.T) {
	const delay = 300 * time.Millisecond
	srv := newTestServer(t)
	planner := &slowKVReporterPlanner{delay: delay, stats: testKVStats()}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	start := time.Now()
	text := getMetrics(t, ts.URL+"/metrics", "")
	elapsed := time.Since(start)

	if elapsed >= 2*delay {
		t.Fatalf("scrape took %v with a %v reporter; want < %v (the reporter must not run twice)", elapsed, delay, 2*delay)
	}
	if got := atomic.LoadInt64(&planner.calls); got != 1 {
		t.Fatalf("KVMemoryStats calls = %d, want exactly 1 per scrape", got)
	}
	for _, want := range []string{
		"fak_gateway_kv_memory_enabled",
		"fak_gateway_kv_memory_resident_bytes",
		"fak_serving_kv_cache_usage_perc",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("scrape missing %q\n--- metrics ---\n%s", want, text)
		}
	}
}

// TestMetricsScrapeBytesStable proves a scrape with no inter-scrape mutation
// renders an identical byte count on the next scrape. The cache is aged past its
// TTL between the two renders so the second scrape performs a fresh reporter
// probe rather than being served from the memo, exercising the deterministic
// shape across two independent snapshots.
// It calls renderMetrics directly (not through the HTTP handler) so the scrape's
// own request counters are not what perturbs the body.
func TestMetricsScrapeBytesStable(t *testing.T) {
	srv := newTestServer(t)
	planner := &slowKVReporterPlanner{delay: 200 * time.Millisecond, stats: testKVStats()}
	srv.planner = planner

	first := srv.renderMetrics()

	srv.kvStatsMu.Lock()
	srv.kvStatsAt = srv.kvStatsAt.Add(-2 * kvStatsCacheTTL)
	srv.kvStatsMu.Unlock()

	second := srv.renderMetrics()
	if len(first) != len(second) {
		t.Fatalf("scrape byte counts differ: first=%d second=%d", len(first), len(second))
	}
	if got := atomic.LoadInt64(&planner.calls); got != 2 {
		t.Fatalf("KVMemoryStats calls = %d, want 2 (one per independent render)", got)
	}
}
