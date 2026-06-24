package gateway

import (
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

// metrics_kvprefix_test.go — the in-kernel KV-prefix reuse family on /metrics
// (fak_gateway_kv_prefix_*). It is the live measurement of the frozen-trajectory cache
// cliff: the planner feeds cacheobs.Default the realized reuse on every served in-kernel
// turn, and the gateway scrapes it here. The process-global tap may carry counts from
// sibling tests, so the family-present checks assert the series exist; the live-read
// asserts an Observe moves the reused-tokens counter and the frozen-regime bucket.
func TestMetricsExposesKVPrefixReuse(t *testing.T) {
	srv := newTestServer(t)

	for _, want := range []string{
		"# TYPE fak_gateway_kv_prefix_turns_total counter",
		"# TYPE fak_gateway_kv_prefix_prompt_tokens_total counter",
		"# TYPE fak_gateway_kv_prefix_reused_tokens_total counter",
		"# TYPE fak_gateway_kv_prefix_turns_by_regime_total counter",
		`fak_gateway_kv_prefix_turns_by_regime_total{regime="frozen"} `,
		`fak_gateway_kv_prefix_turns_by_regime_total{regime="partial"} `,
		`fak_gateway_kv_prefix_turns_by_regime_total{regime="cold"} `,
		"# TYPE fak_gateway_kv_prefix_reuse_ratio gauge",
		"fak_gateway_kv_prefix_reuse_ratio ",
	} {
		if text := srv.renderMetrics(); !strings.Contains(text, want) {
			t.Fatalf("metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}

	// Live read: a frozen-regime turn (990/1000 reused) must move the reused-tokens
	// counter and increment the frozen bucket.
	before := cacheobs.Default.Snapshot()
	cacheobs.Default.Observe(1000, 990)
	after := cacheobs.Default.Snapshot()
	if after.ReusedTokens <= before.ReusedTokens {
		t.Fatalf("reused tokens did not rise after Observe: before=%d after=%d", before.ReusedTokens, after.ReusedTokens)
	}
	if after.FrozenTurns != before.FrozenTurns+1 {
		t.Fatalf("frozen bucket did not increment: before=%d after=%d", before.FrozenTurns, after.FrozenTurns)
	}

	text := srv.renderMetrics()
	line := metricLine(text, "fak_gateway_kv_prefix_reused_tokens_total")
	if line == "" {
		t.Fatalf("no fak_gateway_kv_prefix_reused_tokens_total line:\n%s", text)
	}
	n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "fak_gateway_kv_prefix_reused_tokens_total")))
	if err != nil {
		t.Fatalf("parse %q: %v", line, err)
	}
	if uint64(n) < after.ReusedTokens {
		t.Fatalf("scraped reused tokens %d < observed %d", n, after.ReusedTokens)
	}
}
