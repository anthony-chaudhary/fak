package gateway

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func proxyTurn(prompt int, details *agent.UsageTokenDetails, timings *agent.Timings) *agent.Completion {
	return &agent.Completion{
		Model:   "llama",
		Usage:   agent.Usage{PromptTokens: prompt, CompletionTokens: 10, PromptTokensDetails: details},
		Timings: timings,
	}
}

// TestProxyPrefixReuseByOriginReadsCacheNAndKeepsUnreportedUnknown pins the llama-server
// proxy path: cached_tokens and timings.cache_n both feed the by-origin split, a turn with
// neither is unknown (never cold), and vendor passthrough stays out.
//
// fak-test:runtime fast est=50ms
func TestProxyPrefixReuseByOriginReadsCacheNAndKeepsUnreportedUnknown(t *testing.T) {
	s := newTestServer(t)
	s.upstream = nil
	s.servedSide = localitySelfHosted
	ctx := withPrefixReuseSession(context.Background(), "launch-1")
	sys := agent.Message{Role: agent.RoleSystem, Content: "shared"}
	coord := []agent.Message{sys, {Role: agent.RoleUser, Content: "coordinate"}}
	sub := []agent.Message{sys, {Role: agent.RoleUser, Content: "subtask"}}

	s.observePrefixReuseTurn(ctx, coord, proxyTurn(100, nil, &agent.Timings{CacheN: 0}))                                     // cold via cache_n=0
	s.observePrefixReuseTurn(ctx, sub, proxyTurn(80, nil, &agent.Timings{CacheN: 60}))                                       // cross via cache_n
	s.observePrefixReuseTurn(ctx, coord, proxyTurn(150, &agent.UsageTokenDetails{CachedTokens: 110}, nil))                   // same via cached_tokens
	s.observePrefixReuseTurn(ctx, coord, proxyTurn(200, nil, nil))                                                           // unknown
	s.observePrefixReuseTurn(ctx, sub, proxyTurn(90, &agent.UsageTokenDetails{CachedTokens: 5}, &agent.Timings{CacheN: 70})) // max wins

	s.servedSide = localityVendor
	s.observePrefixReuseTurn(ctx, coord, proxyTurn(500, &agent.UsageTokenDetails{CachedTokens: 400}, nil)) // vendor: skipped

	var b strings.Builder
	s.writePrefixReuseAttributionMetrics(&b)
	lines := map[string]bool{}
	for _, l := range strings.Split(b.String(), "\n") {
		lines[strings.TrimSpace(l)] = true
	}
	for _, want := range []string{
		`fak_gateway_kv_prefix_reused_tokens_by_origin_total{origin="same_session"} 180`,
		`fak_gateway_kv_prefix_reused_tokens_by_origin_total{origin="cross_session"} 60`,
		`fak_gateway_kv_prefix_turns_by_origin_total{origin="cold"} 1`,
		`fak_gateway_kv_prefix_turns_by_origin_total{origin="cross_session"} 1`,
		`fak_gateway_kv_prefix_turns_by_origin_total{origin="same_session"} 2`,
		`fak_gateway_kv_prefix_turns_by_origin_total{origin="unknown"} 1`,
	} {
		if !lines[want] {
			t.Errorf("metrics missing %q\n%s", want, b.String())
		}
	}
}
