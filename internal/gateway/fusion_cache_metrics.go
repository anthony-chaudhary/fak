package gateway

import (
	"fmt"
	"strings"
)

// writeFusionCacheMetrics prices observed prefix reuse against two baselines: no_reuse
// (every prompt token prefilled) and same_session_only (a per-conversation prefix cache
// that cannot share across sibling agents, the next-best alternative). Turns whose
// upstream reported no cached count are excluded from both sides, so unknown is never
// scored as a miss. The counters let a dashboard window the ratio with increase().
func writeFusionCacheMetrics(b *strings.Builder, snap prefixReuseSnapshot) {
	prompt := snap.reportedPromptTokens
	reused := min(snap.sameTokens+snap.crossTokens, prompt)
	prefilled := prompt - reused
	writeHelpType(b, "fak_gateway_fusion_cache_prompt_tokens_total",
		"Prompt tokens of in-kernel and self-hosted proxy turns with a reported cached count: the prefill a no-reuse server would pay.", "counter")
	fmt.Fprintf(b, "fak_gateway_fusion_cache_prompt_tokens_total %d\n", prompt)
	writeHelpType(b, "fak_gateway_fusion_cache_prefilled_tokens_total",
		"Prompt tokens actually prefilled (prompt minus prefix-reused) on the same turns.", "counter")
	fmt.Fprintf(b, "fak_gateway_fusion_cache_prefilled_tokens_total %d\n", prefilled)
	writeHelpType(b, "fak_gateway_fusion_cache_saved_tokens_total",
		"Prefill tokens saved by prefix reuse versus a baseline. no_reuse: all reuse; same_session_only: only reuse a per-conversation cache could not have provided (cross_session).", "counter")
	fmt.Fprintf(b, "fak_gateway_fusion_cache_saved_tokens_total{baseline=\"no_reuse\"} %d\n", reused)
	fmt.Fprintf(b, "fak_gateway_fusion_cache_saved_tokens_total{baseline=\"same_session_only\"} %d\n", min(snap.crossTokens, reused))
	if prefilled == 0 {
		return // ratio undefined with no data or a fully cached window
	}
	writeHelpType(b, "fak_gateway_fusion_cache_advantage_ratio",
		"Baseline prefill tokens divided by actual prefilled tokens since start; above 1 means the fused cache did less prefill work than the baseline. Window it as increase(baseline)/increase(prefilled_tokens_total).", "gauge")
	sameOnly := prefilled + min(snap.crossTokens, reused)
	fmt.Fprintf(b, "fak_gateway_fusion_cache_advantage_ratio{baseline=\"no_reuse\"} %s\n", promFloat(float64(prompt)/float64(prefilled)))
	fmt.Fprintf(b, "fak_gateway_fusion_cache_advantage_ratio{baseline=\"same_session_only\"} %s\n", promFloat(float64(sameOnly)/float64(prefilled)))
}
