package engine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/enginecache"
)

// TestVLLMGovernanceResolvesToEngineVLLM pins the ride-vLLM house-honesty
// contract for issue #40 (vLLM-V1 adapter behind the EngineDriver seam). It is
// the vLLM twin of TestSGLangGovernanceResolvesToEngineSGLang: the serving
// adapter's registered id and the enginecache cache-control referee must agree
// on engine identity, and the honest whole-prefix boundary must be preserved.
//
// Acceptance pinned here (the two items witnessable on a dev box without GPU):
//   - #40 item 4 (cache-control reuse): the adapter reuses enginecache's vLLM
//     coupling; it adds no duplicate flush logic. Identity must match so the
//     referee's /reset_prefix_cache lowering binds to this driver.
//   - #40 item 7 (house-honesty note): exact-span KV governance degrades to
//     whole-prefix reset on a ridden vLLM — enginecache.SupportsExactSpan stays
//     false. This test makes that documented degradation an executable contract,
//     so a future edit that quietly claims exact-span support fails closed.
//
// Generation frame (gen/second-next — architectural option, not default exposure):
//   - Promotion evidence (toward gen/next): the adapter is already registered and
//     lifecycle-capable on main (TestVLLMEngineIsRegisteredLifecycleDriver), drives
//     chat+completions over OpenAI HTTP (TestVLLMHTTPAdapterStreamsChatAndCompletions),
//     feeds the prefix-residency index from KV events
//     (TestVLLMKVEventSubscriptionFeedsResidencyAndCacheMetrics), and normalizes
//     Prometheus into the L2 serving schema (TestVLLMPrometheusNormalization). What
//     remains for promotion is the live-GPU witness (end-to-end /v1/chat/completions
//     AND /v1/messages through a real worker, real KV-event feed, real Prometheus
//     scrape) plus the fak-fronted-vLLM vs raw-vLLM parity-overhead measurement —
//     all of which "would need measurement on real hardware" (not witnessable here).
//   - Demotion/retirement evidence: if the Track-B native engine ships the base
//     items (continuous batching, paged KV, prefix cache) first, this ride's stated
//     value — "serve many GPU nodes before the native engine is ready" — collapses,
//     and the vLLM ride should be demoted toward gen/future or retired rather than
//     promoted.
//   - Invalidating assumption: this driver assumes vLLM's public control plane stays
//     whole-prefix-only. If vLLM ever exposes an exact-span eviction endpoint, the
//     SupportsExactSpan=false assertion below is the intended failure site — it must
//     be flipped deliberately (with the real endpoint wired), never silently.
func TestVLLMGovernanceResolvesToEngineVLLM(t *testing.T) {
	if VLLMEngineID != string(enginecache.EngineVLLM) {
		t.Fatalf("adapter engine id %q != enginecache identity %q", VLLMEngineID, enginecache.EngineVLLM)
	}
	if enginecache.SupportsExactSpan(enginecache.EngineVLLM) {
		t.Fatal("vLLM must not claim exact-span eviction — its public control plane is whole-prefix reset_prefix_cache (issue #40 item 7)")
	}
}
