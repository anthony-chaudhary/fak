package gateway

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestDeepSeekPricingRowsFailClosed pins the two acceptance requirements of the #3009
// price table: a CURRENT price row must exist for BOTH deepseek-v4-pro and
// deepseek-v4-flash (this test fails if either row is removed or zeroed), and every
// other model id must FAIL CLOSED (ok=false, zero row) so no caller prices DeepSeek
// usage off a guessed or stale row.
func TestDeepSeekPricingRowsFailClosed(t *testing.T) {
	want := map[string]DeepSeekCachePricing{
		"deepseek-v4-pro": {
			CacheHitInputPerMTokUSD:  0.003625,
			CacheMissInputPerMTokUSD: 0.435,
			OutputPerMTokUSD:         0.87,
		},
		"deepseek-v4-flash": {
			CacheHitInputPerMTokUSD:  0.0028,
			CacheMissInputPerMTokUSD: 0.14,
			OutputPerMTokUSD:         0.28,
		},
	}
	for model, w := range want {
		p, source, ok := DeepSeekCachePricingFor(model)
		if !ok {
			t.Fatalf("no current DeepSeek price row for %q — the table must carry both V4 rows", model)
		}
		if p != w {
			t.Errorf("%s price row = %+v, want %+v", model, p, w)
		}
		if p.CacheHitInputPerMTokUSD <= 0 || p.CacheMissInputPerMTokUSD <= 0 || p.OutputPerMTokUSD <= 0 {
			t.Errorf("%s carries a zero/negative rate: %+v — a priced row must be fully populated", model, p)
		}
		if p.CacheHitInputPerMTokUSD >= p.CacheMissInputPerMTokUSD {
			t.Errorf("%s hit rate %v >= miss rate %v — the provider's cache discount inverted", model, p.CacheHitInputPerMTokUSD, p.CacheMissInputPerMTokUSD)
		}
		if source == "" {
			t.Errorf("%s price row has no source label — rows must be labeled provider-priced", model)
		}
	}
	// The documented pre-V4 aliases resolve to the flash tier until deprecation.
	for _, alias := range []string{"deepseek-chat", "deepseek-reasoner"} {
		p, _, ok := DeepSeekCachePricingFor(alias)
		if !ok || p != want["deepseek-v4-flash"] {
			t.Errorf("alias %q = (%+v, %v), want the deepseek-v4-flash row", alias, p, ok)
		}
	}
	// Fail closed on everything else — including near-misses and other providers.
	for _, model := range []string{"", "deepseek", "deepseek-v4", "deepseek-v5-pro", "gpt-4o", "claude-opus-4-8"} {
		if p, source, ok := DeepSeekCachePricingFor(model); ok || p != (DeepSeekCachePricing{}) || source != "" {
			t.Errorf("DeepSeekCachePricingFor(%q) = (%+v, %q, %v), want fail-closed zero row", model, p, source, ok)
		}
	}
}

// TestDeepSeekPricingCostUSD prices a warm turn on the provider's own hit/miss split
// and pins the OBSERVED economics: the same turn priced all-miss (the no-cache
// counterfactual) costs strictly more, and the difference is provider-priced discount,
// derived entirely from DeepSeek-published rates and DeepSeek-relayed counters.
func TestDeepSeekPricingCostUSD(t *testing.T) {
	p, _, ok := DeepSeekCachePricingFor("deepseek-v4-pro")
	if !ok {
		t.Fatal("deepseek-v4-pro row missing")
	}
	const hit, miss, out = 800_000, 200_000, 10_000
	got := p.CostUSD(hit, miss, out)
	want := 0.8*0.003625 + 0.2*0.435 + 0.01*0.87
	if !approx(got, want) {
		t.Errorf("CostUSD(warm) = %v, want %v", got, want)
	}
	counterfactual := p.CostUSD(0, hit+miss, out)
	if got >= counterfactual {
		t.Errorf("warm cost %v >= all-miss counterfactual %v — the provider cache discount vanished", got, counterfactual)
	}
}

// TestDeepSeekCacheHitsLandInProviderBucket is the debug-stats attribution fixture for
// #3009: a DeepSeek usage block, parsed off the wire and folded the same way the
// gateway folds every turn (CachedPromptTokens → AdjudicationSummary), must land its
// cache hits in the PROVIDER-observed bucket of the owner split — never in the
// fak-authored bucket. DeepSeek's context caching is on by default, so the hit is the
// provider's own doing unless a separate fak-authored mechanism shaped the request.
func TestDeepSeekCacheHitsLandInProviderBucket(t *testing.T) {
	var usage agent.Usage
	raw := `{"prompt_tokens":1000,"completion_tokens":20,"total_tokens":1020,` +
		`"prompt_cache_hit_tokens":800,"prompt_cache_miss_tokens":200}`
	if err := json.Unmarshal([]byte(raw), &usage); err != nil {
		t.Fatal(err)
	}
	if got := usage.CachedPromptTokens(); got != 800 {
		t.Fatalf("normalized DeepSeek cache hit = %d, want 800", got)
	}
	sum := AdjudicationSummary{CachedPromptTokens: uint64(usage.CachedPromptTokens())}
	ms := sum.MechanismSavings()
	if want := 800 * (1 - CacheReadMultiplier); !approx(ms.ProviderPromptCacheReadTokenEquiv, want) {
		t.Errorf("provider read token-equiv = %v, want %v (hits belong to the provider bucket)", ms.ProviderPromptCacheReadTokenEquiv, want)
	}
	if ms.FakTokenEquiv() != 0 {
		t.Errorf("fak-authored token-equiv = %v, want 0 — DeepSeek's provider-relayed hits must not credit fak", ms.FakTokenEquiv())
	}
	// The /debug/vars owner split renders the same attribution.
	vars := cacheAttributionVars(sum, 0, 0)
	if vars == nil {
		t.Fatal("cache_attribution block missing for a session with provider cache hits")
	}
	if vars.ProviderTokenEquiv <= 0 || vars.FakTokenEquiv != 0 {
		t.Errorf("debug-stats owner split = (provider %v, fak %v), want provider-only attribution", vars.ProviderTokenEquiv, vars.FakTokenEquiv)
	}
}
