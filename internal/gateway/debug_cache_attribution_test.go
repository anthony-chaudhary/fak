package gateway

import "testing"

// The /debug/vars cache_attribution block must carry the SAME provider-vs-fak owner split
// the /metrics fak_cache_saved_token_equiv_by_owner family emits (writeCacheAttributionMetrics
// reads the identical MechanismSavings), so an operator watching a live session reads one
// consistent split — not a provider-only headline (#1849). This pins that the builder folds
// the summary + VDSO the same way the metrics renderer does.
func TestCacheAttributionVarsMatchesMechanismSplit(t *testing.T) {
	sum := AdjudicationSummary{
		CachedPromptTokens:   1000, // provider read rebate = 900 token-equiv
		CacheCreationTokens:  200,  // provider write premium = -50 token-equiv
		CompactionShedTokens: 300,
		KVPrefixReusedTokens: 400,
	}
	// Mirror the /metrics fold exactly: MechanismSavings() + kernel VDSOHits + inline-served.
	const vdsoHits, servedInline = int64(7), uint64(3)
	ms := sum.MechanismSavings()

	got := cacheAttributionVars(sum, vdsoHits, servedInline)
	if got == nil {
		t.Fatal("cacheAttributionVars returned nil for a session with cache activity")
	}
	if !approx(got.ProviderTokenEquiv, ms.ProviderTokenEquiv()) {
		t.Errorf("provider token-equiv = %v, want %v (must match /metrics by_owner)", got.ProviderTokenEquiv, ms.ProviderTokenEquiv())
	}
	if !approx(got.FakTokenEquiv, ms.FakTokenEquiv()) {
		t.Errorf("fak token-equiv = %v, want %v (must match /metrics by_owner)", got.FakTokenEquiv, ms.FakTokenEquiv())
	}
	if !approx(got.TotalTokenEquiv, ms.TotalTokenEquiv()) {
		t.Errorf("total token-equiv = %v, want %v", got.TotalTokenEquiv, ms.TotalTokenEquiv())
	}
	if !approx(got.ProviderPromptCacheReadTokenEquiv, 900) || !approx(got.ProviderPromptCacheWritePremiumTokenEquiv, -50) {
		t.Errorf("provider mechanism split = (read %v, write %v), want (900, -50)",
			got.ProviderPromptCacheReadTokenEquiv, got.ProviderPromptCacheWritePremiumTokenEquiv)
	}
	if got.FakCompactionShedTokens != 300 || got.FakKVPrefixReusedTokens != 400 {
		t.Errorf("fak mechanism split = (shed %d, kv %d), want (300, 400)", got.FakCompactionShedTokens, got.FakKVPrefixReusedTokens)
	}
	// VDSO is an avoided-call counter (not a token-equiv), folded exactly as /metrics does.
	if got.FakVDSOAvoidedCalls != uint64(vdsoHits)+servedInline {
		t.Errorf("vdso avoided calls = %d, want %d (VDSOHits + inline-served)", got.FakVDSOAvoidedCalls, uint64(vdsoHits)+servedInline)
	}
}

// A cold session with no cache activity and no avoided calls emits NO block, so an operator
// sees a quiet surface rather than an all-zero object that reads like a measured "0 saving".
func TestCacheAttributionVarsNilWhenEmpty(t *testing.T) {
	if got := cacheAttributionVars(AdjudicationSummary{}, 0, 0); got != nil {
		t.Fatalf("empty session cache_attribution = %+v, want nil (omitted)", got)
	}
}

// The provider slice alone (fak anchor-starved, #1407) still renders — the block appears with
// fak reading ~0 rather than being suppressed, so the honest provider-vs-fak split is visible
// even when fak has authored nothing yet.
func TestCacheAttributionVarsRendersProviderOnlyWhenFakStarved(t *testing.T) {
	got := cacheAttributionVars(AdjudicationSummary{CachedPromptTokens: 1000}, 0, 0)
	if got == nil {
		t.Fatal("provider-only session must still render the split, got nil")
	}
	if got.FakTokenEquiv != 0 {
		t.Errorf("anchor-starved fak token-equiv = %v, want 0 (honest ~0)", got.FakTokenEquiv)
	}
	if !approx(got.ProviderTokenEquiv, 900) {
		t.Errorf("provider token-equiv = %v, want 900", got.ProviderTokenEquiv)
	}
}
