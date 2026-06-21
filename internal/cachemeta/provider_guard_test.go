package cachemeta

import "testing"

// The core #112 acceptance: a token-counting benchmark cannot double-count a
// provider prompt-prefix hit as a local reuse win. We model exactly that — a
// mixed stream of one real local KV reuse and one provider cache hit — and assert
// the split keeps them apart with no overlap.
func TestSavingsSplitDoesNotDoubleCountProviderHit(t *testing.T) {
	// A genuine local KV-prefix reuse: 64 tokens re-served by a fak-local cache.
	local := FromKVPrefix(KVPrefix{Tokens: make([]int, 64), ModelID: "m", Owner: "kv"})
	local.Metrics.PrefillTokensSaved = 64

	// A provider prompt-prefix hit: 1200 cached tokens reported by Anthropic.
	remote := FromProviderCache(ProviderCache{
		Provider:     "anthropic",
		ModelID:      "claude-opus",
		CachedTokens: 1200,
		PromptTokens: 1500,
	})

	var split SavingsSplit
	for _, e := range []Entry{local, remote} {
		split.Add(e)
	}

	// The local win counts ONLY the local tokens — the 1200 provider tokens must
	// not leak into the local-reuse total.
	if split.LocalReuseTokens != 64 {
		t.Fatalf("local reuse tokens = %d, want 64 (provider tokens must NOT be counted as a local win)", split.LocalReuseTokens)
	}
	// The provider tokens land in the distinct provider counter.
	if split.ProviderReadTokens != 1200 {
		t.Fatalf("provider read tokens = %d, want 1200", split.ProviderReadTokens)
	}
	if split.ProviderHits != 1 {
		t.Fatalf("provider hits = %d, want 1", split.ProviderHits)
	}

	// Partition property: no token is counted on both sides. A naive (buggy)
	// benchmark that summed PrefillTokensSaved over all entries would report
	// 1264 as a "local win"; the guard caps the local side at the true 64.
	naiveAllAsLocal := local.Metrics.PrefillTokensSaved + remote.Metrics.PrefillTokensSaved
	if split.LocalReuseTokens == naiveAllAsLocal {
		t.Fatalf("guard failed: local reuse (%d) equals the naive count-everything total (%d)", split.LocalReuseTokens, naiveAllAsLocal)
	}
}

func TestLocalReuseTokensRejectsProviderEntry(t *testing.T) {
	remote := FromProviderCache(ProviderCache{Provider: "openai", CachedTokens: 900})
	if got := LocalReuseTokens(remote); got != 0 {
		t.Fatalf("LocalReuseTokens on a provider entry = %d, want 0 (no local credit for a remote hit)", got)
	}
	if got := ProviderReadTokens(remote); got != 900 {
		t.Fatalf("ProviderReadTokens on a provider entry = %d, want 900", got)
	}
	if !IsProviderResidency(remote) || !IsProviderPrefixHit(remote) {
		t.Fatalf("provider entry should be provider-residency prompt-prefix: %+v", remote)
	}
}

func TestLocalEntryNotCountedAsProvider(t *testing.T) {
	local := FromKVPrefix(KVPrefix{Tokens: make([]int, 10), ModelID: "m"})
	local.Metrics.PrefillTokensSaved = 10
	if got := ProviderReadTokens(local); got != 0 {
		t.Fatalf("ProviderReadTokens on a local entry = %d, want 0", got)
	}
	if got := LocalReuseTokens(local); got != 10 {
		t.Fatalf("LocalReuseTokens on a local entry = %d, want 10", got)
	}
	if IsProviderResidency(local) || IsProviderPrefixHit(local) {
		t.Fatalf("local KV entry must not be classified as provider: %+v", local)
	}
}
