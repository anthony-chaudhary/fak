package cachemeta

import "testing"

// T13 / L6 acceptance (epic #1147): a reuse-favorable trace reports a positive
// net with the provider-vs-local split intact. We model a reuse-favorable
// session trace — genuine local KV-prefix reuse alongside a remote provider
// prompt-cache hit — and assert the improvement detector surfaces a POSITIVE net
// local win that is net of the provider tokens a double-counting sink would have
// wrongly folded in, with both sides of the split still recoverable.
func TestDetectImprovementPositiveNetSplitIntact(t *testing.T) {
	// 256 tokens a fak-local cache actually re-served (a real local win).
	local := FromKVPrefix(KVPrefix{Tokens: make([]int, 256), ModelID: "m", Owner: "kv"})
	local.Metrics.PrefillTokensSaved = 256

	// A second local reuse on the same trace — improvement accumulates per scope.
	local2 := FromKVPrefix(KVPrefix{Tokens: make([]int, 128), ModelID: "m", Owner: "kv"})
	local2.Metrics.PrefillTokensSaved = 128

	// 4000 provider cache-read tokens reported by Anthropic — cost telemetry, NOT
	// a local win. A naive sum would credit these to the local improvement.
	remote := FromProviderCache(ProviderCache{
		Provider:     "anthropic",
		ModelID:      "claude-opus",
		CachedTokens: 4000,
		PromptTokens: 5000,
	})

	imp := DetectImprovement("session", []Entry{local, remote, local2})

	// The net is POSITIVE — the trace realized a local reuse improvement.
	if !imp.Positive {
		t.Fatalf("reuse-favorable trace should report a positive net: %+v", imp)
	}

	// The net-true local win is the 384 local tokens ONLY — the 4000 provider
	// tokens are excluded (the double-count guard).
	if imp.NetLocalReuseTokens != 384 {
		t.Fatalf("net local reuse = %d, want 384 (provider tokens must NOT inflate the local win)", imp.NetLocalReuseTokens)
	}

	// The provider-vs-local split is intact and recoverable on the verdict.
	if imp.Split.LocalReuseTokens != 384 {
		t.Fatalf("split local = %d, want 384", imp.Split.LocalReuseTokens)
	}
	if imp.Split.ProviderReadTokens != 4000 {
		t.Fatalf("split provider = %d, want 4000 (kept separate, never folded into the local win)", imp.Split.ProviderReadTokens)
	}
	if imp.Split.ProviderHits != 1 {
		t.Fatalf("split provider hits = %d, want 1", imp.Split.ProviderHits)
	}

	// The guard provably subtracted the double-counted provider tokens: a naive
	// "everything is a local win" sink would have reported 4384, not 384.
	if imp.NaiveAllAsLocalTokens != 4384 {
		t.Fatalf("naive-all-as-local = %d, want 4384", imp.NaiveAllAsLocalTokens)
	}
	if imp.DoubleCountedTokens != 4000 {
		t.Fatalf("double-counted (guarded-out) tokens = %d, want 4000", imp.DoubleCountedTokens)
	}
	if imp.NaiveAllAsLocalTokens-imp.DoubleCountedTokens != imp.NetLocalReuseTokens {
		t.Fatalf("net identity broken: %d - %d != %d", imp.NaiveAllAsLocalTokens, imp.DoubleCountedTokens, imp.NetLocalReuseTokens)
	}
	if imp.NetLocalReuseTokens >= imp.NaiveAllAsLocalTokens {
		t.Fatalf("guard subtracted nothing: net %d >= naive %d", imp.NetLocalReuseTokens, imp.NaiveAllAsLocalTokens)
	}

	// Provenance is labeled by what fak controls vs observes.
	prov := imp.Provenance()
	if prov["net_local_reuse_tokens"] != "WITNESSED" || prov["provider_read_tokens"] != "OBSERVED" {
		t.Fatalf("provenance labels wrong: %+v", prov)
	}
}

// A trace whose ONLY savings were provider-side is not a local positive — the
// guard is what stops a remote hit from reading as a local improvement.
func TestDetectImprovementProviderOnlyIsNotALocalPositive(t *testing.T) {
	remote := FromProviderCache(ProviderCache{Provider: "openai", CachedTokens: 9000})

	imp := DetectImprovement("fleet", []Entry{remote})

	if imp.Positive {
		t.Fatalf("provider-only trace must NOT report a local positive: %+v", imp)
	}
	if imp.NetLocalReuseTokens != 0 {
		t.Fatalf("net local reuse = %d, want 0 (no local cache re-served anything)", imp.NetLocalReuseTokens)
	}
	if imp.Split.ProviderReadTokens != 9000 {
		t.Fatalf("provider read tokens = %d, want 9000 (still observed, just not a local win)", imp.Split.ProviderReadTokens)
	}
}

// The token-input constructor reaches the same verdict from already-summed live
// counters and clamps malformed negatives so a bad counter cannot fabricate a
// false positive.
func TestDetectImprovementFromTokens(t *testing.T) {
	imp := DetectImprovementFromTokens("fleet", 1200, 7000)
	if !imp.Positive || imp.NetLocalReuseTokens != 1200 {
		t.Fatalf("from-tokens positive net = %+v, want positive with 1200 local", imp)
	}
	if imp.DoubleCountedTokens != 7000 || imp.NaiveAllAsLocalTokens != 8200 {
		t.Fatalf("from-tokens split wrong: %+v", imp)
	}

	clamped := DetectImprovementFromTokens("session", -5, -10)
	if clamped.Positive || clamped.NetLocalReuseTokens != 0 || clamped.DoubleCountedTokens != 0 {
		t.Fatalf("negative counters must clamp to a non-positive zero verdict: %+v", clamped)
	}
}
