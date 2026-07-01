package metrics

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// Issue #112: a provider prompt-prefix cache hit is folded into the Arm under its
// OWN counters, distinct from the local-reuse fields (VDSOHits / InTokens). A
// token-counting benchmark routing a mixed entry stream through FoldCacheEntry
// must never see a provider hit land in a local counter.
func TestFoldCacheEntryRoutesProviderHitToProviderCounters(t *testing.T) {
	var a Arm

	// A genuine local KV reuse: not provider-resident, so FoldCacheEntry must
	// leave it for the local accounting (returns false, touches no provider field).
	local := cachemeta.FromKVPrefix(cachemeta.KVPrefix{Tokens: make([]int, 64), ModelID: "m", Owner: "kv"})
	local.Metrics.PrefillTokensSaved = 64
	if handled := a.FoldCacheEntry(local); handled {
		t.Fatalf("local KV entry must NOT be handled as provider telemetry")
	}

	// A provider prompt-prefix hit: 1200 cached tokens reported by Anthropic.
	remote := cachemeta.FromProviderCache(cachemeta.ProviderCache{
		Provider:     "anthropic",
		ModelID:      "claude-opus",
		CachedTokens: 1200,
		PromptTokens: 1500,
	})
	if handled := a.FoldCacheEntry(remote); !handled {
		t.Fatalf("provider entry must be handled as provider telemetry")
	}

	// The provider tokens land ONLY in the provider counters.
	if a.ProviderCacheReadTokens != 1200 {
		t.Fatalf("ProviderCacheReadTokens = %d, want 1200", a.ProviderCacheReadTokens)
	}
	if a.ProviderCacheHits != 1 {
		t.Fatalf("ProviderCacheHits = %d, want 1", a.ProviderCacheHits)
	}
	// The local-reuse counters were untouched by the provider fold: the 1200
	// provider tokens did not leak into any local field.
	if a.InTokens != 0 || a.VDSOHits != 0 {
		t.Fatalf("provider fold leaked into local counters: InTokens=%d VDSOHits=%d", a.InTokens, a.VDSOHits)
	}
}

// FoldSavingsSplit copies only the provider side of a cachemeta split into the
// Arm, keeping local reuse out of the provider counters.
func TestFoldSavingsSplitCopiesOnlyProviderSide(t *testing.T) {
	local := cachemeta.FromKVPrefix(cachemeta.KVPrefix{Tokens: make([]int, 10), ModelID: "m"})
	local.Metrics.PrefillTokensSaved = 10
	remote := cachemeta.FromProviderCache(cachemeta.ProviderCache{Provider: "openai", CachedTokens: 900})

	var split cachemeta.SavingsSplit
	split.Add(local)
	split.Add(remote)

	var a Arm
	a.FoldSavingsSplit(split)

	if a.ProviderCacheReadTokens != 900 {
		t.Fatalf("ProviderCacheReadTokens = %d, want 900 (provider side only)", a.ProviderCacheReadTokens)
	}
	if a.ProviderCacheHits != 1 {
		t.Fatalf("ProviderCacheHits = %d, want 1", a.ProviderCacheHits)
	}
	// The local side (10 tokens) must not have been merged into any provider field.
	if a.InTokens != 0 {
		t.Fatalf("local reuse leaked into Arm.InTokens: %d", a.InTokens)
	}
}

// The provider counters serialize under their own JSON labels, distinct from the
// local-reuse fields, so a metrics consumer can report them separately.
func TestProviderCacheCountersSerializeDistinctly(t *testing.T) {
	a := Arm{ProviderCacheHits: 2, ProviderCacheReadTokens: 4096, ProviderCacheCreationTokens: 512, VDSOHits: 7}
	b, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("marshal Arm: %v", err)
	}
	s := string(b)
	for _, key := range []string{`"provider_cache_hits":2`, `"provider_cache_read_tokens":4096`, `"provider_cache_creation_tokens":512`, `"vdso_hits":7`} {
		if !strings.Contains(s, key) {
			t.Fatalf("Arm JSON missing %q: %s", key, s)
		}
	}
}
