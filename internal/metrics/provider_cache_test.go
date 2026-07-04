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

// FoldCacheEntry must also fold the provider's cache-WRITE (creation) tokens into
// the Arm's distinct write axis. A provider charges a write premium the first time
// it caches a prefix; ablate prices that premium off ProviderCacheCreationTokens
// (internal/ablate: ProviderPromptCacheWritePremiumTokenEquiv), so silently
// dropping it on the entry-fold path under-prices the write as zero. A provider
// entry carrying WriteTokens must therefore land in the Arm's creation axis — never
// in a local counter, and never conflated with the read-hit axis.
func TestFoldCacheEntryFoldsProviderCreationTokens(t *testing.T) {
	var a Arm

	remote := cachemeta.FromProviderCache(cachemeta.ProviderCache{
		Provider:     "anthropic",
		ModelID:      "claude-opus",
		CachedTokens: 1200,
		WriteTokens:  512,
		PromptTokens: 1500,
	})
	if handled := a.FoldCacheEntry(remote); !handled {
		t.Fatalf("provider entry must be handled as provider telemetry")
	}

	// Read hits and the write premium land in their OWN distinct provider axes.
	if a.ProviderCacheReadTokens != 1200 {
		t.Fatalf("ProviderCacheReadTokens = %d, want 1200", a.ProviderCacheReadTokens)
	}
	if a.ProviderCacheCreationTokens != 512 {
		t.Fatalf("ProviderCacheCreationTokens = %d, want 512 (write premium must fold, not be dropped)", a.ProviderCacheCreationTokens)
	}
	// The provider write axis is remote cost, never a local reuse win.
	if a.InTokens != 0 || a.VDSOHits != 0 {
		t.Fatalf("provider creation-token fold leaked into local counters: InTokens=%d VDSOHits=%d", a.InTokens, a.VDSOHits)
	}

	// A provider entry with NO write premium leaves the creation axis untouched.
	var b Arm
	noWrite := cachemeta.FromProviderCache(cachemeta.ProviderCache{Provider: "openai", CachedTokens: 900})
	b.FoldCacheEntry(noWrite)
	if b.ProviderCacheCreationTokens != 0 {
		t.Fatalf("ProviderCacheCreationTokens = %d, want 0 when the provider reported no write tokens", b.ProviderCacheCreationTokens)
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
