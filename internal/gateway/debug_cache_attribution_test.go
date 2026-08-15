package gateway

import (
	"encoding/json"
	"strings"
	"testing"
)

// The /debug/vars cache_attribution block must carry the SAME provider-vs-fak owner split
// the /metrics fak_cache_saved_token_equiv_by_owner family emits (writeCacheAttributionMetrics
// reads the identical MechanismSavings), so an operator watching a live session reads one
// consistent split — not a provider-only headline (#1849). This pins that the builder folds
// the summary + VDSO the same way the metrics renderer does.
func TestCacheAttributionVarsMatchesMechanismSplit(t *testing.T) {
	sum := AdjudicationSummary{
		CachedPromptTokens:        1000, // provider read rebate = 900 token-equiv
		CacheCreationTokens:       200,  // provider write premium = -50 token-equiv
		CompactionShedTokens:      300,
		CompactionCacheReadTokens: 200, // warm witness: 200 of the 300 shed priced at the read marginal
		KVPrefixReusedTokens:      400,
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
	// The warm witness rides along so the info tab can explain why the shed prices below its raw count.
	if got.FakCompactionCacheReadTokens != 200 {
		t.Errorf("fak compaction warm witness = %d, want 200 (the cache_read the shed prices on)", got.FakCompactionCacheReadTokens)
	}
	// VDSO is an avoided-call counter (not a token-equiv), folded exactly as /metrics does.
	if got.FakVDSOAvoidedCalls != uint64(vdsoHits)+servedInline {
		t.Errorf("vdso avoided calls = %d, want %d (VDSOHits + inline-served)", got.FakVDSOAvoidedCalls, uint64(vdsoHits)+servedInline)
	}
	if got.FakResponseMemoCalls != uint64(vdsoHits) || got.FakInlineServedCalls != servedInline {
		t.Fatalf("producer provenance memo=%d inline=%d, want memo=%d inline=%d", got.FakResponseMemoCalls, got.FakInlineServedCalls, vdsoHits, servedInline)
	}
}

// The cold-tool-DEFER shed (#3647) must ride the /debug/vars block as its OWN fields, never folded
// into the compaction shed: they are different mechanisms in different currencies — compaction shed
// is priced in token-equiv, defer prices NOTHING (every def still ships; the reduction is
// provider-side, only the hot core loading into context). This pins the three-way separation the
// `fak info` Cache tab renders, and that the deferred tool NAMES survive to the wire so the pane can
// say WHICH tools went cold rather than only how many.
func TestCacheAttributionVarsCarriesColdToolDeferShedDistinctly(t *testing.T) {
	sum := AdjudicationSummary{
		CompactionShedTokens: 300,
		KVPrefixReusedTokens: 400,
		DeferColdTurns:       3,
		DeferColdCount:       12,
		DeferColdToolNames:   []string{"mcp__dos__dos_arbitrate", "mcp__fak__fak_index_docs"},
	}
	got := cacheAttributionVars(sum, 0, 0)
	if got == nil {
		t.Fatal("cacheAttributionVars returned nil for a defer-on session with shed activity")
	}
	if got.FakDeferColdTurns != 3 || got.FakDeferColdCount != 12 {
		t.Errorf("defer shed = (turns %d, count %d), want (3, 12)", got.FakDeferColdTurns, got.FakDeferColdCount)
	}
	if len(got.FakDeferColdToolNames) != 2 || got.FakDeferColdToolNames[0] != "mcp__dos__dos_arbitrate" {
		t.Errorf("deferred tool names = %v, want the producer's sorted set", got.FakDeferColdToolNames)
	}
	// The separation that matters: the defer counters never leak into the compaction-shed field, and
	// the defer count is NOT priced into any token-equiv (defer buys no tokens).
	if got.FakCompactionShedTokens != 300 {
		t.Errorf("compaction shed = %d, want 300 (defer must not be folded in)", got.FakCompactionShedTokens)
	}
	ms := sum.MechanismSavings()
	if !approx(got.FakTokenEquiv, ms.FakTokenEquiv()) {
		t.Errorf("fak token-equiv = %v, want %v — the defer shed must add no token-equiv", got.FakTokenEquiv, ms.FakTokenEquiv())
	}

	// A defer-ON session with NO token activity and NO avoided calls must STILL emit the block:
	// defer can never reach HasAnyTokenActivity (it prices no tokens), so gating on token activity
	// alone would make the lever invisible on exactly the session #3647 asks to surface.
	only := cacheAttributionVars(AdjudicationSummary{DeferColdTurns: 1, DeferColdCount: 4}, 0, 0)
	if only == nil {
		t.Fatal("defer-on session with no token slice must still emit cache_attribution, got nil")
	}
	if only.FakDeferColdCount != 4 || only.FakCompactionShedTokens != 0 {
		t.Errorf("defer-only block = (defer %d, shed %d), want (4, 0)", only.FakDeferColdCount, only.FakCompactionShedTokens)
	}

	// On the WIRE the defer fields are omitempty, so a defer-OFF session keeps the block byte-stable
	// (no all-zero defer keys that would read as a measured "0 tools deferred").
	on, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"fak_defer_cold_turns", "fak_defer_cold_count", "fak_defer_cold_tool_names"} {
		if !strings.Contains(string(on), key) {
			t.Errorf("defer-on wire block missing %q: %s", key, on)
		}
	}
	off, err := json.Marshal(cacheAttributionVars(AdjudicationSummary{CompactionShedTokens: 300}, 0, 0))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(off), "fak_defer_cold") {
		t.Errorf("defer-off wire block must omit the defer keys, got: %s", off)
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
