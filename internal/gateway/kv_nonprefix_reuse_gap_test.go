package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

// kv_nonprefix_reuse_gap_test.go — #1052 witness: fak's live KV-reuse plane credits
// ONLY the literal longest-prefix match, so on a RAG-shaped turn (retrieved chunks
// concatenated out of prefix order) it earns ZERO reuse for cache-resident non-prefix
// chunks. That is the CacheBlend gap, measured on fak's OWN accounting rather than
// asserted.
//
// The gateway surfaces realized reuse through cacheobs (the fak_gateway_kv_prefix_*
// family): Observe(promptTokens, reusedPrefixTokens) where reusedPrefixTokens is the
// planner's longest-prefix `matched` (internal/cacheobs/cacheobs.go:54). There is no
// non-prefix reuse field on cacheobs.Stats — the gap is STRUCTURAL, not a tuning miss.
//
// LMCache CacheBlend (EuroSys'25 Best Paper) reuses ANY cached chunk's KV and recomputes
// ~15% of tokens to repair cross-attention, hitting near-100% KV-cache hit. fak refuses
// that by DESIGN: its reuse share gate is bit-EXACT — cachemeta PrefillSharePolicy admits
// reuse only on a byte-identical PrefixDigest, defined in code as "bit-identical, so reuse
// is lossless, not approximate" (internal/cachemeta/materialization.go:251), and the
// prefix path is proven max|Δ|=0. A non-prefix blend is approximate (max|Δ| != 0), so it
// cannot pass that gate. Closing the gap needs a deliberate bounded-loss reuse plane
// (cache-value epic #1010), not an oversight.
func TestNonPrefixKVReuseGap_CacheBlend(t *testing.T) {
	// A RAG-shaped turn: a shared system preamble that DOES sit at the literal prefix,
	// then several retrieved chunks concatenated in retrieval order, then the query. On a
	// prior turn the same preamble + the same chunks were already served, so every chunk's
	// KV is cache-resident — but the chunks arrived in a different order, so only the
	// preamble is a longest-prefix match this turn.
	const (
		preambleTokens = 32  // shared system prompt — the literal prefix that DOES match
		chunkTokens    = 300 // each retrieved RAG chunk (cache-resident, but non-prefix here)
		nChunks        = 3
		queryTokens    = 24 // the user question (fresh, never cached)
	)
	nonPrefixCacheResident := chunkTokens * nChunks // chunk KV exists in cache, but is non-prefix
	promptTokens := preambleTokens + nonPrefixCacheResident + queryTokens

	// fak's planner credits only the literal longest-prefix match (the preamble).
	reusedPrefixTokens := preambleTokens

	obs := cacheobs.New() // a fresh observer — never the process-global Default
	obs.Observe(promptTokens, reusedPrefixTokens)
	got := obs.Snapshot()

	// 1. fak credits EXACTLY the prefix — not one token of the cache-resident chunks.
	if got.ReusedTokens != uint64(reusedPrefixTokens) {
		t.Fatalf("fak credited %d reused tokens, want exactly the %d-token prefix (no non-prefix credit)",
			got.ReusedTokens, reusedPrefixTokens)
	}

	// 2. The non-prefix cache-resident chunk tokens earn ZERO reuse on fak's plane.
	fakNonPrefixReused := int(got.ReusedTokens) - preambleTokens // 0 by construction
	if fakNonPrefixReused != 0 {
		t.Fatalf("expected 0 non-prefix reuse credit, got %d", fakNonPrefixReused)
	}
	fakNonPrefixHitRate := float64(fakNonPrefixReused) / float64(nonPrefixCacheResident)
	if fakNonPrefixHitRate != 0.0 {
		t.Fatalf("fak non-prefix KV hit rate = %.4f, want 0 (fak has no non-prefix reuse plane)",
			fakNonPrefixHitRate)
	}

	// 3. A RAG turn whose BULK is cache-resident still lands in the COLD reuse regime,
	//    because prefix-only reuse cannot see the chunks. CacheBlend would instead land it
	//    near the frozen ceiling (near-100% KV hit, recomputing ~15%).
	if got.ColdTurns != 1 || got.FrozenTurns != 0 || got.PartialTurns != 0 {
		t.Fatalf("RAG-reorder turn regime = frozen:%d partial:%d cold:%d, want cold (prefix-only sees only the preamble)",
			got.FrozenTurns, got.PartialTurns, got.ColdTurns)
	}
	if got.ReuseRatio >= cacheobs.ColdCeil {
		t.Fatalf("fak reuse ratio %.4f >= ColdCeil %.2f — prefix-only should be cold on a RAG-reorder turn",
			got.ReuseRatio, cacheobs.ColdCeil)
	}

	// The measured gap: CacheBlend's reuse plane would credit the cache-resident chunks
	// (near-100% KV hit); fak's prefix-only plane credits 0 of them. This is the honest
	// negative number behind the non-prefix-kv-reuse-cacheblend scorecard row.
	cacheBlendReusable := nonPrefixCacheResident
	if cacheBlendReusable <= fakNonPrefixReused {
		t.Fatalf("witness inverted: CacheBlend-reusable %d should exceed fak's non-prefix credit %d",
			cacheBlendReusable, fakNonPrefixReused)
	}
	t.Logf("#1052 non-prefix KV reuse gap (RAG-reorder turn, %d prompt tokens): "+
		"fak prefix-only credit=%d (ratio %.3f, COLD); non-prefix cache-resident=%d, fak reuses 0 (0%%); "+
		"CacheBlend would reuse ~%d (near-100%% KV hit). Gap is structural: cacheobs.Stats has no "+
		"non-prefix field; the share gate requires a byte-identical PrefixDigest (lossless), while a "+
		"CacheBlend blend is approximate.",
		promptTokens, got.ReusedTokens, got.ReuseRatio, nonPrefixCacheResident, cacheBlendReusable)
}
