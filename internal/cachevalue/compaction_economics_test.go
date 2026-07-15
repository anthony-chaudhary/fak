package cachevalue

import "testing"

// TestClassifyCompactionCacheHostileWarmPrefix is the #3028 load-bearing case:
// rewriting early history sheds tokens that were WARM (already served as cheap
// cache_read), but busting the prefix forces the provider to re-create it as
// full-price cache-creation writes. The prompt shrank, yet net token-equivalent
// cost ROSE — the rewrite must be reported as cache-hostile, not a win.
func TestClassifyCompactionCacheHostileWarmPrefix(t *testing.T) {
	// Shed 100k tokens that were all warm (worth only 0.1x to keep, so 10k
	// tok-eq saved), but the rewrite paid 50k cache-creation writes to re-warm
	// (1.25x = 62.5k tok-eq). Net = 10k - 62.5k = -52.5k: cache-hostile.
	v := ClassifyCompaction(CompactionSample{
		PromptTokensBefore:         420000,
		PromptTokensAfter:          320000,
		ShedTokens:                 100000,
		ShedWarmTokens:             100000,
		RewriteCacheCreationTokens: 50000,
		RewriteKnown:               true,
	})
	if v.Reason != ReasonCompactionCacheHostile {
		t.Fatalf("reason = %q, want %q (a warm-prefix rewrite that re-warms more than it sheds is cache-hostile)", v.Reason, ReasonCompactionCacheHostile)
	}
	if !v.CacheHostile() {
		t.Errorf("CacheHostile() = false, want true for reason %q", v.Reason)
	}
	if !within(v.ShedValueSavedTokEq, 10000, tol) {
		t.Errorf("shed value = %.3f, want 10000 (100k warm @0.1x)", v.ShedValueSavedTokEq)
	}
	if !within(v.RewriteCostTokEq, 62500, tol) {
		t.Errorf("rewrite cost = %.3f, want 62500 (50k @1.25x)", v.RewriteCostTokEq)
	}
	if !within(v.NetSavedTokEq, -52500, tol) {
		t.Errorf("net = %.3f, want -52500 (shed saved less than the re-warm cost)", v.NetSavedTokEq)
	}
	if v.UncachedTokenDelta != 50000 {
		t.Errorf("uncached-token delta = %d, want 50000 (the re-warm writes the shrink forced)", v.UncachedTokenDelta)
	}
}

// TestClassifyCompactionNetCheaperColdShed: shedding COLD (full-price) tokens with
// a small re-warm cost is genuinely net cheaper — the good case must not be
// mislabeled hostile.
func TestClassifyCompactionNetCheaperColdShed(t *testing.T) {
	// Shed 100k cold tokens (worth 1x = 100k tok-eq) with only 5k re-warm writes
	// (6.25k tok-eq). Net = +93.75k: net cheaper.
	v := ClassifyCompaction(CompactionSample{
		ShedTokens:                 100000,
		ShedWarmTokens:             0,
		RewriteCacheCreationTokens: 5000,
		RewriteKnown:               true,
	})
	if v.Reason != ReasonCompactionNetCheaper {
		t.Fatalf("reason = %q, want %q", v.Reason, ReasonCompactionNetCheaper)
	}
	if v.CacheHostile() {
		t.Errorf("CacheHostile() = true, want false for a net-cheaper cold shed")
	}
	if !within(v.NetSavedTokEq, 93750, tol) {
		t.Errorf("net = %.3f, want 93750", v.NetSavedTokEq)
	}
}

// TestClassifyCompactionEconomicsUnknownAbstains: when the provider did not report
// the post-rewrite cache-creation counter, the shed value is still reported but
// the net verdict abstains — never a phantom-zero re-warm cost that would fake a
// win. This is the honesty fence: fak reports counters, it cannot force a hit.
func TestClassifyCompactionEconomicsUnknownAbstains(t *testing.T) {
	v := ClassifyCompaction(CompactionSample{
		ShedTokens:     100000,
		ShedWarmTokens: 100000,
		RewriteKnown:   false, // provider dollar-blind / codex row
	})
	if v.Reason != ReasonCompactionEconomicsUnknown {
		t.Fatalf("reason = %q, want %q", v.Reason, ReasonCompactionEconomicsUnknown)
	}
	if v.NetKnown {
		t.Errorf("NetKnown = true, want false (no re-warm counter, no net claim)")
	}
	if v.RewriteCostTokEq != 0 || v.NetSavedTokEq != 0 {
		t.Errorf("rewrite cost / net = %.3f / %.3f, want 0 / 0 when net is unknown", v.RewriteCostTokEq, v.NetSavedTokEq)
	}
	// The shed value is still reported — the "tokens shed" story survives even
	// when the "net cheaper" story cannot be proven.
	if !within(v.ShedValueSavedTokEq, 10000, tol) {
		t.Errorf("shed value = %.3f, want 10000 even under an unknown net", v.ShedValueSavedTokEq)
	}
	if v.CacheHostile() {
		t.Errorf("CacheHostile() = true, want false — an abstain is not a hostile flag")
	}
}

// TestClassifyCompactionNeutralNoShed: a pass that shed nothing has no compaction
// economics to judge either way, even with known counters.
func TestClassifyCompactionNeutralNoShed(t *testing.T) {
	v := ClassifyCompaction(CompactionSample{
		ShedTokens:                 0,
		RewriteCacheCreationTokens: 1000,
		RewriteKnown:               true,
	})
	if v.Reason != ReasonCompactionNeutral {
		t.Fatalf("reason = %q, want %q (nothing shed)", v.Reason, ReasonCompactionNeutral)
	}
	if v.CacheHostile() {
		t.Errorf("CacheHostile() = true, want false when nothing was shed")
	}
}

// TestClassifyCompactionWarmClampAndBreakeven guards the warm-split clamp (a
// caller over-reporting warm > shed must not push cold negative) and the
// dead-band (an exactly-break-even net reads NEUTRAL, not a spurious verdict).
func TestClassifyCompactionWarmClampAndBreakeven(t *testing.T) {
	// Warm over-reported (200k > 100k shed) clamps to 100k warm, 0 cold: shed
	// value = 10k. Break-even re-warm: 10k / 1.25x = 8000 creation tokens →
	// cost 10k, net 0 → NEUTRAL (inside the dead-band).
	v := ClassifyCompaction(CompactionSample{
		ShedTokens:                 100000,
		ShedWarmTokens:             200000,
		RewriteCacheCreationTokens: 8000,
		RewriteKnown:               true,
	})
	if !within(v.ShedValueSavedTokEq, 10000, tol) {
		t.Errorf("shed value = %.3f, want 10000 (warm clamped to shed, no negative cold)", v.ShedValueSavedTokEq)
	}
	if !within(v.NetSavedTokEq, 0, tol) {
		t.Errorf("net = %.3f, want ~0 at break-even", v.NetSavedTokEq)
	}
	if v.Reason != ReasonCompactionNeutral {
		t.Errorf("reason = %q, want %q at break-even (dead-band)", v.Reason, ReasonCompactionNeutral)
	}
}
