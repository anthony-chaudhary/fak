package gateway

import (
	"testing"
	"time"
)

// TestRecordCacheCreationTierSplitObservesThe5mTier is the #2789 fix witness on the
// producer side. recordCacheCreationTierSplit used to return early whenever the turn
// was not on the 1h rung, so an all-5m session left every tier counter at zero —
// byte-identical to a session no gateway ever watched. The durable creation row
// therefore had to fall back to the 5m DEFAULT even when fak had, in fact, looked at
// the live tier of every single write. The tier is now recorded for both arms.
func TestRecordCacheCreationTierSplitObservesThe5mTier(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	// Every write went out on the plain 5m tier, and fak watched each one.
	m.recordCacheCreationTierSplit(300, false, false)
	m.recordCacheCreationTierSplit(700, false, false)

	sum := m.adjudicationSummary()
	if sum.CacheCreationTokensTierObserved != 1000 {
		t.Fatalf("CacheCreationTokensTierObserved = %d, want 1000 — an observed 5m tier is still an observation", sum.CacheCreationTokensTierObserved)
	}
	// Observing 5m must never be mistaken for an upgrade: the 1h split and its
	// head/message-prefix arms price at 2.0x and stay strictly the upgraded subset.
	if sum.CacheCreationTokensUpgraded != 0 {
		t.Fatalf("CacheCreationTokensUpgraded = %d, want 0 on an all-5m session", sum.CacheCreationTokensUpgraded)
	}
	if sum.CacheCreationTokensHeadOnly != 0 || sum.CacheCreationTokensMessagePrefix != 0 {
		t.Fatalf("5m writes leaked into the upgrade arms: head %d, message prefix %d",
			sum.CacheCreationTokensHeadOnly, sum.CacheCreationTokensMessagePrefix)
	}
}

// TestRecordCacheCreationTierSplitObservedSpanCoversBothTiers pins the invariant the
// durable row reads: the observed span is the whole watched creation total, of which
// the upgraded split is a subset — never the other way round.
func TestRecordCacheCreationTierSplitObservedSpanCoversBothTiers(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.recordCacheCreationTierSplit(300, false, false) // 5m
	m.recordCacheCreationTierSplit(700, true, false)  // 1h, head only
	m.recordCacheCreationTierSplit(500, true, true)   // 1h, message prefix
	m.recordCacheCreationTierSplit(0, true, true)     // no write: nothing to observe
	m.recordCacheCreationTierSplit(-42, false, false) // guard: a negative counter is not an observation

	sum := m.adjudicationSummary()
	if sum.CacheCreationTokensTierObserved != 1500 {
		t.Fatalf("CacheCreationTokensTierObserved = %d, want 1500 (300 at 5m + 1200 at 1h)", sum.CacheCreationTokensTierObserved)
	}
	if sum.CacheCreationTokensUpgraded != 1200 {
		t.Fatalf("CacheCreationTokensUpgraded = %d, want 1200", sum.CacheCreationTokensUpgraded)
	}
	if sum.CacheCreationTokensUpgraded > sum.CacheCreationTokensTierObserved {
		t.Fatalf("upgraded %d exceeds observed %d — the 1h split must stay a subset of what was observed",
			sum.CacheCreationTokensUpgraded, sum.CacheCreationTokensTierObserved)
	}
	if sum.CacheCreationTokensHeadOnly != 700 || sum.CacheCreationTokensMessagePrefix != 500 {
		t.Fatalf("upgrade arms = head %d, message prefix %d; want 700/500",
			sum.CacheCreationTokensHeadOnly, sum.CacheCreationTokensMessagePrefix)
	}
}

// TestRecordCacheCreationTierSplitNilMetricsIsInert keeps the observation off the
// request path's error budget: a gateway running without metrics must not panic.
func TestRecordCacheCreationTierSplitNilMetricsIsInert(t *testing.T) {
	var m *gatewayMetrics
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("nil metrics panicked: %v", r)
		}
	}()
	m.recordCacheCreationTierSplit(100, false, false)
	m.recordCacheCreationTierSplit(100, true, true)
}
