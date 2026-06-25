package vcachegov

import (
	"strings"
	"testing"
)

// affinity_test.go covers the cross-shard affinity key (consistency + namespacing)
// and its two safety guards: the correlated-collapse rehash detector and the
// warming-burst cap (Law D3).

func TestAffinityKeyDeterministicAndConsistent(t *testing.T) {
	// The load-bearing property: the SAME (tenant, chain, region) tuple must hash
	// identically across the warm and every recall, or chained requests miss.
	a := AffinityKey("acme", "anchor-P0", "us-east-1")
	b := AffinityKey("acme", "anchor-P0", "us-east-1")
	if a != b {
		t.Fatal("identical tuples must hash identically (warm ↔ recall consistency)")
	}
	if a == "" {
		t.Fatal("affinity key must be non-empty")
	}
}

func TestAffinityKeyNamespacedPerTenantChainRegion(t *testing.T) {
	// Each component change MUST flip the key — that is the whole point of
	// namespacing per-tenant/per-chain/per-region (Law D3).
	base := AffinityKey("acme", "chain-1", "us-east-1")
	for _, diff := range []struct {
		tenant, chain, region string
		why                   string
	}{
		{"acme", "chain-1", "us-west-2", "different region"},
		{"acme", "chain-2", "us-east-1", "different chain"},
		{"globex", "chain-1", "us-east-1", "different tenant"},
	} {
		if got := AffinityKey(diff.tenant, diff.chain, diff.region); got == base {
			t.Errorf("%s produced the same key as base — namespacing broken", diff.why)
		}
	}
}

func TestAffinityKeyConcatenationCollisionGuard(t *testing.T) {
	// The NUL separator must keep "ab"+"c" distinct from "a"+"bc".
	if AffinityKey("ab", "c", "r") == AffinityKey("a", "bc", "r") {
		t.Fatal("separator failed: (ab,c) collided with (a,bc)")
	}
}

func TestAffinityHeaderBounded(t *testing.T) {
	h := AffinityHeader("acme", "anchor-P0", "us-east-1")
	if len(h) > 32 {
		t.Fatalf("AffinityHeader len = %d, want ≤32 (provider hint bound)", len(h))
	}
	// Empty tuple still yields a stable, bounded key (no routing hint ever blank).
	empty := AffinityHeader("", "", "")
	if empty == "" || len(empty) > 32 {
		t.Fatalf("empty-tuple header = %q (len %d), want bounded non-empty", empty, len(empty))
	}
	// The header is a stable prefix of the full key, so warm and recall using
	// different surfaces (full vs header) still agree on identity.
	full := AffinityKey("acme", "anchor-P0", "us-east-1")
	if !strings.HasPrefix(full, h) {
		t.Fatal("AffinityHeader must be a prefix of AffinityKey")
	}
}

func TestRehashDetectsCorrelatedCollapse(t *testing.T) {
	// Autoscale rehash invalidates the whole warm set at once: MANY keys collapse
	// together (§9 / Law D3). Feed 4 keys, each 5 cold observations → signal.
	d := NewRehashDetector()
	for key := 0; key < 4; key++ {
		for i := 0; i < 5; i++ {
			d.Observe(affinityKeyLabel(key), false) // all miss
		}
	}
	if !d.ShouldRewarm() {
		t.Fatal("correlated collapse across 4 keys → want ShouldRewarm")
	}
}

func TestRehashIgnoresSingleKeyDrift(t *testing.T) {
	// One key collapsing is normal (eviction, a stale belief) — NOT a rehash. The
	// detector requires the collapse to be CORRELATED across ≥MinKeys.
	d := NewRehashDetector()
	for i := 0; i < 10; i++ {
		d.Observe("only-key", false) // one key, all cold
	}
	if d.ShouldRewarm() {
		t.Fatal("single-key drift must NOT signal rehash (not correlated)")
	}
}

func TestRehashNeedsEnoughSamples(t *testing.T) {
	// Below MinSamplesPerKey the rate is not trusted — avoid a false signal from
	// one unlucky early miss.
	d := NewRehashDetector()
	for key := 0; key < 4; key++ {
		d.Observe(affinityKeyLabel(key), false) // only 1 sample each
	}
	if d.ShouldRewarm() {
		t.Fatal("below MinSamplesPerKey must not signal rehash")
	}
}

func TestRehashHealthySetDoesNotRewarm(t *testing.T) {
	// A warm set hitting well is not collapsing → no re-warm.
	d := NewRehashDetector()
	for key := 0; key < 4; key++ {
		for i := 0; i < 5; i++ {
			d.Observe(affinityKeyLabel(key), true) // all hit
		}
	}
	if d.ShouldRewarm() {
		t.Fatal("healthy warm set must not signal rehash")
	}
}

func TestRehashResetStartsFreshWindow(t *testing.T) {
	d := NewRehashDetector()
	for key := 0; key < 4; key++ {
		for i := 0; i < 5; i++ {
			d.Observe(affinityKeyLabel(key), false)
		}
	}
	d.Reset()
	if d.ShouldRewarm() {
		t.Fatal("after Reset the window must be empty (no signal)")
	}
}

func TestBurstCapThrottlesAndReopens(t *testing.T) {
	// Law D3: cap the warming-burst rate so vCache doesn't self-trigger the rehash
	// that invalidates its own set. 2 warms per 100ms window.
	cap := NewBurstCap(2, 100)
	if !cap.Allow(0) || !cap.Allow(10) {
		t.Fatal("first two warms in window must be allowed")
	}
	if cap.Allow(20) {
		t.Fatal("third warm in the same window must be refused (burst cap)")
	}
	// After the window elapses, the cap reopens to a FRESH budget of 2.
	if !cap.Allow(101) || !cap.Allow(105) {
		t.Fatal("warms after window elapsed must be allowed (fresh budget)")
	}
	if cap.Allow(110) {
		t.Fatal("third warm in the new window must be refused")
	}
}

func TestBurstCapFirstCallAtZeroDoesNotDoubleReset(t *testing.T) {
	// Regression guard for the t=0 ambiguity: a cap whose first observation lands at
	// nowMillis==0 must not treat every later call in-window as a new window.
	cap := NewBurstCap(1, 100)
	if !cap.Allow(0) {
		t.Fatal("first warm at t=0 must be allowed")
	}
	if cap.Allow(5) {
		t.Fatal("second warm at t=5 (same window, cap=1) must be refused")
	}
}

func TestBurstCapRefusesZeroConfig(t *testing.T) {
	// Fail-closed: a zero/negative cap admits nothing (the caller mis-configured).
	for _, c := range []struct {
		max int
		win int64
	}{{0, 100}, {2, 0}, {-1, 100}} {
		if NewBurstCap(c.max, c.win).Allow(0) {
			t.Errorf("BurstCap{%d,%d} must refuse, admitted", c.max, c.win)
		}
	}
}

func affinityKeyLabel(n int) string {
	return string(rune('A' + n))
}
