package radixkv

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// prewarm_test.go is the HOST-FREE witness for the self-hosted tool-latency prewarm
// (issue #810): the closed loop from compute's byte-known WarmNow verdict, through the
// in-tree warm, to a real turn landing HOT — plus the lowest-priority eviction class
// (fence 2) and the demand-reuse promotion. It imports internal/compute (tier 1) from the
// radixkv test (tier 3) on purpose: that downward edge is the production binding the
// serve loop makes (decide in compute, warm here), exercised end-to-end without a model
// or a GPU. The wall-clock RadixAttention-hit-rate readback on a running model stays
// host-gated; the matching-algorithm hit rate witnessed here is model-independent.

// TestPrewarmWarmNowLandsHot is the headline closed loop. A turn has cached its context;
// the tool result about to slot in is byte-known but not yet cached, so the next request's
// prefix is a partial (cold) hit. The scheduler's WarmNow verdict drives the warm during
// the idle tool window, and the real prefix then matches in full — it lands hot instead of
// cold-prefilling the result span.
func TestPrewarmWarmNowLandsHot(t *testing.T) {
	tree := New(0)           // unbounded: isolate the hit-rate effect from eviction
	ctx := seq(1, 40)        // the demand context already served this turn
	result := seq(500, 8)    // the tool result about to slot in — byte-known NOW
	next := cat(ctx, result) // the next /v1/messages prefix, deterministic at dispatch

	_, leaf := servePure(tree, ctx)
	tree.Done(leaf)
	if m := tree.MatchLen(next); m != len(ctx) {
		t.Fatalf("pre-warm: next prefix matched %d, want %d (context hot, result cold)", m, len(ctx))
	}

	// At the tool-call boundary: byte-known prefix, free warm pool, a 500ms tool window, a
	// 50ms warm, 1s residency -> WarmNow (it completes before the request and survives to it).
	cand := compute.PrewarmCandidate{
		PrefixByteKnown:   true,
		WarmPoolFree:      true,
		ToolLatencyMillis: 500,
		WarmMillis:        50,
		ResidencyMillis:   1000,
		PrefixTokens:      len(next),
	}
	dec := compute.DecidePrewarmAdmission(cand)
	if dec.Verdict != compute.WarmNow {
		t.Fatalf("decision = %v, want WarmNow", dec.Verdict)
	}

	// Drive the self-hosted warm off the verdict during the idle tool window.
	warmed := tree.WarmInsert(next, nil)
	if warmed != len(result) {
		t.Fatalf("warmed %d tokens, want %d (only the un-cached result suffix)", warmed, len(result))
	}

	if m := tree.MatchLen(next); m != len(next) {
		t.Fatalf("post-warm: next prefix matched %d, want %d (lands hot)", m, len(next))
	}
	if w := tree.WarmTokens(); w != len(result) {
		t.Fatalf("warm residency %d, want %d (the warmed result span)", w, len(result))
	}
	t.Logf("prewarm: cold hit %d/%d -> hot hit %d/%d after warming %d byte-known tokens",
		len(ctx), len(next), len(next), len(next), warmed)
}

// TestPrewarmOnlyWarmsOnWarmNow proves the binding warms ONLY on WarmNow. Every other
// verdict (unknown prefix, pool pressure, or a defer that means "warm later, not now")
// must leave the tree cold — the caller never touches the cache off a non-WarmNow verdict.
func TestPrewarmOnlyWarmsOnWarmNow(t *testing.T) {
	cases := []struct {
		name string
		cand compute.PrewarmCandidate
		want compute.PrewarmVerdict
	}{
		{"prefix_not_known", compute.PrewarmCandidate{PrefixByteKnown: false, WarmPoolFree: true, ToolLatencyMillis: 500, WarmMillis: 50, ResidencyMillis: 1000}, compute.WarmSkip},
		{"pool_pressure", compute.PrewarmCandidate{PrefixByteKnown: true, WarmPoolFree: false, ToolLatencyMillis: 500, WarmMillis: 50, ResidencyMillis: 1000}, compute.WarmSkip},
		{"defer_lands_too_early", compute.PrewarmCandidate{PrefixByteKnown: true, WarmPoolFree: true, ToolLatencyMillis: 5000, WarmMillis: 50, ResidencyMillis: 100}, compute.WarmDefer},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tree := New(0)
			ctx := seq(1, 20)
			next := cat(ctx, seq(500, 8))
			_, leaf := servePure(tree, ctx)
			tree.Done(leaf)

			dec := compute.DecidePrewarmAdmission(tc.cand)
			if dec.Verdict != tc.want {
				t.Fatalf("verdict = %v, want %v", dec.Verdict, tc.want)
			}
			if dec.Verdict == compute.WarmNow {
				tree.WarmInsert(next, nil)
			}
			if m := tree.MatchLen(next); m != len(ctx) {
				t.Fatalf("%s: result warmed despite %v (matched %d, want %d)", tc.name, tc.want, m, len(ctx))
			}
			if w := tree.WarmTokens(); w != 0 {
				t.Fatalf("%s: warm residency %d, want 0 (no warm off %v)", tc.name, w, tc.want)
			}
		})
	}
}

// TestPrewarmEvictedBeforeDemand is fence 2: an opportunistic warm is reclaimed before any
// demand prefix under memory pressure, even though it was placed AFTER the demand prefix
// (more recent in wall time). The lowest-priority class is always the first victim — it
// never displaces real residency.
func TestPrewarmEvictedBeforeDemand(t *testing.T) {
	tree := New(20) // budget for exactly two 10-token prefixes
	demand := distinctReq(0, 10)
	_, ld := servePure(tree, demand) // a real prefix, served this turn
	tree.Done(ld)

	warm := distinctReq(1, 10)
	if got := tree.WarmInsert(warm, nil); got != 10 {
		t.Fatalf("warmed %d, want 10", got)
	}
	if m := tree.MatchLen(demand); m != 10 {
		t.Fatalf("demand matched %d, want 10 (both fit budget)", m)
	}
	if m := tree.MatchLen(warm); m != 10 {
		t.Fatalf("warm matched %d, want 10 (both fit budget)", m)
	}

	// A NEW demand prefix arrives -> 30 tokens > 20 budget -> one prefix must go.
	d2 := distinctReq(2, 10)
	_, l2 := servePure(tree, d2)
	tree.Done(l2)

	if m := tree.MatchLen(warm); m != 0 {
		t.Errorf("warm should be evicted before demand, but matched %d", m)
	}
	if m := tree.MatchLen(demand); m != 10 {
		t.Errorf("demand must survive an opportunistic warm's pressure, matched %d/10", m)
	}
	if m := tree.MatchLen(d2); m != 10 {
		t.Errorf("new demand should be resident, matched %d/10", m)
	}
}

// TestPrewarmReclaimedUnderFullPool is the belt-and-suspenders fail-safe: a warm into a
// pool already saturated by demand is reclaimed by its OWN eviction pass before it costs a
// single demand token. The decision layer's WarmPoolFree gate should refuse this case, but
// the mechanism stays safe even if it were bypassed.
func TestPrewarmReclaimedUnderFullPool(t *testing.T) {
	tree := New(20)
	a, b := distinctReq(0, 10), distinctReq(1, 10)
	for _, r := range [][]int{a, b} {
		_, l := servePure(tree, r)
		tree.Done(l)
	}
	tree.WarmInsert(distinctReq(2, 10), nil)

	if w := tree.WarmTokens(); w != 0 {
		t.Errorf("warm into a full pool should be reclaimed immediately, residency=%d", w)
	}
	if m := tree.MatchLen(a); m != 10 {
		t.Errorf("oldest demand must survive a warm's pressure, matched %d/10", m)
	}
	if m := tree.MatchLen(b); m != 10 {
		t.Errorf("demand b must survive, matched %d/10", m)
	}
	if st := tree.Stats(); st.Tokens != 20 {
		t.Errorf("tokens=%d, want 20 (warm dropped, demand intact)", st.Tokens)
	}
}

// TestPrewarmPromotedByDemandReuse is the closed loop: when the real turn reuses a warmed
// prefix, the Lookup that matches it freshens its recency, promoting it out of the
// opportunistic class into demand residency. A warm that paid off stops being a
// reclaim-first bet.
func TestPrewarmPromotedByDemandReuse(t *testing.T) {
	tree := New(0)
	warm := seq(1, 16)
	if tree.WarmInsert(warm, nil) != 16 {
		t.Fatal("warm not placed")
	}
	if w := tree.WarmTokens(); w != 16 {
		t.Fatalf("warm residency %d, want 16 (un-promoted)", w)
	}

	m, leaf := servePure(tree, warm) // the real turn lands on the warmed prefix
	tree.Done(leaf)
	if m != 16 {
		t.Fatalf("demand reused %d of the warmed prefix, want 16 (full hot hit)", m)
	}
	if w := tree.WarmTokens(); w != 0 {
		t.Errorf("warm should be promoted to demand residency after reuse, residency=%d", w)
	}
}
