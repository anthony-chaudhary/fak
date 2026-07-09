package agent

import (
	"context"
	"testing"
)

// #3390: the lookup (cacheability) vs retrieve (realized) split at the planner seam.
// generateReused's `matched` is the REALIZED reuse — tokens actually served from cached
// KV. The `cacheable` count is the LOOKUP-side index match, taken before servability
// (nil KV, exact-hit refeed) can trim it. These tests witness the three regimes: a cold
// turn (both zero), a servable hit (both equal), and the load-bearing gap case — a
// prefix that matched the index but had no servable KV, where the old single tap
// folded the whole prefix into "miss" and the split keeps the lookup hit visible.

// splitTurn runs one turn and returns the (cacheable, matched) pair from the result.
func splitTurn(t *testing.T, p *InKernelPlanner, ids []int) (cacheable, matched int) {
	t.Helper()
	res, err := p.generateReusedRecovering(context.Background(), ids, 2, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil)
	if err != nil {
		t.Fatalf("generateReusedRecovering: %v", err)
	}
	if res.cacheable < res.matched {
		t.Fatalf("invariant broken: cacheable %d < matched %d", res.cacheable, res.matched)
	}
	return res.cacheable, res.matched
}

func TestSplitColdTurnBothZero(t *testing.T) {
	p := reusePlanner(true, false, tinyCfg())
	ids := synthIDs(tinyCfg().VocabSize, 16, 1)
	cacheable, matched := splitTurn(t, p, ids)
	if cacheable != 0 || matched != 0 {
		t.Fatalf("cold turn: cacheable=%d matched=%d, want 0/0", cacheable, matched)
	}
}

func TestSplitServableHitAgrees(t *testing.T) {
	cfg := tinyCfg()
	p := reusePlanner(true, false, cfg)
	base := synthIDs(cfg.VocabSize, 16, 1)
	splitTurn(t, p, base) // seed the tree with a real, servable KV prefix

	extended := append(append([]int(nil), base...), synthIDs(cfg.VocabSize, 4, 2)...)
	cacheable, matched := splitTurn(t, p, extended)
	if cacheable != len(base) || matched != len(base) {
		t.Fatalf("servable hit: cacheable=%d matched=%d, want both %d (lookup and retrieve agree)",
			cacheable, matched, len(base))
	}
}

// The gap case: the prefix is IN the index (a pure-accounting insert holds the tokens
// but no KV payload — the shape a lookup hit has after its KV is gone) so lookup
// matches the whole prompt, yet nothing is servable. The realized tap alone reports
// this turn as a total miss; the split must keep the full lookup hit visible.
func TestSplitIndexMatchWithoutKVKeepsCacheableAboveMatched(t *testing.T) {
	cfg := tinyCfg()
	p := reusePlanner(true, false, cfg)
	ids := synthIDs(cfg.VocabSize, 16, 3)

	b, m := p.tree.Lookup(ids)
	leaf := p.tree.Insert(b, ids[m:], nil) // tokens in the index, kv = nil
	p.tree.Done(leaf)

	cacheable, matched := splitTurn(t, p, ids)
	if cacheable != len(ids) {
		t.Fatalf("cacheable = %d, want %d (the full prompt matched the index)", cacheable, len(ids))
	}
	if matched != 0 {
		t.Fatalf("matched = %d, want 0 (nothing was servable)", matched)
	}
}

func TestSplitReuseDisabledBothZero(t *testing.T) {
	p := reusePlanner(false, false, tinyCfg())
	ids := synthIDs(tinyCfg().VocabSize, 16, 4)
	cacheable, matched := splitTurn(t, p, ids)
	if cacheable != 0 || matched != 0 {
		t.Fatalf("reuse off: cacheable=%d matched=%d, want 0/0 (no lookup ever ran)", cacheable, matched)
	}
}
