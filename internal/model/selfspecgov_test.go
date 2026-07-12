package model

import "testing"

// selfspecgov_test.go — the acceptance-governor decision tests (issue #4354). The
// load-bearing case is the cold-cache -> 0 transition: past warmup, an elevated
// per-draft expert page-in cost with low acceptance must drive the draft depth to
// 0 (colibri's DRAFT=0 auto-off), and a warm cache (cheap page-ins, high accept)
// must keep it at N. The rest pin the default-off posture and the warmup invest.

// colibri's figures: a plain decode routes to ~660 experts/token; on a cold cache
// a verified draft adds a heavy marginal page-in tax, on a warm cache almost none.
const govBasePageIns = 660.0

func TestSelfSpecGovernorColdCacheDisables(t *testing.T) {
	g := SelfSpecGovernor{MaxDraftDepth: 4, WarmupDrafts: 32, BasePageInsPerToken: govBasePageIns}

	// COLD cache, PAST warmup: each draft pages in a heavy marginal set of experts
	// (~440 extra, the 660 -> ~1100 jump) and acceptance is low. The page-in tax
	// dwarfs the decode work the few accepted drafts save -> auto-off.
	if d := g.DraftDepth(0.30, 440, 200); d != 0 {
		t.Fatalf("cold cache past warmup: DraftDepth=%d, want 0 (auto-off charged against page-ins)", d)
	}

	// WARM cache, PAST warmup: experts are mostly resident so the per-draft page-in
	// tax is small and acceptance is high -> keep drafting at the configured depth.
	if d := g.DraftDepth(0.85, 20, 200); d != g.MaxDraftDepth {
		t.Fatalf("warm cache past warmup: DraftDepth=%d, want %d (drafting pays)", d, g.MaxDraftDepth)
	}
}

func TestSelfSpecGovernorWarmupInvests(t *testing.T) {
	g := SelfSpecGovernor{MaxDraftDepth: 4, WarmupDrafts: 32, BasePageInsPerToken: govBasePageIns}

	// BELOW warmup, the SAME cold-cache signal that disables post-warmup must NOT
	// disable: a cold cache can only warm if the governor keeps drafting through
	// warmup, so the configured depth governs regardless of the economics.
	if d := g.DraftDepth(0.30, 440, 5); d != g.MaxDraftDepth {
		t.Fatalf("below warmup: DraftDepth=%d, want %d (invest through warmup)", d, g.MaxDraftDepth)
	}
}

func TestSelfSpecGovernorDefaultOff(t *testing.T) {
	// Zero value (MaxDraftDepth == 0): never speculate, for any input — the
	// default-off posture the readiness seam at config.go:789 preserves.
	var g SelfSpecGovernor
	for _, tc := range []struct {
		accept, pageIns float64
		observed        int
	}{{0.9, 0, 0}, {0.9, 5, 1000}, {0.1, 440, 200}} {
		if d := g.DraftDepth(tc.accept, tc.pageIns, tc.observed); d != 0 {
			t.Fatalf("default-off: DraftDepth(%v,%v,%d)=%d, want 0", tc.accept, tc.pageIns, tc.observed, d)
		}
	}
}

func TestSelfSpecGovernorFreePageInsKeepsDrafting(t *testing.T) {
	g := SelfSpecGovernor{MaxDraftDepth: 4, WarmupDrafts: 32, BasePageInsPerToken: govBasePageIns}

	// Fully-resident experts: zero marginal page-ins. Drafting is free on the cost
	// side, so any positive acceptance keeps it on...
	if d := g.DraftDepth(0.05, 0, 500); d != g.MaxDraftDepth {
		t.Fatalf("free page-ins, accept>0: DraftDepth=%d, want %d", d, g.MaxDraftDepth)
	}
	// ...but zero acceptance yields no bonus token, so there is nothing to gain.
	if d := g.DraftDepth(0, 0, 500); d != 0 {
		t.Fatalf("free page-ins, accept=0: DraftDepth=%d, want 0 (no bonus token)", d)
	}
}

func TestSelfSpecGovernorUnarmedCostModelNeverDisables(t *testing.T) {
	// BasePageInsPerToken unset (<= 0): the cost model has no baseline to charge the
	// tax against, so the governor cannot honestly prove a loss and keeps drafting
	// even under a cold-cache signal past warmup.
	g := SelfSpecGovernor{MaxDraftDepth: 4, WarmupDrafts: 32}
	if d := g.DraftDepth(0.30, 440, 200); d != g.MaxDraftDepth {
		t.Fatalf("unarmed cost model: DraftDepth=%d, want %d (no baseline => no auto-off)", d, g.MaxDraftDepth)
	}
}

func TestSelfSpecGovernorDefaultWarmupFloor(t *testing.T) {
	// WarmupDrafts unset falls back to defaultSelfSpecWarmupDrafts: just below it a
	// cold-cache signal still invests; at/above it the same signal disables.
	g := SelfSpecGovernor{MaxDraftDepth: 4, BasePageInsPerToken: govBasePageIns}
	if d := g.DraftDepth(0.30, 440, defaultSelfSpecWarmupDrafts-1); d != g.MaxDraftDepth {
		t.Fatalf("just below default warmup: DraftDepth=%d, want %d", d, g.MaxDraftDepth)
	}
	if d := g.DraftDepth(0.30, 440, defaultSelfSpecWarmupDrafts); d != 0 {
		t.Fatalf("at default warmup with cold signal: DraftDepth=%d, want 0", d)
	}
}

func TestExpectedTokensPerCycle(t *testing.T) {
	// a==1: every draft accepted -> full n+1 tokens. a==0: only the guaranteed
	// token -> 1. Monotonic in a for fixed n, and always >= 1.
	if got := expectedTokensPerCycle(1, 4); got != 5 {
		t.Fatalf("expectedTokensPerCycle(1,4)=%v, want 5", got)
	}
	if got := expectedTokensPerCycle(0, 4); got != 1 {
		t.Fatalf("expectedTokensPerCycle(0,4)=%v, want 1", got)
	}
	if got := expectedTokensPerCycle(0.5, 0); got != 1 {
		t.Fatalf("expectedTokensPerCycle(0.5,0)=%v, want 1 (no drafts)", got)
	}
	lo, hi := expectedTokensPerCycle(0.3, 4), expectedTokensPerCycle(0.8, 4)
	if !(hi > lo && lo >= 1) {
		t.Fatalf("expected monotonic yield >=1: lo(0.3)=%v hi(0.8)=%v", lo, hi)
	}
}
