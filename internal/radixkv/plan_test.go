package radixkv

import "testing"

// overBudgetTree builds a tree whose cached tokens EXCEED its budget: every request is
// served with its lease held (so the insert-time evictToBudget pass finds no unlocked
// victim), then all leases are released. The result is the bounded-eviction plane's
// starting condition — over budget, all leaves unlocked — without a cliff having fired.
func overBudgetTree(budget int, reqs [][]int) *Tree {
	tree := New(budget)
	leaves := make([]*node, 0, len(reqs))
	for _, r := range reqs {
		_, leaf := servePure(tree, r)
		leaves = append(leaves, leaf)
	}
	for _, l := range leaves {
		tree.Done(l)
	}
	return tree
}

func eightByTen() [][]int {
	reqs := make([][]int, 8)
	for i := range reqs {
		reqs[i] = distinctReq(i, 10)
	}
	return reqs
}

// TestBoundedEvictionRatioCapsPerTick: over budget by K=4 victims, ratio 0.25 caps each
// tick at M=2 < K — exactly M evicted per tick, budget reached in ceil(K/M)=2 ticks
// instead of evictToBudget's one-call cliff.
func TestBoundedEvictionRatioCapsPerTick(t *testing.T) {
	tree := overBudgetTree(40, eightByTen()) // 80 tokens cached, budget 40 → K=4 victims of 10

	p1 := tree.PlanBoundedEviction(0.25) // N=8 leaves → cap=ceil(0.25*8)=2
	if p1.OverBudget != 40 || p1.Candidates != 8 || p1.Cap != 2 {
		t.Fatalf("tick1 over/candidates/cap=%d/%d/%d, want 40/8/2", p1.OverBudget, p1.Candidates, p1.Cap)
	}
	if p1.Staged != 2 || p1.StagedTokens != 20 || p1.Remaining != 20 {
		t.Fatalf("tick1 staged/tokens/remaining=%d/%d/%d, want 2/20/20", p1.Staged, p1.StagedTokens, p1.Remaining)
	}
	if got := tree.ConfirmEvictions(); got != 2 {
		t.Fatalf("tick1 confirmed=%d, want 2", got)
	}

	p2 := tree.PlanBoundedEviction(0.25) // N=6 leaves → cap=ceil(1.5)=2; budget reached
	if p2.Staged != 2 || p2.Remaining != 0 {
		t.Fatalf("tick2 staged/remaining=%d/%d, want 2/0", p2.Staged, p2.Remaining)
	}
	if got := tree.ConfirmEvictions(); got != 2 {
		t.Fatalf("tick2 confirmed=%d, want 2", got)
	}

	p3 := tree.PlanBoundedEviction(0.25) // ceil(K/M)=2 ticks sufficed; tick3 is empty
	if p3.Staged != 0 || p3.OverBudget != 0 {
		t.Fatalf("tick3 staged/over=%d/%d, want 0/0 (budget already reached)", p3.Staged, p3.OverBudget)
	}

	st := tree.Stats()
	if st.Tokens != 40 || st.Evictions != 4 || st.BoundedEvictions != 4 {
		t.Fatalf("tokens/evictions/bounded=%d/%d/%d, want 40/4/4", st.Tokens, st.Evictions, st.BoundedEvictions)
	}
}

// TestBoundedEvictionLedgerReconciles: two PARTIAL ticks (planned, never confirmed) leave
// consistent accounting — the second tick resumes past the pending entries without
// double-evicting — and the reconcile confirms every staged victim exactly once: the sum
// of confirmed deletes equals K, and a re-run of the reconcile settles nothing.
func TestBoundedEvictionLedgerReconciles(t *testing.T) {
	tree := overBudgetTree(40, eightByTen()) // K=4 victims

	p1 := tree.PlanBoundedEviction(0.25) // staged, NOT confirmed — the partial tick
	p2 := tree.PlanBoundedEviction(0.25) // resumes: stages the NEXT 2, re-picks nothing
	if p1.Staged != 2 || p2.Staged != 2 {
		t.Fatalf("staged per tick=%d/%d, want 2/2", p1.Staged, p2.Staged)
	}
	if p2.Pending != 4 || p2.Remaining != 0 {
		t.Fatalf("after tick2 pending/remaining=%d/%d, want 4/0", p2.Pending, p2.Remaining)
	}

	st := tree.Stats()
	if st.Tokens != 40 {
		t.Fatalf("tokens=%d, want 40 (detaches already accounted while unconfirmed)", st.Tokens)
	}
	if st.LedgerPending != 4 || st.LedgerConfirmed != 0 || st.BoundedEvictions != 4 {
		t.Fatalf("pending/confirmed/bounded=%d/%d/%d, want 4/0/4",
			st.LedgerPending, st.LedgerConfirmed, st.BoundedEvictions)
	}

	if got := tree.ConfirmEvictions(); got != 4 {
		t.Fatalf("reconcile confirmed=%d, want 4 (== K, none lost)", got)
	}
	if got := tree.ConfirmEvictions(); got != 0 {
		t.Fatalf("re-run reconcile confirmed=%d, want 0 (none double-counted)", got)
	}

	st = tree.Stats()
	if st.LedgerConfirmed != 4 || st.LedgerConfirmedTokens != 40 || st.LedgerPending != 0 {
		t.Fatalf("confirmed/confirmedTokens/pending=%d/%d/%d, want 4/40/0",
			st.LedgerConfirmed, st.LedgerConfirmedTokens, st.LedgerPending)
	}
}

// TestBoundedEvictionAtBudgetEmptyPlan: a tree at/under budget (or unbounded) stages zero
// victims and returns the empty plan; a disabled ratio reports the pressure but stages
// nothing.
func TestBoundedEvictionAtBudgetEmptyPlan(t *testing.T) {
	tree := New(100)
	touchPure(tree, distinctReq(0, 40))
	if p := tree.PlanBoundedEviction(0.5); p != (EvictionPlan{}) {
		t.Fatalf("under-budget plan=%+v, want empty", p)
	}

	unbounded := New(0)
	touchPure(unbounded, distinctReq(0, 40))
	if p := unbounded.PlanBoundedEviction(0.5); p != (EvictionPlan{}) {
		t.Fatalf("unbounded plan=%+v, want empty", p)
	}

	over := overBudgetTree(40, eightByTen())
	p := over.PlanBoundedEviction(0)
	if p.Staged != 0 || p.OverBudget != 40 || p.Remaining != 40 {
		t.Fatalf("disabled-ratio staged/over/remaining=%d/%d/%d, want 0/40/40",
			p.Staged, p.OverBudget, p.Remaining)
	}
	if got := over.Stats().Evictions; got != 0 {
		t.Fatalf("evictions=%d, want 0 (empty and disabled plans evict nothing)", got)
	}
}

// TestBoundedEvictionLedgerBounded: the confirmed-delete ledger never exceeds
// evictionLedgerCap pending entries. Staging saturates at the cap and back-pressures
// (further unconfirmed ticks stage nothing); after the reconcile the drain completes and
// the confirmed total still equals K exactly.
func TestBoundedEvictionLedgerBounded(t *testing.T) {
	reqs := make([][]int, 300)
	for i := range reqs {
		reqs[i] = []int{100000 + i} // 300 single-token leaves
	}
	tree := overBudgetTree(40, reqs) // 300 tokens, budget 40 → K=260 victims

	p1 := tree.PlanBoundedEviction(1.0) // cap=300 but ledger capacity is the bound
	if p1.Staged != evictionLedgerCap || p1.Pending != evictionLedgerCap {
		t.Fatalf("tick1 staged/pending=%d/%d, want %d/%d", p1.Staged, p1.Pending, evictionLedgerCap, evictionLedgerCap)
	}
	p2 := tree.PlanBoundedEviction(1.0) // still unconfirmed: full ledger stages nothing
	if p2.Staged != 0 || p2.Pending != evictionLedgerCap {
		t.Fatalf("tick2 staged/pending=%d/%d, want 0/%d (back-pressure, not growth)", p2.Staged, p2.Pending, evictionLedgerCap)
	}

	c1 := tree.ConfirmEvictions()
	p3 := tree.PlanBoundedEviction(1.0) // capacity freed: the drain resumes
	c2 := tree.ConfirmEvictions()
	if c1+c2 != 260 || p3.Remaining != 0 {
		t.Fatalf("confirmed=%d+%d remaining=%d, want 260 total and 0 remaining", c1, c2, p3.Remaining)
	}

	st := tree.Stats()
	if st.Tokens != 40 || st.LedgerConfirmed != 260 || st.LedgerConfirmedTokens != 260 || st.LedgerPending != 0 {
		t.Fatalf("tokens/confirmed/confirmedTokens/pending=%d/%d/%d/%d, want 40/260/260/0",
			st.Tokens, st.LedgerConfirmed, st.LedgerConfirmedTokens, st.LedgerPending)
	}
}

// TestBoundedEvictionMatchesCliffVictims: the bounded plane picks the SAME victims in the
// SAME order the evictToBudget cliff would for an identically built tree — per tick the
// least-recently-used leaves fall first, and the final survivor set is identical.
func TestBoundedEvictionMatchesCliffVictims(t *testing.T) {
	reqs := eightByTen()

	cliff := overBudgetTree(40, reqs)
	cliff.SetRetention(40) // re-applies the budget → the one-call cliff drain

	bounded := overBudgetTree(40, reqs)
	p1 := bounded.PlanBoundedEviction(0.25)
	if p1.Staged != 2 {
		t.Fatalf("tick1 staged=%d, want 2", p1.Staged)
	}
	for i, r := range reqs { // tick1 must have evicted exactly the 2 LRU-oldest leaves
		want := len(r)
		if i < 2 {
			want = 0
		}
		if m := bounded.MatchLen(r); m != want {
			t.Fatalf("after tick1 req%d matched %d, want %d (oldest-first, same as cliff order)", i, m, want)
		}
	}
	bounded.ConfirmEvictions()
	bounded.PlanBoundedEviction(0.25)
	bounded.ConfirmEvictions()

	for i, r := range reqs { // drained: survivors identical to the cliff's
		cm, bm := cliff.MatchLen(r), bounded.MatchLen(r)
		if cm != bm {
			t.Fatalf("req%d cliff matched %d, bounded matched %d — victim sets diverged", i, cm, bm)
		}
	}
	if ct, bt := cliff.Stats().Tokens, bounded.Stats().Tokens; ct != bt || bt != 40 {
		t.Fatalf("cliff/bounded tokens=%d/%d, want both 40", ct, bt)
	}
}
