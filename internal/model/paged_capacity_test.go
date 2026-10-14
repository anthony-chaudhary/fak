package model

import "testing"

// paged_capacity_test.go — #3386 proof: the capacity-aware, ALL-OR-NOTHING batch reserve.
// PagedKVPool.MaxBlocks (0 == unbounded) bounds the pool's growth for reservations, and
// PagedKV.TryReserveBlocks(n) either mints the WHOLE n-block batch (true) or — on any
// shortfall against free list + remaining growth budget — mints NOTHING and returns false,
// leaving the pool and page table untouched (no partial commit, so no rollback path).
// PagedKV.Reserve is backed by the same check, so a bounded pool fails a too-large
// reservation cleanly instead of growing unbounded, while MaxBlocks == 0 preserves the
// pre-#3386 always-succeeds contract for every existing caller.
//
// Tests intentionally live in their own file (not paged_reserve_test.go) and reuse its
// pagedReserveCfg helper; blockTokens is 4 throughout so token↔block arithmetic is explicit.

// poolSnapshot captures the observable allocator state a refused reservation must not touch.
type poolSnapshot struct {
	nBlocks  int
	nFree    int
	free     []int
	ref      []int
	physical int
}

func snapshotPool(p *PagedKVPool) poolSnapshot {
	return poolSnapshot{
		nBlocks:  len(p.blocks),
		nFree:    len(p.free),
		free:     append([]int(nil), p.free...),
		ref:      append([]int(nil), p.ref...),
		physical: p.PhysicalBlocks(),
	}
}

func assertPoolUnchanged(t *testing.T, tag string, p *PagedKVPool, want poolSnapshot) {
	t.Helper()
	got := snapshotPool(p)
	if got.nBlocks != want.nBlocks {
		t.Fatalf("%s: len(blocks) = %d, want %d (backing store mutated)", tag, got.nBlocks, want.nBlocks)
	}
	if got.nFree != want.nFree {
		t.Fatalf("%s: len(free) = %d, want %d (free list mutated)", tag, got.nFree, want.nFree)
	}
	for i := range want.free {
		if got.free[i] != want.free[i] {
			t.Fatalf("%s: free[%d] = %d, want %d", tag, i, got.free[i], want.free[i])
		}
	}
	for i := range want.ref {
		if got.ref[i] != want.ref[i] {
			t.Fatalf("%s: ref[%d] = %d, want %d", tag, i, got.ref[i], want.ref[i])
		}
	}
	if got.physical != want.physical {
		t.Fatalf("%s: PhysicalBlocks() = %d, want %d", tag, got.physical, want.physical)
	}
}

// (a) Sufficient budget: TryReserveBlocks returns true and mints exactly n owned blocks.
func TestTryReserveBlocksSufficientBudget(t *testing.T) {
	pool := NewPagedKVPool(pagedReserveCfg(), 4)
	pool.MaxBlocks = 8
	s := pool.NewSequence()

	if !s.TryReserveBlocks(5) {
		t.Fatalf("TryReserveBlocks(5) = false with budget 8, want true")
	}
	if got := s.Blocks(); got != 5 {
		t.Fatalf("Blocks() = %d after TryReserveBlocks(5), want exactly 5", got)
	}
	if got := len(pool.blocks); got != 5 {
		t.Fatalf("len(pool.blocks) = %d, want exactly 5 minted", got)
	}
	for i, b := range s.table {
		if pool.ref[b] != 1 {
			t.Fatalf("reserved block %d (id %d) ref = %d, want 1 (owned)", i, b, pool.ref[b])
		}
	}
	if got := len(pool.free); got != 0 {
		t.Fatalf("len(pool.free) = %d after fresh mint, want 0", got)
	}
	// Len is untouched: reserved blocks sit past the live tail.
	if got := s.Len(); got != 0 {
		t.Fatalf("Len() = %d after reserve, want 0", got)
	}
}

// (b) Insufficient budget (MaxBlocks set low): false, and the pool is observably unchanged —
// free-list (count and content), len(blocks), refcounts, and the page table all identical.
func TestTryReserveBlocksInsufficientBudgetMutatesNothing(t *testing.T) {
	pool := NewPagedKVPool(pagedReserveCfg(), 4)
	pool.MaxBlocks = 2
	s := pool.NewSequence()
	if !s.TryReserveBlocks(1) {
		t.Fatalf("TryReserveBlocks(1) = false with budget 2, want true")
	}

	before := snapshotPool(pool)
	tableBefore := s.Blocks()

	if s.TryReserveBlocks(2) { // 0 free + budget (2-1)=1 < 2
		t.Fatalf("TryReserveBlocks(2) = true with only 1 block of headroom, want false")
	}
	assertPoolUnchanged(t, "refused batch", pool, before)
	if got := s.Blocks(); got != tableBefore {
		t.Fatalf("Blocks() = %d after refused reserve, want %d (page table mutated)", got, tableBefore)
	}
}

// (c) MaxBlocks == 0: unbounded, the pre-#3386 contract — every batch succeeds, and Reserve
// grows exactly as before.
func TestTryReserveBlocksUnboundedDefaultPreserved(t *testing.T) {
	pool := NewPagedKVPool(pagedReserveCfg(), 4) // MaxBlocks left at zero value
	s := pool.NewSequence()

	if !s.TryReserveBlocks(64) {
		t.Fatalf("TryReserveBlocks(64) = false on unbounded pool, want true")
	}
	if got := s.Blocks(); got != 64 {
		t.Fatalf("Blocks() = %d, want 64", got)
	}

	// Reserve on an unbounded pool still always satisfies: 1000 tokens at blockTokens=4.
	s2 := pool.NewSequence()
	s2.Reserve(1000)
	if got, want := s2.Blocks(), pool.blocksForTokens(1000); got != want {
		t.Fatalf("unbounded Reserve(1000): Blocks() = %d, want %d", got, want)
	}
}

// (d) Partial availability — some free blocks, but free + growth budget still short of n:
// false, and NOTHING is minted or consumed from the free list.
func TestTryReserveBlocksPartialAvailabilityAllOrNothing(t *testing.T) {
	pool := NewPagedKVPool(pagedReserveCfg(), 4)
	pool.MaxBlocks = 4

	owner := pool.NewSequence()
	if !owner.TryReserveBlocks(2) {
		t.Fatalf("setup: TryReserveBlocks(2) = false, want true")
	}
	freed := pool.NewSequence()
	if !freed.TryReserveBlocks(1) {
		t.Fatalf("setup: TryReserveBlocks(1) = false, want true")
	}
	freed.Free() // 1 block on the free list; len(blocks)=3, growth budget = 4-3 = 1

	before := snapshotPool(pool)
	if before.nFree != 1 || before.nBlocks != 3 {
		t.Fatalf("setup: free=%d blocks=%d, want free=1 blocks=3", before.nFree, before.nBlocks)
	}

	s := pool.NewSequence()
	if s.TryReserveBlocks(3) { // available = 1 free + 1 growth = 2 < 3
		t.Fatalf("TryReserveBlocks(3) = true with only 2 satisfiable, want false")
	}
	assertPoolUnchanged(t, "partial availability", pool, before)
	if got := s.Blocks(); got != 0 {
		t.Fatalf("Blocks() = %d after refused partial reserve, want 0 (nothing minted)", got)
	}

	// The exact fit (free + growth) still succeeds afterwards: all-or-nothing, not sticky-failed.
	if !s.TryReserveBlocks(2) {
		t.Fatalf("TryReserveBlocks(2) = false with 2 satisfiable, want true")
	}
	if got := len(pool.blocks); got != 4 {
		t.Fatalf("len(pool.blocks) = %d after exact-fit reserve, want 4 (== MaxBlocks)", got)
	}
	if got := len(pool.free); got != 0 {
		t.Fatalf("len(pool.free) = %d after exact-fit reserve, want 0 (free block reused)", got)
	}
}

// Reserve is backed by the same all-or-nothing check: on a bounded pool a reservation that
// cannot be FULLY satisfied is a clean no-op instead of unbounded growth, and a later
// reservation that fits still succeeds.
func TestReserveBackedByCapacityCheck(t *testing.T) {
	pool := NewPagedKVPool(pagedReserveCfg(), 4)
	pool.MaxBlocks = 2
	s := pool.NewSequence()

	before := snapshotPool(pool)
	s.Reserve(12) // wants 3 blocks > budget 2 — must mint nothing
	assertPoolUnchanged(t, "over-budget Reserve", pool, before)
	if got := s.Blocks(); got != 0 {
		t.Fatalf("Blocks() = %d after over-budget Reserve, want 0", got)
	}

	s.Reserve(8) // wants 2 blocks == budget — succeeds in full
	if got := s.Blocks(); got != 2 {
		t.Fatalf("Blocks() = %d after in-budget Reserve(8), want 2", got)
	}
}

// Non-positive n is trivially satisfied: true, nothing minted.
func TestTryReserveBlocksNonPositiveNoOp(t *testing.T) {
	pool := NewPagedKVPool(pagedReserveCfg(), 4)
	pool.MaxBlocks = 1
	s := pool.NewSequence()
	before := snapshotPool(pool)
	for _, n := range []int{0, -3} {
		if !s.TryReserveBlocks(n) {
			t.Fatalf("TryReserveBlocks(%d) = false, want true (trivially satisfied)", n)
		}
	}
	assertPoolUnchanged(t, "non-positive n", pool, before)
	if got := s.Blocks(); got != 0 {
		t.Fatalf("Blocks() = %d after non-positive reserves, want 0", got)
	}
}

// Integration: blocks minted by TryReserveBlocks are real reserved capacity — subsequent
// Appends draw from the page table without minting new pool blocks, and the appended bytes
// gather back bit-exact (the #34 Reserve contract carried through the #3386 path).
func TestTryReserveBlocksThenAppendReusesReservedBlocks(t *testing.T) {
	pool := NewPagedKVPool(pagedReserveCfg(), 4)
	pool.MaxBlocks = 2
	wantK := make([][]float32, pool.nLayers)
	wantV := make([][]float32, pool.nLayers)
	for l := range wantK {
		wantK[l], wantV[l] = []float32{}, []float32{}
	}
	s := pool.NewSequence()
	if !s.TryReserveBlocks(2) {
		t.Fatalf("TryReserveBlocks(2) = false with budget 2, want true")
	}
	for tok := 0; tok < 8; tok++ { // exactly fills 2 blocks of 4 tokens
		appendPagedToken(s, wantK, wantV, tok)
	}
	if got := len(pool.blocks); got != 2 {
		t.Fatalf("len(pool.blocks) = %d after appends into reserved blocks, want 2 (no mid-append mint)", got)
	}
	assertGathersEqual(t, "reserved-then-appended", s, wantK, wantV)
}

// fak-test:runtime fast est=1ms lane=default
func TestPagedKVDownwardCap(t *testing.T) {
	t.Parallel()
	// Keep the feature witness compilable on the parent: absence is a missing
	// capability, not a claim that a serving path already used this setter.
	type downwardCap interface{ LowerMaxBlocks(int) bool }
	lowerer := func(t *testing.T, p *PagedKVPool) downwardCap {
		t.Helper()
		c, ok := any(p).(downwardCap)
		if !ok {
			t.Fatal("paged reservation pool has no downward-only cap capability")
		}
		return c
	}

	t.Run("sticky reservation ceiling", func(t *testing.T) {
		p := NewPagedKVPool(pagedReserveCfg(), 4)
		p.MaxBlocks = 8
		c := lowerer(t, p)
		if !c.LowerMaxBlocks(2) || c.LowerMaxBlocks(2) || c.LowerMaxBlocks(4) || c.LowerMaxBlocks(-1) {
			t.Fatal("lowering must be strict, nonnegative and idempotent")
		}
		if p.MaxBlocks != 8 {
			t.Fatalf("configured MaxBlocks = %d, want unchanged 8", p.MaxBlocks)
		}
		s := p.NewSequence()
		before := snapshotPool(p)
		p.MaxBlocks = 1
		if s.TryReserveBlocks(2) || c.LowerMaxBlocks(1) {
			t.Fatal("stricter configured cap was ignored or equal lowering was accepted")
		}
		p.MaxBlocks = 0 // Legacy unbounded configuration cannot erase the latch.
		if s.TryReserveBlocks(3) {
			t.Fatal("reservation grew past the latched ceiling")
		}
		s.Reserve(12)
		assertPoolUnchanged(t, "lowered TryReserveBlocks and Reserve", p, before)
		if s.Blocks() != 0 {
			t.Fatal("refused reservation changed the page table")
		}
		if !s.TryReserveBlocks(2) || p.TotalBlocks() != 2 {
			t.Fatal("exact lowered-cap reservation did not succeed")
		}
		p.MaxBlocks = 16
		if c.LowerMaxBlocks(16) || s.TryReserveBlocks(1) {
			t.Fatal("a larger configuration or lowering call regrew the pool")
		}
	})

	t.Run("reuse above a lowered and zero ceiling", func(t *testing.T) {
		p := NewPagedKVPool(pagedReserveCfg(), 4)
		p.MaxBlocks = 8
		live, wantK, wantV := newPagedSeq(p, 3)
		spare := p.NewSequence()
		if !spare.TryReserveBlocks(2) {
			t.Fatal("setup reservation failed")
		}
		spare.Free() // Three backing blocks, two already allocated and reusable.
		before := snapshotPool(p)
		c := lowerer(t, p)
		if !c.LowerMaxBlocks(1) || !c.LowerMaxBlocks(0) || c.LowerMaxBlocks(0) || c.LowerMaxBlocks(1) {
			t.Fatal("explicit zero must be a strict, persistent lowering")
		}
		assertPoolUnchanged(t, "lowering below existing backing", p, before)
		assertGathersEqual(t, "lowering preserves KV", live, wantK, wantV)
		s := p.NewSequence()
		if s.TryReserveBlocks(3) {
			t.Fatal("zero ceiling allowed new backing allocation")
		}
		s.Reserve(12)
		assertPoolUnchanged(t, "refusal preserves reusable blocks", p, before)
		if s.Blocks() != 0 {
			t.Fatal("refused batch partially reserved free blocks")
		}
		if !s.TryReserveBlocks(2) || p.TotalBlocks() != 3 || p.FreeBlocks() != 0 {
			t.Fatal("zero ceiling must permit reuse without growing backing")
		}
		p.MaxBlocks = -1
		if s.TryReserveBlocks(1) || !s.TryReserveBlocks(0) || !s.TryReserveBlocks(-1) {
			t.Fatal("zero ceiling lost growth refusal or nonpositive no-op behavior")
		}
		assertGathersEqual(t, "reuse preserves live KV", live, wantK, wantV)
	})

	t.Run("only successful lowering installs a latch", func(t *testing.T) {
		for _, configured := range []int{-1, 0, 2} {
			p := NewPagedKVPool(pagedReserveCfg(), 4)
			p.MaxBlocks = configured
			c := lowerer(t, p)
			if c.LowerMaxBlocks(-1) {
				t.Fatal("negative lowering was accepted")
			}
			if configured > 0 && (c.LowerMaxBlocks(configured) || c.LowerMaxBlocks(configured+1)) {
				t.Fatal("equal or higher than configured lowering was accepted")
			}
			if p.MaxBlocks != configured {
				t.Fatal("rejected lowering changed the configured cap")
			}
			p.MaxBlocks = 6
			if !p.NewSequence().TryReserveBlocks(6) {
				t.Fatal("rejected lowering installed a hidden latch")
			}
		}
		for _, configured := range []int{-1, 0} {
			p := NewPagedKVPool(pagedReserveCfg(), 4)
			p.MaxBlocks = configured
			if !lowerer(t, p).LowerMaxBlocks(0) || p.NewSequence().TryReserveBlocks(1) || p.MaxBlocks != configured {
				t.Fatal("unbounded configuration did not accept an explicit zero ceiling")
			}
		}
	})
}

// fak-test:runtime fast est=1ms lane=default
func TestPagedKVDownwardCapLegacyConfiguration(t *testing.T) {
	t.Parallel()
	// This control stays green on the parent: without a successful lowering,
	// MaxBlocks remains the same mutable optional cap, including negative values.
	for _, configured := range []int{-1, 0, 2} {
		p := NewPagedKVPool(pagedReserveCfg(), 4)
		p.MaxBlocks = configured
		s := p.NewSequence()
		if got, want := s.TryReserveBlocks(3), configured <= 0; got != want {
			t.Fatalf("configured %d: reserve = %v, want %v", configured, got, want)
		}
		if p.MaxBlocks != configured {
			t.Fatal("reservation changed the configured cap")
		}
		p.MaxBlocks = 8
		if !s.TryReserveBlocks(3) {
			t.Fatal("legacy configuration increase stopped permitting growth")
		}
	}
}
