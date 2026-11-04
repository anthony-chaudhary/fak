// eviction_test.go — witnesses for the splash bump-topology KV-eviction port
// (internal/kvmmu/eviction.go): leaf-only eligibility, the recency bump when a
// leaf leaves the resident tier, the release bump, continuation scans, and the
// allocation-free intrusive last-used order.

package kvmmu_test

import (
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvmmu"
)

func expectPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s: expected a panic, got none", name)
		}
	}()
	fn()
}

// TestOnlyResidentLeavesAreEvictable pins the eligibility rule: a block with
// resident children or users is not a candidate; dropping a leaf re-exposes
// its parent, bumped to the leaf's stamp.
func TestOnlyResidentLeavesAreEvictable(t *testing.T) {
	ev := kvmmu.NewEvictor(kvmmu.NewRecencyClock())
	// R -> A -> C and R -> B. Publish stamps: R=1, A=2, C=3, B=4.
	ev.Publish(1, 0, true) // R
	ev.Publish(2, 1, true) // A
	ev.Publish(3, 2, true) // C
	ev.Publish(4, 1, true) // B

	want := map[int64]bool{1: false, 2: false, 3: true, 4: true}
	for id, leaf := range want {
		if got := ev.ResidentLeaf(id); got != leaf {
			t.Fatalf("ResidentLeaf(%d) = %v, want %v", id, got, leaf)
		}
	}
	cand, ok := ev.EvictionCandidate(0)
	if !ok || cand.ID != 3 || cand.LastUsed != 3 {
		t.Fatalf("first candidate = %+v (ok=%v), want C (id 3, stamp 3)", cand, ok)
	}

	// Drop C's page: C leaves the order, A becomes a leaf and bumps to C's stamp.
	ev.DropPage(3)
	if ev.ResidentLeaf(3) {
		t.Fatal("dropped block still reports ResidentLeaf")
	}
	cand, ok = ev.EvictionCandidate(0)
	if !ok || cand.ID != 2 || cand.LastUsed != 3 {
		t.Fatalf("after dropping C, candidate = %+v (ok=%v), want A (id 2) bumped to stamp 3", cand, ok)
	}
}

// TestReleasedLeafIsNewest pins the release bump: a released leaf becomes the
// newest member (it was just touched), while a still-held block stays out of
// the order entirely.
func TestReleasedLeafIsNewest(t *testing.T) {
	ev := kvmmu.NewEvictor(kvmmu.NewRecencyClock())
	ev.Publish(1, 0, true) // root, stamp 1
	ev.Publish(2, 1, true) // leaf, stamp 2
	ev.Publish(3, 1, true) // leaf, stamp 3

	ev.Retain(2) // held: out of the order, stamp 4
	cand, ok := ev.EvictionCandidate(0)
	if !ok || cand.ID != 3 {
		t.Fatalf("with 2 held, candidate = %+v (ok=%v), want only leaf 3", cand, ok)
	}

	ev.Release(2) // released leaf takes a fresh stamp (5) and re-enters as newest
	cand, ok = ev.EvictionCandidate(0)
	if !ok || cand.ID != 3 {
		t.Fatalf("oldest candidate after release = %+v (ok=%v), want 3 (still older)", cand, ok)
	}
	next, ok := ev.EvictionCandidate(3)
	if !ok || next.ID != 2 || next.LastUsed <= 3 {
		t.Fatalf("candidate after 3 = %+v (ok=%v), want released leaf 2 at a newer stamp", next, ok)
	}
}

// TestTransferExcludesFromOrder pins the transfer guard: a block moving
// between tiers is neither evicted nor replaced until the transfer clears.
func TestTransferExcludesFromOrder(t *testing.T) {
	ev := kvmmu.NewEvictor(kvmmu.NewRecencyClock())
	ev.Publish(1, 0, true)
	ev.Publish(2, 1, true)

	ev.SetTransferring(2, true)
	if _, ok := ev.EvictionCandidate(0); ok {
		t.Fatal("transferring block still offered as an eviction candidate")
	}
	expectPanic(t, "SetTransferring no-op", func() { ev.SetTransferring(2, true) })

	ev.SetTransferring(2, false)
	cand, ok := ev.EvictionCandidate(0)
	if !ok || cand.ID != 2 {
		t.Fatalf("after transfer cleared, candidate = %+v (ok=%v), want 2", cand, ok)
	}
}

// TestEvictionNeverAllocates is the load-bearing property of the port: once
// blocks are published, steady-state eviction work (tier moves, user churn,
// recency refreshes, candidate scans) performs zero heap allocations — the
// order nodes were allocated once, at publish.
func TestEvictionNeverAllocates(t *testing.T) {
	ev := kvmmu.NewEvictor(kvmmu.NewRecencyClock())
	ev.Publish(1, 0, true)
	for i := int64(2); i <= 64; i++ {
		ev.Publish(i, 1, true)
	}
	fn := func() {
		ev.DropPage(2)
		ev.AdoptPage(2)
		ev.Retain(3)
		ev.Release(3)
		ev.Touch(4)
		ev.EvictionCandidate(0)
		ev.EvictionCandidate(4)
	}
	if a := testing.AllocsPerRun(100, fn); a != 0 {
		t.Fatalf("steady-state eviction ops allocated %.2f B/op, want 0", a)
	}
}

// TestRecencyOrderNeverAllocates witnesses the same property on the order
// itself: link/unlink/scan cycles over preallocated nodes allocate nothing.
func TestRecencyOrderNeverAllocates(t *testing.T) {
	o := kvmmu.NewRecencyOrder()
	const n = 256
	nodes := make([]*kvmmu.RecencyNode, n)
	for i := range nodes {
		nodes[i] = kvmmu.NewRecencyNode()
	}
	for i, nd := range nodes {
		o.Link(nd, uint64(i+1), int64(i+1))
	}
	stamps := make([]uint64, n)
	for i := range stamps {
		stamps[i] = uint64(1_000_000 + i)
	}
	k := 0
	fn := func() {
		nd := nodes[k%n]
		nd.Unlink()
		o.Oldest()
		o.Next(kvmmu.EvictionCandidate{ID: 1, LastUsed: 1})
		o.Link(nd, stamps[k%n], int64(k+1))
		k++
	}
	if a := testing.AllocsPerRun(1000, fn); a != 0 {
		t.Fatalf("order link/unlink cycle allocated %.2f B/op, want 0", a)
	}
}

// TestEvictionCandidateContinuationScan pins the scan contract: EvictionCandidate
// walks the order oldest-first and continues from a previous candidate without
// a lookup, skipping members that left the order in between.
func TestEvictionCandidateContinuationScan(t *testing.T) {
	ev := kvmmu.NewEvictor(kvmmu.NewRecencyClock())
	ev.Publish(1, 0, true)
	ev.Publish(2, 1, true)
	ev.Publish(3, 1, true)
	ev.Publish(4, 1, true)

	c1, ok := ev.EvictionCandidate(0)
	if !ok || c1.ID != 2 {
		t.Fatalf("candidate 1 = %+v (ok=%v), want 2", c1, ok)
	}
	c2, ok := ev.EvictionCandidate(c1.ID)
	if !ok || c2.ID != 3 {
		t.Fatalf("candidate 2 = %+v (ok=%v), want 3", c2, ok)
	}
	c3, ok := ev.EvictionCandidate(c2.ID)
	if !ok || c3.ID != 4 {
		t.Fatalf("candidate 3 = %+v (ok=%v), want 4", c3, ok)
	}
	if _, ok := ev.EvictionCandidate(c3.ID); ok {
		t.Fatal("scan past the last member reported a candidate")
	}

	// Skip a member that left the order mid-scan.
	ev.DropPage(3)
	skipped, ok := ev.EvictionCandidate(2)
	if !ok || skipped.ID != 4 {
		t.Fatalf("continuation after dropped 3 = %+v (ok=%v), want 4", skipped, ok)
	}
}

// TestParentInheritsChildRecency pins the bump rule in both directions: a
// parent colder than its child advances to the child's stamp; a parent hotter
// than its child keeps its own; and an inherited stamp never jumps ahead of a
// chain used later.
func TestParentInheritsChildRecency(t *testing.T) {
	t.Run("parent colder than child advances", func(t *testing.T) {
		ev := kvmmu.NewEvictor(kvmmu.NewRecencyClock())
		ev.Publish(1, 0, true) // P, stamp 1
		ev.Publish(2, 1, true) // Q, stamp 2
		ev.DropPage(2)
		if got := ev.LastUsed(1); got != 2 {
			t.Fatalf("parent stamp after child drop = %d, want 2 (inherited from child)", got)
		}
	})

	t.Run("parent hotter than child keeps its own stamp", func(t *testing.T) {
		ev := kvmmu.NewEvictor(kvmmu.NewRecencyClock())
		ev.Publish(1, 0, true) // P, stamp 1
		ev.Publish(2, 1, true) // Q, stamp 2
		ev.Touch(1)            // P refreshed to stamp 3
		ev.DropPage(2)
		if got := ev.LastUsed(1); got != 3 {
			t.Fatalf("parent stamp after child drop = %d, want 3 (its own newer stamp kept)", got)
		}
	})

	t.Run("inherited stamp does not jump ahead of later chains", func(t *testing.T) {
		ev := kvmmu.NewEvictor(kvmmu.NewRecencyClock())
		ev.Publish(1, 0, true) // S, stamp 1 (used first)
		ev.Publish(2, 0, true) // P, stamp 2
		ev.Publish(3, 2, true) // Q, stamp 3
		ev.Publish(4, 0, true) // T, stamp 4 (used last)
		ev.DropPage(3)         // P inherits Q's stamp 3
		if got := ev.LastUsed(2); got != 3 {
			t.Fatalf("P stamp = %d, want 3", got)
		}
		// Order must be S(1), P(3), T(4): the bump moved P forward but not
		// past the chain used after it.
		first, ok := ev.EvictionCandidate(0)
		if !ok || first.ID != 1 {
			t.Fatalf("oldest = %+v (ok=%v), want S (id 1)", first, ok)
		}
		second, ok := ev.EvictionCandidate(first.ID)
		if !ok || second.ID != 2 || second.LastUsed != 3 {
			t.Fatalf("next = %+v (ok=%v), want P (id 2) at inherited stamp 3", second, ok)
		}
		third, ok := ev.EvictionCandidate(second.ID)
		if !ok || third.ID != 4 {
			t.Fatalf("next = %+v (ok=%v), want T (id 4)", third, ok)
		}
	})
}

// TestEraseGuardsAndReexposesParent pins the erase contract: only a block
// without children, users, or an in-flight transfer may be erased, and erasing
// a leaf re-exposes its parent.
func TestEraseGuardsAndReexposesParent(t *testing.T) {
	ev := kvmmu.NewEvictor(kvmmu.NewRecencyClock())
	ev.Publish(1, 0, true)
	ev.Publish(2, 1, true)
	ev.Publish(3, 2, true)

	expectPanic(t, "erase with children", func() { ev.Erase(1) })
	expectPanic(t, "erase with children", func() { ev.Erase(2) })

	ev.Retain(3)
	expectPanic(t, "erase with users", func() { ev.Erase(3) })
	ev.Release(3)

	ev.SetTransferring(3, true)
	expectPanic(t, "erase while transferring", func() { ev.Erase(3) })
	ev.SetTransferring(3, false)

	ev.Erase(3)
	if !ev.ResidentLeaf(2) {
		t.Fatal("erasing a leaf did not re-expose its parent")
	}
	ev.Erase(2)
	ev.Erase(1)
	if ev.Blocks() != 0 {
		t.Fatalf("Blocks() = %d after full erase, want 0", ev.Blocks())
	}
	expectPanic(t, "unknown block", func() { ev.LastUsed(99) })
}

// TestDiskOnlyChainAdoption pins the tier transitions: disk-only blocks may
// hang off resident or disk-only parents, only resident blocks enter the
// order, and a page cannot be adopted under a non-resident parent.
func TestDiskOnlyChainAdoption(t *testing.T) {
	ev := kvmmu.NewEvictor(kvmmu.NewRecencyClock())
	ev.Publish(1, 0, true)
	ev.Publish(2, 1, false) // disk-only child of a resident parent

	// A parent whose only child is disk-only is still a resident leaf.
	if !ev.ResidentLeaf(1) {
		t.Fatal("resident parent with only a disk-only child should be a leaf")
	}
	if ev.ResidentLeaf(2) {
		t.Fatal("disk-only block reported as a resident leaf")
	}

	ev.AdoptPage(2)
	if ev.ResidentLeaf(1) {
		t.Fatal("adopting a child's page must remove the parent from the leaf order")
	}

	// Disk-only chain under a disk-only parent.
	ev.Publish(3, 0, true)
	ev.Publish(4, 3, false)
	ev.DropPage(3)
	expectPanic(t, "adopt under non-resident parent", func() { ev.AdoptPage(4) })
}

// TestRecencyOrderDirect pins the order's own semantics: oldest-first walk,
// id tie-break on equal stamps, unlink/relink, and misuse panics.
func TestRecencyOrderDirect(t *testing.T) {
	o := kvmmu.NewRecencyOrder()
	n1 := kvmmu.NewRecencyNode()
	n2 := kvmmu.NewRecencyNode()
	n3 := kvmmu.NewRecencyNode()
	n4 := kvmmu.NewRecencyNode()

	if _, ok := o.Oldest(); ok {
		t.Fatal("empty order reported a member")
	}
	if o.NewestID() != 0 {
		t.Fatal("empty order reported a newest id")
	}

	o.Link(n1, 10, 1)
	o.Link(n2, 20, 2)
	o.Link(n3, 30, 3)
	if o.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", o.Len())
	}
	oldest, _ := o.Oldest()
	if oldest.ID != 1 || oldest.LastUsed != 10 {
		t.Fatalf("Oldest() = %+v, want (1, 10)", oldest)
	}
	if o.NewestID() != 3 {
		t.Fatalf("NewestID() = %d, want 3", o.NewestID())
	}
	after, ok := o.Next(kvmmu.EvictionCandidate{ID: 1, LastUsed: 10})
	if !ok || after.ID != 2 || after.LastUsed != 20 {
		t.Fatalf("Next((1,10)) = %+v (ok=%v), want (2, 20)", after, ok)
	}

	// Equal stamps break by id.
	o.Link(n4, 20, 21)
	after, ok = o.Next(kvmmu.EvictionCandidate{ID: 2, LastUsed: 20})
	if !ok || after.ID != 21 || after.LastUsed != 20 {
		t.Fatalf("tie-break Next((2,20)) = %+v (ok=%v), want (21, 20)", after, ok)
	}

	// Unlink the middle (4 members are linked: n1, n2, n3, n4), then relink
	// n2 with a newer stamp.
	n2.Unlink()
	if o.Len() != 3 {
		t.Fatalf("Len() after unlink = %d, want 3", o.Len())
	}
	after, ok = o.Next(kvmmu.EvictionCandidate{ID: 1, LastUsed: 10})
	if !ok || after.ID != 21 {
		t.Fatalf("Next((1,10)) after unlink = %+v (ok=%v), want (21, 20)", after, ok)
	}
	o.Link(n2, 40, 2)
	if o.NewestID() != 2 {
		t.Fatalf("NewestID() after relink = %d, want 2", o.NewestID())
	}

	expectPanic(t, "double link", func() { o.Link(n1, 50, 1) })
	n3.Unlink()
	expectPanic(t, "unlink unlinked", func() { n3.Unlink() })
}

// TestRecencyClockMonotonic pins the clock: strictly increasing stamps, first
// stamp 1, and distinct draws under concurrent use.
func TestRecencyClockMonotonic(t *testing.T) {
	c := kvmmu.NewRecencyClock()
	if v := c.Next(); v != 1 {
		t.Fatalf("first stamp = %d, want 1", v)
	}
	prev := c.Next()
	for i := 0; i < 1000; i++ {
		v := c.Next()
		if v <= prev {
			t.Fatalf("stamp went backwards: %d after %d", v, prev)
		}
		prev = v
	}

	const workers, per = 8, 2500
	var mu sync.Mutex
	seen := make(map[uint64]struct{})
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < per; i++ {
				v := c.Next()
				mu.Lock()
				seen[v] = struct{}{}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if len(seen) != workers*per {
		t.Fatalf("concurrent stamps collided: %d distinct of %d", len(seen), workers*per)
	}
}
