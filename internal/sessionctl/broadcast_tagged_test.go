package sessionctl

// broadcast_tagged_test.go — #5640 (epic #5632): the tag registry now has a producer,
// so these assert the two properties that producer must not break.
//
// 1. A tagged session resolves; an untagged one still resolves to nothing. The
//    fail-closed rule is the reason a broadcast cannot silently mutate work nobody
//    scoped into it, and "make the selector actually match something" is exactly the
//    change most likely to relax it by accident.
// 2. The registry stays bounded across session churn. Tagging without untagging trades
//    a correctness bug for a leak, and the untag edge is not guaranteed to fire for
//    every trace (session.Table evicts a cold record with no terminal transition), so
//    the ceiling has to hold on its own.

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// sessionTagCount reads the live registry size. In-package so the bound is asserted on
// the real map rather than on a count the code under test computes for itself.
func sessionTagCount() int {
	broadcastMu.Lock()
	defer broadcastMu.Unlock()
	return len(broadcastTags)
}

// assertTagRegistryConsistent checks the map and its recency ring agree. A desync (a
// clear that forgot the ring, a re-tag that pushed twice) would let the ring pin dead
// traces and defeat the ceiling long before the map count showed it.
func assertTagRegistryConsistent(t *testing.T) {
	t.Helper()
	broadcastMu.Lock()
	tags, idx, order := len(broadcastTags), len(broadcastIndex), broadcastOrder.Len()
	broadcastMu.Unlock()
	if tags != idx || tags != order {
		t.Fatalf("tag registry desync: tags=%d index=%d order=%d (want all equal)", tags, idx, order)
	}
}

// TestBroadcastSelectorMatchesTaggedSessions is the issue's named sessionctl witness: a
// populated registry resolves a lane selector onto exactly the sessions carrying that
// lane, an untagged session is still unreachable, and a lane nobody carries resolves to
// a Matched=0 report rather than a quiet success.
func TestBroadcastSelectorMatchesTaggedSessions(t *testing.T) {
	tbl := &session.Table{}
	seedSession(t, tbl, "tag-alpha-1", BroadcastMeta{Lane: "alpha", Wave: "w1"})
	seedSession(t, tbl, "tag-alpha-2", BroadcastMeta{Lane: "alpha", Labels: []string{"nightrun"}})
	seedSession(t, tbl, "tag-beta-1", BroadcastMeta{Lane: "beta", Wave: "w1"})
	seedSession(t, tbl, "tag-untagged-1", BroadcastMeta{})
	assertTagRegistryConsistent(t)

	rep, ref := Broadcast(tbl, BroadcastSelector{Lane: "alpha"}, OpPause, "lane quiesce")
	if ref != nil {
		t.Fatalf("broadcast refused: %v", ref)
	}
	if rep.Matched != 2 || rep.Applied != 2 {
		t.Fatalf("lane=alpha: matched=%d applied=%d, want 2 and 2", rep.Matched, rep.Applied)
	}
	for _, trace := range []string{"tag-alpha-1", "tag-alpha-2"} {
		if got := tbl.Get(trace).Run; got != session.Paused {
			t.Fatalf("%s run = %v, want paused", trace, got)
		}
	}
	// The whole point of the fail-closed rule: neither the other lane nor the session
	// nobody tagged may be touched by a lane-scoped op.
	for _, trace := range []string{"tag-beta-1", "tag-untagged-1"} {
		if got := tbl.Get(trace).Run; got != session.Running {
			t.Fatalf("%s run = %v, want running (untouched by a lane=alpha broadcast)", trace, got)
		}
	}

	// A lane nobody carries: a report that resolved NOTHING, distinguishable from the
	// two-session apply above by Matched alone. serve_fleetbus.go turns this into the
	// operator-facing refusal rather than an "applied, 0 affected" phantom.
	empty, ref := Broadcast(tbl, BroadcastSelector{Lane: "nobody-carries-this"}, OpPause, "lane quiesce")
	if ref != nil {
		t.Fatalf("zero-match broadcast refused at the edge: %v", ref)
	}
	if empty.Matched != 0 || empty.Applied != 0 || len(empty.Results) != 0 {
		t.Fatalf("unknown lane: matched=%d applied=%d rows=%d, want 0/0/0", empty.Matched, empty.Applied, len(empty.Results))
	}
}

// TestSessionTagRegistryStaysBoundedAcrossChurn is the acceptance gate's leak assertion:
// a count after N create/destroy cycles, and a hard ceiling for the traces whose destroy
// edge never fires at all.
func TestSessionTagRegistryStaysBoundedAcrossChurn(t *testing.T) {
	base := sessionTagCount()

	// Balanced churn: every tag is cleared, so the registry returns to where it started.
	// This is the path a healthy serve takes — admit, run, stop.
	const cycles = 20_000
	for i := 0; i < cycles; i++ {
		trace := fmt.Sprintf("churn-%d", i)
		TagSession(trace, BroadcastMeta{Lane: "churn"})
		ClearSessionTag(trace)
	}
	if got := sessionTagCount(); got != base {
		t.Fatalf("after %d create/destroy cycles count = %d, want the %d it started at", cycles, got, base)
	}
	assertTagRegistryConsistent(t)

	// Unbalanced churn: nothing is ever cleared, the shape a table eviction or a killed
	// process leaves behind. The ceiling, not the caller, has to hold here.
	overflow := MaxSessionTags + 500
	for i := 0; i < overflow; i++ {
		TagSession(fmt.Sprintf("leak-%d", i), BroadcastMeta{Lane: "leak"})
	}
	if got := sessionTagCount(); got != MaxSessionTags {
		t.Fatalf("after %d untagged-teardown sessions count = %d, want the %d ceiling", overflow, got, MaxSessionTags)
	}
	assertTagRegistryConsistent(t)

	// Eviction is fail-closed in the same direction as everything else: the coldest
	// trace is now UNtagged, so it matches no selector — never a stale match.
	if _, ok := SessionTag("leak-0"); ok {
		t.Fatal("leak-0 survived eviction; the coldest tag must be the one dropped")
	}
	if _, ok := SessionTag(fmt.Sprintf("leak-%d", overflow-1)); !ok {
		t.Fatal("the most recently tagged trace was evicted; recency order is inverted")
	}

	for i := 0; i < overflow; i++ {
		ClearSessionTag(fmt.Sprintf("leak-%d", i))
	}
	if got := sessionTagCount(); got != base {
		t.Fatalf("after cleanup count = %d, want %d", got, base)
	}
	assertTagRegistryConsistent(t)
}
