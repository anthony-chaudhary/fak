package worklog

import (
	"fmt"
	"testing"
)

// TestAppendStampsMonotonicSeq: the feed, not the producer, assigns the cursor,
// and it is strictly increasing across kinds.
func TestAppendStampsMonotonicSeq(t *testing.T) {
	f := NewFeed(0)
	a, ok := f.Append(WorkChange{Kind: KindCommit, SHA: "aaa", Claim: "fix x", Verdict: "OK", Witness: "diff-witnessed"})
	if !ok || a.Seq != 1 {
		t.Fatalf("first append: ok=%v seq=%d, want ok=true seq=1", ok, a.Seq)
	}
	b, ok := f.Append(WorkChange{Kind: KindCommit, SHA: "bbb", Claim: "add y", Verdict: "CLAIM_UNWITNESSED", Witness: "subject-only"})
	if !ok || b.Seq != 2 {
		t.Fatalf("second append: ok=%v seq=%d, want ok=true seq=2", ok, b.Seq)
	}
}

// TestAppendIdempotentByKey: a replayed commit is a no-op returning the original
// Seq; but a genuinely new verdict for the same SHA (a flip) is a NEW event.
func TestAppendIdempotentByKey(t *testing.T) {
	f := NewFeed(0)
	first, _ := f.Append(WorkChange{Kind: KindCommit, SHA: "aaa", Verdict: "OK"})

	replay, ok := f.Append(WorkChange{Kind: KindCommit, SHA: "aaa", Verdict: "OK"})
	if ok {
		t.Errorf("replayed commit should dedupe (ok=false), got ok=true")
	}
	if replay.Seq != first.Seq {
		t.Errorf("replay seq=%d, want original %d", replay.Seq, first.Seq)
	}

	flip, ok := f.Append(WorkChange{Kind: KindVerdictFlip, SHA: "aaa", Verdict: "CLAIM_UNWITNESSED", PrevVerdict: "OK"})
	if !ok {
		t.Errorf("a new verdict flip for the same SHA must be a new event, got dedupe")
	}
	if flip.Seq <= first.Seq {
		t.Errorf("flip seq=%d must exceed commit seq=%d", flip.Seq, first.Seq)
	}

	replayFlip, ok := f.Append(WorkChange{Kind: KindVerdictFlip, SHA: "aaa", Verdict: "CLAIM_UNWITNESSED"})
	if ok {
		t.Errorf("replayed identical flip should dedupe, got ok=true")
	}
	if replayFlip.Seq != flip.Seq {
		t.Errorf("replayed flip seq=%d, want %d", replayFlip.Seq, flip.Seq)
	}
}

// TestDrainCursorAndSince: Drain returns Seq>since and the feed-head cursor
// (not the max of the returned slice), so a caught-up consumer drains empty.
func TestDrainCursorAndSince(t *testing.T) {
	f := NewFeed(0)
	f.Append(WorkChange{Kind: KindCommit, SHA: "a"})
	f.Append(WorkChange{Kind: KindCommit, SHA: "b"})
	f.Append(WorkChange{Kind: KindCommit, SHA: "c"})

	all, cursor := f.Drain("", 0)
	if len(all) != 3 || cursor != 3 {
		t.Fatalf("drain all: n=%d cursor=%d, want 3,3", len(all), cursor)
	}

	tail, cursor := f.Drain("", 2)
	if len(tail) != 1 || tail[0].SHA != "c" || cursor != 3 {
		t.Fatalf("drain since 2: got n=%d cursor=%d, want 1 (c),3", len(tail), cursor)
	}

	empty, cursor := f.Drain("", 3)
	if len(empty) != 0 || cursor != 3 {
		t.Fatalf("caught-up drain: n=%d cursor=%d, want 0,3", len(empty), cursor)
	}
}

// TestPrincipalVisibility: a tenant sees its own + global (principal-less)
// changes, never a peer's; an empty drainer sees everything.
func TestPrincipalVisibility(t *testing.T) {
	f := NewFeed(0)
	f.Append(WorkChange{Kind: KindCommit, SHA: "global"}) // principal-less broadcast
	f.Append(WorkChange{Kind: KindCommit, SHA: "a-only", Principal: "tenant-a"})
	f.Append(WorkChange{Kind: KindCommit, SHA: "b-only", Principal: "tenant-b"})

	a, _ := f.Drain("tenant-a", 0)
	if len(a) != 2 {
		t.Fatalf("tenant-a should see global + own = 2, got %d", len(a))
	}
	for _, c := range a {
		if c.SHA == "b-only" {
			t.Errorf("tenant-a must not see peer change %q", c.SHA)
		}
	}

	admin, _ := f.Drain("", 0)
	if len(admin) != 3 {
		t.Fatalf("admin/empty principal should see all 3, got %d", len(admin))
	}
}

// TestBoundedRetentionAndGap: past capacity the oldest payloads are evicted, and
// a consumer whose cursor is below the retained window sees a Seq gap
// (returned[0].Seq > sinceSeq+1) — the documented re-sync trigger. Idempotency
// survives eviction: a replayed evicted key still dedupes.
func TestBoundedRetentionAndGap(t *testing.T) {
	f := NewFeed(2)
	f.Append(WorkChange{Kind: KindCommit, SHA: "s1"})
	f.Append(WorkChange{Kind: KindCommit, SHA: "s2"})
	f.Append(WorkChange{Kind: KindCommit, SHA: "s3"}) // evicts s1

	got, cursor := f.Drain("", 0)
	if len(got) != 2 || got[0].SHA != "s2" || cursor != 3 {
		t.Fatalf("retained window: n=%d first=%s cursor=%d, want 2 (s2..),3", len(got), got[0].SHA, cursor)
	}
	// A consumer sitting at cursor 0 sees the gap: first retained Seq is 2, not 1.
	if got[0].Seq <= 1 {
		t.Errorf("expected a Seq gap (first retained > 1), got first Seq=%d", got[0].Seq)
	}

	// Replaying an evicted key still dedupes (seen map outlives the ring).
	if _, ok := f.Append(WorkChange{Kind: KindCommit, SHA: "s1"}); ok {
		t.Errorf("replay of evicted key s1 should still dedupe")
	}
}

// TestFoldLatestVerdict: the read-model folds a SHA that was OK then flipped to
// its latest (refuted) verdict, and ignores changes carrying no verdict.
func TestFoldLatestVerdict(t *testing.T) {
	changes := []WorkChange{
		{Seq: 1, Kind: KindCommit, SHA: "aaa", Verdict: "OK"},
		{Seq: 2, Kind: KindCommit, SHA: "bbb", Verdict: "OK"},
		{Seq: 3, Kind: KindVerdictFlip, SHA: "aaa", Verdict: "CLAIM_UNWITNESSED", PrevVerdict: "OK"},
		{Seq: 4, Kind: KindCommit, SHA: "ccc"}, // no verdict — must not appear
	}
	got := FoldLatestVerdict(changes)
	if got["aaa"] != "CLAIM_UNWITNESSED" {
		t.Errorf("aaa should fold to the flipped verdict, got %q", got["aaa"])
	}
	if got["bbb"] != "OK" {
		t.Errorf("bbb should remain OK, got %q", got["bbb"])
	}
	if _, ok := got["ccc"]; ok {
		t.Errorf("ccc carries no verdict and must be absent from the read-model")
	}
}

// TestSeenMapBounded proves the idempotency map does not grow without limit on a
// bounded feed: appending far more distinct changes than the seen-retention window
// keeps len(seen) at O(cap), NOT O(total-appended). The load-bearing invariant is
// preserved — a change whose payload is still in the ring always dedupes — and a
// change within the wider retention window still dedupes too; only very old,
// long-evicted keys are allowed to lapse (the same bounded-retention tradeoff the
// payload ring itself makes).
func TestSeenMapBounded(t *testing.T) {
	f := NewFeed(4) // ring cap 4 -> seen retention 4*seenRetentionFactor keys.
	const n = 500
	for i := 0; i < n; i++ {
		if _, ok := f.Append(WorkChange{Kind: KindCommit, SHA: fmt.Sprintf("sha-%d", i)}); !ok {
			t.Fatalf("append %d of a distinct change must be accepted", i)
		}
	}
	// seen is bounded to O(cap), not O(n).
	if bound := f.cap*seenRetentionFactor + f.cap; len(f.seen) > bound {
		t.Fatalf("len(seen) = %d after %d distinct appends, want <= %d (the map grew unbounded)", len(f.seen), n, bound)
	}
	// INVARIANT: a change still in the ring MUST still dedupe (must never regress).
	ringHead := fmt.Sprintf("sha-%d", n-1)
	if _, ok := f.Append(WorkChange{Kind: KindCommit, SHA: ringHead}); ok {
		t.Errorf("replay of a still-ringed change %q must dedupe", ringHead)
	}
	// A ring-evicted change still inside the seen-retention window also dedupes.
	recent := fmt.Sprintf("sha-%d", n-2*f.cap)
	if _, ok := f.Append(WorkChange{Kind: KindCommit, SHA: recent}); ok {
		t.Errorf("replay of a within-retention change %q must still dedupe", recent)
	}
	// A very old, long-evicted change is allowed to lapse (proof the map was pruned).
	if _, ok := f.Append(WorkChange{Kind: KindCommit, SHA: "sha-0"}); !ok {
		t.Errorf("the oldest change should have aged out of seen (bounded retention), but it still deduped")
	}
}
