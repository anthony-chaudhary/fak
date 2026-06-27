package session

import (
	"sync"
	"testing"
)

// revision_test.go — the every-revision observer seam (#630): WatchRevisions must
// fire exactly once per Rev bump, in strict Rev order, for EVERY write verb (not
// just the notable run-state transitions WatchTransitions covers), delivering the
// freshly-written State. This is the source of the gateway's /v1/fak/session/changes
// drive-state stream, so the per-write, in-order, exactly-once contract is the thing
// the feed depends on.

func TestWatchRevisionsFiresOnEveryWrite(t *testing.T) {
	tbl := NewTable()
	var got []State
	tbl.WatchRevisions(func(s State) { got = append(got, s) })

	// One write through each distinct verb that bumps Rev.
	tbl.SetPriority("s", 3)                                   // Rev 1
	tbl.SetBudget("s", Budget{TurnsLeft: 5, TokensLeft: 100}) // Rev 2
	tbl.SetPace("s", Pace{MaxTokensPerTurn: 64})              // Rev 3
	tbl.Transition("s", Paused, "operator-hold")              // Rev 4
	tbl.Transition("s", Running, "")                          // Rev 5
	tbl.DebitUsage("s", Usage{OutputTokens: 10})              // Rev 6

	if len(got) != 6 {
		t.Fatalf("expected 6 revision events (one per write), got %d: %+v", len(got), got)
	}
	for i, ev := range got {
		if ev.TraceID != "s" {
			t.Errorf("event %d: TraceID = %q, want %q", i, ev.TraceID, "s")
		}
		if want := uint64(i + 1); ev.Rev != want {
			t.Errorf("event %d: Rev = %d, want %d (revisions must arrive in strict order)", i, ev.Rev, want)
		}
	}
	// The delivered State is the post-write value, not a stale snapshot.
	if got[0].Priority != 3 {
		t.Errorf("event 0 Priority = %d, want 3 (the value the verb set)", got[0].Priority)
	}
	if got[3].Run != Paused || got[3].Reason != "operator-hold" {
		t.Errorf("event 3 = {Run:%v Reason:%q}, want {Paused operator-hold}", got[3].Run, got[3].Reason)
	}
}

func TestWatchRevisionsNoOpUntilWiredAndAfterClear(t *testing.T) {
	tbl := NewTable()

	// No observer wired (the default): writes must not panic and must be byte-identical.
	tbl.SetPriority("s", 1)
	if got := tbl.Get("s"); got.Rev != 1 || got.Priority != 1 {
		t.Fatalf("unwatched write lost: %+v", got)
	}

	var n int
	tbl.WatchRevisions(func(State) { n++ })
	tbl.SetPriority("s", 2) // fires
	tbl.WatchRevisions(nil) // clear the seam
	tbl.SetPriority("s", 3) // must NOT fire
	if n != 1 {
		t.Fatalf("observer fired %d times, want exactly 1 (one between wire and clear)", n)
	}

	// A nil receiver is a no-op (never panics).
	var nilTbl *Table
	nilTbl.WatchRevisions(func(State) { t.Fatal("nil-receiver observer must never fire") })
}

// TestWatchRevisionsConcurrentInOrderPerSession asserts that concurrent writers to
// the SAME session deliver revisions in strict Rev order — the in-lock delivery
// guarantee a cursor feed relies on (an after-unlock fan-out could reorder).
func TestWatchRevisionsConcurrentInOrderPerSession(t *testing.T) {
	tbl := NewTable()
	var mu sync.Mutex
	var revs []uint64
	tbl.WatchRevisions(func(s State) {
		mu.Lock()
		revs = append(revs, s.Rev)
		mu.Unlock()
	})

	const writers = 50
	var wg sync.WaitGroup
	wg.Add(writers)
	for i := 0; i < writers; i++ {
		go func(p int) {
			defer wg.Done()
			tbl.SetPriority("s", p)
		}(i)
	}
	wg.Wait()

	if len(revs) != writers {
		t.Fatalf("got %d revisions, want %d (one per write)", len(revs), writers)
	}
	// Delivery order under the lock is the write order, so the recorded Revs are a
	// strictly increasing 1..writers run.
	for i, r := range revs {
		if want := uint64(i + 1); r != want {
			t.Fatalf("revision %d out of order: Rev = %d, want %d", i, r, want)
		}
	}
}
