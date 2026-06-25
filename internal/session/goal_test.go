package session

import (
	"encoding/json"
	"sync"
	"testing"
)

// TestGoalIsZeroAndSetGoalCarried proves the #849 data structure: the zero Goal is the
// safe "no root" default, SetGoal bumps Rev and Snapshot carries the goal verbatim, and
// a terminal session rejects the write (no resurrection). Mirrors the TurnIntent test.
func TestGoalIsZeroAndSetGoalCarried(t *testing.T) {
	if !(Goal{}).IsZero() {
		t.Fatal("zero Goal must report IsZero (the safe 'no active goal' default)")
	}
	root := Goal{ID: "goal-abc123", Priority: 2, Budget: 50_000}
	if root.IsZero() {
		t.Fatal("a populated Goal must NOT report IsZero")
	}

	tbl := NewTable()
	st, ok := tbl.SetGoal("s", root)
	if !ok || st.Rev != 1 {
		t.Fatalf("SetGoal ok=%v Rev=%d, want true/1", ok, st.Rev)
	}
	if st.Goal != root {
		t.Fatalf("recorded goal = %+v, want %+v", st.Goal, root)
	}

	// The scheduler's read (Snapshot) carries the root verbatim.
	snap := tbl.Snapshot()
	if len(snap) != 1 || snap[0].Goal != root {
		t.Fatalf("Snapshot goal = %+v, want it to carry the root verbatim", snap)
	}

	// Re-setting bumps Rev again (monotonic version a concurrent reader keys on).
	st2, ok := tbl.SetGoal("s", Goal{ID: "goal-def456"})
	if !ok || st2.Rev != 2 {
		t.Fatalf("second SetGoal ok=%v Rev=%d, want true/2", ok, st2.Rev)
	}

	// A terminal session rejects the goal write — no resurrection, no goal change.
	tbl2 := NewTable()
	if _, ok := tbl2.Transition("t", Stopped, "done"); !ok {
		t.Fatal("setup: could not stop session")
	}
	if _, ok := tbl2.SetGoal("t", root); ok {
		t.Fatal("a terminal session must reject SetGoal")
	}
}

// TestStateJSONOmitsZeroGoal proves the zero-value-unchanged acceptance: a goal-less
// State marshals byte-identically to today (the omitzero tag), and a populated goal is
// present. This is the regression guard that #849 adds no resident bytes by default.
func TestStateJSONOmitsZeroGoal(t *testing.T) {
	raw, err := json.Marshal(DefaultState("s"))
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"trace_id":"s","run":0,"budget":{"turns_left":-1,"tokens_left":-1},"priority":0,"pace":{"max_tokens_per_turn":0,"min_turn_gap_ms":0},"rev":0}`
	if string(raw) != want {
		t.Fatalf("zero goal must be absent from JSON\n got: %s\nwant: %s", raw, want)
	}

	st := DefaultState("s")
	st.Goal = Goal{ID: "g1", Priority: 3}
	raw, err = json.Marshal(st)
	if err != nil {
		t.Fatal(err)
	}
	var round State
	if err := json.Unmarshal(raw, &round); err != nil {
		t.Fatal(err)
	}
	if round.Goal != st.Goal {
		t.Fatalf("goal did not round-trip through JSON: got %+v want %+v", round.Goal, st.Goal)
	}
}

// TestSetGoalRevMonotonicUnderConcurrency proves the CAS-safe setter keeps Rev strictly
// monotonic under concurrent writers, and a concurrent reader (Get/Snapshot) never sees
// a torn or non-increasing version. N goroutines each SetGoal; readers poll Rev and
// assert it never decreases. The final Rev equals the number of accepted writes.
func TestSetGoalRevMonotonicUnderConcurrency(t *testing.T) {
	tbl := NewTable()
	const writers = 32

	var wg sync.WaitGroup
	// Concurrent readers: assert Rev is non-decreasing across reads (monotonic version).
	stop := make(chan struct{})
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var last uint64
			for {
				select {
				case <-stop:
					return
				default:
				}
				cur := tbl.Get("s").Rev
				if cur < last {
					t.Errorf("Rev went backwards for a concurrent reader: %d -> %d", last, cur)
					return
				}
				last = cur
			}
		}()
	}

	wg.Add(writers)
	for i := 0; i < writers; i++ {
		go func(i int) {
			defer wg.Done()
			tbl.SetGoal("s", Goal{ID: "g", Priority: i})
		}(i)
	}
	// Let writers finish, then stop readers.
	doneWriters := make(chan struct{})
	go func() {
		// drain only the writer count by waiting on a second group would be cleaner, but
		// a short spin until Rev settles is sufficient and avoids a second WaitGroup.
		for tbl.Get("s").Rev < writers {
		}
		close(doneWriters)
	}()
	<-doneWriters
	close(stop)
	wg.Wait()

	if got := tbl.Get("s").Rev; got != writers {
		t.Fatalf("final Rev = %d, want %d (one bump per accepted SetGoal)", got, writers)
	}
}
