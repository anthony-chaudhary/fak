package session

import (
	"encoding/json"
	"testing"
)

// TestPendingTurnIsZeroAndSetPendingTurnCarried proves the #1363 write-ahead
// checkpoint primitive: the zero PendingTurn is the safe "nothing in flight"
// default, SetPendingTurn bumps Rev and Snapshot carries the checkpoint verbatim,
// and a terminal session rejects the write (no resurrection). Mirrors the
// TestGoalIsZeroAndSetGoalCarried / TestTurnIntent pattern.
func TestPendingTurnIsZeroAndSetPendingTurnCarried(t *testing.T) {
	if !(PendingTurn{}).IsZero() {
		t.Fatal("zero PendingTurn must report IsZero (the safe 'nothing in flight' default)")
	}
	pt := PendingTurn{Attempt: 2, LastStatus: 429, StartedAtUnixNano: 1_780_000_000_000_000_000}
	if pt.IsZero() {
		t.Fatal("a populated PendingTurn must NOT report IsZero")
	}

	tbl := NewTable()
	st, ok := tbl.SetPendingTurn("s", pt)
	if !ok || st.Rev != 1 {
		t.Fatalf("SetPendingTurn ok=%v Rev=%d, want true/1", ok, st.Rev)
	}
	if st.PendingTurn != pt {
		t.Fatalf("recorded pending turn = %+v, want %+v", st.PendingTurn, pt)
	}

	snap := tbl.Snapshot()
	if len(snap) != 1 || snap[0].PendingTurn != pt {
		t.Fatalf("Snapshot pending turn = %+v, want it to carry the checkpoint verbatim", snap)
	}

	// Clearing (the zero value, once the turn completes) bumps Rev again.
	st2, ok := tbl.SetPendingTurn("s", PendingTurn{})
	if !ok || st2.Rev != 2 {
		t.Fatalf("clearing SetPendingTurn ok=%v Rev=%d, want true/2", ok, st2.Rev)
	}
	if !st2.PendingTurn.IsZero() {
		t.Fatalf("cleared pending turn must be zero, got %+v", st2.PendingTurn)
	}

	// A terminal session rejects the checkpoint write — no resurrection.
	tbl2 := NewTable()
	if _, ok := tbl2.Transition("t", Stopped, "done"); !ok {
		t.Fatal("setup: could not stop session")
	}
	if _, ok := tbl2.SetPendingTurn("t", pt); ok {
		t.Fatal("a terminal session must reject SetPendingTurn")
	}
}

// TestStateJSONOmitsZeroPendingTurn proves the zero-value-unchanged acceptance: a
// State with no in-flight turn marshals with no pending_turn bytes (the omitzero
// tag), and a populated checkpoint round-trips through JSON. The regression guard
// that #1363 adds no resident bytes by default.
func TestStateJSONOmitsZeroPendingTurn(t *testing.T) {
	raw, err := json.Marshal(DefaultState("s"))
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"trace_id":"s","run":0,"budget":{"turns_left":-1,"tokens_left":-1},"priority":0,"pace":{"max_tokens_per_turn":0,"min_turn_gap_ms":0},"rev":0}`
	if string(raw) != want {
		t.Fatalf("zero pending turn must be absent from JSON\n got: %s\nwant: %s", raw, want)
	}

	st := DefaultState("s")
	st.PendingTurn = PendingTurn{Attempt: 1, LastStatus: 503, StartedAtUnixNano: 1_780_000_000_000_000_000}
	raw, err = json.Marshal(st)
	if err != nil {
		t.Fatal(err)
	}
	var round State
	if err := json.Unmarshal(raw, &round); err != nil {
		t.Fatal(err)
	}
	if round.PendingTurn != st.PendingTurn {
		t.Fatalf("pending turn did not round-trip through JSON: got %+v want %+v", round.PendingTurn, st.PendingTurn)
	}
}
