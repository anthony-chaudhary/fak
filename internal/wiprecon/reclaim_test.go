package wiprecon

import (
	"reflect"
	"testing"
)

// TestReclaimableSelectsOnlyTheRecoveryQueue: the worklist is exactly the RECLAIM rows.
// A QUARANTINE row leaking in would put a delta git says does not apply in front of an
// operator as "recoverable"; a DISCARD_WITNESSED row leaking in would offer work that
// already landed. Both are the wrong queue.
func TestReclaimableSelectsOnlyTheRecoveryQueue(t *testing.T) {
	ds := []Decision{
		{Session: "a", Action: ActSkip},
		{Session: "b", Action: ActReclaim},
		{Session: "c", Action: ActQuarantine},
		{Session: "d", Action: ActDiscardWitnessed},
		{Session: "e", Action: ActReclaim},
	}
	got := Reclaimable(ds)
	want := []string{"b", "e"}
	if len(got) != len(want) {
		t.Fatalf("want %d reclaimable rows, got %d: %+v", len(want), len(got), got)
	}
	for i, w := range want {
		if got[i].Session != w || got[i].Action != ActReclaim {
			t.Errorf("row %d: want session %s/RECLAIM, got %s/%s", i, w, got[i].Session, got[i].Action)
		}
	}
	// Empty but non-nil, so a JSON consumer never has to handle null.
	if none := Reclaimable([]Decision{{Session: "x", Action: ActQuarantine}}); none == nil || len(none) != 0 {
		t.Errorf("want empty non-nil for a queue with no RECLAIM rows, got %#v", none)
	}
	if none := Reclaimable(nil); none == nil || len(none) != 0 {
		t.Errorf("want empty non-nil for a nil decision set, got %#v", none)
	}
}

// TestRankReclaimIsMostDecayedFirst is the ordering contract the whole worklist exists
// for: the row closest to decaying out of RECLAIM must be the one an operator sees
// first. Drift (how far HEAD has moved past the checkpoint's base) is the decay clock,
// so it is the primary key; age breaks ties; session makes it deterministic.
func TestRankReclaimIsMostDecayedFirst(t *testing.T) {
	rows := []ReclaimRow{
		{Session: "fresh", TrunkDistance: 0, AgeHours: 1},
		{Session: "worst", TrunkDistance: 300, AgeHours: 1},
		{Session: "mid-young", TrunkDistance: 12, AgeHours: 2},
		{Session: "mid-old", TrunkDistance: 12, AgeHours: 99},
	}
	got := RankReclaim(rows)
	want := []string{"worst", "mid-old", "mid-young", "fresh"}
	for i, w := range want {
		if got[i].Session != w {
			t.Fatalf("rank[%d] = %s, want %s (full order %v)", i, got[i].Session, w, sessions(got))
		}
	}
	// Totality: ranking never drops a row, so a caller acting on the head still sees
	// the whole tail.
	if len(got) != len(rows) {
		t.Fatalf("ranking dropped rows: %d in, %d out", len(rows), len(got))
	}
	// The input slice must be left alone — the caller still holds the unranked view.
	if rows[0].Session != "fresh" {
		t.Errorf("RankReclaim mutated its input: rows[0]=%s", rows[0].Session)
	}
}

// TestRankReclaimSortsUnknownDriftLast is the honesty rule behind DriftUnknown: a row
// whose base could not be resolved has NO evidence of urgency and must never displace a
// row with measured drift. A zero sentinel would have done the opposite of DriftUnknown
// here only by accident — this pins the intended end of the queue.
func TestRankReclaimSortsUnknownDriftLast(t *testing.T) {
	rows := []ReclaimRow{
		{Session: "unknown-and-ancient", TrunkDistance: DriftUnknown, AgeHours: 10000},
		{Session: "measured-zero", TrunkDistance: 0, AgeHours: 1},
		{Session: "measured-five", TrunkDistance: 5, AgeHours: 1},
	}
	got := RankReclaim(rows)
	want := []string{"measured-five", "measured-zero", "unknown-and-ancient"}
	if !reflect.DeepEqual(sessions(got), want) {
		t.Fatalf("order = %v, want %v", sessions(got), want)
	}
	if DriftUnknown >= 0 {
		t.Fatalf("DriftUnknown must be negative so it sorts last under a descending drift key; got %d", DriftUnknown)
	}
}

// TestRankReclaimDeterministicAcrossInputOrder: two callers reading the same repo must
// be handed the same queue, whatever order the refs came back in.
func TestRankReclaimDeterministicAcrossInputOrder(t *testing.T) {
	rows := []ReclaimRow{
		{Session: "b", TrunkDistance: 7, AgeHours: 3},
		{Session: "a", TrunkDistance: 7, AgeHours: 3},
		{Session: "c", TrunkDistance: 7, AgeHours: 3},
	}
	fwd := RankReclaim(rows)
	rev := RankReclaim([]ReclaimRow{rows[2], rows[1], rows[0]})
	if !reflect.DeepEqual(fwd, rev) {
		t.Fatalf("non-deterministic across input order:\n a=%v\n b=%v", sessions(fwd), sessions(rev))
	}
	if want := []string{"a", "b", "c"}; !reflect.DeepEqual(sessions(fwd), want) {
		t.Errorf("equal-drift equal-age rows must tie-break by session: %v, want %v", sessions(fwd), want)
	}
	if empty := RankReclaim(nil); empty == nil || len(empty) != 0 {
		t.Errorf("want empty non-nil for a nil worklist, got %#v", empty)
	}
}

func sessions(rows []ReclaimRow) []string {
	out := make([]string, 0, len(rows))
	for _, r := range rows {
		out = append(out, r.Session)
	}
	return out
}
