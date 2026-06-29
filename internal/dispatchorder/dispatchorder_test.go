package dispatchorder

import (
	"fmt"
	"testing"
)

// base is a fixed "now" the tests reason against; recency/cooldown are offsets from it.
const base = 1_000_000

// dispoOf returns the disposition the planner gave the unit with id, or "" if absent.
func dispoOf(r Result, id string) Disposition {
	for _, x := range r.Order {
		if x.ID == id {
			return x.Disposition
		}
	}
	return ""
}

// TestSupersedeKeepsFreshest is the headline scenario: 25 tasks spawned for the SAME target
// (one shared key) collapse to the single freshest unit — that one is kept and picked, the
// other 24 are superseded (not eligible, not re-attempted), and the pick is the freshest.
func TestSupersedeKeepsFreshest(t *testing.T) {
	var cands []Candidate
	for i := 0; i < 25; i++ {
		cands = append(cands, Candidate{
			ID:          fmt.Sprintf("task-%02d", i),
			Key:         "X",
			UpdatedUnix: int64(base - 10_000 + i*100), // task-24 is the most recently updated
		})
	}
	r := Plan(Input{Candidates: cands, NowUnix: base})

	if r.KeepCount != 1 || r.SupersededCount != 24 {
		t.Fatalf("counts = keep %d superseded %d, want 1/24", r.KeepCount, r.SupersededCount)
	}
	if r.Pick() != "task-24" {
		t.Errorf("pick = %q, want task-24 (the freshest)", r.Pick())
	}
	if dispoOf(r, "task-24") != DispKeep {
		t.Errorf("freshest task-24 disposition = %q, want keep", dispoOf(r, "task-24"))
	}
	if dispoOf(r, "task-00") != DispSuperseded {
		t.Errorf("stale task-00 disposition = %q, want superseded", dispoOf(r, "task-00"))
	}
	// Every superseded unit names the winner.
	for _, x := range r.Order {
		if x.Disposition == DispSuperseded && x.SupersededBy != "task-24" {
			t.Errorf("%s superseded_by = %q, want task-24", x.ID, x.SupersededBy)
		}
	}
}

// TestDistinctKeysAllKeptFreshestFirst: units with DIFFERENT keys are all distinct targets, so
// all are kept and ordered freshest-first (by recency, not by id).
func TestDistinctKeysAllKeptFreshestFirst(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "a", Key: "A", UpdatedUnix: base - 300},
		{ID: "b", Key: "B", UpdatedUnix: base - 100}, // freshest
		{ID: "c", Key: "C", UpdatedUnix: base - 200},
	}})
	if r.KeepCount != 3 {
		t.Fatalf("keep = %d, want 3 (distinct keys never supersede)", r.KeepCount)
	}
	want := []string{"b", "c", "a"} // freshest-first
	for i, id := range want {
		if r.Keep[i] != id {
			t.Errorf("keep[%d] = %q, want %q (freshest-first order)", i, r.Keep[i], id)
		}
	}
}

// TestPreferOldestOrdersDistinctKeysOldestFirst: with PreferOldest, distinct kept units are
// ordered OLDEST-first by creation, even when the oldest ticket has the FRESHEST update — the
// backlog-draining policy that makes the dispatcher pick the longest-waiting ticket first.
func TestPreferOldestOrdersDistinctKeysOldestFirst(t *testing.T) {
	r := Plan(Input{NowUnix: base, PreferOldest: true, Candidates: []Candidate{
		{ID: "new", Key: "A", CreatedUnix: base - 100, UpdatedUnix: base - 10},
		{ID: "old", Key: "B", CreatedUnix: base - 900, UpdatedUnix: base - 5}, // oldest created, freshest update
		{ID: "mid", Key: "C", CreatedUnix: base - 500, UpdatedUnix: base - 50},
	}})
	if r.KeepCount != 3 {
		t.Fatalf("keep = %d, want 3 (distinct keys never supersede)", r.KeepCount)
	}
	want := []string{"old", "mid", "new"} // oldest-created first, regardless of update recency
	for i, id := range want {
		if r.Keep[i] != id {
			t.Errorf("keep[%d] = %q, want %q (oldest-first)", i, r.Keep[i], id)
		}
	}
	if r.Pick() != "old" {
		t.Errorf("pick = %q, want old (oldest ticket, even though it has the freshest update)", r.Pick())
	}
}

// TestPreferOldestStillKeepsFreshestWithinKey: PreferOldest changes only the ORDER of distinct
// survivors; within ONE supersede key the freshest duplicate still wins (the others are
// superseded), so the de-dup contract is preserved.
func TestPreferOldestStillKeepsFreshestWithinKey(t *testing.T) {
	r := Plan(Input{NowUnix: base, PreferOldest: true, Candidates: []Candidate{
		{ID: "dup-old", Key: "X", CreatedUnix: base - 900, UpdatedUnix: base - 500},
		{ID: "dup-new", Key: "X", CreatedUnix: base - 100, UpdatedUnix: base - 50}, // freshest update of target X
	}})
	if r.Pick() != "dup-new" {
		t.Errorf("pick = %q, want dup-new (freshest within the key, even under PreferOldest)", r.Pick())
	}
	if dispoOf(r, "dup-old") != DispSuperseded {
		t.Errorf("dup-old disposition = %q, want superseded", dispoOf(r, "dup-old"))
	}
}

func TestCollisionPricedFanoutSerializesExclusiveOverlapBeforeLaunch(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "gateway-old", Key: "A", Lane: "gateway", Tree: []string{"internal/gateway/**"}, UpdatedUnix: base - 300},
		{ID: "gateway-fresh", Key: "B", Lane: "gateway", Tree: []string{"internal/gateway/http.go"}, UpdatedUnix: base - 100},
		{ID: "docs", Key: "C", Lane: "docs", Tree: []string{"docs/**"}, UpdatedUnix: base - 200},
	}})

	if r.KeepCount != 2 || r.CollisionCount != 1 {
		t.Fatalf("counts = keep %d collision %d, want 2/1", r.KeepCount, r.CollisionCount)
	}
	if dispoOf(r, "gateway-fresh") != DispKeep || dispoOf(r, "docs") != DispKeep {
		t.Fatalf("safe set dispositions: gateway-fresh=%q docs=%q, want keep/keep",
			dispoOf(r, "gateway-fresh"), dispoOf(r, "docs"))
	}
	if dispoOf(r, "gateway-old") != DispCollisionRisk {
		t.Fatalf("gateway-old disposition = %q, want collision_risk", dispoOf(r, "gateway-old"))
	}
	if r.CollisionsAvoided != 1 || r.LanesUtilized != 2 || r.SerializationWasted != 1 || r.SafeConcurrency != 2 {
		t.Fatalf("S0 counts = avoided %d lanes %d wasted %d safe %d, want 1/2/1/2",
			r.CollisionsAvoided, r.LanesUtilized, r.SerializationWasted, r.SafeConcurrency)
	}
	if len(r.Collisions) != 1 || r.Collisions[0].Reason != ReasonCollisionRisk {
		t.Fatalf("collisions = %+v, want one %s edge", r.Collisions, ReasonCollisionRisk)
	}
}

func TestCollisionPricedFanoutAllowsSharedSharedOverlap(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "reader-a", Key: "A", Lane: "docs", Tree: []string{"docs/**"}, Mode: "shared", UpdatedUnix: base - 10},
		{ID: "reader-b", Key: "B", Lane: "docs", Tree: []string{"docs/notes/**"}, Mode: "shared", UpdatedUnix: base - 20},
	}})
	if r.KeepCount != 2 || r.CollisionCount != 0 || len(r.Collisions) != 0 {
		t.Fatalf("shared/shared overlap counts = keep %d collision %d edges %d, want 2/0/0",
			r.KeepCount, r.CollisionCount, len(r.Collisions))
	}
}

func TestCollisionPricedFanoutUnknownTreeCollidesConservatively(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "unknown", Key: "A", Lane: "gateway", UpdatedUnix: base - 10},
		{ID: "known", Key: "B", Lane: "docs", Tree: []string{"docs/**"}, UpdatedUnix: base - 20},
	}})
	if r.KeepCount != 1 || r.CollisionCount != 1 || r.CollisionsAvoided != 1 {
		t.Fatalf("unknown-tree pricing = keep %d collision %d avoided %d, want 1/1/1",
			r.KeepCount, r.CollisionCount, r.CollisionsAvoided)
	}
	if dispoOf(r, "unknown") != DispKeep || dispoOf(r, "known") != DispCollisionRisk {
		t.Fatalf("dispositions unknown=%q known=%q, want keep/collision_risk",
			dispoOf(r, "unknown"), dispoOf(r, "known"))
	}
}

// TestEmptyKeyNeverSuperseded: an empty Key opts a unit out of collapse — two empty-key units
// are both kept even though they would otherwise look like duplicates.
func TestEmptyKeyNeverSuperseded(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "u1", Key: "", UpdatedUnix: base - 50},
		{ID: "u2", Key: "", UpdatedUnix: base - 60},
	}})
	if r.KeepCount != 2 || r.SupersededCount != 0 {
		t.Errorf("empty-key units: keep %d superseded %d, want 2/0", r.KeepCount, r.SupersededCount)
	}
}

// TestLiveWinnerYieldsNoKeep: when the freshest unit for a key is already running, the group
// yields NO keep this tick — the older duplicates are superseded (not run), and the live one is
// reported live, so the dispatcher does not start a stale duplicate behind a running fresh one.
func TestLiveWinnerYieldsNoKeep(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "fresh", Key: "X", UpdatedUnix: base - 100, Live: true},
		{ID: "stale", Key: "X", UpdatedUnix: base - 500},
	}})
	if r.KeepCount != 0 {
		t.Fatalf("keep = %d, want 0 (freshest is live)", r.KeepCount)
	}
	if dispoOf(r, "fresh") != DispLive {
		t.Errorf("fresh disposition = %q, want live", dispoOf(r, "fresh"))
	}
	if dispoOf(r, "stale") != DispSuperseded {
		t.Errorf("stale disposition = %q, want superseded", dispoOf(r, "stale"))
	}
	if r.Pick() != "" {
		t.Errorf("pick = %q, want empty (nothing dispatchable)", r.Pick())
	}
}

// TestCooldownHoldsFreshestNoFallback: the freshest unit attempted within the cooldown window is
// held (cooling) — and the planner does NOT fall back to an older duplicate, so the group still
// yields no keep. This is the deliberate v1 posture (freshest-or-wait).
func TestCooldownHoldsFreshestNoFallback(t *testing.T) {
	r := Plan(Input{NowUnix: base, CooldownSeconds: 600, Candidates: []Candidate{
		{ID: "fresh", Key: "X", UpdatedUnix: base - 100, LastAttemptUnix: base - 60}, // attempted 60s ago < 600s
		{ID: "stale", Key: "X", UpdatedUnix: base - 900},
	}})
	if dispoOf(r, "fresh") != DispCooling {
		t.Fatalf("fresh disposition = %q, want cooling", dispoOf(r, "fresh"))
	}
	if dispoOf(r, "stale") != DispSuperseded {
		t.Errorf("stale disposition = %q, want superseded (no fallback to older dup)", dispoOf(r, "stale"))
	}
	if r.Pick() != "" || r.KeepCount != 0 {
		t.Errorf("pick=%q keep=%d, want empty/0 (freshest cooling, no fallback)", r.Pick(), r.KeepCount)
	}
}

// TestCooldownExpiredKeeps: once the cooldown window passes, the freshest unit is keepable again.
func TestCooldownExpiredKeeps(t *testing.T) {
	r := Plan(Input{NowUnix: base, CooldownSeconds: 600, Candidates: []Candidate{
		{ID: "fresh", Key: "X", UpdatedUnix: base - 100, LastAttemptUnix: base - 700}, // 700s ago > 600s
	}})
	if r.Pick() != "fresh" {
		t.Errorf("pick = %q, want fresh (cooldown expired)", r.Pick())
	}
}

// TestUpdatedBeatsCreatedAndId: recency is UPDATED time, not creation and not id — a unit
// created later but updated earlier loses to one updated more recently. This is the fix for the
// old "freshest = largest issue number" behavior.
func TestUpdatedBeatsCreatedAndId(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "zzz-new", Key: "X", CreatedUnix: base - 10, UpdatedUnix: base - 900}, // newest+highest id, but stale update
		{ID: "aaa-old", Key: "X", CreatedUnix: base - 999, UpdatedUnix: base - 50}, // oldest+lowest id, freshest update
	}})
	if r.Pick() != "aaa-old" {
		t.Errorf("pick = %q, want aaa-old (most recently UPDATED, not highest id/newest created)", r.Pick())
	}
}

// TestDeterministicAndTotal: identical inputs give identical results, and the empty input yields
// a defined empty result.
func TestDeterministicAndTotal(t *testing.T) {
	in := Input{NowUnix: base, Candidates: []Candidate{
		{ID: "a", Key: "K", UpdatedUnix: base - 1},
		{ID: "b", Key: "K", UpdatedUnix: base - 2},
		{ID: "c", Key: "", UpdatedUnix: base - 3},
	}}
	a, b := Plan(in), Plan(in)
	if fmt.Sprint(a) != fmt.Sprint(b) {
		t.Errorf("Plan is not deterministic:\n%v\n%v", a, b)
	}
	empty := Plan(Input{NowUnix: base})
	if len(empty.Order) != 0 || empty.Pick() != "" {
		t.Errorf("empty input = %+v, want empty/no-pick", empty)
	}
}

// TestNegativeCooldownDisables: a negative cooldown disables the hold — an attempted freshest is
// still keepable.
func TestNegativeCooldownDisables(t *testing.T) {
	r := Plan(Input{NowUnix: base, CooldownSeconds: -1, Candidates: []Candidate{
		{ID: "fresh", Key: "X", UpdatedUnix: base - 10, LastAttemptUnix: base - 1},
	}})
	if r.Pick() != "fresh" {
		t.Errorf("pick = %q, want fresh (cooldown disabled)", r.Pick())
	}
}
