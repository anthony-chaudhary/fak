package dispatchorder

// The #2558 witness: a sustained-STALL ("wedged") objective is DEPRIORITIZED in the dispatch
// order — within its priority tier it sorts after fresh work, with ReasonWedgedObjective
// ledgered — but it is never refused: it stays DispKeep, stays in Order and Keep, and a lone
// wedged candidate is still picked. The trajctl curve signal arrives as PLAIN DATA
// (Candidate.ObjectiveSignal), so this pureRoot leaf keeps importing nothing internal.

import (
	"encoding/json"
	"strings"
	"testing"
)

// reasonOf returns the reason the planner ledgered for the unit with id, or "" if absent.
func reasonOf(r Result, id string) string {
	for _, x := range r.Order {
		if x.ID == id {
			return x.Reason
		}
	}
	return ""
}

// TestWedgedObjectiveSortsBelowFreshSamePriorityTier is the #2558 Done condition: at the same
// priority tier, a wedged (sustained-STALL) objective sorts BELOW fresh candidates — even when
// the wedged unit is the FRESHEST by recency (re-attempts keep bumping a wedged unit's update
// stamp, so recency alone would keep re-feeding it) — and carries the ledgered reason. Both
// wedged units stay kept and ranked: a demotion, never a refusal.
func TestWedgedObjectiveSortsBelowFreshSamePriorityTier(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "wedged-fresher", Key: "A", ObjectiveSignal: "STALL", UpdatedUnix: base - 10}, // freshest of all
		{ID: "fresh", Key: "B", UpdatedUnix: base - 200},                                   // no objective signal
		{ID: "healthy", Key: "C", ObjectiveSignal: "HEALTHY", UpdatedUnix: base - 100},
		{ID: "wedged-staler", Key: "D", ObjectiveSignal: "STALL", UpdatedUnix: base - 300},
	}})
	if r.KeepCount != 4 {
		t.Fatalf("keep = %d, want 4 (deprioritization never drops a unit)", r.KeepCount)
	}
	// Fresh work first (by recency among themselves), then the wedged units (by recency among
	// themselves) — the wedged-vs-fresh key is inside the priority tier, outside recency.
	want := []string{"healthy", "fresh", "wedged-fresher", "wedged-staler"}
	for i, id := range want {
		if r.Keep[i] != id {
			t.Errorf("keep[%d] = %q, want %q (fresh before wedged, then recency)", i, r.Keep[i], id)
		}
	}
	if r.Pick() != "healthy" {
		t.Errorf("pick = %q, want healthy (never a wedged unit while a fresh alternative exists)", r.Pick())
	}
	for _, id := range []string{"wedged-fresher", "wedged-staler"} {
		if got := reasonOf(r, id); got != ReasonWedgedObjective {
			t.Errorf("%s reason = %q, want %q (the ledgered deprioritization reason)", id, got, ReasonWedgedObjective)
		}
		if got := dispoOf(r, id); got != DispKeep {
			t.Errorf("%s disposition = %q, want keep (deprioritized, not refused)", id, got)
		}
	}
	for _, id := range []string{"fresh", "healthy"} {
		if got := reasonOf(r, id); got != ReasonFreshest {
			t.Errorf("%s reason = %q, want %q (fresh work is untouched)", id, got, ReasonFreshest)
		}
	}
}

// TestLoneWedgedObjectiveNotRefused: with no fresh alternative, a wedged unit is still kept,
// ranked first, and picked, with the ordinary ReasonFreshest — the reason token is attached
// only when a fresh alternative actually outranks it, so deprioritization never becomes an
// implicit refusal or starvation.
func TestLoneWedgedObjectiveNotRefused(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "wedged-solo", Key: "A", ObjectiveSignal: "STALL", UpdatedUnix: base - 10},
	}})
	if r.KeepCount != 1 || r.Pick() != "wedged-solo" {
		t.Fatalf("keep = %d pick = %q, want 1/wedged-solo (a lone wedged unit still dispatches)", r.KeepCount, r.Pick())
	}
	if got := reasonOf(r, "wedged-solo"); got != ReasonFreshest {
		t.Errorf("reason = %q, want %q (no fresh alternative => no deprioritization to ledger)", got, ReasonFreshest)
	}
	// All-wedged sets behave the same: nobody is demoted below anybody fresh, order is recency.
	r = Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "w-old", Key: "A", ObjectiveSignal: "STALL", UpdatedUnix: base - 200},
		{ID: "w-new", Key: "B", ObjectiveSignal: "STALL", UpdatedUnix: base - 20},
	}})
	if r.KeepCount != 2 || r.Pick() != "w-new" {
		t.Fatalf("all-wedged keep = %d pick = %q, want 2/w-new (recency order, nothing refused)", r.KeepCount, r.Pick())
	}
	for _, id := range []string{"w-new", "w-old"} {
		if got := reasonOf(r, id); got != ReasonFreshest {
			t.Errorf("%s reason = %q, want %q (no fresh alternative exists)", id, got, ReasonFreshest)
		}
	}
}

// TestPriorityTierLeadsWedgedDemotion: the priority tier stays the OUTER key — a wedged P0
// still outranks fresh unlabeled work, and since nothing fresh outranks it, no wedged reason
// is ledgered (it was not, in fact, deprioritized).
func TestPriorityTierLeadsWedgedDemotion(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "wedged-p0", Key: "A", Priority: 1000, ObjectiveSignal: "STALL", UpdatedUnix: base - 500},
		{ID: "fresh-noprio", Key: "B", UpdatedUnix: base - 10},
	}})
	if r.Pick() != "wedged-p0" {
		t.Fatalf("pick = %q, want wedged-p0 (priority is the outer key; the wedge demotes only within a tier)", r.Pick())
	}
	if got := reasonOf(r, "wedged-p0"); got != ReasonFreshest {
		t.Errorf("wedged-p0 reason = %q, want %q (not outranked by fresh work => not deprioritized)", got, ReasonFreshest)
	}
}

// TestWedgedDemotionAppliesUnderPreferOldest: the wedged-vs-fresh key sits inside the priority
// tier but OUTSIDE the recency/PreferOldest tie-break, so even the backlog-draining oldest-first
// policy dispatches fresh work ahead of an older wedged unit.
func TestWedgedDemotionAppliesUnderPreferOldest(t *testing.T) {
	r := Plan(Input{NowUnix: base, PreferOldest: true, Candidates: []Candidate{
		{ID: "wedged-oldest", Key: "A", ObjectiveSignal: "STALL", CreatedUnix: base - 900, UpdatedUnix: base - 800},
		{ID: "fresh-newer", Key: "B", CreatedUnix: base - 100, UpdatedUnix: base - 50},
	}})
	if r.Pick() != "fresh-newer" {
		t.Fatalf("pick = %q, want fresh-newer (fresh outranks wedged even under prefer-oldest)", r.Pick())
	}
	if got := reasonOf(r, "wedged-oldest"); got != ReasonWedgedObjective {
		t.Errorf("wedged-oldest reason = %q, want %q", got, ReasonWedgedObjective)
	}
}

// TestWedgedSignalCaseInsensitive: the plain-data token matches case-insensitively and
// whitespace-tolerantly, so a caller passing "stall" or " STALL " still deprioritizes.
func TestWedgedSignalCaseInsensitive(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "wedged-lower", Key: "A", ObjectiveSignal: " stall ", UpdatedUnix: base - 10},
		{ID: "fresh", Key: "B", UpdatedUnix: base - 200},
	}})
	if r.Pick() != "fresh" {
		t.Fatalf("pick = %q, want fresh (lowercase/padded stall token still wedges)", r.Pick())
	}
	if got := reasonOf(r, "wedged-lower"); got != ReasonWedgedObjective {
		t.Errorf("wedged-lower reason = %q, want %q", got, ReasonWedgedObjective)
	}
}

// TestEmptyObjectiveSignalNoRegression is the additive-no-regression witness for the field:
// candidates with an empty (or non-STALL) ObjectiveSignal order exactly as before the field
// existed — pure freshest-first, ReasonFreshest everywhere — and the zero value serializes with
// NO objective_signal key, so the pinned all-zero golden (TestAllZeroPriorityByteIdenticalResult)
// keeps holding byte-identity for every existing caller.
func TestEmptyObjectiveSignalNoRegression(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "a", Key: "A", UpdatedUnix: base - 300},
		{ID: "b", Key: "B", ObjectiveSignal: "", UpdatedUnix: base - 100}, // explicit empty == absent
		{ID: "c", Key: "C", ObjectiveSignal: "DRIFT", UpdatedUnix: base - 200},
	}})
	want := []string{"b", "c", "a"} // freshest-first, untouched
	for i, id := range want {
		if r.Keep[i] != id {
			t.Errorf("keep[%d] = %q, want %q (empty/non-STALL signals reproduce today's order)", i, r.Keep[i], id)
		}
	}
	for _, x := range r.Order {
		if x.Reason == ReasonWedgedObjective {
			t.Errorf("%s carries %q without any STALL candidate", x.ID, ReasonWedgedObjective)
		}
	}
	b, err := json.Marshal(Candidate{ID: "zero"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "objective_signal") {
		t.Errorf("zero-value Candidate JSON = %s, want no objective_signal key (omitempty guards the golden)", b)
	}
}
