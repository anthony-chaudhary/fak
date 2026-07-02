package ctxplan

import (
	"reflect"
	"testing"
)

// TestReleasedSpanElidedRecoverableAndBudgetFreed is the core property (#2225): releasing
// a span (a) elides it with reason ElideReleased and its recovery handle intact, (b) frees
// its budget for a span the work still needs, and (c) keeps the plan Faithful — the
// release lane is a residency decision, never destruction.
func TestReleasedSpanElidedRecoverableAndBudgetFreed(t *testing.T) {
	cands := []Candidate{
		cand("stale", 1, 5, 5.0), // denser — wins the knapsack when not released
		cand("live", 2, 5, 3.0),  // loses the knapsack to "stale" under budget 8
	}

	without := Optimize(cands, Budget{Tokens: 8}, nil, ObjGreedy)
	if len(without.Selected) != 1 || without.Selected[0].ID != "stale" {
		t.Fatalf("precondition: without a release the denser span must win, selected=%+v", without.Selected)
	}

	released := map[string]bool{"stale": true}
	p := OptimizeWithReleases(cands, Budget{Tokens: 8}, nil, released, ObjGreedy)

	if len(p.Selected) != 1 || p.Selected[0].ID != "live" {
		t.Errorf("the released span's budget must admit the live span, selected=%+v", p.Selected)
	}
	found := false
	for _, e := range p.Elided {
		if e.ID == "stale" {
			found = true
			if e.Reason != ElideReleased {
				t.Errorf("released span must carry reason %q, got %q", ElideReleased, e.Reason)
			}
			if e.Digest == "" {
				t.Errorf("released span must keep its recovery handle — a release is not destruction")
			}
		}
	}
	if !found {
		t.Fatalf("released span must be elided, not dropped; elided=%+v", p.Elided)
	}
	if w := Audit(p); !w.Faithful {
		t.Errorf("a plan with a released span must stay faithful, witness=%+v", w)
	}
}

// TestPinOutranksRelease is the over-retain fence: a span both pinned and released stays
// RESIDENT (a false-retain costs tokens; a false-free costs context), and the conflict is
// reported as PinHeld — never silently resolved either way. Structural roots are pinned,
// so this is also what makes them un-releasable.
func TestPinOutranksRelease(t *testing.T) {
	cands := []Candidate{cand("root", 1, 3, 2.0), cand("other", 2, 3, 1.0)}
	pins := map[string]bool{"root": true}
	released := map[string]bool{"root": true}

	p := OptimizeWithReleases(cands, Budget{Tokens: 10}, pins, released, ObjGreedy)

	resident := false
	for _, s := range p.Selected {
		if s.ID == "root" && s.Pinned {
			resident = true
		}
	}
	if !resident {
		t.Fatalf("a pinned span must survive its own release, selected=%+v", p.Selected)
	}
	for _, e := range p.Elided {
		if e.ID == "root" {
			t.Errorf("a pinned span must not be elided by a release, elided=%+v", p.Elided)
		}
	}

	report := buildReleaseReport(p, []string{"root"})
	if !reflect.DeepEqual(report.PinHeld, []string{"root"}) || len(report.Honored) != 0 {
		t.Errorf("the pin-vs-release conflict must be reported PinHeld, report=%+v", report)
	}
}

// TestReleaseTrustLanePrecedence: sealed/tombstoned always win — a released poison span is
// elided as sealed (the trust verdict), not laundered into the softer "released" class,
// and the report calls it Gated (the release changed nothing).
func TestReleaseTrustLanePrecedence(t *testing.T) {
	sealed := cand("poison", 1, 3, 2.0)
	sealed.Cell.Sealed = true
	p := OptimizeWithReleases([]Candidate{sealed}, Budget{Tokens: 10}, nil, map[string]bool{"poison": true}, ObjGreedy)

	if len(p.Elided) != 1 || p.Elided[0].Reason != ElideSealed {
		t.Fatalf("a sealed span must keep reason %q under a release, elided=%+v", ElideSealed, p.Elided)
	}
	report := buildReleaseReport(p, []string{"poison"})
	if !reflect.DeepEqual(report.Gated, []string{"poison"}) {
		t.Errorf("a released sealed span is Gated, report=%+v", report)
	}
}

// TestOptimizeDelegationIsExact: Optimize must be byte-identical to OptimizeWithReleases
// with a nil release set — one planner, two entry points, zero divergence.
func TestOptimizeDelegationIsExact(t *testing.T) {
	cands := []Candidate{cand("a", 1, 3, 2.0), cand("b", 2, 4, 1.0), cand("c", 3, 5, 3.0)}
	p1 := Optimize(cands, Budget{Tokens: 7}, map[string]bool{"b": true}, ObjGreedy)
	p2 := OptimizeWithReleases(cands, Budget{Tokens: 7}, map[string]bool{"b": true}, nil, ObjGreedy)
	if !reflect.DeepEqual(p1, p2) {
		t.Errorf("Optimize and OptimizeWithReleases(nil) diverged:\n%+v\nvs\n%+v", p1, p2)
	}
}

// TestPlanQueryReleaseReport drives the AGENT-FACING path end to end: the model declares
// releases on its PlanQuery, and the PlanView that comes back both honors them in the
// plan and reports the disposition of every declared id — including the unknown id, which
// is an advisory no-op (a stale id cannot poison the plan). Run twice for determinism.
func TestPlanQueryReleaseReport(t *testing.T) {
	poison := mkSpan("poison", "tool", "quarantined dump", 3)
	poison.Sealed = true
	spans := []Span{
		mkSpan("old", "tool", "superseded draft of the fix", 1),
		mkSpan("keep", "user", "the standing constraint", 2),
		poison,
		mkSpan("live", "tool", "current failing test output", 4),
	}
	q := PlanQuery{
		Budget:   &Budget{Tokens: 100},
		Pins:     []string{"keep"},
		Releases: []string{"old", "keep", "poison", "ghost"},
	}

	view := q.Plan(spans, nil)
	again := q.Plan(spans, nil)
	if !reflect.DeepEqual(view, again) {
		t.Errorf("the same query must yield a byte-identical view (determinism)")
	}

	if view.Releases == nil {
		t.Fatalf("a query that declared releases must get a ReleaseReport back")
	}
	r := *view.Releases
	if !reflect.DeepEqual(r.Honored, []string{"old"}) {
		t.Errorf("Honored = %v, want [old]", r.Honored)
	}
	if !reflect.DeepEqual(r.PinHeld, []string{"keep"}) {
		t.Errorf("PinHeld = %v, want [keep]", r.PinHeld)
	}
	if !reflect.DeepEqual(r.Gated, []string{"poison"}) {
		t.Errorf("Gated = %v, want [poison]", r.Gated)
	}
	if !reflect.DeepEqual(r.Unknown, []string{"ghost"}) {
		t.Errorf("Unknown = %v, want [ghost]", r.Unknown)
	}

	elidedReleased := false
	for _, e := range view.Elided {
		if e.ID == "old" && e.Reason == ElideReleased {
			elidedReleased = true
		}
	}
	if !elidedReleased {
		t.Errorf("the honored release must show in the view's elided set, elided=%+v", view.Elided)
	}
	if !view.Faithful {
		t.Errorf("a view with releases must stay faithful")
	}
}

// TestClassifyReleases is the recant witness: a released span the turn demand-paged back
// in (an Outcome fault) proves the declaration wrong — RECANTED; one that stayed cold is
// VINDICATED. The two classes are disjoint and deterministic (deduped, sorted).
func TestClassifyReleases(t *testing.T) {
	got := ClassifyReleases([]string{"b", "a", "b"}, Outcome{Faults: []string{"b", "x"}})
	want := ReleaseOutcome{Vindicated: []string{"a"}, Recanted: []string{"b"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ClassifyReleases = %+v, want %+v", got, want)
	}
}

// TestDropRecantedReleases: a carried-forward forecast sheds exactly the releases the
// witnessed outcome refuted (so one wrong declaration converges to ONE fault, not one per
// turn), preserves the survivors' order, and is a no-op when nothing was recanted.
func TestDropRecantedReleases(t *testing.T) {
	f := Forecast{Releases: []string{"c", "a", "b"}}
	out := f.DropRecantedReleases(Outcome{Faults: []string{"b", "x"}})
	if !reflect.DeepEqual(out.Releases, []string{"c", "a"}) {
		t.Errorf("Releases after recant = %v, want [c a]", out.Releases)
	}

	same := f.DropRecantedReleases(Outcome{Faults: []string{"x"}})
	if !reflect.DeepEqual(same, f) {
		t.Errorf("no recant must be a no-op, got %+v", same)
	}
	empty := Forecast{}.DropRecantedReleases(Outcome{Faults: []string{"a"}})
	if len(empty.Releases) != 0 {
		t.Errorf("an empty release set stays empty, got %v", empty.Releases)
	}
}

// TestForecastFingerprintReleases: plan identity must move when the release SET changes
// and only then — an empty set fingerprints byte-identically to the pre-release-lane
// forecast (it plans identically), order/duplicates don't matter (a set), and a release
// can never collide with a pin of the same id (distinct sections).
func TestForecastFingerprintReleases(t *testing.T) {
	base := Forecast{Intents: []string{"refund fee"}, Horizon: 2}

	withNil := base
	withEmpty := base
	withEmpty.Releases = []string{}
	if ForecastFingerprint(withNil) != ForecastFingerprint(withEmpty) {
		t.Errorf("an empty release set must not move the fingerprint")
	}

	withRel := base
	withRel.Releases = []string{"a", "b"}
	if ForecastFingerprint(withRel) == ForecastFingerprint(base) {
		t.Errorf("a non-empty release set must move the fingerprint")
	}

	shuffled := base
	shuffled.Releases = []string{"b", "a", "a"}
	if ForecastFingerprint(withRel) != ForecastFingerprint(shuffled) {
		t.Errorf("releases are a set: order and duplicates must not move the fingerprint")
	}

	pinned := base
	pinned.Pins = []string{"a"}
	releasedSame := base
	releasedSame.Releases = []string{"a"}
	if ForecastFingerprint(pinned) == ForecastFingerprint(releasedSame) {
		t.Errorf("a pin and a release of the same id are different forecasts")
	}
}
