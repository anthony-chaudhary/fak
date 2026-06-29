package ctxplan

import "testing"

// TestRecommendBudgetHarderGetsLargerW is the headline acceptance check: a harder task — more
// intents, more pins, more observed faults — recommends a LARGER working set W than an easy
// one. A fixed Budget.Tokens could not express this; the adaptive budget must.
func TestRecommendBudgetHarderGetsLargerW(t *testing.T) {
	bounds := DefaultBudgetBounds()

	easy := Difficulty{Intents: 0, Pins: 0, Horizon: 1, FaultRate: 0}
	hard := Difficulty{Intents: 8, Pins: 6, Horizon: 4, FaultRate: 1}

	wEasy := RecommendBudget(easy, bounds).Tokens
	wHard := RecommendBudget(hard, bounds).Tokens

	if !(wHard > wEasy) {
		t.Fatalf("harder task must recommend a larger W: easy=%d hard=%d", wEasy, wHard)
	}
	// The easy end should sit at the floor and the saturated-hard end at the ceiling, so the
	// recommendation actually USES the spectrum rather than hugging one point.
	if wEasy != bounds.Floor {
		t.Errorf("easiest task should recommend the floor W=%d, got %d", bounds.Floor, wEasy)
	}
	if wHard != bounds.Ceil {
		t.Errorf("saturated-hard task should recommend the ceiling W=%d, got %d", bounds.Ceil, wHard)
	}
}

// TestRecommendBudgetDeterministic: the same inputs must yield byte-identical W (no randomness,
// no wall clock) — replanning the same turn twice cannot drift the budget.
func TestRecommendBudgetDeterministic(t *testing.T) {
	bounds := BudgetBounds{Floor: 256, Ceil: 4096}
	d := Difficulty{Intents: 5, Pins: 3, Horizon: 2, FaultRate: 0.4}

	first := RecommendBudget(d, bounds).Tokens
	for i := 0; i < 64; i++ {
		if got := RecommendBudget(d, bounds).Tokens; got != first {
			t.Fatalf("non-deterministic: run %d gave %d, want %d", i, got, first)
		}
	}
	// A middling task must land strictly between floor and ceiling — proves the interpolation
	// actually moves, it is not pinned to an endpoint.
	if !(first > bounds.Floor && first < bounds.Ceil) {
		t.Fatalf("middling task should land between bounds (%d, %d), got %d", bounds.Floor, bounds.Ceil, first)
	}
}

// TestRecommendBudgetBounded: W is never 0 and never unbounded, even for pathological signals
// and degenerate bounds — the BOUNDED guarantee.
func TestRecommendBudgetBounded(t *testing.T) {
	bounds := BudgetBounds{Floor: 100, Ceil: 1000}

	// A pathological over-count must clamp to the ceiling, never exceed it.
	huge := Difficulty{Intents: 100000, Pins: 100000, Horizon: 100000, FaultRate: 999}
	if w := RecommendBudget(huge, bounds).Tokens; w != bounds.Ceil {
		t.Errorf("pathological-hard task must clamp to ceiling %d, got %d", bounds.Ceil, w)
	}

	// Negative / NaN-ish signals must fail closed to the floor, never below it, never 0.
	neg := Difficulty{Intents: -5, Pins: -5, Horizon: -5, FaultRate: -3}
	if w := RecommendBudget(neg, bounds).Tokens; w != bounds.Floor {
		t.Errorf("negative-signal task must clamp to floor %d, got %d", bounds.Floor, w)
	}

	// A zero/empty bounds set must never produce W=0: it collapses to a fixed W of at least 1.
	empty := RecommendBudget(huge, BudgetBounds{}).Tokens
	if empty < 1 {
		t.Fatalf("zero bounds must still recommend W>=1 (never 0), got %d", empty)
	}

	// An INVERTED bounds set (ceil < floor) collapses to the floor — a single fixed W, the
	// documented worst case, never an unbounded or negative W.
	inverted := RecommendBudget(huge, BudgetBounds{Floor: 800, Ceil: 200}).Tokens
	if inverted != 800 {
		t.Errorf("inverted bounds must collapse to the floor (800), got %d", inverted)
	}
}

// TestRecommendBudgetMonotone: increasing any single difficulty signal never DECREASES W — the
// frontier never folds back (more budget for a harder task, never less). This is the witness
// the issue asks for: more difficulty never hurts the resident-view sizing.
func TestRecommendBudgetMonotone(t *testing.T) {
	bounds := DefaultBudgetBounds()
	base := Difficulty{Intents: 2, Pins: 1, Horizon: 1, FaultRate: 0.1}
	w0 := RecommendBudget(base, bounds).Tokens

	bumps := []struct {
		name string
		d    Difficulty
	}{
		{"more intents", func() Difficulty { d := base; d.Intents += 3; return d }()},
		{"more pins", func() Difficulty { d := base; d.Pins += 3; return d }()},
		{"longer horizon", func() Difficulty { d := base; d.Horizon += 2; return d }()},
		{"higher fault rate", func() Difficulty { d := base; d.FaultRate += 0.5; return d }()},
	}
	for _, b := range bumps {
		if w := RecommendBudget(b.d, bounds).Tokens; w < w0 {
			t.Errorf("%s LOWERED W (%d < base %d) — frontier folded back", b.name, w, w0)
		}
	}
}

// TestDifficultyFromOutcomeClosedLoop: a turn that FAULTED a lot (the forecast under-budgeted)
// must size the NEXT turn's W UP versus a turn that hit everything resident — the closed-loop
// rung the planner already learns over (learn.go), now driving the budget.
func TestDifficultyFromOutcomeClosedLoop(t *testing.T) {
	bounds := DefaultBudgetBounds()
	f := Forecast{Intents: []string{"refund", "fee"}, Horizon: 2}

	// Last turn confirmed the forecast: everything it needed was resident (all hits, no faults).
	confirmed := Outcome{Hits: []string{"a", "b", "c", "d"}}
	// Last turn thrashed: it demand-paged most of what it needed back in (mostly faults).
	thrashed := Outcome{Hits: []string{"a"}, Faults: []string{"b", "c", "d", "e", "f"}}

	wConfirmed := RecommendBudgetFromOutcome(f, confirmed, bounds).Tokens
	wThrashed := RecommendBudgetFromOutcome(f, thrashed, bounds).Tokens

	if !(wThrashed > wConfirmed) {
		t.Fatalf("a high-fault turn must size the next W up: confirmed=%d thrashed=%d", wConfirmed, wThrashed)
	}

	// The fault rate must be the witnessed faults/(hits+faults), not invented.
	dThrashed := DifficultyFromOutcome(f, thrashed)
	if want := 5.0 / 6.0; dThrashed.FaultRate != want {
		t.Errorf("FaultRate = %v, want %v (5 faults / 6 referenced)", dThrashed.FaultRate, want)
	}
	// A turn that touched nothing resident-or-elided yields a 0 fault rate (cold/neutral), not
	// a divide-by-zero.
	if d := DifficultyFromOutcome(f, Outcome{}); d.FaultRate != 0 {
		t.Errorf("empty outcome FaultRate = %v, want 0", d.FaultRate)
	}
}

// TestRecommendBudgetForForecastStaticPath: the first-turn convenience path sizes W from the
// forecast's declared shape alone (no outcome), and a forecast with more intents + pins gets a
// larger W than a bare one — proving the static signals feed through.
func TestRecommendBudgetForForecastStaticPath(t *testing.T) {
	bounds := DefaultBudgetBounds()

	bare := Forecast{Intents: []string{"x"}}
	broad := Forecast{
		Intents: []string{"a", "b", "c", "d", "e"},
		Pins:    []string{"p1", "p2", "p3"},
		Horizon: 3,
	}

	wBare := RecommendBudgetForForecast(bare, bounds).Tokens
	wBroad := RecommendBudgetForForecast(broad, bounds).Tokens

	if !(wBroad > wBare) {
		t.Fatalf("a broader forecast should recommend a larger W: bare=%d broad=%d", wBare, wBroad)
	}
	if wBare < bounds.Floor || wBroad > bounds.Ceil {
		t.Fatalf("both recommendations must stay within bounds [%d,%d]: bare=%d broad=%d",
			bounds.Floor, bounds.Ceil, wBare, wBroad)
	}
	// DifficultyFromForecast must report the static signals verbatim and a 0 fault rate (no
	// outcome yet).
	d := DifficultyFromForecast(broad)
	if d.Intents != 5 || d.Pins != 3 || d.Horizon != 3 || d.FaultRate != 0 {
		t.Errorf("static difficulty mismatch: %+v", d)
	}
}

// TestRecommendBudgetFeedsOptimize ties the recommendation back to the planner it sizes: the
// recommended Budget is a normal Budget Optimize accepts, and a larger recommended W keeps at
// least as many spans resident as a smaller one (the MONOTONE-recall frontier — more budget
// never hurts recall). This proves the recommendation is advisory-but-real: it changes
// efficiency, never correctness.
func TestRecommendBudgetFeedsOptimize(t *testing.T) {
	// Five equal-cost, non-pinned candidates; each costs 100 tokens and carries a positive
	// benefit (the greedy planner skips zero-benefit spans regardless of budget, so the budget
	// only BINDS on spans worth a resident slot — give each one).
	cands := make([]Candidate, 0, 5)
	for i := 0; i < 5; i++ {
		id := string(rune('a' + i))
		cands = append(cands, Candidate{
			Cell:    Span{ID: id, Step: i + 1, Descriptor: "span " + id, Bytes: 100},
			Cost:    100,
			Benefit: 1.0,
		})
	}

	easyBudget := RecommendBudget(Difficulty{Intents: 0, Pins: 0, Horizon: 1, FaultRate: 0}, BudgetBounds{Floor: 100, Ceil: 500})
	hardBudget := RecommendBudget(Difficulty{Intents: 8, Pins: 6, Horizon: 4, FaultRate: 1}, BudgetBounds{Floor: 100, Ceil: 500})

	easyPlan := Optimize(cands, easyBudget, nil, ObjGreedy)
	hardPlan := Optimize(cands, hardBudget, nil, ObjGreedy)

	if len(hardPlan.Selected) < len(easyPlan.Selected) {
		t.Fatalf("a larger recommended W must keep >= as many spans resident: easy kept %d, hard kept %d",
			len(easyPlan.Selected), len(hardPlan.Selected))
	}
	// Recoverability is preserved at BOTH budgets: every candidate is partitioned into
	// selected-or-elided, so the tighter budget lost no fact, it only paged more out.
	for _, p := range []Plan{easyPlan, hardPlan} {
		if got := len(p.Selected) + len(p.Elided); got != len(cands) {
			t.Errorf("plan dropped a candidate: selected+elided=%d, want %d", got, len(cands))
		}
	}
}
