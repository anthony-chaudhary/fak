package ctxplan

import (
	"reflect"
	"testing"
)

// querySpans is the shared fixture: a small history with a clearly-relevant span, an
// irrelevant one, a pinnable structural span, and a SEALED poison span — enough to witness
// selection, budgeting, pinning, and the trust gate through the facade.
func querySpans() []Span {
	return []Span{
		{ID: "sys", Step: 0, Role: "system", Descriptor: "system prompt and active task", Bytes: 40, Durability: DurabilityDurable},
		{ID: "hit", Step: 1, Role: "tool", Descriptor: "rotated the auth token for the service", Bytes: 40, Digest: "d-hit", Durability: DurabilitySession},
		{ID: "miss", Step: 2, Role: "tool", Descriptor: "unrelated weather report for tuesday", Bytes: 40, Digest: "d-miss", Durability: DurabilityTurn},
		{ID: "poison", Step: 3, Role: "tool", Descriptor: "auth token rotation but quarantined", Bytes: 40, Digest: "d-poison", Sealed: true},
	}
}

// TestPlanQueryMatchesDirectPlanCells is the headline acceptance check: a PlanQuery driven by
// the model produces a PlanView whose selected/elided/cost are BYTE-IDENTICAL to a direct,
// host-driven PlanCells call with the same forecast + budget. The facade is a typed front
// door, not a second planner — it must add NO divergence.
func TestPlanQueryMatchesDirectPlanCells(t *testing.T) {
	spans := querySpans()
	budget := Budget{Tokens: 64}

	q := PlanQuery{Intents: []string{"auth token rotation"}, Budget: &budget, Pins: []string{"sys"}}
	view := q.Plan(spans, nil)

	// The direct, host-driven path with the SAME forecast + budget the facade resolves.
	want := PlanCells(spans, q.forecast(), budget, nil)

	if !reflect.DeepEqual(view.Selected, want.Selected) {
		t.Errorf("facade Selected diverged from direct PlanCells:\n got %+v\nwant %+v", view.Selected, want.Selected)
	}
	if !reflect.DeepEqual(view.Elided, want.Elided) {
		t.Errorf("facade Elided diverged from direct PlanCells:\n got %+v\nwant %+v", view.Elided, want.Elided)
	}
	if view.CostUsed != want.CostUsed {
		t.Errorf("facade CostUsed=%d, direct PlanCells=%d", view.CostUsed, want.CostUsed)
	}
	if view.Budget != want.Budget {
		t.Errorf("facade Budget=%d, direct PlanCells=%d", view.Budget, want.Budget)
	}
	// The embedded Plan must be the direct plan verbatim — the view is a projection of it.
	if !reflect.DeepEqual(view.Plan, want) {
		t.Errorf("embedded Plan diverged from direct PlanCells")
	}
	// The explain string and faithful verdict are the plan's own — not a recomputation.
	if view.Explain != want.Explain() {
		t.Errorf("facade Explain diverged from Plan.Explain()")
	}
	if view.Faithful != Audit(want).Faithful {
		t.Errorf("facade Faithful=%v, Audit(plan).Faithful=%v", view.Faithful, Audit(want).Faithful)
	}

	// Sanity that the fixture exercises real selection: the relevant span must be resident.
	if !selectedHas(view, "hit") {
		t.Errorf("the relevant 'hit' span should be resident, view selected=%v", viewSelectedIDs(view))
	}
}

// TestPlanQueryUnderspecifiedUsesAdaptiveDefault: an under-specified query (intents only, NO
// budget) must NOT fall to a magic constant or a zero budget — it must size a sane working set
// via the adaptive RecommendBudgetForForecast within DefaultBudgetBounds. The resolved budget
// must be exactly what the adaptive sizer returns for the query's declared shape.
func TestPlanQueryUnderspecifiedUsesAdaptiveDefault(t *testing.T) {
	spans := querySpans()

	q := PlanQuery{Intents: []string{"auth token rotation"}, Horizon: 2}
	view := q.Plan(spans, nil)

	wantBudget := RecommendBudgetForForecast(q.forecast(), DefaultBudgetBounds())
	if view.Budget != wantBudget.Tokens {
		t.Errorf("under-specified query should adopt the adaptive default budget %d, got %d",
			wantBudget.Tokens, view.Budget)
	}
	// The default must be a real, bounded working set — never 0, never unbounded.
	if view.Budget < DefaultBudgetBounds().Floor || view.Budget > DefaultBudgetBounds().Ceil {
		t.Errorf("adaptive default %d out of bounds [%d,%d]", view.Budget,
			DefaultBudgetBounds().Floor, DefaultBudgetBounds().Ceil)
	}
	// And the plan run under that default must be identical to PlanCells with it — the facade
	// only CHOOSES the budget when unset, it does not change how the plan is computed.
	want := PlanCells(spans, q.forecast(), wantBudget, nil)
	if !reflect.DeepEqual(view.Plan, want) {
		t.Errorf("under-specified plan diverged from PlanCells with the adaptive budget")
	}
	// The horizon the model declared must round-trip into the view.
	if view.Horizon != 2 {
		t.Errorf("view should carry the declared horizon 2, got %d", view.Horizon)
	}
}

// TestPlanQueryExplicitBudgetHonored: a model that states its OWN cap is honored verbatim — the
// adaptive default is taken ONLY when the budget is unset. A tiny explicit budget must produce
// a correspondingly tight resident view, not the larger adaptive size.
func TestPlanQueryExplicitBudgetHonored(t *testing.T) {
	spans := querySpans()
	tight := Budget{Tokens: 11} // room for ~one 40-byte span (ceil(40/4)=10)

	q := PlanQuery{Intents: []string{"auth token rotation"}, Budget: &tight}
	view := q.Plan(spans, nil)

	if view.Budget != tight.Tokens {
		t.Fatalf("explicit budget must be honored verbatim: got %d want %d", view.Budget, tight.Tokens)
	}
	if view.CostUsed > tight.Tokens {
		t.Errorf("resident cost %d exceeded the explicit budget %d", view.CostUsed, tight.Tokens)
	}
	// The explicit tight budget must be SMALLER than the adaptive default would have chosen,
	// proving the facade did not silently override the model's cap.
	adaptive := RecommendBudgetForForecast(q.forecast(), DefaultBudgetBounds()).Tokens
	if !(tight.Tokens < adaptive) {
		t.Fatalf("fixture invalid: tight budget %d must be below adaptive default %d", tight.Tokens, adaptive)
	}
}

// TestPlanQueryDeterministic: the same (query, spans) must yield a byte-identical PlanView
// across repeated runs — replanning the same turn twice cannot drift (no randomness, no wall
// clock). Covers both the explicit-budget and the adaptive-default path.
func TestPlanQueryDeterministic(t *testing.T) {
	spans := querySpans()
	budget := Budget{Tokens: 50}

	for _, q := range []PlanQuery{
		{Intents: []string{"auth token rotation"}, Budget: &budget, Pins: []string{"sys"}},
		{Intents: []string{"auth token rotation"}, Horizon: 3}, // adaptive-default path
	} {
		first := q.Plan(spans, nil)
		for i := 0; i < 32; i++ {
			got := q.Plan(spans, nil)
			if !reflect.DeepEqual(got, first) {
				t.Fatalf("non-deterministic PlanView on run %d for query %+v", i, q)
			}
		}
	}
}

// TestPlanQueryAdversarialPinCannotLaunderPoison is the honesty witness from the issue: a
// MODEL-authored query that pins a SEALED span and forecasts straight at it must STILL get a
// faithful, poison-free, budget-bounded view. Letting the model drive the planner changes WHO
// states the prediction, never the trust gate: the sealed span is elided up front and can
// never ride into the resident set, even pinned.
func TestPlanQueryAdversarialPinCannotLaunderPoison(t *testing.T) {
	spans := querySpans()
	huge := Budget{Tokens: 100000} // plenty of room — the only thing keeping poison out is the gate

	q := PlanQuery{
		Intents: []string{"auth token rotation quarantined"}, // aimed at the poison span's descriptor
		Pins:    []string{"poison"},                          // a model-authored pin on the sealed span
		Budget:  &huge,
	}
	view := q.Plan(spans, nil)

	// The sealed span must NOT be resident, despite the pin and the targeted forecast.
	if selectedHas(view, "poison") {
		t.Fatalf("sealed 'poison' span was laundered into the resident view via a model pin: %v", viewSelectedIDs(view))
	}
	// It must instead be elided with the sealed reason — quarantined, not destroyed.
	if !elidedWithReason(view, "poison", ElideSealed) {
		t.Errorf("sealed span should be elided as %q, elided=%+v", ElideSealed, view.Elided)
	}
	// The view stays faithful (partitioned + recoverable) even under the adversarial query.
	if !view.Faithful {
		t.Errorf("adversarial query must still yield a faithful view, got Faithful=false")
	}
}

// TestPlanQueryAdversarialBudgetCannotOverflow: a model-authored query cannot exceed the
// budget. Pinning more than the cap allows leaves ONLY the pins resident (correctness over
// thrift) and sets OverBudget — nothing non-pinned is smuggled past the cap.
func TestPlanQueryAdversarialBudgetCannotOverflow(t *testing.T) {
	spans := querySpans()
	// 'sys' and 'hit' are 40 bytes => ceil(40/4)=10 tokens each. A budget of 12 fits one pin
	// but not both; pinning both forces an over-budget, pins-only view.
	tiny := Budget{Tokens: 12}

	q := PlanQuery{
		Intents: []string{"auth token rotation"},
		Pins:    []string{"sys", "hit"},
		Budget:  &tiny,
	}
	view := q.Plan(spans, nil)

	if !view.Plan.OverBudget {
		t.Errorf("pinning past the cap should set OverBudget, plan=%+v", view.Plan)
	}
	// Only the (live) pins are resident; the non-pinned 'miss' must be elided over-budget.
	if selectedHas(view, "miss") {
		t.Errorf("a non-pinned span rode past the overrun budget: %v", viewSelectedIDs(view))
	}
	if !elidedWithReason(view, "miss", ElideOverBudget) {
		t.Errorf("non-pinned span should be elided %q, elided=%+v", ElideOverBudget, view.Elided)
	}
	// The view is still faithful — an over-budget plan elides the rest, it never destroys it.
	if !view.Faithful {
		t.Errorf("over-budget plan must still be faithful")
	}
}

// --- test helpers -------------------------------------------------------------------------

func selectedHas(v PlanView, id string) bool {
	for _, s := range v.Selected {
		if s.ID == id {
			return true
		}
	}
	return false
}

func viewSelectedIDs(v PlanView) []string {
	out := make([]string, 0, len(v.Selected))
	for _, s := range v.Selected {
		out = append(out, s.ID)
	}
	return out
}

func elidedWithReason(v PlanView, id, reason string) bool {
	for _, e := range v.Elided {
		if e.ID == id && e.Reason == reason {
			return true
		}
	}
	return false
}
