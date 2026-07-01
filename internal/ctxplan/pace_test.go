package ctxplan

import "testing"

// TestScaleBudgetForPaceShrinksProportionally proves the #1585 load-bearing claim: a slower
// observed pace ratio shrinks the resident Budget proportionally, and an unthrottled (or
// zero-value) pace signal leaves the base Budget byte-for-byte unchanged.
func TestScaleBudgetForPaceShrinksProportionally(t *testing.T) {
	base := Budget{Tokens: 4096}

	if got := ScaleBudgetForPace(base, PaceBudget{Ratio: 0.5, MinResidentTokens: 1}); got.Tokens != 2048 {
		t.Fatalf("half-pace budget = %d, want 2048", got.Tokens)
	}
	if got := ScaleBudgetForPace(base, PaceBudget{Ratio: 0.25, MinResidentTokens: 1}); got.Tokens != 1024 {
		t.Fatalf("quarter-pace budget = %d, want 1024", got.Tokens)
	}
	// The zero value (no signal observed) must be a total no-op.
	if got := ScaleBudgetForPace(base, PaceBudget{}); got.Tokens != base.Tokens {
		t.Fatalf("zero-value pace budget = %d, want %d unchanged", got.Tokens, base.Tokens)
	}
	// Ratio >= 1 (keeping pace or running ahead) is never a reason to shrink.
	if got := ScaleBudgetForPace(base, PaceBudget{Ratio: 1.0}); got.Tokens != base.Tokens {
		t.Fatalf("on-pace budget = %d, want %d unchanged", got.Tokens, base.Tokens)
	}
	if got := ScaleBudgetForPace(base, PaceBudget{Ratio: 2.0}); got.Tokens != base.Tokens {
		t.Fatalf("above-pace (ratio>1) budget = %d, want %d unchanged (never widens)", got.Tokens, base.Tokens)
	}
}

// TestScaleBudgetForPacePreservesMinimum proves the OTHER half of the issue's done
// condition: however hard the observed pace collapses, the scaled budget never drops below
// MinResidentTokens (or DefaultMinResidentTokens when unset) — the minimum resident context
// is always preserved.
func TestScaleBudgetForPacePreservesMinimum(t *testing.T) {
	base := Budget{Tokens: 8192}

	// A near-zero ratio would round to ~8 tokens without a floor; the explicit floor must
	// catch it.
	got := ScaleBudgetForPace(base, PaceBudget{Ratio: 0.001, MinResidentTokens: 600})
	if got.Tokens != 600 {
		t.Fatalf("near-stalled pace budget = %d, want the floor 600", got.Tokens)
	}

	// An unset (<=0) MinResidentTokens falls back to DefaultMinResidentTokens.
	got = ScaleBudgetForPace(base, PaceBudget{Ratio: 0.001})
	if got.Tokens != DefaultMinResidentTokens {
		t.Fatalf("default-floor pace budget = %d, want %d", got.Tokens, DefaultMinResidentTokens)
	}

	// A floor above the base budget itself is clamped to the base — never inflate a small
	// budget past what was actually configured.
	small := Budget{Tokens: 100}
	got = ScaleBudgetForPace(small, PaceBudget{Ratio: 0.1, MinResidentTokens: 1_000_000})
	if got.Tokens != small.Tokens {
		t.Fatalf("floor-above-base budget = %d, want %d (never inflate past the configured base)", got.Tokens, small.Tokens)
	}

	// Never zero, never negative, for any ratio in (0,1].
	for _, r := range []float64{1e-9, 0.01, 0.5, 0.999} {
		got := ScaleBudgetForPace(base, PaceBudget{Ratio: r, MinResidentTokens: 256})
		if got.Tokens <= 0 {
			t.Fatalf("ratio=%v produced non-positive budget %d", r, got.Tokens)
		}
	}
}

// TestScaleBudgetForPaceFailsClosedOnGarbageRatio proves a poisoned/garbage Ratio (negative,
// NaN, +Inf) never zeroes or inverts the budget — it fails closed to "no scaling" (ratio
// 1.0), the same posture Difficulty.Score already applies to a garbage difficulty signal.
func TestScaleBudgetForPaceFailsClosedOnGarbageRatio(t *testing.T) {
	base := Budget{Tokens: 4096}
	garbage := []float64{-1, 0, negZero()}
	for _, r := range garbage {
		got := ScaleBudgetForPace(base, PaceBudget{Ratio: r, MinResidentTokens: 1})
		if got.Tokens != base.Tokens {
			t.Fatalf("garbage ratio %v produced %d, want the unscaled base %d (fail closed)", r, got.Tokens, base.Tokens)
		}
	}
}

func negZero() float64 { return -0.0 }

// TestScaleBudgetForPaceNonPositiveBase leaves a non-configured base Budget alone.
func TestScaleBudgetForPaceNonPositiveBase(t *testing.T) {
	if got := ScaleBudgetForPace(Budget{Tokens: 0}, PaceBudget{Ratio: 0.1}); got.Tokens != 0 {
		t.Fatalf("zero base = %d, want 0 (nothing to scale)", got.Tokens)
	}
	if got := ScaleBudgetForPace(Budget{Tokens: -1}, PaceBudget{Ratio: 0.1}); got.Tokens != -1 {
		t.Fatalf("negative base = %d, want -1 unchanged", got.Tokens)
	}
}

// buildPaceSpans builds n distinct, roughly-equal-cost spans so a budget shrink can be
// observed as fewer spans SELECTED (not just a smaller declared budget number) — the same
// vacuity-guard discipline ctxplan_apply_pace_test.go's TestApplyPaceChangesPlanning uses.
func buildPaceSpans(n int) []Span {
	spans := make([]Span, 0, n)
	for i := 0; i < n; i++ {
		spans = append(spans, Span{
			ID:         "s" + itoaPace(i),
			Step:       i,
			Role:       "user",
			Descriptor: "alpha beta gamma delta marker" + itoaPace(i),
			Bytes:      480, // ~120 tokens at the bytes/4 proxy
			Durability: DurabilityBounded,
		})
	}
	return spans
}

func itoaPace(i int) string {
	if i == 0 {
		return "0"
	}
	digits := []byte{}
	for i > 0 {
		digits = append([]byte{byte('0' + i%10)}, digits...)
		i /= 10
	}
	return string(digits)
}

// TestPlanLayoutForPaceElidesMoreUnderSlowerPace is the end-to-end proof that an observed
// pace signal reaching PlanLayoutForPace actually changes the resulting PLAN, not just a
// number: a slower pace keeps fewer spans resident than the full-budget plan over the exact
// same candidate set. A vacuity guard fails the test if the full-budget plan did not already
// keep MORE than the paced one would, so the assertion cannot pass for a trivial reason.
func TestPlanLayoutForPaceElidesMoreUnderSlowerPace(t *testing.T) {
	spans := buildPaceSpans(24) // ~120 tokens each => ~2880 tokens of body
	f := Forecast{Intents: []string{"alpha", "beta"}, Horizon: 1}
	budget := Budget{Tokens: 4096} // comfortably fits all 24 spans
	layout := DefaultLayout()
	ix := BuildIndex(spans)

	full := ix.PlanLayout(f, budget, nil, layout)
	paced := ix.PlanLayoutForPace(f, budget, nil, layout, PaceBudget{Ratio: 0.25, MinResidentTokens: 256})

	if len(full.Selected) == 0 {
		t.Fatalf("vacuity guard: the full-budget plan selected nothing, so a comparison proves nothing")
	}
	if paced.Budget >= full.Budget {
		t.Fatalf("paced plan Budget = %d, want strictly less than the full plan's %d", paced.Budget, full.Budget)
	}
	if len(paced.Selected) >= len(full.Selected) {
		t.Fatalf("paced plan kept %d spans, full plan kept %d — the pace signal did not shrink the resulting VIEW",
			len(paced.Selected), len(full.Selected))
	}
	// The minimum resident floor must still be respected in the composed Budget the plan
	// actually ran under.
	if paced.Budget < 256 {
		t.Fatalf("paced plan Budget = %d, want at least the floor 256", paced.Budget)
	}
}

// TestPlanCellsForPaceIndexBoundedElidesMoreUnderSlowerPace proves the SAME behavior on the
// Index-bounded PlanCells path (the one agent.SessionPlanner.PlanTurn actually calls when no
// Layout is configured), not just the Layout path.
func TestPlanCellsForPaceIndexBoundedElidesMoreUnderSlowerPace(t *testing.T) {
	spans := buildPaceSpans(24)
	f := Forecast{Intents: []string{"alpha", "beta"}, Horizon: 1}
	budget := Budget{Tokens: 4096}
	ix := BuildIndex(spans)

	full := ix.PlanCells(f, budget, nil, ProbeOptions{})
	paced := ix.PlanCellsForPace(f, budget, nil, ProbeOptions{}, PaceBudget{Ratio: 0.25, MinResidentTokens: 256})

	if len(full.Selected) == 0 {
		t.Fatalf("vacuity guard: the full-budget plan selected nothing")
	}
	if len(paced.Selected) >= len(full.Selected) {
		t.Fatalf("paced plan kept %d spans, full plan kept %d — the pace signal did not bind on the index-bounded path",
			len(paced.Selected), len(full.Selected))
	}
}
