package metrics

import "testing"

// TestFoldBudgetSpendVsTarget is the #2091 done-condition witness at the fold
// layer: a real per-task spend snapshot + a soft target folds into a readout
// that reports spend, remaining, percent-used, and the per-category breakdown.
func TestFoldBudgetSpendVsTarget(t *testing.T) {
	spend := BudgetSpend{
		InputTokens:  30000,
		OutputTokens: 10000,
		CachedTokens: 25000,
		Turns:        18,
		ToolCalls:    54,
	}
	r := FoldBudget("sess-abc", spend, BudgetTarget{Tokens: 100000, Turns: 40})

	if r.Schema != BudgetReadoutSchema {
		t.Errorf("schema = %q, want %q", r.Schema, BudgetReadoutSchema)
	}
	if r.Session != "sess-abc" {
		t.Errorf("session = %q, want sess-abc", r.Session)
	}
	// Token spend is input+output; cached is a saving, never folded into spend.
	if r.Tokens.Spent != 40000 {
		t.Errorf("token spend = %d, want 40000 (input+output, cached excluded)", r.Tokens.Spent)
	}
	if r.CachedTokens != 25000 {
		t.Errorf("cached tokens = %d, want 25000", r.CachedTokens)
	}
	if !r.Tokens.HasTarget || r.Tokens.Remaining != 60000 {
		t.Errorf("token remaining = %d (has_target=%v), want 60000 remaining", r.Tokens.Remaining, r.Tokens.HasTarget)
	}
	if r.Tokens.PercentUsed != 40 || r.Tokens.Over {
		t.Errorf("token percent_used = %.1f over=%v, want 40.0 not-over", r.Tokens.PercentUsed, r.Tokens.Over)
	}
	if r.Turns.Spent != 18 || r.Turns.Remaining != 22 {
		t.Errorf("turns spent/remaining = %d/%d, want 18/22", r.Turns.Spent, r.Turns.Remaining)
	}
}

// TestFoldBudgetCategoriesAreLabeled pins the coarse breakdown and its provenance
// discipline: model token categories are OBSERVED (provider-relayed), fak's own
// counts are WITNESSED, and the gate passes only a fully-labeled readout.
func TestFoldBudgetCategoriesAreLabeled(t *testing.T) {
	r := FoldBudget("s", BudgetSpend{InputTokens: 1, OutputTokens: 2, CachedTokens: 3, Turns: 4, ToolCalls: 5}, BudgetTarget{})

	want := map[string]struct {
		spent uint64
		prov  SpendProvenance
	}{
		"model_input_tokens":   {1, SpendObserved},
		"model_output_tokens":  {2, SpendObserved},
		"cached_prompt_tokens": {3, SpendObserved},
		"served_turns":         {4, SpendWitnessed},
		"tool_calls":           {5, SpendWitnessed},
	}
	got := map[string]bool{}
	for _, c := range r.Categories {
		w, ok := want[c.Name]
		if !ok {
			t.Errorf("unexpected category %q", c.Name)
			continue
		}
		got[c.Name] = true
		if c.Spent != w.spent {
			t.Errorf("category %q spent = %d, want %d", c.Name, c.Spent, w.spent)
		}
		if c.Provenance != w.prov {
			t.Errorf("category %q provenance = %q, want %q", c.Name, c.Provenance, w.prov)
		}
		if c.Unit == "" {
			t.Errorf("category %q has empty unit", c.Name)
		}
	}
	for name := range want {
		if !got[name] {
			t.Errorf("missing category %q", name)
		}
	}
	if defects := GateBudgetLabeled(r); len(defects) > 0 {
		t.Errorf("GateBudgetLabeled failed a fully-labeled readout: %v", defects)
	}
}

// TestFoldBudgetNoTargetAndOver covers the two edge axes: no soft target (report
// raw spend, no invented percentage) and spend past a set target (Over, negative
// remaining), so an agent self-pacing off this readout is never misled.
func TestFoldBudgetNoTargetAndOver(t *testing.T) {
	noTarget := FoldBudget("s", BudgetSpend{InputTokens: 500, OutputTokens: 500}, BudgetTarget{})
	if noTarget.Tokens.HasTarget || noTarget.Tokens.PercentUsed != 0 || noTarget.Tokens.Remaining != 0 {
		t.Errorf("no-target axis invented target data: %+v", noTarget.Tokens)
	}
	if noTarget.Tokens.Spent != 1000 {
		t.Errorf("no-target spend = %d, want 1000", noTarget.Tokens.Spent)
	}

	over := FoldBudget("s", BudgetSpend{InputTokens: 90000, OutputTokens: 30000}, BudgetTarget{Tokens: 100000})
	if !over.Tokens.Over || over.Tokens.Remaining != -20000 {
		t.Errorf("over-budget axis = %+v, want Over with remaining -20000", over.Tokens)
	}
	if over.Tokens.PercentUsed != 120 {
		t.Errorf("over-budget percent_used = %.1f, want 120.0", over.Tokens.PercentUsed)
	}
}

// TestGateBudgetCatchesUnlabeled proves the gate refuses a category with a
// provenance outside the closed set — the honesty fence, not decoration.
func TestGateBudgetCatchesUnlabeled(t *testing.T) {
	r := BudgetReadout{Categories: []BudgetCategory{
		{Name: "bad", Spent: 1, Unit: "tokens", Provenance: SpendProvenance("GUESSED")},
		{Name: "no_unit", Spent: 1, Provenance: SpendWitnessed},
	}}
	if defects := GateBudgetLabeled(r); len(defects) != 2 {
		t.Errorf("GateBudgetLabeled defects = %d (%v), want 2", len(defects), defects)
	}
}
