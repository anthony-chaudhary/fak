package session

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1s
func TestOutputBudgetOverspendStaysFinite(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name                 string
		before, charge, want int
	}{
		{"overspend", 5, 6, 0},
		{"exact", 5, 5, 0},
		{"ordinary", 5, 4, 1},
		{"already_zero", 0, 1, 0},
		{"maximum_charge", 5, maxInt, 0},
		{"unbounded", Unbounded, 6, Unbounded},
		{"negative_unbounded", -5, 6, -5},
		{"zero_charge", 5, 0, 5},
		{"negative_charge", 5, -1, 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			table := NewTable()
			const trace = "finite-output-budget"
			table.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: tc.before, ContextTokensLeft: 100})
			before := table.Get(trace)
			state := table.Debit(trace, tc.charge)
			if state.Budget.TokensLeft != tc.want {
				t.Errorf("finite output budget crossed sentinel: before=%d charge=%d left=%d want=%d", tc.before, tc.charge, state.Budget.TokensLeft, tc.want)
			}
			expectedBudget := before.Budget
			expectedBudget.TokensLeft = tc.want
			if state.Budget != expectedBudget || state.Run != Running || state.Reason != before.Reason || state.ContinuationID != before.ContinuationID {
				t.Errorf("debit changed boundary timing or unrelated state: before=%+v after=%+v", before, state)
			}
			if tc.charge <= 0 {
				if !reflect.DeepEqual(state, before) {
					t.Errorf("nonpositive output-only usage changed state: before=%+v after=%+v", before, state)
				}
			} else if cost, ok := state.Cost.at(0); !ok || cost != (TurnCost{OutputTokens: tc.charge}) {
				t.Errorf("reported cost was clamped with balance: cost=%+v present=%v charge=%d", cost, ok, tc.charge)
			}
			verdict := table.Decide(trace)
			if tc.want == 0 {
				if verdict.Proceed || !verdict.Stop || verdict.Reason != ReasonBudgetTokens || verdict.State.Budget.TokensLeft != 0 {
					t.Errorf("exhausted finite budget admitted next boundary: %+v", verdict)
				}
			} else if !verdict.Proceed || verdict.Stop || verdict.State.Budget.TokensLeft != tc.want {
				t.Errorf("ordinary or unbounded next boundary changed: %+v", verdict)
			}
		})
	}
}
