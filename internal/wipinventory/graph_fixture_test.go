package wipinventory

import "testing"

func graphHistory(transitions ...Transition) History {
	return History{Schema: WIPUnitSchema, Transitions: transitions}
}

func TestLogicalWIPGraphFixtures(t *testing.T) {
	a, b, c, d, e := id(1), id(2), id(3), id(4), id(5)
	cases := []struct {
		name       string
		history    History
		wantActive int
		wantDebt   []AccountingDebtKind
	}{
		{
			name: "split 1 to 2",
			history: graphHistory(
				tr(TransitionCreate, nil, []WIPUnitID{a}),
				tr(TransitionCreate, nil, []WIPUnitID{b}),
				tr(TransitionCreate, nil, []WIPUnitID{c}),
				tr(TransitionSplit, []WIPUnitID{a}, []WIPUnitID{b, c}),
			),
			wantActive: 2,
		},
		{
			name: "merge 2 to 1",
			history: graphHistory(
				tr(TransitionCreate, nil, []WIPUnitID{a}),
				tr(TransitionCreate, nil, []WIPUnitID{b}),
				tr(TransitionCreate, nil, []WIPUnitID{c}),
				tr(TransitionMerge, []WIPUnitID{a, b}, []WIPUnitID{c}),
			),
			wantActive: 1,
		},
		{
			name: "chained supersession",
			history: graphHistory(
				tr(TransitionCreate, nil, []WIPUnitID{a}), tr(TransitionCreate, nil, []WIPUnitID{b}),
				tr(TransitionCreate, nil, []WIPUnitID{c}), tr(TransitionSupersede, []WIPUnitID{a}, []WIPUnitID{b}),
				tr(TransitionSupersede, []WIPUnitID{b}, []WIPUnitID{c}),
			),
			wantActive: 1,
		},
		{
			name: "park resume",
			history: graphHistory(
				tr(TransitionCreate, nil, []WIPUnitID{a}), tr(TransitionPark, []WIPUnitID{a}, []WIPUnitID{a}),
				tr(TransitionRecover, []WIPUnitID{a}, []WIPUnitID{a}),
			),
			wantActive: 1,
		},
		{
			name: "abandon and land",
			history: graphHistory(
				tr(TransitionCreate, nil, []WIPUnitID{a}), tr(TransitionCreate, nil, []WIPUnitID{b}),
				tr(TransitionAbandon, []WIPUnitID{a}, []WIPUnitID{a}), tr(TransitionLand, []WIPUnitID{b}, []WIPUnitID{b}),
			),
			wantActive: 0,
		},
		{
			name: "handoff preserves count",
			history: graphHistory(
				tr(TransitionCreate, nil, []WIPUnitID{a}), tr(TransitionHandoff, []WIPUnitID{a}, []WIPUnitID{a}),
			),
			wantActive: 1,
		},
		{
			name: "duplicate successors",
			history: graphHistory(
				tr(TransitionCreate, nil, []WIPUnitID{a}), tr(TransitionCreate, nil, []WIPUnitID{b}),
				tr(TransitionSplit, []WIPUnitID{a}, []WIPUnitID{b, b}),
			),
			wantActive: 2,
			wantDebt:   []AccountingDebtKind{AccountingDebtDuplicateSuccessor},
		},
		{
			name: "cycle",
			history: graphHistory(
				tr(TransitionCreate, nil, []WIPUnitID{a}), tr(TransitionCreate, nil, []WIPUnitID{b}), tr(TransitionCreate, nil, []WIPUnitID{c}),
				tr(TransitionSplit, []WIPUnitID{a}, []WIPUnitID{b, c}), tr(TransitionMerge, []WIPUnitID{b, c}, []WIPUnitID{a}),
			),
			wantActive: 2,
			wantDebt:   []AccountingDebtKind{AccountingDebtCycle},
		},
		{
			name: "incomplete transition",
			history: graphHistory(
				tr(TransitionCreate, nil, []WIPUnitID{a}), tr(TransitionSupersede, []WIPUnitID{a}, nil),
			),
			wantActive: 1,
			wantDebt:   []AccountingDebtKind{AccountingDebtIncompleteTransition},
		},
		{
			name: "partial merge retirement",
			history: graphHistory(
				tr(TransitionCreate, nil, []WIPUnitID{a}), tr(TransitionCreate, nil, []WIPUnitID{b}), tr(TransitionCreate, nil, []WIPUnitID{c}),
				tr(TransitionLand, []WIPUnitID{a}, []WIPUnitID{a}), tr(TransitionMerge, []WIPUnitID{a, b}, []WIPUnitID{c}),
			),
			wantActive: 2,
			wantDebt:   []AccountingDebtKind{AccountingDebtPartialMergeRetirement},
		},
		{
			name: "ambiguous multi parent ownership",
			history: graphHistory(
				tr(TransitionCreate, nil, []WIPUnitID{a}), tr(TransitionCreate, nil, []WIPUnitID{b}),
				tr(TransitionCreate, nil, []WIPUnitID{c}), tr(TransitionCreate, nil, []WIPUnitID{d}), tr(TransitionCreate, nil, []WIPUnitID{e}),
				tr(TransitionSplit, []WIPUnitID{a}, []WIPUnitID{c, d}), tr(TransitionMerge, []WIPUnitID{b, e}, []WIPUnitID{c}),
			),
			wantActive: 4,
			wantDebt:   []AccountingDebtKind{AccountingDebtAmbiguousMultiParentOwner},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := AccountHistory(tc.history)
			if got.ActiveCount != tc.wantActive {
				t.Errorf("active=%d, want %d", got.ActiveCount, tc.wantActive)
			}
			if len(got.Debt) != len(tc.wantDebt) {
				t.Fatalf("debt=%+v, want kinds %v", got.Debt, tc.wantDebt)
			}
			for index, want := range tc.wantDebt {
				if got.Debt[index].Kind != want {
					t.Errorf("debt[%d].kind=%q, want %q", index, got.Debt[index].Kind, want)
				}
			}
		})
	}
}
