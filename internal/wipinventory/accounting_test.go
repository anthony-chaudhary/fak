package wipinventory

import "testing"

func TestAccountHistoryReceiptDeltas(t *testing.T) {
	one, two, three := id(1), id(2), id(3)
	history := graphHistory(
		tr(TransitionCreate, nil, []WIPUnitID{one}),
		tr(TransitionCreate, nil, []WIPUnitID{two}),
		tr(TransitionCreate, nil, []WIPUnitID{three}),
		tr(TransitionSplit, []WIPUnitID{one}, []WIPUnitID{two, three}),
	)
	got := AccountHistory(history)
	if got.ActiveCount != 2 || len(got.Debt) != 0 {
		t.Fatalf("active=%d debt=%v, want 2 and none", got.ActiveCount, got.Debt)
	}
	wantDeltas := []int{1, 1, 1, -1}
	if len(got.Receipts) != len(wantDeltas) {
		t.Fatalf("receipts=%d, want %d", len(got.Receipts), len(wantDeltas))
	}
	for index, want := range wantDeltas {
		if got.Receipts[index].Delta != want {
			t.Fatalf("receipt %d delta=%d, want %d", index, got.Receipts[index].Delta, want)
		}
		if got.Receipts[index].After-got.Receipts[index].Before != got.Receipts[index].Delta {
			t.Fatalf("receipt %d has unattributable delta: %+v", index, got.Receipts[index])
		}
	}
	if len(got.Units) != 3 || got.Units[0].ID != one || got.Units[0].State != AccountedUnitSuperseded {
		t.Fatalf("history not retained deterministically: %+v", got.Units)
	}
}

func TestAccountHistoryDoesNotPartiallyApplyDebt(t *testing.T) {
	one := id(1)
	got := AccountHistory(graphHistory(
		tr(TransitionCreate, nil, []WIPUnitID{one}),
		tr(TransitionSplit, []WIPUnitID{one}, nil),
	))
	if got.ActiveCount != 1 {
		t.Fatalf("active=%d, want 1 after rejected incomplete transition", got.ActiveCount)
	}
	if len(got.Debt) != 1 || got.Debt[0].Kind != AccountingDebtIncompleteTransition {
		t.Fatalf("debt=%+v, want incomplete transition", got.Debt)
	}
	if len(got.Receipts) != 1 {
		t.Fatalf("receipts=%d, want only accepted create", len(got.Receipts))
	}
}
