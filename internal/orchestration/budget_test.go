package orchestration

import (
	"errors"
	"reflect"
	"testing"
)

func TestHierarchicalBudgetFoldConservesRoot(t *testing.T) {
	t.Run("concurrent siblings cannot each inherit the root cap", func(t *testing.T) {
		_, err := FoldBudgetEvents(Budget{MaxWorkers: 4, MaxTokens: 100}, []BudgetEvent{
			{Kind: BudgetReserve, NodeID: "child-a", ParentID: RootBudgetNodeID, Workers: 1, Tokens: 80},
			{Kind: BudgetReserve, NodeID: "child-b", ParentID: RootBudgetNodeID, Workers: 1, Tokens: 80},
		})
		assertBudgetRefusal(t, err, BudgetRootExhausted)
	})

	t.Run("a grandchild is conserved and terminal refunds are idempotent", func(t *testing.T) {
		events := []BudgetEvent{
			{Kind: BudgetReserve, NodeID: "parent", ParentID: RootBudgetNodeID, Workers: 2, Tokens: 80},
			{Kind: BudgetConsume, NodeID: "parent", Workers: 1, Tokens: 10},
			{Kind: BudgetReserve, NodeID: "grandchild", ParentID: "parent", Workers: 1, Tokens: 30},
			{Kind: BudgetConsume, NodeID: "grandchild", Workers: 1, Tokens: 20},
			{Kind: BudgetClose, NodeID: "grandchild"},
			{Kind: BudgetCancel, NodeID: "grandchild"},
			{Kind: BudgetReserve, NodeID: "sibling", ParentID: RootBudgetNodeID, Workers: 1, Tokens: 20},
			{Kind: BudgetCancel, NodeID: "sibling"},
			{Kind: BudgetCancel, NodeID: "sibling"},
			{Kind: BudgetClose, NodeID: "parent"},
			{Kind: BudgetRelease, NodeID: "parent"},
		}

		got, err := FoldBudgetEvents(Budget{MaxWorkers: 4, MaxTokens: 100}, events)
		if err != nil {
			t.Fatalf("FoldBudgetEvents: %v", err)
		}
		wantRoot := BudgetTotals{
			Limit:     Budget{MaxWorkers: 4, MaxTokens: 100},
			Remaining: Budget{MaxWorkers: 4, MaxTokens: 70},
			Consumed:  Budget{MaxTokens: 30},
		}
		if got.Root != wantRoot {
			t.Fatalf("root totals = %+v, want %+v", got.Root, wantRoot)
		}
		if got.Nodes["grandchild"].ParentID != "parent" || got.Nodes["grandchild"].State != BudgetNodeClosed {
			t.Fatalf("grandchild state = %+v", got.Nodes["grandchild"])
		}
		if got.Nodes["grandchild"].Consumed != (Budget{MaxTokens: 20}) || got.Nodes["grandchild"].Refunded != (Budget{MaxWorkers: 1, MaxTokens: 10}) {
			t.Fatalf("grandchild accounting = %+v", got.Nodes["grandchild"])
		}
		if got.Nodes["sibling"].State != BudgetNodeCancelled || got.Nodes["sibling"].Refunded != (Budget{MaxWorkers: 1, MaxTokens: 20}) {
			t.Fatalf("cancelled sibling accounting = %+v", got.Nodes["sibling"])
		}
		if got.Nodes["parent"].State != BudgetNodeClosed || got.Nodes["parent"].Consumed != (Budget{MaxTokens: 30}) || got.Nodes["parent"].Refunded != (Budget{MaxWorkers: 2, MaxTokens: 50}) {
			t.Fatalf("parent accounting = %+v", got.Nodes["parent"])
		}

		replayed, err := FoldBudgetEvents(Budget{MaxWorkers: 4, MaxTokens: 100}, events)
		if err != nil {
			t.Fatalf("replay: %v", err)
		}
		if !reflect.DeepEqual(got, replayed) {
			t.Fatalf("replay differs:\nfirst  %+v\nsecond %+v", got, replayed)
		}
	})

	t.Run("parent and event bounds fail with closed reasons", func(t *testing.T) {
		_, err := FoldBudgetEvents(Budget{MaxWorkers: 4, MaxTokens: 200}, []BudgetEvent{
			{Kind: BudgetReserve, NodeID: "parent", ParentID: RootBudgetNodeID, Workers: 2, Tokens: 80},
			{Kind: BudgetReserve, NodeID: "child-a", ParentID: "parent", Workers: 1, Tokens: 50},
			{Kind: BudgetReserve, NodeID: "child-b", ParentID: "parent", Workers: 1, Tokens: 40},
		})
		assertBudgetRefusal(t, err, BudgetParentExhausted)

		_, err = FoldBudgetEvents(Budget{MaxWorkers: 1, MaxTokens: 100}, []BudgetEvent{
			{Kind: BudgetReserve, NodeID: "negative", ParentID: RootBudgetNodeID, Workers: 1, Tokens: -1},
		})
		assertBudgetRefusal(t, err, BudgetNegativeAmount)
	})
}

func assertBudgetRefusal(t *testing.T, err error, want BudgetRefusalReason) {
	t.Helper()
	var refusal *BudgetRefusal
	if !errors.As(err, &refusal) {
		t.Fatalf("error = %v, want BudgetRefusal(%s)", err, want)
	}
	if refusal.Reason != want {
		t.Fatalf("refusal reason = %s, want %s (%v)", refusal.Reason, want, err)
	}
}
