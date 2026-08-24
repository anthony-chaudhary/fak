package harnesscompose

import (
	"errors"
	"testing"
)

func TestComposeReceiptsDispatchesBoundedArbitrationWithProvenance(t *testing.T) {
	receipts := []ProfileReceipt{
		{Kind: ReceiptModel, ID: "planner", Verified: true, EvidenceRef: "child:a", Asset: Asset{Kind: "route", ID: "planner", Ref: "fast"}},
		{Kind: ReceiptModel, ID: "planner", Verified: true, EvidenceRef: "child:b", Asset: Asset{Kind: "route", ID: "planner", Ref: "deep"}},
		{Kind: ReceiptTool, ID: "shell", Verified: true, EvidenceRef: "child:c", Asset: Asset{Kind: "tool", ID: "shell", Ref: "workspace-only"}},
	}
	calls := 0
	result, conflicts, err := ComposeReceiptsWithArbitration(receipts, 1, func(request ArbitrationRequest) (ProfileReceipt, error) {
		calls++
		if request.Ordinal != 1 || request.Limit != 1 {
			t.Fatalf("request bounds = %d/%d, want 1/1", request.Ordinal, request.Limit)
		}
		if got := request.Conflict.Candidates; len(got) != 2 || got[0].EvidenceRef != "child:a" || got[1].EvidenceRef != "child:b" {
			t.Fatalf("conflict provenance = %#v", got)
		}
		return ProfileReceipt{Kind: ReceiptModel, ID: "planner", Verified: true, EvidenceRef: "arbiter:decision-1", Asset: Asset{Kind: "route", ID: "planner", Ref: "deep"}}, nil
	})
	if err != nil {
		t.Fatalf("ComposeReceiptsWithArbitration: %v", err)
	}
	if calls != 1 || len(conflicts) != 1 {
		t.Fatalf("calls/conflicts = %d/%d, want 1/1", calls, len(conflicts))
	}
	if len(result.Assets) != 2 {
		t.Fatalf("assets = %d, want 2", len(result.Assets))
	}
}

func TestComposeReceiptsRefusesConflictWhenArbitrationBoundIsZero(t *testing.T) {
	receipts := []ProfileReceipt{
		{Kind: ReceiptModel, ID: "planner", Verified: true, EvidenceRef: "child:a", Asset: Asset{Kind: "route", ID: "planner", Ref: "fast"}},
		{Kind: ReceiptModel, ID: "planner", Verified: true, EvidenceRef: "child:b", Asset: Asset{Kind: "route", ID: "planner", Ref: "deep"}},
	}
	called := false
	_, conflicts, err := ComposeReceiptsWithArbitration(receipts, 0, func(ArbitrationRequest) (ProfileReceipt, error) {
		called = true
		return ProfileReceipt{}, nil
	})
	if !errors.Is(err, ErrArbitrationRequired) {
		t.Fatalf("error = %v, want ErrArbitrationRequired", err)
	}
	if called {
		t.Fatal("arbiter called despite zero request bound")
	}
	if len(conflicts) != 1 || len(conflicts[0].Candidates) != 2 {
		t.Fatalf("conflicts = %#v, want surfaced provenance", conflicts)
	}
}
