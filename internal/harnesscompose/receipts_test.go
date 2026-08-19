package harnesscompose

import (
	"errors"
	"reflect"
	"testing"
)

func TestComposeReceiptsIsDeterministicAcrossInputOrder(t *testing.T) {
	receipts := []ProfileReceipt{
		{Kind: ReceiptProof, ID: "completion", Verified: true, EvidenceRef: "test:proof", Asset: Asset{Kind: "workflow", ID: "completion", Value: "build-and-test"}},
		{Kind: ReceiptTool, ID: "shell", Verified: true, EvidenceRef: "policy:tools", Asset: Asset{Kind: "tool", ID: "shell", Ref: "workspace-only"}},
		{Kind: ReceiptArchitecture, ID: "runtime", Verified: true, EvidenceRef: "artifact:architecture", Asset: Asset{Kind: "instruction", ID: "runtime", Value: "fak-native"}},
		{Kind: ReceiptConstraints, ID: "authority", Verified: true, EvidenceRef: "policy:authority", Asset: Asset{Kind: "policy", ID: "authority", Grants: []string{"repo.read"}, Denies: []string{"network"}}},
		{Kind: ReceiptModel, ID: "planner", Verified: true, EvidenceRef: "route:planner", Asset: Asset{Kind: "route", ID: "planner", Ref: "balanced"}},
	}
	forward, err := ComposeReceipts(receipts)
	if err != nil {
		t.Fatalf("ComposeReceipts(forward): %v", err)
	}
	reversed := append([]ProfileReceipt(nil), receipts...)
	for i, j := 0, len(reversed)-1; i < j; i, j = i+1, j-1 {
		reversed[i], reversed[j] = reversed[j], reversed[i]
	}
	backward, err := ComposeReceipts(reversed)
	if err != nil {
		t.Fatalf("ComposeReceipts(reversed): %v", err)
	}
	if !reflect.DeepEqual(forward, backward) {
		t.Fatalf("input permutation changed plan:\nforward=%#v\nbackward=%#v", forward, backward)
	}
	if len(forward.Assets) != 5 {
		t.Fatalf("assets = %d, want five typed receipt classes", len(forward.Assets))
	}
}

func TestComposeReceiptsRefusesUnverifiedAndDuplicateInputs(t *testing.T) {
	valid := ProfileReceipt{Kind: ReceiptTool, ID: "shell", Verified: true, EvidenceRef: "test:1", Asset: Asset{Kind: "tool", ID: "shell", Ref: "workspace-only"}}
	tests := []struct {
		name     string
		receipts []ProfileReceipt
	}{
		{name: "unverified", receipts: []ProfileReceipt{{Kind: ReceiptTool, ID: "shell", EvidenceRef: "test:1", Asset: valid.Asset}}},
		{name: "missing evidence", receipts: []ProfileReceipt{{Kind: ReceiptTool, ID: "shell", Verified: true, Asset: valid.Asset}}},
		{name: "duplicate", receipts: []ProfileReceipt{valid, valid}},
		{name: "unknown kind", receipts: []ProfileReceipt{{Kind: "transcript", ID: "raw", Verified: true, EvidenceRef: "test:1", Asset: valid.Asset}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := ComposeReceipts(tt.receipts); !errors.Is(err, ErrInvalidReceipt) {
				t.Fatalf("error = %v, want ErrInvalidReceipt", err)
			}
		})
	}
}
