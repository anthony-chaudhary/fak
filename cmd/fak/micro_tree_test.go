package main

import (
	"strings"
	"testing"
)

func TestMicroTreeTextShowsRecursiveContextAndBudgetFlow(t *testing.T) {
	report := microTreeReport{Nodes: []microTreeNode{
		{ID: "root", Turns: microTreeBudget{Used: 1, Limit: 3}, Tokens: microTreeBudget{Used: 120, Limit: 1000}, CostMicroUSD: microTreeBudget{Used: 20, Limit: 200}, Capabilities: []string{"read", "search"}, ReceiptBytes: 96, VerifierState: "verified", RootTokens: 40},
		{ID: "child", ParentID: "root", Turns: microTreeBudget{Used: 2, Limit: 3}, Tokens: microTreeBudget{Used: 300, Limit: 600}, CostMicroUSD: microTreeBudget{Used: 60, Limit: 100}, Capabilities: []string{"read"}, ReceiptBytes: 128, VerifierState: "pending", RootTokens: 12},
		{ID: "grandchild", ParentID: "child", Turns: microTreeBudget{Used: 1, Limit: 1}, Tokens: microTreeBudget{Used: 80, Limit: 200}, CostMicroUSD: microTreeBudget{Used: 10, Limit: 40}, Capabilities: []string{"read"}, ReceiptBytes: 64, VerifierState: "verified", RootTokens: 8},
	}}

	got, err := microTreeText(report)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"MICRO TASK TREE (receipt-only; child transcripts excluded)",
		"root depth=0 turns=1/3 tokens=120/1000 cost_microusd=20/200 capabilities=read,search receipt_bytes=96 verifier=verified root_context_tokens=40",
		"  child depth=1 turns=2/3 tokens=300/600 cost_microusd=60/100 capabilities=read receipt_bytes=128 verifier=pending root_context_tokens=12",
		"    grandchild depth=2 turns=1/1 tokens=80/200 cost_microusd=10/40 capabilities=read receipt_bytes=64 verifier=verified root_context_tokens=8",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("render missing %q:\n%s", want, got)
		}
	}
}

func TestMicroTreeTextRefusesRecursiveEnvelopeExpansion(t *testing.T) {
	report := microTreeReport{Nodes: []microTreeNode{
		{ID: "root", Turns: microTreeBudget{Limit: 3}, Tokens: microTreeBudget{Limit: 100}, CostMicroUSD: microTreeBudget{Limit: 50}, Capabilities: []string{"read"}, VerifierState: "verified"},
		{ID: "child", ParentID: "root", Turns: microTreeBudget{Limit: 2}, Tokens: microTreeBudget{Limit: 101}, CostMicroUSD: microTreeBudget{Limit: 50}, Capabilities: []string{"read"}, VerifierState: "pending"},
	}}

	_, err := microTreeText(report)
	if err == nil || !strings.Contains(err.Error(), "REFUSE node \"child\" expands parent budget") {
		t.Fatalf("want budget expansion refusal, got %v", err)
	}
}
