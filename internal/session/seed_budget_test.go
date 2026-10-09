package session

import "testing"

func TestSeedBudgetNeverResetsADebitedSession(t *testing.T) {
	tbl := NewTable()
	b := Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 1000}
	if _, seeded := tbl.SeedBudget("unkeyed-a", b); !seeded {
		t.Fatal("first SeedBudget on an unseen trace did not seed")
	}
	tbl.DebitUsage("unkeyed-a", Usage{ContextTokens: 400})
	left := tbl.Get("unkeyed-a").Budget.ContextTokensLeft
	if _, seeded := tbl.SeedBudget("unkeyed-a", b); seeded {
		t.Fatal("SeedBudget re-seeded a trace the table already holds")
	}
	if got := tbl.Get("unkeyed-a").Budget.ContextTokensLeft; got != left {
		t.Fatalf("ContextTokensLeft = %d after a second seed, want the debited %d", got, left)
	}
}
