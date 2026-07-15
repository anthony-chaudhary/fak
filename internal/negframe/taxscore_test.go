package negframe

import "testing"

func TestNegationTaxScorecardFallsAfterReframe(t *testing.T) {
	dirty := BuildNegationTax([]HotPathString{{Name: "hot", Tier: TierPerTurn, Text: "Never delete the audit."}})
	clean := BuildNegationTax([]HotPathString{{Name: "hot", Tier: TierPerTurn, Text: "Preserve the audit."}})
	dirtyDebt := dirty.Corpus[NegationTaxDebtKey].(int)
	cleanDebt := clean.Corpus[NegationTaxDebtKey].(int)
	if dirtyDebt <= cleanDebt || cleanDebt != 0 {
		t.Fatalf("dirty=%d clean=%d", dirtyDebt, cleanDebt)
	}
	if dirty.Schema != NegationTaxSchema {
		t.Fatalf("payload=%+v", dirty)
	}
	if dirty.Corpus["weighted_debt"].(int) <= clean.Corpus["weighted_debt"].(int) {
		t.Fatalf("weighted debt did not fall")
	}
}
