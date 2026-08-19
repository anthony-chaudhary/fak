package worktype

import "testing"

func TestFoldSpendKeepsUnknownInDenominator(t *testing.T) {
	r := FoldSpend([]SpendRow{{SessionID: "a", TraceID: "ta", PatternID: "wp.issue-to-patch", Tokens: 100, Cost: 80, Outcome: "accepted_witness"}, {SessionID: "b", PatternID: "wp.issue-to-patch", Tokens: 200, Cost: 150, Outcome: "no_change"}, {SessionID: "c", Tokens: 700, Cost: 700, Outcome: "failed"}}, SeedPatternIDs())
	if r.Coverage.RowCount != 3 || r.Coverage.ClassifiedRows != 2 || r.Coverage.TotalTokens != 1000 || r.Coverage.CoveredTokens != 300 {
		t.Fatalf("%+v", r.Coverage)
	}
	if r.HighestSpendClass != "unknown" {
		t.Fatalf("%+v", r)
	}
	for _, g := range r.Groups {
		if g.PatternID == "wp.issue-to-patch" && g.AcceptedRate != .5 {
			t.Fatalf("%+v", g)
		}
	}
}
