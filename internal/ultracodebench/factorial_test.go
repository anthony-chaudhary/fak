package ultracodebench

import "testing"

func TestEvaluateFactorialCampaignSeparatesEffects(t *testing.T) {
	c := factorialFixture()
	r, err := EvaluateFactorialCampaign(c, []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Widths) != 2 || r.Widths[0].Verdict != "GAIN" || r.Widths[0].GenericPrefix.Estimate != -10 || r.Widths[0].MicroScope.Estimate != -30 || r.Widths[0].Combined.Estimate != -50 || r.Widths[0].Interaction.Estimate != -10 {
		t.Fatalf("report = %+v", r)
	}
	if r.Widths[0].GenericPrefix.StdError == 0 || r.HillClimb.ChosenWidth != 2 || r.HillClimb.StopWidth != 0 {
		t.Fatalf("uncertainty/climb = %+v", r)
	}
}

func TestEvaluateFactorialCampaignAbstainsOnUnequalOutcomeAndStops(t *testing.T) {
	c := factorialFixture()
	for i := range c.Cells {
		if c.Cells[i].Width == 2 && c.Cells[i].Treatment == "D" {
			c.Cells[i].Replicates[0].Accepted = false
		}
	}
	r, err := EvaluateFactorialCampaign(c, []int{1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	if r.Widths[1].Verdict != "ABSTAIN" || r.HillClimb.ChosenWidth != 1 || r.HillClimb.StopWidth != 2 {
		t.Fatalf("report = %+v", r)
	}
}

func factorialFixture() FactorialCampaign {
	c := FactorialCampaign{Schema: FactorialCampaignSchema, EvidenceKind: "synthetic_fixture", SourceArtifact: "fixture", Model: "small", Runtime: "runtime", Tokenizer: "tokenizer", TaskDigest: "task", OutcomeDigest: "accepted", Metric: "work", OrderPolicy: "alternating"}
	for _, width := range []int{1, 2, 4} {
		for order, x := range []struct {
			name, context, cache string
			work                 float64
		}{{"A", "full", "cold", 100}, {"D", "scoped", "warm", 50}, {"C", "scoped", "cold", 70}, {"B", "full", "warm", 90}} {
			cell := FactorialCell{Width: width, Treatment: x.name, Context: x.context, Cache: x.cache, Order: order + 1}
			for _, delta := range []float64{-1, 1} {
				cell.Replicates = append(cell.Replicates, FactorialReplicate{Accepted: true, OutcomeDigest: "accepted", Work: x.work + delta, InputTokens: 100, CachedTokens: map[bool]int64{true: 50}[x.cache == "warm"], ResetReceipt: "unloaded", WarmupReceipt: map[bool]string{true: "warmed"}[x.cache == "warm"]})
			}
			c.Cells = append(c.Cells, cell)
		}
	}
	return c
}
