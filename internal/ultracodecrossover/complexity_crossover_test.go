package ultracodecrossover

import "testing"

func TestComplexityCampaignStopsAfterTwoFailures(t *testing.T) {
	campaign := validComplexityCampaign()
	for i := range campaign.Rungs[3].Cells {
		if campaign.Rungs[3].Cells[i].Context == "scoped" {
			campaign.Rungs[3].Cells[i].Accepted = false
		}
	}
	for i := range campaign.Rungs[4].Cells {
		if campaign.Rungs[4].Cells[i].Context == "scoped" {
			campaign.Rungs[4].Cells[i].Accepted = false
		}
	}
	report, err := EvaluateComplexityCampaign(campaign)
	if err != nil {
		t.Fatal(err)
	}
	if report.LastEqualOutcomeRung != 3 || report.FirstFailureRung != 4 || report.StoppedAfterRung != 5 {
		t.Fatalf("unexpected crossover: %+v", report)
	}
	if len(report.Rungs) != 5 || report.Rungs[3].Verdict != "ABSTAIN" { //boundarylint:ignore CHANGE_DETECTOR_TEST the crossover fixture defines five ordered rungs and pins the fourth rung verdict
		t.Fatalf("abstentions were not preserved: %+v", report.Rungs)
	}
}

func TestComplexityCampaignRejectsIncompleteEnvelope(t *testing.T) {
	campaign := validComplexityCampaign()
	campaign.Rungs[0].Cells = campaign.Rungs[0].Cells[1:]
	if _, err := EvaluateComplexityCampaign(campaign); err == nil {
		t.Fatal("expected missing-cell error")
	}
}

func validComplexityCampaign() ComplexityCampaign {
	c := ComplexityCampaign{Schema: ComplexityCrossoverSchema, CampaignVersion: "test-v1", Model: "model", Runtime: "runtime", Tokenizer: "tokenizer", CachePosture: "cold/warm", SourceReceipt: "receipt", PromotionEvidence: "repeat", DemotionEvidence: "failure", InvalidatingAssumption: "telemetry"}
	for n := 1; n <= 6; n++ {
		r := ComplexityRung{Number: n, Name: "rung", DependencyDepth: n - 1, Task: "task", FrozenCheck: "check"}
		for _, w := range []int{1, 2, 4, 8} {
			for _, cache := range []string{"cold", "warm"} {
				for _, ctx := range []string{"full", "scoped"} {
					r.Cells = append(r.Cells, ComplexityCell{Width: w, Cache: cache, Context: ctx, Accepted: true, OutcomeDigest: "same", InputTokens: 100, CachedTokens: 10, SourceReceipt: "receipt"})
				}
			}
		}
		c.Rungs = append(c.Rungs, r)
	}
	return c
}
