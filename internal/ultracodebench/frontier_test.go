package ultracodebench

import "testing"

func TestEvaluateAccessFrontierDecomposesSavingsAndStopsHillClimb(t *testing.T) {
	r, err := EvaluateAccessFrontier(AccessFrontierFixture(), []int{1, 2, 4, 8})
	if err != nil {
		t.Fatal(err)
	}
	var scout4, writer4 AccessCellResult
	for _, c := range r.Cells {
		if c.Mode == "scout_writer" && c.Width == 4 {
			scout4 = c
		}
		if c.Mode == "multi_writer" && c.Width == 4 {
			writer4 = c
		}
	}
	if scout4.Verdict != "GAIN" || scout4.ScopeAvoidedTokens != 27000 || scout4.CacheAvoidedTokens != 11000 || scout4.TotalAvoidedTokens != 38000 {
		t.Fatalf("scout width 4 = %+v", scout4)
	}
	if writer4.Verdict != "ABSTAIN" || writer4.TotalAvoidedTokens != 0 {
		t.Fatalf("writer width 4 = %+v", writer4)
	}
	for _, h := range r.HillClimb {
		if h.Mode == "multi_writer" && (h.ChosenWidth != 2 || h.StopWidth != 4) {
			t.Fatalf("multi-writer climb = %+v", h)
		}
	}
}

func TestEvaluateAccessFrontierRejectsMissingTelemetry(t *testing.T) {
	f := AccessFrontierFixture()
	f.Cells[1].ScopedContextInputTokens = 0
	r, err := EvaluateAccessFrontier(f, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	if r.Cells[1].Verdict != "ABSTAIN" {
		t.Fatalf("cell = %+v", r.Cells[1])
	}
}

func TestEvaluateAccessFrontierRequiresObservedSource(t *testing.T) {
	f := AccessFrontierFixture()
	f.EvidenceKind = "observed_run"
	f.SourceArtifact = ""
	if _, err := EvaluateAccessFrontier(f, []int{1}); err == nil {
		t.Fatal("expected observed source error")
	}
}
