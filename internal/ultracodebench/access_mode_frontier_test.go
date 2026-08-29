package ultracodebench

import "testing"

func TestEvaluateAccessModeFrontierMapsCompleteMatrix(t *testing.T) {
	report, err := EvaluateAccessModeFrontier(AccessModeFrontierFixture(), []int{1, 2, 4, 8})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Artifacts) != 2 {
		t.Fatalf("artifacts=%d", len(report.Artifacts))
	}
	for _, artifact := range report.Artifacts {
		if len(artifact.Cells) != 12 { //boundarylint:ignore CHANGE_DETECTOR_TEST the access-frontier artifact schema contains exactly twelve required cells
			t.Fatalf("%s cells=%d", artifact.EvidenceKind, len(artifact.Cells))
		}
	}
	verdict := func(artifact, mode string, width int) AccessModeCellResult {
		for _, a := range report.Artifacts {
			if a.EvidenceKind == artifact {
				for _, c := range a.Cells {
					if c.Mode == mode && c.Width == width {
						return c
					}
				}
			}
		}
		t.Fatalf("missing %s %s width %d", artifact, mode, width)
		return AccessModeCellResult{}
	}
	if got := verdict("offline_fixture", "scout_plus_writer", 2); got.Verdict != "GAIN" || got.CoordinationMS <= 0 || got.BilledTokens <= 0 {
		t.Fatalf("scout2=%+v", got)
	}
	if got := verdict("offline_fixture", "multi_writer", 4); got.Verdict != "ABSTAIN" || got.Reasons[0] != "unequal_or_unaccepted_outcome" {
		t.Fatalf("writer4=%+v", got)
	}
	if got := verdict("scrubbed_live_artifact", "scout_plus_writer", 8); got.Verdict != "ABSTAIN" || got.Reasons[0] != "missing_or_unobserved_telemetry" {
		t.Fatalf("live=%+v", got)
	}
}

func TestEvaluateAccessModeFrontierDoesNotTreatMissingAsZero(t *testing.T) {
	input := AccessModeFrontierFixture()
	input.Artifacts[0].Cells[4].Metrics.LeaseWaitMS = OptionalInt64{}
	report, err := EvaluateAccessModeFrontier(input, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	if got := report.Artifacts[0].Cells[1]; got.Verdict != "ABSTAIN" || got.Metrics.LeaseWaitMS.Observed {
		t.Fatalf("cell=%+v", got)
	}
}

func TestEvaluateAccessModeFrontierNoGainIncludesCoordinationTax(t *testing.T) {
	input := AccessModeFrontierFixture()
	for i := range input.Artifacts[0].Cells {
		c := &input.Artifacts[0].Cells[i]
		if c.Mode == "scout_plus_writer" && c.Width == 2 {
			c.Metrics.ReconcileMS = observed(5000)
		}
	}
	report, err := EvaluateAccessModeFrontier(input, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	if got := report.Artifacts[0].Cells[1]; got.Verdict != "NO_GAIN" || got.CoordinationMS != 5770 {
		t.Fatalf("cell=%+v", got)
	}
}
