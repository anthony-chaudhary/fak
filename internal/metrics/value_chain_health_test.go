package metrics

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/valuechain"
)

func TestValueChainHealthCapturedRealData(t *testing.T) {
	manifest, observations, err := valuechain.Read(
		filepath.Join("..", "..", "examples", "value-chain", "support-manifest.json"),
		filepath.Join("..", "..", "examples", "value-chain", "support-observations.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	report, err := valuechain.Audit(manifest, observations)
	if err != nil {
		t.Fatal(err)
	}

	got := RenderValueChainHealth(ScoreValueChainHealth(report))
	const want = "Vertical value-chain health: A\n" +
		"- invocation_outcomes: success=4 refusal=0 error=0\n" +
		"- adoption: candidate=shared sessions=2 turns=10\n" +
		"- failure_rate: uncovered_candidate_turns=0.00%\n" +
		"- drift: paired_cost_per_turn_delta=-40.00%\n"
	if got != want {
		t.Fatalf("captured scorecard output mismatch\n--- got ---\n%s--- want ---\n%s", got, want)
	}
}

func TestValueChainHealthGradesMissingAndRegressedWitnesses(t *testing.T) {
	if got := ScoreValueChainHealth(valuechain.Report{}); got.Grade != "F" {
		t.Fatalf("missing witness grade = %q, want F", got.Grade)
	}

	drift := 5.0
	report := valuechain.Report{
		Arms: []valuechain.ArmReport{{
			Arm:      "candidate",
			Sessions: 1,
			Turns:    4,
			BillingEvidence: valuechain.Coverage{
				Covered: 3,
				Total:   4,
				Ratio:   0.75,
			},
		}},
		Comparison: &valuechain.Comparison{
			Candidate:           "candidate",
			CostPerTurnDeltaPct: &drift,
		},
	}
	got := ScoreValueChainHealth(report)
	if got.Grade != "C" || got.FailureRate != 0.25 {
		t.Fatalf("regressed witness = grade %q failure %.2f, want C and 0.25", got.Grade, got.FailureRate)
	}
}
