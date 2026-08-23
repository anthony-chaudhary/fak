package metrics

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/valuechain"
)

// ValueChainHealth folds an audit witness into adoption, evidence-failure, and economic-drift signals.
type ValueChainHealth struct {
	Grade         string
	Candidate     string
	Sessions      int
	Turns         int64
	FailureRate   float64
	CostDriftPct  *float64
	NamedEvidence []string
}

// ScoreValueChainHealth grades the non-default arm of a value-chain audit.
func ScoreValueChainHealth(report valuechain.Report) ValueChainHealth {
	health := ValueChainHealth{Grade: "F"}
	if report.Comparison == nil {
		health.NamedEvidence = []string{
			"adoption: no candidate comparison witness",
			"failure_rate: no candidate billing witness",
			"drift: no paired cost-per-turn witness",
		}
		return health
	}

	health.Candidate = report.Comparison.Candidate
	health.CostDriftPct = report.Comparison.CostPerTurnDeltaPct
	for _, arm := range report.Arms {
		if arm.Arm != health.Candidate {
			continue
		}
		health.Sessions = arm.Sessions
		health.Turns = arm.Turns
		if arm.BillingEvidence.Total > 0 {
			health.FailureRate = 1 - arm.BillingEvidence.Ratio
		}
		break
	}

	drift := "unknown"
	if health.CostDriftPct != nil {
		drift = fmt.Sprintf("%+.2f%%", *health.CostDriftPct)
	}
	health.NamedEvidence = []string{
		fmt.Sprintf("adoption: candidate=%s sessions=%d turns=%d", health.Candidate, health.Sessions, health.Turns),
		fmt.Sprintf("failure_rate: uncovered_candidate_turns=%.2f%%", health.FailureRate*100),
		fmt.Sprintf("drift: paired_cost_per_turn_delta=%s", drift),
	}

	switch {
	case health.Sessions == 0 || health.Turns == 0:
		health.Grade = "F"
	case health.FailureRate > 0 || (health.CostDriftPct != nil && *health.CostDriftPct > 0):
		health.Grade = "C"
	case health.CostDriftPct == nil:
		health.Grade = "B"
	default:
		health.Grade = "A"
	}
	return health
}

// RenderValueChainHealth emits the compact operator witness used by CI.
func RenderValueChainHealth(health ValueChainHealth) string {
	return fmt.Sprintf("Vertical value-chain health: %s\n- %s\n- %s\n- %s\n",
		health.Grade,
		health.NamedEvidence[0],
		health.NamedEvidence[1],
		health.NamedEvidence[2],
	)
}
