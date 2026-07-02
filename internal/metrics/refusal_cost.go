package metrics

import (
	"sort"
	"strings"
)

// RefusalClearanceObservation is one historical refusal outcome for a closed
// refusal reason. Cleared=false means the agent escalated, abandoned the run, or
// otherwise failed to recover from that reason inside the observed window.
type RefusalClearanceObservation struct {
	Reason       string `json:"reason"`
	Cleared      bool   `json:"cleared"`
	TurnsToClear int    `json:"turns_to_clear,omitempty"`
}

// RefusalCostEnvelope is the agent-facing inline readout carried beside a
// refusal. The reason/fix come from the guard/DOS refusal table; the two cost
// fields come from measured recovery telemetry.
type RefusalCostEnvelope struct {
	Reason             string  `json:"reason"`
	Fix                string  `json:"fix"`
	MedianTurnsToClear float64 `json:"median_turns_to_clear"`
	RecoveryRate       float64 `json:"recovery_rate"`
	Samples            int     `json:"samples"`
	Recovered          int     `json:"recovered"`
}

// FoldRefusalCost folds historical observations for one reason into the compact
// envelope an agent can use for recover-vs-escalate decisions. Observations for
// other reasons are ignored. Negative turn counts are treated as not recovered.
func FoldRefusalCost(reason, fix string, observations []RefusalClearanceObservation) RefusalCostEnvelope {
	token := normalizeRefusalReason(reason)
	out := RefusalCostEnvelope{
		Reason: token,
		Fix:    strings.TrimSpace(fix),
	}
	var recoveredTurns []int
	for _, observation := range observations {
		if normalizeRefusalReason(observation.Reason) != token {
			continue
		}
		out.Samples++
		if observation.Cleared && observation.TurnsToClear >= 0 {
			out.Recovered++
			recoveredTurns = append(recoveredTurns, observation.TurnsToClear)
		}
	}
	if out.Samples > 0 {
		out.RecoveryRate = float64(out.Recovered) / float64(out.Samples)
	}
	out.MedianTurnsToClear = medianTurns(recoveredTurns)
	return out
}

// RecommendRecovery returns true when the measured cost envelope is good enough
// for an agent to keep recovering locally instead of escalating immediately.
func (e RefusalCostEnvelope) RecommendRecovery(maxMedianTurns float64, minRecoveryRate float64) bool {
	if e.Samples == 0 || e.Recovered == 0 {
		return false
	}
	return e.MedianTurnsToClear <= maxMedianTurns && e.RecoveryRate >= minRecoveryRate
}

func normalizeRefusalReason(reason string) string {
	reason = strings.TrimSpace(reason)
	reason = strings.ReplaceAll(reason, "-", "_")
	reason = strings.ReplaceAll(reason, " ", "_")
	return strings.ToUpper(reason)
}

func medianTurns(turns []int) float64 {
	if len(turns) == 0 {
		return 0
	}
	ordered := append([]int(nil), turns...)
	sort.Ints(ordered)
	mid := len(ordered) / 2
	if len(ordered)%2 == 1 {
		return float64(ordered[mid])
	}
	return float64(ordered[mid-1]+ordered[mid]) / 2
}
