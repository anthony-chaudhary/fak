package modelroute

import "strings"

const (
	CapacityReasonModelCeiling  = "MODEL_CAPACITY_CEILING"
	CapacityReasonContextWindow = "MODEL_CONTEXT_WINDOW_CEILING"
)

type CapacitySignal struct {
	Blocked            bool    `json:"blocked"`
	Reason             string  `json:"reason,omitempty"`
	RequiredModelB     float64 `json:"required_model_b,omitempty"`
	LocalModelCeilingB float64 `json:"local_model_ceiling_b,omitempty"`
	RequiredContext    int     `json:"required_context,omitempty"`
	UsableContext      int     `json:"usable_context,omitempty"`
}

type CapacityTarget struct {
	Name      string  `json:"name"`
	Model     string  `json:"model"`
	Pool      string  `json:"pool"`
	ModelB    float64 `json:"model_b,omitempty"`
	Context   int     `json:"context,omitempty"`
	Available bool    `json:"available"`
}
type CapacityReroute struct {
	Rerouted bool           `json:"rerouted"`
	Reason   string         `json:"reason,omitempty"`
	From     string         `json:"from,omitempty"`
	To       CapacityTarget `json:"to,omitempty"`
	Score    int            `json:"score,omitempty"`
}

// RerouteCapacity selects the smallest available target satisfying the blocked
// dimension, preferring a fleet GPU pool on exact score ties.
func RerouteCapacity(from string, signal CapacitySignal, targets []CapacityTarget) CapacityReroute {
	if !signal.Blocked {
		return CapacityReroute{}
	}
	best := -1
	bestScore := int(^uint(0) >> 1)
	for i, t := range targets {
		if !t.Available || t.Name == "" || t.Name == from {
			continue
		}
		if signal.Reason == CapacityReasonModelCeiling && t.ModelB < signal.RequiredModelB {
			continue
		}
		if signal.Reason == CapacityReasonContextWindow && t.Context < signal.RequiredContext {
			continue
		}
		score := 0
		if signal.Reason == CapacityReasonModelCeiling {
			score = int(t.ModelB * 100)
		} else {
			score = t.Context
		}
		if strings.Contains(strings.ToLower(t.Pool), "gpu") {
			score--
		}
		if score < bestScore {
			best, bestScore = i, score
		}
	}
	if best < 0 {
		return CapacityReroute{Reason: signal.Reason, From: from}
	}
	return CapacityReroute{Rerouted: true, Reason: signal.Reason, From: from, To: targets[best], Score: bestScore}
}
